"""F&O Agent: deterministic strategy signal + OpenAI tool loop for mock option choice (no live orders)."""
import calendar
from datetime import date
import html
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_claw.constants import (
    DEFAULT_INTERVALS,
    DEFAULT_INTRADAY_INTERVAL,
    ENVELOPE_EMA_PERIOD,
    ENVELOPE_PCT,
    FO_BROKERAGE_PER_LOT_RT_DEFAULT,
    FO_CLOSED_AT_REALISED,
    FO_ENVELOPE_BANDWIDTH_MAX_PCT,
    FO_ENVELOPE_BANDWIDTH_MIN_PCT,
    FO_ENVELOPE_BANDWIDTH_STEP,
    FO_INDEX_UNDERLYING_LABELS,
    FO_OPTION_STOP_LOSS_PCT,
    FO_OPTION_TARGET_PCT,
    FO_STRATEGY_ENVELOPE,
    FO_STRATEGY_MA_CROSS,
    FO_STRATEGY_OPTIONS,
    FO_TAXES_PER_LOT_RT_DEFAULT,
    FO_UNDERLYING_OPTIONS,
    MA_EMA_FAST,
    MA_EMA_SLOW,
    NFO_EXCHANGE,
    NSE_EXCHANGE,
)
from trade_claw.fo_openai_agent import configure_fo_agent_logging, get_openai_api_key, run_fo_agent_pipeline
from trade_claw.kite_session import get_kite_credentials
from trade_claw.pl_style import pl_markdown, pl_title_color
from trade_claw.strategies import add_ma_ema_line_traces, add_ma_envelope_line_traces

_default_model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")


def _foa_underlying_select_label(u: str) -> str:
    if u in FO_INDEX_UNDERLYING_LABELS:
        return f"{FO_INDEX_UNDERLYING_LABELS[u]} — `{u}`"
    return u


def _abbrev_rupee(x: float) -> str:
    if x == 0:
        return "₹0"
    sgn = "-" if x < 0 else ""
    v = abs(float(x))
    if v >= 1e7:
        return f"{sgn}₹{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"{sgn}₹{v / 1e5:.2f} L"
    if v >= 1e3:
        return f"{sgn}₹{v / 1e3:.1f} K"
    return f"{sgn}₹{v:,.0f}"


def _abbrev_pl_html(x: float) -> str:
    full = html.escape(f"₹{x:+,.2f}")
    short = html.escape(_abbrev_rupee(x))
    if x > 0:
        col = "#15803d"
    elif x < 0:
        col = "#b91c1c"
    else:
        col = "#64748b"
    return f'<span title="{full}" style="color:{col};font-weight:600">{short}</span>'


def _resolve_openai_key() -> str | None:
    k = get_openai_api_key()
    if k:
        return k.strip()
    try:
        s = st.secrets.get("OPENAI_API_KEY") if getattr(st, "secrets", None) else None
        if s:
            return str(s).strip()
    except Exception:
        pass
    return None


def render_fo_agent_options(kite):
    configure_fo_agent_logging()

    if st.session_state.nse_instruments is None:
        with st.spinner("Loading NSE instruments..."):
            try:
                st.session_state.nse_instruments = kite.instruments(NSE_EXCHANGE)
            except Exception as e:
                st.error(f"Failed to load NSE instruments: {e}")
                st.stop()
    if st.session_state.get("nfo_instruments") is None:
        with st.spinner("Loading NFO instruments (options master)..."):
            try:
                st.session_state.nfo_instruments = kite.instruments(NFO_EXCHANGE)
            except Exception as e:
                st.error(f"Failed to load NFO instruments: {e}")
                st.stop()

    nse = st.session_state.nse_instruments
    nfo = st.session_state.nfo_instruments
    symbol_to_name = {i["tradingsymbol"]: i.get("name", i["tradingsymbol"]) for i in nse}

    st.title("F&O Agent (OpenAI + mock trades only)")
    today = date.today()
    _, _last_dom = calendar.monthrange(today.year, today.month)
    month_start = date(today.year, today.month, 1)
    month_end = date(today.year, today.month, _last_dom)
    session_upper = min(today, month_end)

    st.info(
        "**Mock backtest only.** The model only sees **`search_instruments`**, **`get_historical_data`**, "
        "and **`submit_mock_trade_choice`** (no `place_order` / GTT / other write tools).\n\n"
        "When **`KITE_MCP_ENABLED=1`** (or `KITE_MCP_STREAMABLE_URL` / `KITE_MCP_COMMAND` is set), "
        "the two read tools call the **Zerodha Kite MCP server**. "
        "Otherwise the same parameters run in-process via KiteConnect. "
        "`submit_mock_trade_choice` is always app-side only (mock).\n\n"
        f"**Current month:** {month_start.isoformat()} to {session_upper.isoformat()} (session date cannot exceed today)."
    )

    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        if st.button("Intraday home", key="foa_nav_all10"):
            st.session_state.view = "all10"
            st.session_state.selected_symbol = None
            st.rerun()
    with nav2:
        if st.button("Stock list", key="foa_nav_dash"):
            st.session_state.view = "dashboard"
            st.session_state.selected_symbol = None
            st.rerun()
    with nav3:
        if st.button("F&O Options (rules)", key="foa_nav_fo"):
            st.session_state.view = "fo_options"
            st.session_state.selected_symbol = None
            st.rerun()
    with nav4:
        if st.button("Reports", key="foa_nav_rep"):
            st.session_state.view = "reports"
            st.session_state.selected_symbol = None
            st.rerun()

    _ndef = "NIFTY" if "NIFTY" in FO_UNDERLYING_OPTIONS else FO_UNDERLYING_OPTIONS[0]
    _u_idx = FO_UNDERLYING_OPTIONS.index(_ndef) if _ndef in FO_UNDERLYING_OPTIONS else 0
    underlying = st.selectbox(
        "Underlying (index or Nifty 50 stock)",
        options=FO_UNDERLYING_OPTIONS,
        index=_u_idx,
        key="foa_underlying",
        format_func=_foa_underlying_select_label,
    )

    strategy_choice = st.selectbox(
        "Underlying strategy",
        options=FO_STRATEGY_OPTIONS,
        index=FO_STRATEGY_OPTIONS.index(FO_STRATEGY_ENVELOPE),
        key="foa_strategy_choice",
    )
    strategy_is_envelope = strategy_choice == FO_STRATEGY_ENVELOPE

    intraday_intervals = [i for i in DEFAULT_INTERVALS if i != "day"]
    chosen_date = st.date_input(
        "Session date (this month only, not after today)",
        value=session_upper,
        min_value=month_start,
        max_value=session_upper,
        key="foa_date",
    )
    _int_idx = (
        intraday_intervals.index(DEFAULT_INTRADAY_INTERVAL)
        if DEFAULT_INTRADAY_INTERVAL in intraday_intervals
        else 0
    )
    chosen_interval = st.selectbox(
        "Minute interval",
        options=intraday_intervals,
        index=_int_idx,
        key="foa_interval",
    )

    if strategy_is_envelope:
        envelope_bw_pct = st.slider(
            "Envelope bandwidth (% each side of EMA)",
            min_value=float(FO_ENVELOPE_BANDWIDTH_MIN_PCT),
            max_value=float(FO_ENVELOPE_BANDWIDTH_MAX_PCT),
            value=float(round(100 * ENVELOPE_PCT, 4)),
            step=float(FO_ENVELOPE_BANDWIDTH_STEP),
            key="foa_envelope_bandwidth_pct",
        )
        envelope_pct = envelope_bw_pct / 100.0
    else:
        envelope_bw_pct = float(round(100 * ENVELOPE_PCT, 4))
        envelope_pct = ENVELOPE_PCT
        st.caption(
            f"EMA crossover uses **fast {MA_EMA_FAST}** / **slow {MA_EMA_SLOW}**."
        )

    bc1, bc2 = st.columns(2)
    with bc1:
        brokerage_per_lot_rt = st.number_input(
            "Brokerage ₹ / lot (round trip)",
            min_value=0.0,
            value=float(FO_BROKERAGE_PER_LOT_RT_DEFAULT),
            step=5.0,
            key="foa_brokerage_per_lot",
        )
    with bc2:
        taxes_per_lot_rt = st.number_input(
            "Taxes & charges ₹ / lot (round trip)",
            min_value=0.0,
            value=float(FO_TAXES_PER_LOT_RT_DEFAULT),
            step=5.0,
            key="foa_taxes_per_lot",
        )

    st.caption(
        f"OpenAI model: **`{_default_model}`** (override with `OPENAI_MODEL` in `.env`). "
        "Set **`OPENAI_API_KEY`** in `.env` or Streamlit secrets."
    )

    api_key = _resolve_openai_key()
    if not api_key:
        st.warning("Add **OPENAI_API_KEY** to `.env` or Streamlit secrets to run the agent.")

    load = st.button("Load / run agent", type="primary", key="foa_load")

    if underlying in FO_INDEX_UNDERLYING_LABELS:
        name = str(FO_INDEX_UNDERLYING_LABELS[underlying]).split("(")[0].strip()
    else:
        name = symbol_to_name.get(underlying, underlying)

    if load and api_key:
        k_api, _k_sec = get_kite_credentials()
        acc_tok = st.session_state.get("access_token")
        with st.spinner("Running deterministic strategy + OpenAI agent (mock only)..."):
            try:
                result = run_fo_agent_pipeline(
                    kite=kite,
                    nse_instruments=nse,
                    nfo_instruments=nfo,
                    underlying=underlying,
                    name=name,
                    session_date=chosen_date,
                    chosen_interval=chosen_interval,
                    strategy_is_envelope=strategy_is_envelope,
                    envelope_pct=envelope_pct,
                    strategy_choice=strategy_choice,
                    brokerage_per_lot_rt=float(brokerage_per_lot_rt),
                    taxes_per_lot_rt=float(taxes_per_lot_rt),
                    openai_api_key=api_key,
                    kite_api_key=str(k_api).strip() if k_api else None,
                    kite_access_token=str(acc_tok).strip() if acc_tok else None,
                )
                st.session_state["fo_agent_last_result"] = result
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.session_state["fo_agent_last_result"] = None
        st.rerun()

    res = st.session_state.get("fo_agent_last_result")

    if res is None:
        st.caption("Click **Load / run agent** after configuring inputs.")
        return

    # ---- Trace (no secrets) ----
    with st.expander("Agent trace (tool calls)", expanded=False):
        for i, step in enumerate(res.trace):
            st.json({"index": i, **step})

    if not res.success:
        st.error(res.error_message or "Failed")
        intent = res.intent or {}
        df_u = intent.get("df_u")
        if df_u is not None and not getattr(df_u, "empty", True):
            st.markdown("### Underlying chart (strategy lines)")
            sk_render = "envelope" if strategy_is_envelope else "ma"
            fig_u = go.Figure()
            fig_u.add_trace(
                go.Candlestick(
                    x=df_u["date"],
                    open=df_u["open"],
                    high=df_u["high"],
                    low=df_u["low"],
                    close=df_u["close"],
                    name="Underlying",
                )
            )
            if sk_render == "envelope":
                add_ma_envelope_line_traces(fig_u, df_u, ema_period=ENVELOPE_EMA_PERIOD, pct=envelope_pct)
            else:
                add_ma_ema_line_traces(fig_u, df_u)
            fig_u.update_layout(title="Underlying", height=320, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_u, use_container_width=True)
        if intent.get("analysis_text"):
            st.markdown("**Strategy analysis**")
            st.caption(str(intent.get("analysis_text"))[:3000])
        if res.final_rationale:
            st.markdown("**Agent rationale (partial)**")
            st.write(res.final_rationale[:4000])
        return

    row = res.row
    t = row.get("trade")
    df_u = row.get("df_u")
    df_o = row.get("df_o")

    st.markdown("### Agent rationale")
    st.markdown((res.final_rationale or "—").strip() or "—")

    if t:
        total_trades = 1
        total_finished = 1 if t.get("Closed at") in FO_CLOSED_AT_REALISED else 0
        realised_pl = t["P/L"] if t.get("Closed at") in FO_CLOSED_AT_REALISED else 0.0
        unrealised_pl = t["P/L"] if t.get("Closed at") == "EOD" else 0.0
        total_pl = t["P/L"]
        total_traded_value = t.get("Value", 0.0)
        total_txn_cost = t.get("Txn cost", 0.0)

        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        with m1:
            st.metric("Option trades", f"{total_trades:,}")
        with m2:
            _nt = 1 if t.get("Closed at") == "Target" else 0
            _ns = 1 if t.get("Closed at") == "Stop" else 0
            st.metric(
                "Target / stop (realised)",
                f"{total_finished:,}",
                help=f"1 if closed at target or stop ({_nt}T/{_ns}S); 0 if EOD only.",
            )
        with m3:
            st.markdown(
                f"**Realised P/L (net)**<br/>{_abbrev_pl_html(realised_pl)}",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"**EOD P/L (net)**<br/>{_abbrev_pl_html(unrealised_pl)}",
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f"**Total P/L (net)**<br/>{_abbrev_pl_html(total_pl)}",
                unsafe_allow_html=True,
            )
        with m6:
            st.metric("Premium deployed", _abbrev_rupee(total_traded_value))
        with m7:
            st.metric("Txn brk+tax", _abbrev_rupee(total_txn_cost))

        st.markdown(f"**Net P/L:** {pl_markdown(total_pl)}")
        _sl_txt = (
            f"stop −{100 * FO_OPTION_STOP_LOSS_PCT:.0f}% on bar low (same-bar: target wins)"
            if FO_OPTION_STOP_LOSS_PCT > 0
            else "no stop (set FO_OPTION_STOP_LOSS_PCT>0 in env)"
        )
        st.caption(
            f"Exit rule: target +{100 * FO_OPTION_TARGET_PCT:.0f}% on premium high; {_sl_txt}; else EOD. "
            "1 lot; costs per lot as entered."
        )

    st.divider()
    st.markdown(f"### Charts — **{underlying}** on **{chosen_date.isoformat()}**")
    sk_render = "envelope" if strategy_is_envelope else "ma"
    if df_u is not None and not df_u.empty:
        fig_u = go.Figure()
        fig_u.add_trace(
            go.Candlestick(
                x=df_u["date"],
                open=df_u["open"],
                high=df_u["high"],
                low=df_u["low"],
                close=df_u["close"],
                name="Underlying",
            )
        )
        if sk_render == "envelope":
            add_ma_envelope_line_traces(fig_u, df_u, ema_period=ENVELOPE_EMA_PERIOD, pct=envelope_pct)
            spot_title = "Spot (envelope)"
        else:
            add_ma_ema_line_traces(fig_u, df_u)
            spot_title = f"Spot (EMA {MA_EMA_FAST}/{MA_EMA_SLOW})"
        _ei_raw = (t.get("entry_bar_idx") if t else None) or row.get("_chart_entry_bar_idx")
        try:
            ei = int(_ei_raw) if _ei_raw is not None else None
        except (TypeError, ValueError):
            ei = None
        sig_m = (t.get("Signal") if t else None) or row.get("_chart_spot_signal")
        du = df_u["date"]
        if ei is not None and sig_m in ("BUY", "SELL") and 0 <= ei < len(du):
            fig_u.add_trace(
                go.Scatter(
                    x=[du.iloc[ei]],
                    y=[float(df_u.iloc[ei]["close"])],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up" if sig_m == "BUY" else "triangle-down",
                        size=14,
                        color="lime" if sig_m == "BUY" else "tomato",
                        line=dict(width=1, color="black"),
                    ),
                    name=sig_m,
                )
            )
        fig_u.update_layout(title=spot_title, height=360, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_u, use_container_width=True)

    if t and df_o is not None and not df_o.empty:
        st.markdown("**Option premium (mock exit)**")
        fig_o = go.Figure()
        fig_o.add_trace(
            go.Candlestick(
                x=df_o["date"],
                open=df_o["open"],
                high=df_o["high"],
                low=df_o["low"],
                close=df_o["close"],
                name=t["Option"],
            )
        )
        oei = t.get("opt_entry_idx")
        oxi = t.get("exit_bar_idx")
        ddt = df_o["date"]
        if oei is not None and 0 <= oei < len(ddt):
            fig_o.add_trace(
                go.Scatter(
                    x=[ddt.iloc[oei]],
                    y=[t["Entry"]],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="cyan", line=dict(width=1, color="black")),
                    name="Entry",
                )
            )
        if oxi is not None and 0 <= oxi < len(ddt):
            fig_o.add_trace(
                go.Scatter(
                    x=[ddt.iloc[oxi]],
                    y=[t["Exit"]],
                    mode="markers",
                    marker=dict(symbol="diamond", size=10, color="gold", line=dict(width=1, color="orange")),
                    name="Exit",
                )
            )
        _net = float(t["P/L"])
        _gross = float(t.get("P/L gross", _net))
        _txn = float(t.get("Txn cost", 0.0))
        fig_o.update_layout(
            title=dict(
                text=(
                    f"{t['Leg']} @ {t.get('Strike', 0):,.0f} · Net ₹{_net:+,.2f} "
                    f"(gross ₹{_gross:+,.2f} − txn ₹{_txn:,.0f}) · {t['Closed at']}"
                ),
                font=dict(color=pl_title_color(_net), size=13),
            ),
            height=360,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig_o, use_container_width=True)

    st.markdown("### Summary row")
    disp = pd.DataFrame(
        [
            {
                "Session date": row.get("Session date"),
                "Option": row.get("Option"),
                "Strike": row.get("Strike"),
                "Signal": row.get("Signal"),
                "Closed at": row.get("Closed at") if t else "—",
                "P/L": row.get("P/L") if t else "—",
            }
        ]
    )
    st.dataframe(disp, use_container_width=True, hide_index=True)
