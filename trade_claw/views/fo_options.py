"""F&O: underlying strategy (envelope or EMA cross) → long CE/PE; premium target or EOD; lot costs."""
import calendar
from datetime import date, datetime
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_claw.constants import (
    DEFAULT_INTERVALS,
    ENVELOPE_EMA_PERIOD,
    ENVELOPE_PCT,
    FO_BROKERAGE_PER_LOT_RT_DEFAULT,
    FO_CLOSED_AT_REALISED,
    FO_ENVELOPE_BANDWIDTH_MAX_PCT,
    FO_ENVELOPE_BANDWIDTH_MIN_PCT,
    FO_ENVELOPE_BANDWIDTH_STEP,
    FO_OPTION_TARGET_PCT,
    FO_STRATEGY_ENVELOPE,
    FO_STRATEGY_MA_CROSS,
    FO_STRATEGY_OPTIONS,
    FO_STRIKE_POLICY_LABELS,
    FO_STRIKE_POLICY_STEPS,
    FO_TAXES_PER_LOT_RT_DEFAULT,
    FO_UNDERLYING_OPTIONS,
    MA_EMA_FAST,
    MA_EMA_SLOW,
    NFO_EXCHANGE,
    NSE_EXCHANGE,
)
from trade_claw.fo_runner import iter_calendar_dates_in_month, run_fo_underlying_one_day
from trade_claw.pl_style import pl_markdown, pl_title_color, style_pl_dataframe
from trade_claw.strategies import add_ma_ema_line_traces, add_ma_envelope_line_traces


def _abbrev_rupee(x: float) -> str:
    """Short rupee label for metric cards (hover shows full precision)."""
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
    """Abbreviated P/L with native hover (title) showing full ₹ precision."""
    full = html.escape(f"₹{x:+,.2f}")
    short = html.escape(_abbrev_rupee(x))
    if x > 0:
        col = "#15803d"
    elif x < 0:
        col = "#b91c1c"
    else:
        col = "#64748b"
    return f'<span title="{full}" style="color:{col};font-weight:600">{short}</span>'


def render_fo_options(kite):
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

    st.title("F&O options – intraday underlying signal (long premium only)")
    today = date.today()
    _, _last_dom = calendar.monthrange(today.year, today.month)
    month_start = date(today.year, today.month, 1)
    month_end = date(today.year, today.month, _last_dom)
    session_upper = min(today, month_end)

    st.info(
        f"**This page uses the current calendar month only** ({month_start.strftime('%b %d')} → "
        f"{month_end.strftime('%b %d, %Y')}), with session dates **not after today** "
        f"({session_upper.isoformat()}). "
        "Option contracts come from Kite’s **live NFO list** (nearest expiry on/after each session day); "
        "older months are disabled because expired series don’t match today’s instrument master."
    )
    st.caption(
        "**Envelope:** first **close** above upper / below lower band (not wick-only). "
        "→ long ATM **CE** / **PE**. **EMA cross:** first golden / death cross in session → **CE** / **PE**. "
        "No short options. Exit: premium **+"
        f"{100 * FO_OPTION_TARGET_PCT:.0f}%** vs bar **high**, else **EOD**. "
        "Mock size: **always 1 lot** per symbol (exchange `lot_size` × premium). "
        "P/L **net** = gross premium P/L − (brokerage + tax) × 1 lot."
    )

    nav1, nav2, nav3, nav4, nav5 = st.columns(5)
    with nav1:
        if st.button("Intraday home", key="fo_nav_all10"):
            st.session_state.view = "all10"
            st.session_state.selected_symbol = None
            st.rerun()
    with nav2:
        if st.button("Stock list", key="fo_nav_dash"):
            st.session_state.view = "dashboard"
            st.session_state.selected_symbol = None
            st.rerun()
    with nav3:
        if st.button("Reports", key="fo_nav_rep"):
            st.session_state.view = "reports"
            st.session_state.selected_symbol = None
            st.rerun()
    with nav4:
        if st.button("Index ETFs", key="fo_nav_etf"):
            st.session_state.view = "index_etfs"
            st.session_state.selected_symbol = None
            st.rerun()
    with nav5:
        if st.button("F&O Agent", key="fo_nav_fo_agent"):
            st.session_state.view = "fo_agent"
            st.session_state.selected_symbol = None
            st.rerun()

    _ndef = "NIFTY" if "NIFTY" in FO_UNDERLYING_OPTIONS else FO_UNDERLYING_OPTIONS[0]
    _u_idx = FO_UNDERLYING_OPTIONS.index(_ndef) if _ndef in FO_UNDERLYING_OPTIONS else 0
    underlying = st.selectbox(
        "Underlying (one scrip at a time)",
        options=FO_UNDERLYING_OPTIONS,
        index=_u_idx,
        key="fo_underlying_single",
    )
    st.caption(
        "The **month table** below replays the same rules on **each weekday from the 1st through today** "
        "(weekends and **future** dates in the month are skipped; holidays show as few/no bars)."
    )

    strategy_choice = st.selectbox(
        "Underlying strategy",
        options=FO_STRATEGY_OPTIONS,
        index=FO_STRATEGY_OPTIONS.index(FO_STRATEGY_ENVELOPE),
        key="fo_strategy_choice",
    )
    strategy_is_envelope = strategy_choice == FO_STRATEGY_ENVELOPE

    intraday_intervals = [i for i in DEFAULT_INTERVALS if i != "day"]
    chosen_date = st.date_input(
        "Session date (this month only, not after today)",
        value=session_upper,
        min_value=month_start,
        max_value=session_upper,
        key="fo_date",
    )
    chosen_interval = st.selectbox(
        "Minute interval",
        options=intraday_intervals,
        index=intraday_intervals.index("5minute"),
        key="fo_interval",
    )
    if strategy_is_envelope:
        envelope_bw_pct = st.slider(
            "Envelope bandwidth (% each side of EMA)",
            min_value=float(FO_ENVELOPE_BANDWIDTH_MIN_PCT),
            max_value=float(FO_ENVELOPE_BANDWIDTH_MAX_PCT),
            value=float(round(100 * ENVELOPE_PCT, 4)),
            step=float(FO_ENVELOPE_BANDWIDTH_STEP),
            key="fo_envelope_bandwidth_pct",
            help=(
                "Upper band = EMA × (1 + p), lower = EMA × (1 − p), with p = this % ÷ 100. "
                f"Intraday home default is {100 * ENVELOPE_PCT:.2f}%."
            ),
        )
        envelope_pct = envelope_bw_pct / 100.0
    else:
        envelope_bw_pct = float(round(100 * ENVELOPE_PCT, 4))
        envelope_pct = ENVELOPE_PCT
        st.caption(
            f"EMA crossover uses **fast {MA_EMA_FAST}** / **slow {MA_EMA_SLOW}** (same as intraday home). "
            "First cross in the session sets the signal."
        )

    bc1, bc2 = st.columns(2)
    with bc1:
        brokerage_per_lot_rt = st.number_input(
            "Brokerage ₹ / lot (round trip)",
            min_value=0.0,
            value=float(FO_BROKERAGE_PER_LOT_RT_DEFAULT),
            step=5.0,
            key="fo_brokerage_per_lot",
            help="Total brokerage you assign per lot for buy+sell combined.",
        )
    with bc2:
        taxes_per_lot_rt = st.number_input(
            "Taxes & charges ₹ / lot (round trip)",
            min_value=0.0,
            value=float(FO_TAXES_PER_LOT_RT_DEFAULT),
            step=5.0,
            key="fo_taxes_per_lot",
            help="STT, stamp, exchange—your lump sum per lot for the full round trip.",
        )

    strike_policy_label = st.selectbox(
        "Option strike vs spot",
        options=FO_STRIKE_POLICY_LABELS,
        index=0,
        key="fo_strike_policy",
        help=(
            "Chooses listed strike relative to spot for the traded leg (long CE on BUY, long PE on SELL). "
            "ATM = nearest exchange strike to spot (e.g. ~22k spot → ~22k strike). "
            "OTM/ITM steps move along the strike list for that expiry. "
            "Expiry is always **on or after the session day**, picked from Kite’s current NFO chain (nearest first)."
        ),
    )
    steps_from_atm = FO_STRIKE_POLICY_STEPS[FO_STRIKE_POLICY_LABELS.index(strike_policy_label)]

    manual_strike_raw = st.number_input(
        "Manual strike override (0 = use strike policy only)",
        min_value=0,
        max_value=10_000_000,
        step=50,
        value=0,
        key="fo_manstrike_single",
        help="When >0, nearest listed strike to this value is used for the selected underlying (session + month scan).",
    )
    manual_strike_val = float(manual_strike_raw) if manual_strike_raw and float(manual_strike_raw) > 0 else None

    name = symbol_to_name.get(underlying, underlying) if underlying not in ("NIFTY", "BANKNIFTY") else underlying

    rows_out: list[dict] = []
    month_rows: list[dict] = []

    with st.spinner(
        f"Running {strategy_choice} + options for **{underlying}** "
        f"(session + month-to-date {month_start:%Y-%m-%d} … {session_upper:%Y-%m-%d})..."
    ):
        rows_out.append(
            run_fo_underlying_one_day(
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
                steps_from_atm=steps_from_atm,
                strike_policy_label=strike_policy_label,
                manual_strike_val=manual_strike_val,
                brokerage_per_lot_rt=brokerage_per_lot_rt,
                taxes_per_lot_rt=taxes_per_lot_rt,
                include_chart_data=True,
            )
        )
        for d in iter_calendar_dates_in_month(today):
            if d > today:
                month_rows.append({
                    "Session date": d,
                    "Underlying": underlying,
                    "Name": name,
                    "Strategy": "—",
                    "Leg": "—",
                    "Option": "—",
                    "Strike": float("nan"),
                    "Strike pick": "—",
                    "Note": "Future date — not run",
                    "Signal": "—",
                    "Lots": 0,
                    "Lot size": float("nan"),
                    "Qty": 0,
                    "Entry": float("nan"),
                    "Target prem.": float("nan"),
                    "Txn cost": 0.0,
                    "P/L gross": 0.0,
                    "Closed at": "—",
                    "Exit": float("nan"),
                    "P/L": 0.0,
                    "Value": 0.0,
                    "df_u": None,
                    "df_o": None,
                    "trade": None,
                })
                continue
            if d.weekday() >= 5:
                month_rows.append({
                    "Session date": d,
                    "Underlying": underlying,
                    "Name": name,
                    "Strategy": "—",
                    "Leg": "—",
                    "Option": "—",
                    "Strike": float("nan"),
                    "Strike pick": "—",
                    "Note": "Weekend — not run",
                    "Signal": "—",
                    "Lots": 0,
                    "Lot size": float("nan"),
                    "Qty": 0,
                    "Entry": float("nan"),
                    "Target prem.": float("nan"),
                    "Txn cost": 0.0,
                    "P/L gross": 0.0,
                    "Closed at": "—",
                    "Exit": float("nan"),
                    "P/L": 0.0,
                    "Value": 0.0,
                    "df_u": None,
                    "df_o": None,
                    "trade": None,
                })
                continue
            month_rows.append(
                run_fo_underlying_one_day(
                    kite=kite,
                    nse_instruments=nse,
                    nfo_instruments=nfo,
                    underlying=underlying,
                    name=name,
                    session_date=d,
                    chosen_interval=chosen_interval,
                    strategy_is_envelope=strategy_is_envelope,
                    envelope_pct=envelope_pct,
                    strategy_choice=strategy_choice,
                    steps_from_atm=steps_from_atm,
                    strike_policy_label=strike_policy_label,
                    manual_strike_val=manual_strike_val,
                    brokerage_per_lot_rt=brokerage_per_lot_rt,
                    taxes_per_lot_rt=taxes_per_lot_rt,
                    include_chart_data=False,
                )
            )

    trade_rows = [r["trade"] for r in rows_out if r.get("trade")]
    total_trades = len(trade_rows)
    total_finished = sum(1 for t in trade_rows if t.get("Closed at") in FO_CLOSED_AT_REALISED)
    realised_pl = sum(t["P/L"] for t in trade_rows if t.get("Closed at") in FO_CLOSED_AT_REALISED)
    unrealised_pl = sum(t["P/L"] for t in trade_rows if t.get("Closed at") == "EOD")
    total_pl = realised_pl + unrealised_pl
    total_traded_value = sum(t["Value"] for t in trade_rows)
    total_txn_cost = sum(t.get("Txn cost", 0.0) for t in trade_rows)

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    with m1:
        st.metric(
            "Option trades",
            f"{total_trades:,}",
            help=f"{total_trades:,} executed mock option trade(s). Hover label for full count.",
        )
    with m2:
        st.metric(
            "Hit premium target",
            f"{total_finished:,}",
            help=f"{total_finished:,} trade(s) hit premium target (vs EOD).",
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
        st.metric(
            "Premium deployed",
            _abbrev_rupee(total_traded_value),
            help=f"Full: ₹{total_traded_value:,.2f}",
        )
    with m7:
        st.metric(
            "Txn brk+tax",
            _abbrev_rupee(total_txn_cost),
            help=f"Full: ₹{total_txn_cost:,.2f}",
        )
    if strategy_is_envelope:
        st.caption(
            f"Strategy: **{FO_STRATEGY_ENVELOPE}** — EMA{ENVELOPE_EMA_PERIOD} ±{envelope_bw_pct:.2f}% · "
            f"option target = entry × (1 + {FO_OPTION_TARGET_PCT:.2f}). "
            f"Deduction = (brokerage + taxes) × **1 lot** per trade."
        )
    else:
        st.caption(
            f"Strategy: **{FO_STRATEGY_MA_CROSS}** — EMA {MA_EMA_FAST} / {MA_EMA_SLOW} · "
            f"option target = entry × (1 + {FO_OPTION_TARGET_PCT:.2f}). "
            f"Deduction = (brokerage + taxes) × **1 lot** per trade."
        )
    st.divider()

    st.markdown(f"### Charts — **{underlying}** on **{chosen_date.isoformat()}**")
    st.caption(
        "Envelope or EMA lines match your **Underlying strategy** dropdown. "
        "Option chart appears only when a mock option trade was built for the session date."
    )
    sk_render = "envelope" if strategy_is_envelope else "ma"
    for r in rows_out:
        df_u = r.get("df_u")
        if df_u is None:
            continue
        u = r["Underlying"]
        nm = r["Name"]
        t = r.get("trade")
        df_o = r.get("df_o")
        if t:
            _stk = t.get("Strike")
            _sk = ""
            if _stk is not None and not pd.isna(_stk):
                _sk = f" @{float(_stk):,.0f}"
            exp_sub = f"{t['Leg']}{_sk} `{t['Option']}`"
        else:
            note = str(r.get("Note", "No option trade"))
            exp_sub = (note[:72] + "…") if len(note) > 72 else note
        cap = (t.get("why") if t else None) or r.get("_analysis_text") or r.get("Note", "")
        with st.expander(f"**{u}** — {nm} · {exp_sub}", expanded=False):
            st.caption(cap)
            c1, c2 = st.columns(2)
            with c1:
                if sk_render == "envelope":
                    st.markdown("**Underlying + envelope**")
                    spot_title = "Spot (envelope)"
                else:
                    st.markdown("**Underlying + EMA crossover**")
                    spot_title = f"Spot (EMA {MA_EMA_FAST}/{MA_EMA_SLOW})"
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
                _ei_raw = (t.get("entry_bar_idx") if t else None) or r.get("_chart_entry_bar_idx")
                try:
                    ei = int(_ei_raw) if _ei_raw is not None else None
                except (TypeError, ValueError):
                    ei = None
                sig_m = (t.get("Signal") if t else None) or r.get("_chart_spot_signal")
                du = df_u["date"]
                if (
                    ei is not None
                    and sig_m in ("BUY", "SELL")
                    and 0 <= ei < len(du)
                ):
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
                fig_u.update_layout(
                    title=spot_title,
                    height=320,
                    xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig_u, use_container_width=True)
            with c2:
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
                        height=320,
                        xaxis_rangeslider_visible=False,
                    )
                    st.plotly_chart(fig_o, use_container_width=True)
                else:
                    st.markdown("**Option premium**")
                    st.info(
                        "No mock option trade for this session (no signal, data error, or below 1 lot). "
                        "See the summary table below for the exact reason."
                    )

    st.markdown("### Selected session (detail row)")
    display_cols = [
        "Session date",
        "Underlying",
        "Name",
        "Strategy",
        "Leg",
        "Option",
        "Strike",
        "Strike pick",
        "Note",
        "Signal",
        "Lots",
        "Lot size",
        "Qty",
        "Entry",
        "Target prem.",
        "Txn cost",
        "P/L gross",
        "Closed at",
        "Exit",
        "P/L",
        "Value",
    ]
    disp_rows = []
    for r in rows_out:
        row = {k: r[k] for k in display_cols if k in r}
        sd = row.get("Session date")
        if sd is not None and hasattr(sd, "isoformat"):
            row["Session date"] = sd.isoformat()
        disp_rows.append(row)
    disp = pd.DataFrame(disp_rows)
    if not disp.empty:
        fmt = {
            "Strike": "{:,.0f}",
            "Lots": "{:,.0f}",
            "Lot size": "{:,.0f}",
            "Qty": "{:,.0f}",
            "Entry": "₹{:,.2f}",
            "Target prem.": "₹{:,.2f}",
            "Txn cost": "₹{:,.2f}",
            "P/L gross": "₹{:+,.2f}",
            "Exit": "₹{:,.2f}",
            "P/L": "₹{:+,.2f}",
            "Value": "₹{:,.2f}",
        }
        styled = style_pl_dataframe(
            disp.style.format(fmt, na_rep="—"),
            "P/L",
            "P/L gross",
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    if trade_rows:
        _tpl = sum(t["P/L"] for t in trade_rows)
        st.markdown(f"**Session net P/L (if trade executed):** {pl_markdown(_tpl)}")

    st.divider()
    month_trade_rows = [r["trade"] for r in month_rows if r.get("trade")]
    month_n_trades = len(month_trade_rows)
    month_hit_target = sum(1 for t in month_trade_rows if t.get("Closed at") in FO_CLOSED_AT_REALISED)
    month_eod_exits = sum(1 for t in month_trade_rows if t.get("Closed at") == "EOD")
    month_realised_pl = sum(t["P/L"] for t in month_trade_rows if t.get("Closed at") in FO_CLOSED_AT_REALISED)
    month_eod_pl = sum(t["P/L"] for t in month_trade_rows if t.get("Closed at") == "EOD")
    month_total_pl = month_realised_pl + month_eod_pl
    month_txn_total = sum(t.get("Txn cost", 0.0) for t in month_trade_rows)
    _skip_notes = frozenset({"Weekend — not run", "Future date — not run"})
    month_weekdays = sum(1 for r in month_rows if r.get("Note") not in _skip_notes)

    st.markdown(f"### Current month (MTD) — **{today.strftime('%B %Y')}** · `{underlying}`")
    st.caption(
        f"Same strategy, interval, strike policy, brokerage/taxes, and manual strike as above. "
        f"**{month_weekdays}** weekday(s) from month start through **{session_upper.isoformat()}** "
        f"with a full run (weekends & future dates excluded)."
    )
    mx1, mx2, mx3, mx4, mx5, mx6 = st.columns(6)
    with mx1:
        st.metric("MTD trades", f"{month_n_trades:,}", help="Mock option trades with a filled position (month to date).")
    with mx2:
        st.metric("Target hit", f"{month_hit_target:,}", help="Exits on premium target (vs EOD).")
    with mx3:
        st.metric("EOD exits", f"{month_eod_exits:,}", help="Held to end of session.")
    with mx4:
        st.markdown(
            f"**MTD net P/L**<br/>{_abbrev_pl_html(month_total_pl)}",
            unsafe_allow_html=True,
        )
    with mx5:
        st.metric(
            "MTD txn (brk+tax)",
            _abbrev_rupee(month_txn_total),
            help=f"Full: ₹{month_txn_total:,.2f}",
        )
    with mx6:
        st.markdown(
            f"**Realised / EOD (net)**<br/>{_abbrev_pl_html(month_realised_pl)} / {_abbrev_pl_html(month_eod_pl)}",
            unsafe_allow_html=True,
        )

    month_disp_cols = [
        "Session date",
        "Note",
        "Signal",
        "Leg",
        "Option",
        "Strike",
        "Strike pick",
        "Closed at",
        "Txn cost",
        "P/L gross",
        "P/L",
    ]
    month_disp_rows = []
    for r in month_rows:
        row = {k: r[k] for k in month_disp_cols if k in r}
        sd = row.get("Session date")
        if sd is not None and hasattr(sd, "isoformat"):
            row["Session date"] = sd.isoformat()
        month_disp_rows.append(row)
    month_df = pd.DataFrame(month_disp_rows)
    if not month_df.empty:
        mfmt = {
            "Strike": "{:,.0f}",
            "Txn cost": "₹{:,.2f}",
            "P/L gross": "₹{:+,.2f}",
            "P/L": "₹{:+,.2f}",
        }
        mstyled = style_pl_dataframe(
            month_df.style.format(mfmt, na_rep="—"),
            "P/L",
            "P/L gross",
        )
        st.dataframe(mstyled, use_container_width=True, hide_index=True)
    if month_trade_rows:
        _mpl = sum(t["P/L"] for t in month_trade_rows)
        st.markdown(f"**MTD net P/L (sum of executed trades this month):** {pl_markdown(_mpl)}")
