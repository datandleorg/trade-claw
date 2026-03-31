"""F&O: underlying strategy (envelope or EMA cross) → long CE/PE; premium target, stop, or EOD; lot costs."""
from datetime import date, timedelta
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from trade_claw.constants import (
    DEFAULT_INTERVALS,
    DEFAULT_INTRADAY_INTERVAL,
    ENVELOPE_EMA_PERIOD,
    FO_BROKERAGE_PER_LOT_RT_DEFAULT,
    FO_CLOSED_AT_REALISED,
    FO_DEFAULT_UNDERLYINGS,
    FO_INDEX_UNDERLYING_KEYS,
    FO_INDEX_UNDERLYING_LABELS,
    FO_ENVELOPE_BANDWIDTH_MAX_PCT,
    FO_ENVELOPE_BANDWIDTH_MIN_PCT,
    FO_ENVELOPE_BANDWIDTH_STEP,
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
from trade_claw.fo_options_persist import fingerprint_params, save_fo_options_snapshot
from trade_claw.fo_runner import run_fo_underlying_one_day
from trade_claw.mock_market_signal import (
    fo_option_stop_loss_pct_runtime,
    fo_option_target_pct_runtime,
    fo_options_default_envelope_bandwidth_pct,
    fo_options_default_option_stop_loss_pct_ui,
    fo_options_default_option_target_pct_ui,
    mock_agent_envelope_pct,
)
from trade_claw.pl_style import pl_markdown, pl_title_color, style_pl_dataframe
from trade_claw.strategies import add_ma_ema_line_traces, add_ma_envelope_line_traces

FO_OPTIONS_DETAIL_COLS = [
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
    "Stop prem.",
    "Txn cost",
    "P/L gross",
    "Closed at",
    "Exit",
    "P/L",
    "Value",
]

# Web-audio beep loop (iframe removed on dismiss / rerun → tone stops)
_FO_ALERT_BEEP_HTML = """
<div style="height:0;width:0;overflow:hidden" aria-hidden="true">
<script>
(function () {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return;
  const ctx = new Ctx();
  function ping() {
    if (ctx.state === "suspended") ctx.resume();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = "sine";
    o.frequency.value = 880;
    o.connect(g);
    g.connect(ctx.destination);
    const t = ctx.currentTime;
    g.gain.setValueAtTime(0.12, t);
    g.gain.exponentialRampToValueAtTime(0.01, t + 0.14);
    o.start(t);
    o.stop(t + 0.14);
  }
  ping();
  setInterval(ping, 480);
})();
</script>
</div>
"""


def _fo_num_for_alert_fp(x) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def _fo_trade_alert_fingerprint(r: dict) -> str:
    lots = int(pd.to_numeric(r.get("Lots"), errors="coerce") or 0)
    if lots < 1:
        return ""
    return "|".join(
        [
            str(r.get("Option", "") or ""),
            _fo_num_for_alert_fp(r.get("Strike")),
            str(r.get("Leg", "") or ""),
            str(r.get("Signal", "") or ""),
            _fo_num_for_alert_fp(r.get("Entry")),
        ]
    )


def _fo_trade_alert_payload_lines(r: dict) -> list[str]:
    pairs = [
        ("Underlying", "Underlying"),
        ("Name", "Name"),
        ("Leg", "Leg"),
        ("Option", "Option"),
        ("Strike", "Strike"),
        ("Signal", "Signal"),
        ("Entry", "Entry (₹)"),
        ("Target prem.", "Target (₹)"),
        ("Stop prem.", "Stop (₹)"),
        ("Lots", "Lots"),
        ("Qty", "Qty"),
        ("Closed at", "Closed at"),
    ]
    lines: list[str] = []
    for k, label in pairs:
        v = r.get(k)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, float):
            lines.append(f"**{label}:** {v:,.2f}" if k in ("Entry", "Strike", "Target prem.", "Stop prem.") else f"**{label}:** {v}")
        else:
            lines.append(f"**{label}:** {v}")
    return lines


def _fo_sync_slider_session_from_env_defaults() -> None:
    """
    Push env-derived defaults into widget session_state when they change.

    Fixes stale Streamlit slider keys across reruns; defaults match current ``.env`` via
    :mod:`trade_claw.env_trading_params` (call-time reads).
    """
    ss = st.session_state
    sig = "|".join(
        [
            f"{mock_agent_envelope_pct():.12g}",
            f"{fo_options_default_option_target_pct_ui():.12g}",
            f"{fo_options_default_option_stop_loss_pct_ui():.12g}",
        ]
    )
    if ss.get("fo_options_env_defaults_sig") == sig:
        return
    ss["fo_options_env_defaults_sig"] = sig
    ss["fo_envelope_bandwidth_pct"] = float(round(fo_options_default_envelope_bandwidth_pct(), 4))
    ss["fo_option_target_pct_ui"] = float(fo_options_default_option_target_pct_ui())
    ss["fo_option_stop_loss_pct_ui"] = float(fo_options_default_option_stop_loss_pct_ui())


def _fo_run_context(kite, nse, nfo, symbol_to_name: dict) -> dict:
    ss = st.session_state
    underlying = ss["fo_underlying_single"]
    strategy_choice = ss["fo_strategy_choice"]
    strategy_is_envelope = strategy_choice == FO_STRATEGY_ENVELOPE
    chosen_date = ss["fo_date"]
    chosen_interval = ss["fo_interval"]
    if strategy_is_envelope:
        envelope_bw_pct = float(ss["fo_envelope_bandwidth_pct"])
        envelope_pct = envelope_bw_pct / 100.0
    else:
        envelope_pct = mock_agent_envelope_pct()
        envelope_bw_pct = float(round(100 * envelope_pct, 4))
    brokerage = float(ss["fo_brokerage_per_lot"])
    taxes = float(ss["fo_taxes_per_lot"])
    option_target_pct_ui = float(ss["fo_option_target_pct_ui"])
    option_stop_loss_pct_ui = float(ss["fo_option_stop_loss_pct_ui"])
    option_target_pct = option_target_pct_ui / 100.0
    option_stop_loss_pct = option_stop_loss_pct_ui / 100.0
    strike_policy_label = ss["fo_strike_policy"]
    steps_from_atm = FO_STRIKE_POLICY_STEPS[FO_STRIKE_POLICY_LABELS.index(strike_policy_label)]
    manual_raw = ss.get("fo_manstrike_single", 0)
    manual_strike_val = float(manual_raw) if manual_raw and float(manual_raw) > 0 else None
    name = _fo_display_name(symbol_to_name, underlying)
    common_kw = {
        "kite": kite,
        "nse_instruments": nse,
        "nfo_instruments": nfo,
        "session_date": chosen_date,
        "chosen_interval": chosen_interval,
        "strategy_is_envelope": strategy_is_envelope,
        "envelope_pct": envelope_pct,
        "strategy_choice": strategy_choice,
        "steps_from_atm": steps_from_atm,
        "strike_policy_label": strike_policy_label,
        "manual_strike_val": manual_strike_val,
        "brokerage_per_lot_rt": brokerage,
        "taxes_per_lot_rt": taxes,
        "option_stop_loss_pct": option_stop_loss_pct,
        "option_target_pct": option_target_pct,
    }
    return {
        "common_kw": common_kw,
        "name": name,
        "chosen_date": chosen_date,
        "underlying": underlying,
        "envelope_pct": envelope_pct,
        "envelope_bw_pct": envelope_bw_pct,
        "strategy_is_envelope": strategy_is_envelope,
        "strategy_choice": strategy_choice,
        "option_target_pct": option_target_pct,
        "option_stop_loss_pct": option_stop_loss_pct,
        "option_target_pct_ui": option_target_pct_ui,
        "option_stop_loss_pct_ui": option_stop_loss_pct_ui,
    }


def _fo_run_selected_once(ctx: dict) -> tuple[list[dict], list, dict, dict, str]:
    rows_out = [
        run_fo_underlying_one_day(
            underlying=ctx["underlying"],
            name=ctx["name"],
            include_chart_data=True,
            **ctx["common_kw"],
        )
    ]
    trade_rows = [r["trade"] for r in rows_out if r.get("trade")]
    total_trades = len(trade_rows)
    total_finished = sum(1 for t in trade_rows if t.get("Closed at") in FO_CLOSED_AT_REALISED)
    realised_pl = sum(t["P/L"] for t in trade_rows if t.get("Closed at") in FO_CLOSED_AT_REALISED)
    unrealised_pl = sum(t["P/L"] for t in trade_rows if t.get("Closed at") == "EOD")
    total_pl = realised_pl + unrealised_pl
    total_traded_value = sum(t["Value"] for t in trade_rows)
    total_txn_cost = sum(t.get("Txn cost", 0.0) for t in trade_rows)
    metrics = {
        "total_trades": total_trades,
        "total_finished": total_finished,
        "target_exits": sum(1 for t in trade_rows if t.get("Closed at") == "Target"),
        "stop_exits": sum(1 for t in trade_rows if t.get("Closed at") == "Stop"),
        "eod_exits": sum(1 for t in trade_rows if t.get("Closed at") == "EOD"),
        "realised_pl_net": realised_pl,
        "eod_pl_net": unrealised_pl,
        "total_pl_net": total_pl,
        "total_traded_value": total_traded_value,
        "total_txn_cost": total_txn_cost,
    }
    run_params = {
        "underlying": ctx["underlying"],
        "name": ctx["name"],
        "session_date": ctx["chosen_date"].isoformat(),
        "chosen_interval": ctx["common_kw"]["chosen_interval"],
        "strategy_choice": ctx["strategy_choice"],
        "strategy_is_envelope": ctx["strategy_is_envelope"],
        "envelope_pct": ctx["envelope_pct"],
        "envelope_bw_pct": ctx["envelope_bw_pct"],
        "envelope_ema_period": ENVELOPE_EMA_PERIOD,
        "ma_ema_fast": MA_EMA_FAST,
        "ma_ema_slow": MA_EMA_SLOW,
        "brokerage_per_lot_rt": ctx["common_kw"]["brokerage_per_lot_rt"],
        "taxes_per_lot_rt": ctx["common_kw"]["taxes_per_lot_rt"],
        "option_stop_loss_pct": ctx["option_stop_loss_pct"],
        "option_target_pct": ctx["option_target_pct"],
        "option_target_pct_default_constant": fo_option_target_pct_runtime(),
        "option_stop_loss_pct_default_constant": fo_option_stop_loss_pct_runtime(),
        "strike_policy_label": ctx["common_kw"]["strike_policy_label"],
        "steps_from_atm": ctx["common_kw"]["steps_from_atm"],
        "manual_strike_val": ctx["common_kw"]["manual_strike_val"],
    }
    fp_now = fingerprint_params(run_params)
    return rows_out, trade_rows, metrics, run_params, fp_now


def _fo_persist_fo_snapshot(run_params: dict, rows_out: list[dict], metrics: dict) -> None:
    fp_now = fingerprint_params(run_params)
    last_fp = st.session_state.get("fo_options_last_persisted_fp")
    if last_fp != fp_now:
        try:
            saved_path = save_fo_options_snapshot(
                params=run_params,
                rows_out=rows_out,
                metrics=metrics,
            )
            st.session_state["fo_options_last_persisted_fp"] = fp_now
            st.session_state["fo_options_last_saved_path"] = str(saved_path)
            toast = getattr(st, "toast", None)
            if callable(toast):
                st.toast(f"Saved run snapshot: {saved_path.name}", icon="💾")
            else:
                st.sidebar.success(f"Saved run: `{saved_path.name}`")
        except OSError as e:
            st.warning(f"Could not save run snapshot to disk: {e}")


def _fo_watch_process_trade_alert(r0: dict) -> None:
    cur = _fo_trade_alert_fingerprint(r0)
    prev = st.session_state.get("fo_watch_prev_fp", "")
    ack = st.session_state.get("fo_watch_ack_fp", "")
    if not cur:
        st.session_state["fo_watch_prev_fp"] = ""
        st.session_state["fo_watch_ack_fp"] = ""
        st.session_state["fo_show_trade_alert"] = False
        st.session_state["fo_alert_sound_fp"] = None
        st.session_state["fo_audio_started_fp"] = None
        return
    if cur != prev:
        st.session_state["fo_watch_prev_fp"] = cur
        if cur != ack:
            st.session_state["fo_show_trade_alert"] = True
            st.session_state["fo_alert_payload_lines"] = _fo_trade_alert_payload_lines(r0)
            st.session_state["fo_alert_sound_fp"] = cur
            st.session_state["fo_audio_started_fp"] = None


def _fo_render_trade_alert_banner() -> None:
    if not st.session_state.get("fo_show_trade_alert"):
        return
    snd = st.session_state.get("fo_alert_sound_fp")
    if snd is not None and st.session_state.get("fo_audio_started_fp") != snd:
        components.html(_FO_ALERT_BEEP_HTML, height=0)
        st.session_state["fo_audio_started_fp"] = snd
    with st.container(border=True):
        st.markdown("#### Trade detected (selected scrip)")
        for line in st.session_state.get("fo_alert_payload_lines") or []:
            st.markdown(line)
        if st.button("Dismiss — stop alert tone", key="fo_watch_dismiss_alert"):
            st.session_state["fo_show_trade_alert"] = False
            st.session_state["fo_watch_ack_fp"] = st.session_state.get("fo_watch_prev_fp", "")
            st.session_state["fo_alert_sound_fp"] = None
            st.session_state["fo_audio_started_fp"] = None
            st.rerun()


def _fo_render_session_block(
    ctx: dict,
    rows_out: list[dict],
    trade_rows: list,
    metrics: dict,
    *,
    watch_mode: bool,
) -> None:
    """Metrics, strategy caption, charts, and detail table for the selected underlying run."""
    total_trades = metrics["total_trades"]
    total_finished = metrics["total_finished"]
    realised_pl = metrics["realised_pl_net"]
    unrealised_pl = metrics["eod_pl_net"]
    total_pl = metrics["total_pl_net"]
    total_traded_value = metrics["total_traded_value"]
    total_txn_cost = metrics["total_txn_cost"]
    underlying = ctx["underlying"]
    chosen_date = ctx["chosen_date"]
    envelope_pct = ctx["envelope_pct"]
    envelope_bw_pct = ctx["envelope_bw_pct"]
    strategy_is_envelope = ctx["strategy_is_envelope"]
    option_target_pct = ctx["option_target_pct"]
    option_stop_loss_pct = ctx["option_stop_loss_pct"]
    sk_render = "envelope" if strategy_is_envelope else "ma"
    if watch_mode:
        st.caption("Last refresh (auto) · selected scrip only")

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    with m1:
        st.metric(
            "Option trades",
            f"{total_trades:,}",
            help=f"{total_trades:,} executed mock option trade(s). Hover label for full count.",
        )
    with m2:
        _nt = sum(1 for t in trade_rows if t.get("Closed at") == "Target")
        _ns = sum(1 for t in trade_rows if t.get("Closed at") == "Stop")
        st.metric(
            "Target / stop (realised)",
            f"{total_finished:,}",
            help=f"{total_finished:,} exited on target or stop ({_nt} target, {_ns} stop). EOD is separate.",
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
            f"target prem = entry × (1 + {option_target_pct:.2f})"
            + (
                f", stop prem = entry × (1 − {option_stop_loss_pct:.2f})"
                if option_stop_loss_pct > 0
                else " (no stop — slider at 0%)"
            )
            + ". Deduction = (brokerage + taxes) × **1 lot** per trade."
        )
    else:
        st.caption(
            f"Strategy: **{FO_STRATEGY_MA_CROSS}** — EMA {MA_EMA_FAST} / {MA_EMA_SLOW} · "
            f"target prem = entry × (1 + {option_target_pct:.2f})"
            + (
                f", stop prem = entry × (1 − {option_stop_loss_pct:.2f})"
                if option_stop_loss_pct > 0
                else " (no stop — slider at 0%)"
            )
            + ". Deduction = (brokerage + taxes) × **1 lot** per trade."
        )
    st.divider()

    st.markdown(f"### Charts — **{underlying}** on **{chosen_date.isoformat()}**")
    st.caption(
        "Envelope or EMA lines match your **Underlying strategy** dropdown. "
        "Option chart appears only when a mock option trade was built for the session date."
    )
    for r in rows_out:
        _u = str(r.get("Underlying", "row"))
        _fo_expander_charts_single_row(
            r,
            chart_key_prefix=f"fo_main_{_u}",
            sk_render=sk_render,
            envelope_pct=envelope_pct,
            option_target_frac=option_target_pct,
            option_stop_loss_frac=option_stop_loss_pct,
            expanded=False,
            table_relative="below",
        )

    st.markdown("### Selected session (detail row)")
    disp_rows = []
    for r in rows_out:
        row = {k: r[k] for k in FO_OPTIONS_DETAIL_COLS if k in r}
        sd = row.get("Session date")
        if sd is not None and hasattr(sd, "isoformat"):
            row["Session date"] = sd.isoformat()
        disp_rows.append(row)
    disp = pd.DataFrame(disp_rows)
    if not disp.empty:
        _fmt = _fo_table_formatters()
        styled = style_pl_dataframe(
            disp.style.format(_fmt, na_rep="—"),
            "P/L",
            "P/L gross",
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    if trade_rows:
        _tpl = sum(t["P/L"] for t in trade_rows)
        st.markdown(f"**Session net P/L (if trade executed):** {pl_markdown(_tpl)}")


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


def _fo_display_name(symbol_to_name: dict, u: str) -> str:
    if u in FO_INDEX_UNDERLYING_LABELS:
        return str(FO_INDEX_UNDERLYING_LABELS[u]).split("(")[0].strip()
    return symbol_to_name.get(u, u)


def _fo_underlying_select_label(u: str) -> str:
    """Selectbox: index rows show descriptive label + key; stocks show symbol only."""
    if u in FO_INDEX_UNDERLYING_LABELS:
        return f"{FO_INDEX_UNDERLYING_LABELS[u]} — `{u}`"
    return u


def _fo_sanitize_st_key(s: str, max_len: int = 120) -> str:
    """Streamlit element keys: stable alphanumeric + underscore (avoid duplicate ID collisions)."""
    out = []
    for ch in str(s):
        if ch.isalnum():
            out.append(ch)
        elif ch in ("-", ".", " "):
            out.append("_")
    x = "".join(out).strip("_") or "x"
    return x[:max_len]


def _fo_expander_charts_single_row(
    r: dict,
    *,
    chart_key_prefix: str,
    sk_render: str,
    envelope_pct: float,
    option_target_frac: float | None = None,
    option_stop_loss_frac: float | None = None,
    expanded: bool = False,
    table_relative: str = "below",
) -> None:
    """Underlying + option candle charts and strategy caption for one mock F&O row."""
    _kpre = _fo_sanitize_st_key(chart_key_prefix)
    _chart_sig = (
        f"{float(envelope_pct):.8f}_"
        f"{float(option_target_frac if option_target_frac is not None else -1.0):.6f}_"
        f"{float(option_stop_loss_frac if option_stop_loss_frac is not None else -1.0):.6f}"
    )
    df_u = r.get("df_u")
    u = r.get("Underlying", "—")
    nm = r.get("Name", "")
    t = r.get("trade")
    df_o = r.get("df_o")
    if df_u is None:
        note = str(r.get("Note", "No underlying data"))
        exp_sub = (note[:72] + "…") if len(note) > 72 else note
        with st.expander(f"**{u}** — {nm} · {exp_sub}", expanded=expanded):
            st.caption(r.get("_analysis_text") or note)
        return
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
    with st.expander(f"**{u}** — {nm} · {exp_sub}", expanded=expanded):
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
            fig_u.update_layout(
                title=spot_title,
                height=320,
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig_u, use_container_width=True, key=f"{_kpre}_spot_{_chart_sig}")
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
                _tp = None
                _sp = None
                try:
                    _ev = float(t.get("Entry"))
                except (TypeError, ValueError):
                    _ev = float("nan")
                if (
                    option_target_frac is not None
                    and not pd.isna(_ev)
                    and _ev > 0
                ):
                    _tp = _ev * (1.0 + float(option_target_frac))
                if _tp is None:
                    _tp = t.get("Target prem.")
                if (
                    option_stop_loss_frac is not None
                    and float(option_stop_loss_frac) > 0
                    and not pd.isna(_ev)
                    and _ev > 0
                ):
                    _sp = _ev * (1.0 - float(option_stop_loss_frac))
                if _sp is None:
                    _sp = t.get("Stop prem.")
                if _tp is not None and not pd.isna(_tp):
                    fig_o.add_hline(
                        y=float(_tp),
                        line_dash="dash",
                        line_color="rgba(34,197,94,0.85)",
                        annotation_text="Target",
                        annotation_position="right",
                    )
                _draw_stop = _sp is not None and not pd.isna(_sp)
                if _draw_stop and option_stop_loss_frac is not None and float(option_stop_loss_frac) <= 0:
                    _draw_stop = False
                if _draw_stop:
                    fig_o.add_hline(
                        y=float(_sp),
                        line_dash="dash",
                        line_color="rgba(239,68,68,0.85)",
                        annotation_text="Stop",
                        annotation_position="right",
                    )
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
                st.plotly_chart(fig_o, use_container_width=True, key=f"{_kpre}_opt_{_chart_sig}")
            else:
                st.markdown("**Option premium**")
                st.info(
                    "No mock option trade for this session (no signal, data error, or below 1 lot). "
                    f"See the row in the table {table_relative} for the exact reason."
                )


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


def _fo_table_formatters() -> dict:
    return {
        "Strike": "{:,.0f}",
        "Lots": "{:,.0f}",
        "Lot size": "{:,.0f}",
        "Qty": "{:,.0f}",
        "Entry": "₹{:,.2f}",
        "Target prem.": "₹{:,.2f}",
        "Stop prem.": "₹{:,.2f}",
        "Txn cost": "₹{:,.2f}",
        "P/L gross": "₹{:+,.2f}",
        "Exit": "₹{:,.2f}",
        "P/L": "₹{:+,.2f}",
        "Value": "₹{:,.2f}",
    }


def _fo_rows_to_styled_dataframe(rows: list[dict]) -> tuple[pd.DataFrame, float, int, pd.DataFrame]:
    """Build display frame, total P/L, signal count, and subset with Lots > 0."""
    disp_rows = []
    for r in rows:
        row = {k: r[k] for k in FO_OPTIONS_DETAIL_COLS if k in r}
        sd = row.get("Session date")
        if sd is not None and hasattr(sd, "isoformat"):
            row["Session date"] = sd.isoformat()
        disp_rows.append(row)
    df = pd.DataFrame(disp_rows)
    if df.empty:
        return df, 0.0, 0, df
    if "Lots" in df.columns:
        lots_num = pd.to_numeric(df["Lots"], errors="coerce").fillna(0)
        traded = df[lots_num > 0]
    else:
        traded = df
    sum_pl = float(
        pd.to_numeric(df["P/L"], errors="coerce").fillna(0).sum() if "P/L" in df.columns else 0.0
    )
    if "Signal" in df.columns:
        n_sig = int((df["Signal"].fillna("—").astype(str) != "—").sum())
    else:
        n_sig = 0
    return df, sum_pl, n_sig, traded


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
    date_min = today - timedelta(days=365)
    date_max = today

    st.info(
        "**Session date** can be any day in the last year through today. "
        "Option contracts come from Kite’s **live NFO list** (nearest expiry on/after the session day). "
        "Very old dates may not resolve if the chain no longer matches history. "
        "Each time you **change parameters**, the run is saved under **`data/fo_options_runs/`** (JSON) for analysis — "
        "see **`FO_OPTIONS_RUNS_DIR`** in `.env.example` to override the folder."
    )
    with st.expander("NSE **index options** — reference (cash-settled, European-style)", expanded=False):
        st.markdown(
            """
**Index options** use an **index** (not a single stock) as the underlying. On NSE they are **cash-settled**
and are common for hedging and volatility trading.

| Index | Liquidity | Volatility | Typical use |
| :--- | :--- | :--- | :--- |
| **NIFTY 50** | Very high | Medium | Balanced / hedging |
| **BANK NIFTY** | Very high | Very high | Short-term / aggressive |
| **MIDCP NIFTY** | Growing | Medium–high | Midcap exposure |

**In this app:** pick an index from the **Underlying** dropdown (`NIFTY`, `BANKNIFTY`, `MIDCPNIFTY`).
Weekly and monthly expiries exist on the exchange; we pick the **nearest NFO expiry on or after** the session date.

**Note:** Lot sizes and exact NSE index **spot** symbols come from Kite’s instrument master. If a new index fails to load,
check `trade_claw/fo_support.py` (`_INDEX_NSE_SPOT_CANDIDATES` / `_INDEX_NFO_NAMES`).
            """
        )

    if st.button("Mock AI engine (home)", key="fo_nav_mock_home"):
        st.switch_page("app.py")

    _ndef = "NIFTY" if "NIFTY" in FO_UNDERLYING_OPTIONS else FO_UNDERLYING_OPTIONS[0]
    _u_idx = FO_UNDERLYING_OPTIONS.index(_ndef) if _ndef in FO_UNDERLYING_OPTIONS else 0
    underlying = st.selectbox(
        "Underlying (index or Nifty 50 stock)",
        options=FO_UNDERLYING_OPTIONS,
        index=_u_idx,
        key="fo_underlying_single",
        format_func=_fo_underlying_select_label,
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
        "Session date",
        value=today,
        min_value=date_min,
        max_value=date_max,
        key="fo_date",
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
        key="fo_interval",
    )
    _fo_sync_slider_session_from_env_defaults()
    if strategy_is_envelope:
        envelope_bw_pct = st.slider(
            "Envelope bandwidth (% each side of EMA)",
            min_value=float(FO_ENVELOPE_BANDWIDTH_MIN_PCT),
            max_value=float(FO_ENVELOPE_BANDWIDTH_MAX_PCT),
            step=float(FO_ENVELOPE_BANDWIDTH_STEP),
            key="fo_envelope_bandwidth_pct",
            help=(
                "Upper band = EMA × (1 + p), lower = EMA × (1 − p), with p = this % ÷ 100. "
                f"Default follows **`MOCK_AGENT_ENVELOPE_PCT`** when set; else mock default "
                f"({100 * mock_agent_envelope_pct():.2f}% each side, clamped to slider max)."
            ),
        )
        envelope_pct = envelope_bw_pct / 100.0
    else:
        envelope_pct = mock_agent_envelope_pct()
        envelope_bw_pct = float(round(100 * envelope_pct, 4))
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

    sl1, sl2 = st.columns(2)
    with sl1:
        option_target_pct_ui = st.slider(
            "Target above entry (premium %)",
            min_value=0.5,
            max_value=200.0,
            step=0.5,
            key="fo_option_target_pct_ui",
            help=(
                "Exit when option bar **high** ≥ entry × (1 + %/100). "
                "Resolved in **`trade_claw.env_trading_params`**: **`MOCK_ENGINE_OPTION_TARGET_PCT`** if set, else **`FO_OPTION_TARGET_PCT`** "
                f"({100 * fo_option_target_pct_runtime():.1f}%). Same-bar as stop: **target wins**."
            ),
        )
    with sl2:
        option_stop_loss_pct_ui = st.slider(
            "Stop loss below entry (premium %)",
            min_value=0.0,
            max_value=50.0,
            step=0.5,
            key="fo_option_stop_loss_pct_ui",
            help=(
                "**0** = no stop (only target or EOD). Otherwise exit when option bar **low** ≤ entry × (1 − %/100). "
                "Resolved in **`trade_claw.env_trading_params`**: **`MOCK_ENGINE_OPTION_STOP_PCT`** / **`MOCK_ENGINE_STOP_LOSS_CLAMP_PCT`** if set, "
                "else **`FO_OPTION_STOP_LOSS_PCT`**. Same-bar as target: **target wins** (same as cash BUY)."
            ),
        )
    option_target_pct = option_target_pct_ui / 100.0
    option_stop_loss_pct = option_stop_loss_pct_ui / 100.0

    st.caption(
        "**Envelope:** first **close** above upper / below lower band (not wick-only). "
        "→ long ATM **CE** / **PE**. **EMA cross:** first golden / death cross in session → **CE** / **PE**. "
        "No short options. Exit: **target** = premium **+"
        f"{option_target_pct_ui:.1f}%** on bar **high**; **stop** = **−{option_stop_loss_pct_ui:.1f}%** on bar **low** (0% = no stop); else **EOD**. "
        "Mock size: **always 1 lot** per symbol (exchange `lot_size` × premium). "
        "P/L **net** = gross premium P/L − (brokerage + tax) × 1 lot."
    )

    st.selectbox(
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

    st.number_input(
        "Manual strike override (0 = use strike policy only)",
        min_value=0,
        max_value=10_000_000,
        step=50,
        value=0,
        key="fo_manstrike_single",
        help="When >0, nearest listed strike to this value is used for the selected underlying.",
    )

    st.divider()
    watch_sel = st.checkbox(
        "Watch selected scrip: auto-refresh every 5 seconds",
        key="fo_watch_5s",
        help=(
            "Re-runs **only** the **Underlying** dropdown with your current settings. "
            "Best on **today’s session** while new minute bars arrive. "
            "Beeps (browser permitting) and shows a banner when a **mock trade first appears** (Lots > 0). "
            "**Dismiss** stops the tone."
        ),
    )

    ctx = _fo_run_context(kite, nse, nfo, symbol_to_name)
    _fo_common_kw = ctx["common_kw"]
    underlying = ctx["underlying"]
    chosen_date = ctx["chosen_date"]
    envelope_pct = ctx["envelope_pct"]
    envelope_bw_pct = ctx["envelope_bw_pct"]
    strategy_is_envelope = ctx["strategy_is_envelope"]
    strategy_choice = ctx["strategy_choice"]
    option_target_pct = ctx["option_target_pct"]
    option_stop_loss_pct = ctx["option_stop_loss_pct"]
    sk_render = "envelope" if strategy_is_envelope else "ma"

    if watch_sel:
        st.caption(
            f"**Live watch:** refreshing **{underlying}** every **5s** (selected scrip only). "
            "Charts and metrics in the block below update on each tick. "
            "**All scrips / index** sections further down refresh only after you change a control or turn watch off."
        )

        @st.fragment(run_every=timedelta(seconds=5))
        def _fo_watch_fragment():
            ctx_w = _fo_run_context(kite, nse, nfo, symbol_to_name)
            rows_local, trade_rows_l, metrics_l, run_params_l, fp_l = _fo_run_selected_once(ctx_w)
            st.session_state["fo_options_fp_now"] = fp_l
            st.session_state["fo_watch_cached_row"] = dict(rows_local[0])
            _fo_persist_fo_snapshot(run_params_l, rows_local, metrics_l)
            _fo_watch_process_trade_alert(rows_local[0])
            _fo_render_trade_alert_banner()
            _fo_render_session_block(ctx_w, rows_local, trade_rows_l, metrics_l, watch_mode=True)

        _fo_watch_fragment()
    else:
        for k in (
            "fo_show_trade_alert",
            "fo_alert_sound_fp",
            "fo_audio_started_fp",
            "fo_alert_payload_lines",
            "fo_watch_prev_fp",
            "fo_watch_ack_fp",
        ):
            st.session_state.pop(k, None)
        with st.spinner(
            f"Running {strategy_choice} + options for **{underlying}** on **{chosen_date}**..."
        ):
            rows_out, trade_rows, metrics, run_params, fp_now = _fo_run_selected_once(ctx)
        st.session_state["fo_options_fp_now"] = fp_now
        st.session_state["fo_watch_cached_row"] = dict(rows_out[0])
        _fo_persist_fo_snapshot(run_params, rows_out, metrics)
        _fo_render_session_block(ctx, rows_out, trade_rows, metrics, watch_mode=False)

    fp_now = str(st.session_state.get("fo_options_fp_now") or "")
    _cached = st.session_state.get("fo_watch_cached_row")
    rows_out = [_cached] if _cached is not None else []

    st.divider()
    st.markdown("### All scrips — same strategy parameters")
    st.caption(
        "Mock trade per underlying in **FO_UNDERLYING_OPTIONS** using the **session date**, **minute interval**, "
        "**underlying strategy**, **strike policy**, **manual strike**, **stop %**, and **per-lot costs** above. "
        "**Load / refresh** pulls underlying + option history for every scrip (slow, large session cache) and "
        "enables the chart expanders below. Top-of-page charts still reflect only the **Underlying** dropdown."
    )
    _all_cache = st.session_state.get("fo_all_scrips_cache")
    if _all_cache and _all_cache.get("fp") != fp_now:
        st.warning("All-scrips table was built with different parameters. Click **Load / refresh all scrips** to update.")
    _refresh_all = st.button("Load / refresh all scrips", key="fo_all_scrips_refresh")
    if _refresh_all:
        _all_rows: list[dict] = []
        with st.spinner(
            f"Fetching all {len(FO_UNDERLYING_OPTIONS)} underlyings (hist + option mock + chart data) — may take several minutes…"
        ):
            for u in FO_UNDERLYING_OPTIONS:
                nm = _fo_display_name(symbol_to_name, u)
                if u == underlying and st.session_state.get("fo_watch_cached_row"):
                    _all_rows.append(dict(st.session_state["fo_watch_cached_row"]))
                elif u == underlying and rows_out:
                    _all_rows.append(dict(rows_out[0]))
                else:
                    _all_rows.append(
                        run_fo_underlying_one_day(
                            underlying=u,
                            name=nm,
                            include_chart_data=True,
                            **_fo_common_kw,
                        )
                    )
        st.session_state["fo_all_scrips_cache"] = {"fp": fp_now, "rows": _all_rows}

    _show_all = st.session_state.get("fo_all_scrips_cache")
    if _show_all and _show_all.get("fp") == fp_now and _show_all.get("rows"):
        _arows = _show_all["rows"]
        _df_all, _sum_pl, _n_sig, _traded = _fo_rows_to_styled_dataframe(_arows)
        if not _df_all.empty:
            st.metric(
                "All-scrips net P/L (sum of mock trades)",
                _abbrev_rupee(_sum_pl),
                help=f"Full: ₹{_sum_pl:+,.2f}. Rows with a signal: {_n_sig}.",
            )
            _styled_all = style_pl_dataframe(
                _df_all.style.format(_fo_table_formatters(), na_rep="—"),
                "P/L",
                "P/L gross",
            )
            st.dataframe(_styled_all, use_container_width=True, hide_index=True)
            st.caption(f"{len(_df_all)} underlyings · {len(_traded)} with executed mock position (Lots > 0)")
            st.markdown("#### Charts & trade details per scrip")
            st.caption("Expand each row for underlying + option candles, entry/exit markers, target/stop lines, and rationale.")
            for _ai, r in enumerate(_arows):
                _u = str(r.get("Underlying", f"r{_ai}"))
                _fo_expander_charts_single_row(
                    r,
                    chart_key_prefix=f"fo_all_{_ai}_{_u}",
                    sk_render=sk_render,
                    envelope_pct=envelope_pct,
                    option_target_frac=option_target_pct,
                    option_stop_loss_frac=option_stop_loss_pct,
                    expanded=False,
                    table_relative="above",
                )
    elif not _refresh_all:
        st.info("Click **Load / refresh all scrips** to build the full-universe table and per-scrip charts.")

    st.divider()
    st.markdown("### Index underlyings (all NSE index keys) — same parameters")
    st.caption(
        f"Index **F&O** only: **{', '.join(FO_INDEX_UNDERLYING_KEYS)}** with the same session and strategy settings. "
        "Uses a separate **Load / refresh** so you can inspect indices without running the full stock universe."
    )
    _ix_cache = st.session_state.get("fo_index_scrips_cache")
    if _ix_cache and _ix_cache.get("fp") != fp_now:
        st.warning(
            "Index table was built with different parameters. Click **Load / refresh index underlyings** to update."
        )
    _refresh_ix = st.button("Load / refresh index underlyings", key="fo_index_scrips_refresh")
    if _refresh_ix:
        _ix_rows: list[dict] = []
        with st.spinner(f"Index suite ({len(FO_DEFAULT_UNDERLYINGS)} underlyings) — mock F&O with chart data…"):
            for u in FO_DEFAULT_UNDERLYINGS:
                nm = _fo_display_name(symbol_to_name, u)
                if u == underlying and st.session_state.get("fo_watch_cached_row"):
                    _ix_rows.append(dict(st.session_state["fo_watch_cached_row"]))
                elif u == underlying and rows_out:
                    _ix_rows.append(dict(rows_out[0]))
                else:
                    _ix_rows.append(
                        run_fo_underlying_one_day(
                            underlying=u,
                            name=nm,
                            include_chart_data=True,
                            **_fo_common_kw,
                        )
                    )
        st.session_state["fo_index_scrips_cache"] = {"fp": fp_now, "rows": _ix_rows}

    _show_ix = st.session_state.get("fo_index_scrips_cache")
    if _show_ix and _show_ix.get("fp") == fp_now and _show_ix.get("rows"):
        _irows = _show_ix["rows"]
        _df_ix, _sum_ix, _n_sig_ix, _traded_ix = _fo_rows_to_styled_dataframe(_irows)
        if not _df_ix.empty:
            st.metric(
                "Index underlyings net P/L (sum)",
                _abbrev_rupee(_sum_ix),
                help=f"Full: ₹{_sum_ix:+,.2f}. Rows with a signal: {_n_sig_ix}.",
            )
            _styled_ix = style_pl_dataframe(
                _df_ix.style.format(_fo_table_formatters(), na_rep="—"),
                "P/L",
                "P/L gross",
            )
            st.dataframe(_styled_ix, use_container_width=True, hide_index=True)
            st.caption(f"{len(_df_ix)} index underlyings · {len(_traded_ix)} with executed mock position (Lots > 0)")
        st.markdown("#### Charts & trade details (index)")
        for _ii, r in enumerate(_irows):
            _u = str(r.get("Underlying", f"r{_ii}"))
            _fo_expander_charts_single_row(
                r,
                chart_key_prefix=f"fo_idx_{_ii}_{_u}",
                sk_render=sk_render,
                envelope_pct=envelope_pct,
                option_target_frac=option_target_pct,
                option_stop_loss_frac=option_stop_loss_pct,
                expanded=False,
                table_relative="above",
            )
    elif not _refresh_ix:
        st.info("Click **Load / refresh index underlyings** for the index suite table and charts.")

    _last_path = st.session_state.get("fo_options_last_saved_path")
    if _last_path:
        st.caption(f"**Last saved snapshot:** `{_last_path}`")
