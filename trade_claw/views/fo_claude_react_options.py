"""Claude Vision + ReAct F&O mock page — LLM-only analysis, tool-driven chart overlays."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_claw.constants import (
    DEFAULT_INTERVALS,
    DEFAULT_INTRADAY_INTERVAL,
    FO_BROKERAGE_PER_LOT_RT_DEFAULT,
    FO_STRIKE_POLICY_LABELS,
    FO_STRIKE_POLICY_STEPS,
    FO_TAXES_PER_LOT_RT_DEFAULT,
    FO_UNDERLYING_OPTIONS,
    FO_INDEX_UNDERLYING_LABELS,
    NFO_EXCHANGE,
    NSE_EXCHANGE,
)
from trade_claw.fo_claude_react_agent import run_claude_fo_react_pipeline
from trade_claw.plotly_ohlc import (
    add_candlestick_trace,
    add_volume_bar_trace,
    create_ohlc_volume_figure,
    finalize_ohlc_volume_figure,
)


def _underlying_select_label(u: str) -> str:
    if u in FO_INDEX_UNDERLYING_LABELS:
        return f"{FO_INDEX_UNDERLYING_LABELS[u]} — `{u}`"
    return u


def _render_observability_timeline(obs: list[dict], chart_pngs: list[bytes]) -> None:
    st.markdown("### Orchestration (live)")
    st.caption(
        "Chronological trace: assistant text, tool calls with arguments, tool results, and chart thumbnails "
        "when a step references a chart index."
    )
    for ev in obs:
        step = ev.get("step", "?")
        kind = str(ev.get("kind", "?"))
        title = f"Step {step} — {kind}"
        with st.expander(title, expanded=False):
            if kind == "assistant_text":
                st.markdown(ev.get("text") or "")
            elif kind == "tool_call":
                st.json({"name": ev.get("name"), "arguments": ev.get("arguments")})
            elif kind == "tool_result":
                st.code(str(ev.get("result") or ""), language="json")
            elif kind == "chart":
                idx = ev.get("chart_index")
                label = str(ev.get("label", "chart"))
                if idx is not None and isinstance(idx, int) and 0 <= idx < len(chart_pngs):
                    st.image(chart_pngs[idx], caption=f"{label} (index {idx})")
                else:
                    st.caption(f"{label} — {ev.get('bytes', '?')} bytes (index {idx})")
            else:
                show = {k: v for k, v in ev.items() if k not in ("step", "kind")}
                st.json(show if show else ev)


def _plot_underlying_session(df_u: pd.DataFrame, *, title: str) -> go.Figure:
    fig, pr, vr = create_ohlc_volume_figure(df_u, include_volume=True)
    add_candlestick_trace(fig, df_u, name="Underlying", price_row=pr, volume_row=vr)
    if vr is not None:
        add_volume_bar_trace(fig, df_u, volume_row=vr)
    finalize_ohlc_volume_figure(fig, height=400)
    fig.update_layout(title=title)
    return fig


def render_fo_claude_react_options(kite) -> None:
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

    st.title("F&O — Claude Vision + ReAct (LLM-only mock)")
    st.caption(
        "100% LLM technical narrative and trade vs no-trade — **no** deterministic strategy gate. "
        "Claude sees candlestick PNGs with tool-driven overlays (EMA, VWAP, levels). "
        "Requires **ANTHROPIC_API_KEY** and **kaleido** for chart export."
    )

    today = date.today()
    _ndef = "NIFTY" if "NIFTY" in FO_UNDERLYING_OPTIONS else FO_UNDERLYING_OPTIONS[0]
    _u_idx = FO_UNDERLYING_OPTIONS.index(_ndef) if _ndef in FO_UNDERLYING_OPTIONS else 0
    underlying = st.selectbox(
        "Underlying",
        options=FO_UNDERLYING_OPTIONS,
        index=_u_idx,
        key="fo_claude_underlying",
        format_func=_underlying_select_label,
    )
    name = symbol_to_name.get(underlying, underlying)

    chosen_date = st.date_input(
        "Session date",
        value=today,
        min_value=today - timedelta(days=365),
        max_value=today,
        key="fo_claude_date",
    )

    intraday_intervals = [i for i in DEFAULT_INTERVALS if i != "day"]
    _int_idx = (
        intraday_intervals.index(DEFAULT_INTRADAY_INTERVAL)
        if DEFAULT_INTRADAY_INTERVAL in intraday_intervals
        else 0
    )
    chosen_interval = st.selectbox(
        "Minute interval",
        options=intraday_intervals,
        index=_int_idx,
        key="fo_claude_interval",
    )

    sp_idx = st.selectbox(
        "Option strike vs spot",
        options=list(range(len(FO_STRIKE_POLICY_LABELS))),
        format_func=lambda i: FO_STRIKE_POLICY_LABELS[i],
        index=0,
        key="fo_claude_strike_policy",
    )
    steps_from_atm = FO_STRIKE_POLICY_STEPS[sp_idx]
    strike_policy_label = FO_STRIKE_POLICY_LABELS[sp_idx]

    man = st.number_input(
        "Manual strike override (0 = policy only)",
        min_value=0,
        max_value=10_000_000,
        step=50,
        value=0,
        key="fo_claude_manstrike",
    )
    manual_strike_val = float(man) if man and man > 0 else None

    bc1, bc2 = st.columns(2)
    with bc1:
        brokerage_per_lot_rt = float(
            st.number_input(
                "Brokerage ₹ / lot (round trip)",
                min_value=0.0,
                value=float(FO_BROKERAGE_PER_LOT_RT_DEFAULT),
                step=5.0,
                key="fo_claude_brokerage",
            )
        )
    with bc2:
        taxes_per_lot_rt = float(
            st.number_input(
                "Taxes & charges ₹ / lot (round trip)",
                min_value=0.0,
                value=float(FO_TAXES_PER_LOT_RT_DEFAULT),
                step=5.0,
                key="fo_claude_taxes",
            )
        )

    sl1, sl2 = st.columns(2)
    with sl1:
        option_target_pct_ui = float(
            st.slider(
                "Target above entry (premium %)",
                min_value=0.5,
                max_value=200.0,
                step=0.5,
                key="fo_claude_target_pct_ui",
            )
        )
    with sl2:
        option_stop_loss_pct_ui = float(
            st.slider(
                "Stop loss below entry (premium %)",
                min_value=0.0,
                max_value=50.0,
                step=0.5,
                key="fo_claude_stop_pct_ui",
            )
        )
    option_target_pct = option_target_pct_ui / 100.0
    option_stop_loss_pct = option_stop_loss_pct_ui / 100.0

    st.divider()
    run = st.button("Run Claude ReAct (mock)", type="primary", key="fo_claude_run")

    if not run:
        return

    with st.spinner(f"Claude ReAct for {underlying} on {chosen_date}..."):
        result = run_claude_fo_react_pipeline(
            kite=kite,
            nse_instruments=nse,
            nfo_instruments=nfo,
            underlying=underlying,
            name=name,
            session_date=chosen_date,
            chosen_interval=chosen_interval,
            steps_from_atm=steps_from_atm,
            strike_policy_label=strike_policy_label,
            manual_strike_val=manual_strike_val,
            brokerage_per_lot_rt=brokerage_per_lot_rt,
            taxes_per_lot_rt=taxes_per_lot_rt,
            option_target_pct=option_target_pct,
            option_stop_loss_pct=option_stop_loss_pct,
        )

    rd = result.get("run_dir")
    if rd:
        st.caption(f"Run artifacts: `{rd}`")

    if not result.get("success"):
        st.error(result.get("error_message") or "Run failed")
        obs = result.get("observability") or []
        chart_pngs = result.get("chart_pngs") or []
        if obs:
            _render_observability_timeline(obs, chart_pngs)
        if chart_pngs:
            st.markdown("#### Charts generated this run")
            for i, png in enumerate(chart_pngs):
                st.image(png, caption=f"Chart {i}")
        if result.get("trace"):
            with st.expander("Raw trace", expanded=False):
                st.json(result["trace"])
        return

    row = result.get("row") or {}
    fd = result.get("final_decision") or {}
    chart_pngs = result.get("chart_pngs") or []
    obs = result.get("observability") or []

    st.markdown("## Result")
    m1, m2, m3 = st.columns(3)
    pl = row.get("P/L")
    try:
        pl_v = float(pl) if pl is not None and not (isinstance(pl, float) and pd.isna(pl)) else 0.0
    except (TypeError, ValueError):
        pl_v = 0.0
    lots = row.get("Lots", 0)
    try:
        lots_v = int(pd.to_numeric(lots, errors="coerce") or 0)
    except (TypeError, ValueError):
        lots_v = 0
    m1.metric("Lots", lots_v)
    m2.metric("Net P/L (mock)", f"₹{pl_v:+,.2f}")
    m3.metric("Strategy", str(row.get("Strategy", "—"))[:48])

    st.markdown("### Final decision (structured)")
    st.json(fd)

    detail = {k: row.get(k) for k in (
        "Session date",
        "Underlying",
        "Name",
        "Strategy",
        "Leg",
        "Option",
        "Strike",
        "Signal",
        "Lots",
        "Entry",
        "Target prem.",
        "Stop prem.",
        "Closed at",
        "Exit",
        "P/L",
        "Txn cost",
    ) if k in row}
    st.dataframe(pd.DataFrame([detail]), use_container_width=True, hide_index=True)

    df_u = row.get("df_u")
    if isinstance(df_u, pd.DataFrame) and not df_u.empty:
        st.markdown("### Underlying session")
        st.plotly_chart(
            _plot_underlying_session(df_u, title=f"{underlying} — {chosen_date}"),
            use_container_width=True,
            key=f"fo_claude_u_{underlying}_{chosen_date}",
        )

    trade = row.get("trade")
    df_o = row.get("df_o")
    if isinstance(trade, dict) and isinstance(df_o, pd.DataFrame) and not df_o.empty:
        st.markdown("### Option (mock)")
        fig, pr, vr = create_ohlc_volume_figure(df_o, include_volume=True)
        add_candlestick_trace(fig, df_o, name=str(trade.get("Option", "option")), price_row=pr, volume_row=vr)
        if vr is not None:
            add_volume_bar_trace(fig, df_o, volume_row=vr)
        finalize_ohlc_volume_figure(fig, height=380)
        st.plotly_chart(fig, use_container_width=True, key=f"fo_claude_o_{underlying}")

    st.markdown("### Charts passed to the model (vision)")
    if chart_pngs:
        cols = st.columns(min(3, len(chart_pngs)))
        for i, png in enumerate(chart_pngs):
            cols[i % len(cols)].image(png, caption=f"Chart {i}")
    else:
        st.warning("No PNGs in this run (check kaleido / chart export).")

    _render_observability_timeline(obs, chart_pngs)

    with st.expander("Execution trace (compact)", expanded=False):
        st.json(result.get("trace") or [])

