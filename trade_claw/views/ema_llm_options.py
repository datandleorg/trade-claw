"""EMA trend/crossover + mandatory LLM option pick page."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_claw.constants import (
    DEFAULT_INTERVALS,
    DEFAULT_INTRADAY_INTERVAL,
    FO_DEFAULT_UNDERLYINGS,
    FO_UNDERLYING_OPTIONS,
    NFO_EXCHANGE,
    NSE_EXCHANGE,
)
from trade_claw.ema_llm_runner import run_ema_llm_backtest_one_day
from trade_claw.plotly_ohlc import (
    add_candlestick_trace,
    add_volume_bar_trace,
    create_ohlc_volume_figure,
    finalize_ohlc_volume_figure,
)
from trade_claw.strategies import add_ma_ema_line_traces


def _params_fingerprint(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _render_single_run_output(
    out: dict,
    *,
    trend_ema_period: int,
    fast_ema_period: int,
    slow_ema_period: int,
    chart_key_prefix: str,
) -> None:
    if out.get("error"):
        st.error(out["error"])
        return
    st.caption(out.get("note") or "")
    trades = out.get("trades") or []
    executed = [t for t in trades if t.get("status") == "executed"]
    skipped = [t for t in trades if t.get("status") != "executed"]
    total_pl = sum(float(t.get("pl_net", 0.0)) for t in executed)
    m1, m2, m3 = st.columns(3)
    m1.metric("Signals", len(trades))
    m2.metric("Executed trades", len(executed))
    m3.metric("Net P/L", f"₹{total_pl:+,.2f}")

    if skipped:
        st.markdown("### Skipped signals")
        for s in skipped:
            st.warning(f"{s.get('flow_id','—')}: {s.get('reason','Skipped')}")
            llm_output = s.get("llm_output")
            if llm_output:
                st.markdown("**LLM output**")
                st.code(json.dumps(llm_output, indent=2, ensure_ascii=False), language="json")

    df_u = out["df_u"]
    for i, t in enumerate(executed, start=1):
        with st.expander(f"Trade {i}: {t['signal']} {t['option']} ({t['closed_at']})", expanded=False):
            st.markdown(
                f"**Entry:** ₹{t['entry']:.2f}  |  **Target:** ₹{t['target']:.2f}  |  "
                f"**Stop:** ₹{t['stop']:.2f}  |  **Exit:** ₹{t['exit']:.2f}  |  "
                f"**Net P/L:** ₹{t['pl_net']:+,.2f}"
            )
            st.caption(f"Flow: {t.get('flow_id','—')} · {t.get('flow_dir','')}")
            st.markdown("**LLM output**")
            st.code(
                json.dumps(
                    t.get("llm_output")
                    or {
                        "tradingsymbol": t.get("option"),
                        "stop_loss": round(float(t.get("stop", 0.0)), 2),
                        "target": round(float(t.get("target", 0.0)), 2),
                        "rationale": t.get("rationale", ""),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                language="json",
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Script chart (underlying)**")
                fig_u = _plot_underlying_chart(
                    df_u,
                    trend_period=trend_ema_period,
                    fast_period=fast_ema_period,
                    slow_period=slow_ema_period,
                    signal_idx=int(t["signal_bar_idx"]),
                    signal=t["signal"],
                )
                st.plotly_chart(fig_u, use_container_width=True, key=f"{chart_key_prefix}_script_{i}")
            with c2:
                st.markdown("**Mock trade chart (option)**")
                fig_o = _plot_option_chart(t["df_o"], t)
                st.plotly_chart(fig_o, use_container_width=True, key=f"{chart_key_prefix}_opt_{i}")


def _bulk_summary_rows(cache_rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    totals = {"signals": 0, "executed": 0, "skipped": 0, "net_pl": 0.0}
    for r in cache_rows:
        out = r.get("out") or {}
        trades = out.get("trades") or []
        executed = [t for t in trades if t.get("status") == "executed"]
        skipped = [t for t in trades if t.get("status") != "executed"]
        net_pl = sum(float(t.get("pl_net", 0.0)) for t in executed)
        rows.append(
            {
                "Underlying": r.get("underlying"),
                "Signals": len(trades),
                "Executed": len(executed),
                "Skipped": len(skipped),
                "Net P/L": net_pl,
                "Status": "OK" if not out.get("error") else f"ERR: {out.get('error')}",
            }
        )
        totals["signals"] += len(trades)
        totals["executed"] += len(executed)
        totals["skipped"] += len(skipped)
        totals["net_pl"] += net_pl
    return pd.DataFrame(rows), totals


def _plot_underlying_chart(
    df_u: pd.DataFrame,
    *,
    trend_period: int,
    fast_period: int,
    slow_period: int,
    signal_idx: int | None,
    signal: str | None,
):
    fig, pr, vr = create_ohlc_volume_figure(df_u, include_volume=True)
    add_candlestick_trace(fig, df_u, name="Underlying", price_row=pr, volume_row=vr)
    close = pd.to_numeric(df_u["close"], errors="coerce")
    trend = close.ewm(span=trend_period, adjust=False).mean()
    if vr is not None:
        fig.add_trace(
            go.Scatter(x=df_u["date"], y=trend, mode="lines", name=f"Trend EMA {trend_period}", line=dict(color="#22c55e", width=2)),
            row=pr,
            col=1,
        )
        add_ma_ema_line_traces(fig, df_u, fast_period=fast_period, slow_period=slow_period, row=pr, col=1)
        add_volume_bar_trace(fig, df_u, volume_row=vr)
    else:
        fig.add_trace(go.Scatter(x=df_u["date"], y=trend, mode="lines", name=f"Trend EMA {trend_period}", line=dict(color="#22c55e", width=2)))
        add_ma_ema_line_traces(fig, df_u, fast_period=fast_period, slow_period=slow_period)
    if signal_idx is not None and signal in ("BUY", "SELL") and 0 <= signal_idx < len(df_u):
        mk = dict(
            x=[df_u["date"].iloc[signal_idx]],
            y=[float(df_u["close"].iloc[signal_idx])],
            mode="markers",
            marker=dict(
                symbol="triangle-up" if signal == "BUY" else "triangle-down",
                size=12,
                color="lime" if signal == "BUY" else "tomato",
                line=dict(width=1, color="black"),
            ),
            name=signal,
        )
        if vr is not None:
            fig.add_trace(go.Scatter(**mk), row=pr, col=1)
        else:
            fig.add_trace(go.Scatter(**mk))
    finalize_ohlc_volume_figure(fig, height=370)
    return fig


def _plot_option_chart(df_o: pd.DataFrame, trade: dict):
    fig, pr, vr = create_ohlc_volume_figure(df_o, include_volume=True)
    add_candlestick_trace(fig, df_o, name=trade["option"], price_row=pr, volume_row=vr)
    if vr is not None:
        add_volume_bar_trace(fig, df_o, volume_row=vr)
    eidx = int(trade["opt_entry_idx"])
    xidx = int(trade["opt_exit_idx"])
    if 0 <= eidx < len(df_o):
        ek = dict(
            x=[df_o["date"].iloc[eidx]],
            y=[trade["entry"]],
            mode="markers",
            marker=dict(symbol="triangle-up", size=11, color="cyan", line=dict(width=1, color="black")),
            name="Entry",
        )
        if vr is not None:
            fig.add_trace(go.Scatter(**ek), row=pr, col=1)
        else:
            fig.add_trace(go.Scatter(**ek))
    if 0 <= xidx < len(df_o):
        xk = dict(
            x=[df_o["date"].iloc[xidx]],
            y=[trade["exit"]],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color="gold", line=dict(width=1, color="orange")),
            name="Exit",
        )
        if vr is not None:
            fig.add_trace(go.Scatter(**xk), row=pr, col=1)
        else:
            fig.add_trace(go.Scatter(**xk))
    hl_kw = dict(row=pr, col=1) if vr is not None else {}
    fig.add_hline(y=float(trade["target"]), line_dash="dash", line_color="rgba(34,197,94,0.85)", annotation_text="Target", **hl_kw)
    fig.add_hline(y=float(trade["stop"]), line_dash="dash", line_color="rgba(239,68,68,0.85)", annotation_text="Stop", **hl_kw)
    finalize_ohlc_volume_figure(fig, height=370)
    return fig


def render_ema_llm_options(kite):
    if st.session_state.nse_instruments is None:
        with st.spinner("Loading NSE instruments..."):
            st.session_state.nse_instruments = kite.instruments(NSE_EXCHANGE)
    if st.session_state.get("nfo_instruments") is None:
        with st.spinner("Loading NFO instruments..."):
            st.session_state.nfo_instruments = kite.instruments(NFO_EXCHANGE)

    nse = st.session_state.nse_instruments
    nfo = st.session_state.nfo_instruments

    st.title("EMA Trend + LLM Options (single-day backtest)")
    st.caption(
        "Trend from configurable EMA, entry from fast/slow EMA cross, and mandatory LLM option + SL/target selection. "
        "Brokerage is fixed at ₹40 per trade."
    )

    today = date.today()
    underlying = st.selectbox("Underlying", options=FO_UNDERLYING_OPTIONS, index=0, key="ema_llm_underlying")
    chosen_date = st.date_input(
        "Session date",
        value=today,
        min_value=today - timedelta(days=365),
        max_value=today,
        key="ema_llm_date",
    )
    intraday_intervals = [i for i in DEFAULT_INTERVALS if i != "day"]
    idx = intraday_intervals.index(DEFAULT_INTRADAY_INTERVAL) if DEFAULT_INTRADAY_INTERVAL in intraday_intervals else 0
    chosen_interval = st.selectbox("Minute interval", intraday_intervals, index=idx, key="ema_llm_interval")

    c1, c2, c3 = st.columns(3)
    with c1:
        trend_ema_period = int(st.number_input("Trend EMA period", min_value=5, max_value=300, value=50, step=1, key="ema_llm_trend_period"))
    with c2:
        fast_ema_period = int(st.number_input("Fast EMA period", min_value=2, max_value=100, value=9, step=1, key="ema_llm_fast_period"))
    with c3:
        slow_ema_period = int(st.number_input("Slow EMA period", min_value=3, max_value=200, value=21, step=1, key="ema_llm_slow_period"))

    t1, t2 = st.columns(2)
    with t1:
        target_pct_ui = float(st.slider("Target above entry (premium %)", min_value=1.0, max_value=200.0, value=25.0, step=0.5, key="ema_llm_target_pct_ui"))
    with t2:
        stop_pct_ui = float(st.slider("Stop below entry (premium %)", min_value=1.0, max_value=60.0, value=10.0, step=0.5, key="ema_llm_stop_pct_ui"))

    f1, f2, f3 = st.columns(3)
    with f1:
        use_vwap_filter = st.checkbox("Use VWAP filter", value=True, key="ema_llm_use_vwap")
    with f2:
        use_strong_candle = st.checkbox("Use strong crossover candle filter", value=True, key="ema_llm_use_strong")
    with f3:
        strong_body = float(st.slider("Strong candle min body/range", min_value=0.1, max_value=0.95, value=0.55, step=0.05, key="ema_llm_body_frac"))

    s1, s2, s3 = st.columns(3)
    with s1:
        sideways_window = int(st.number_input("Sideways window bars", min_value=5, max_value=80, value=14, step=1, key="ema_llm_sw_window"))
    with s2:
        sideways_flat_slope = float(st.slider("Sideways flat trend slope (%)", min_value=0.01, max_value=1.0, value=0.08, step=0.01, key="ema_llm_sw_slope"))
    with s3:
        sideways_crosses = int(st.number_input("Sideways max EMA crossings", min_value=2, max_value=10, value=3, step=1, key="ema_llm_sw_crosses"))

    st.info("Brokerage + costs are fixed to **₹40 per trade**.")
    run = st.button("Run EMA + LLM backtest", type="primary", key="ema_llm_run")
    if fast_ema_period >= slow_ema_period:
        st.error("Fast EMA must be less than Slow EMA.")
        return

    run_payload = {
        "session_date": chosen_date.isoformat(),
        "interval": chosen_interval,
        "trend_ema_period": trend_ema_period,
        "fast_ema_period": fast_ema_period,
        "slow_ema_period": slow_ema_period,
        "target_pct_ui": target_pct_ui,
        "stop_pct_ui": stop_pct_ui,
        "use_vwap_filter": use_vwap_filter,
        "use_strong_candle": use_strong_candle,
        "strong_body": strong_body,
        "sideways_window": sideways_window,
        "sideways_flat_slope": sideways_flat_slope,
        "sideways_crosses": sideways_crosses,
    }
    fp_now = _params_fingerprint(run_payload)

    if run:
        with st.spinner(f"Running EMA+LLM backtest for {underlying} on {chosen_date}..."):
            out = run_ema_llm_backtest_one_day(
                kite=kite,
                nse_instruments=nse,
                nfo_instruments=nfo,
                underlying=underlying,
                session_date=chosen_date,
                chosen_interval=chosen_interval,
                trend_ema_period=trend_ema_period,
                fast_ema_period=fast_ema_period,
                slow_ema_period=slow_ema_period,
                target_pct=target_pct_ui / 100.0,
                stop_pct=stop_pct_ui / 100.0,
                use_vwap_filter=use_vwap_filter,
                use_strong_candle=use_strong_candle,
                strong_candle_min_body_frac=strong_body,
                brokerage_per_trade=40.0,
                sideways_window=sideways_window,
                sideways_flat_slope_pct=sideways_flat_slope,
                sideways_max_crosses=sideways_crosses,
            )
        st.markdown("## Selected scrip run")
        _render_single_run_output(
            out,
            trend_ema_period=trend_ema_period,
            fast_ema_period=fast_ema_period,
            slow_ema_period=slow_ema_period,
            chart_key_prefix=f"ema_single_{underlying}",
        )

    st.divider()
    st.markdown("## All scrips — same parameters")
    all_cache = st.session_state.get("ema_llm_all_scrips_cache")
    if all_cache and all_cache.get("fp") != fp_now:
        st.warning("All-scrips cache was built with different parameters. Click refresh to update.")
    refresh_all = st.button("Load / refresh all scrips", key="ema_llm_all_scrips_refresh")
    if refresh_all:
        rows = []
        with st.spinner(f"Running full EMA+LLM flow for {len(FO_UNDERLYING_OPTIONS)} scrips..."):
            for u in FO_UNDERLYING_OPTIONS:
                out_u = run_ema_llm_backtest_one_day(
                    kite=kite,
                    nse_instruments=nse,
                    nfo_instruments=nfo,
                    underlying=u,
                    session_date=chosen_date,
                    chosen_interval=chosen_interval,
                    trend_ema_period=trend_ema_period,
                    fast_ema_period=fast_ema_period,
                    slow_ema_period=slow_ema_period,
                    target_pct=target_pct_ui / 100.0,
                    stop_pct=stop_pct_ui / 100.0,
                    use_vwap_filter=use_vwap_filter,
                    use_strong_candle=use_strong_candle,
                    strong_candle_min_body_frac=strong_body,
                    brokerage_per_trade=40.0,
                    sideways_window=sideways_window,
                    sideways_flat_slope_pct=sideways_flat_slope,
                    sideways_max_crosses=sideways_crosses,
                )
                rows.append({"underlying": u, "out": out_u})
        st.session_state["ema_llm_all_scrips_cache"] = {"fp": fp_now, "rows": rows}

    show_all = st.session_state.get("ema_llm_all_scrips_cache")
    if show_all and show_all.get("fp") == fp_now and show_all.get("rows"):
        df_sum, totals = _bulk_summary_rows(show_all["rows"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("All signals", totals["signals"])
        c2.metric("All executed", totals["executed"])
        c3.metric("All skipped", totals["skipped"])
        c4.metric("All net P/L", f"₹{totals['net_pl']:+,.2f}")
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
        st.markdown("### All-scrip details")
        for row in show_all["rows"]:
            u = row["underlying"]
            with st.expander(f"{u} details", expanded=False):
                _render_single_run_output(
                    row["out"],
                    trend_ema_period=trend_ema_period,
                    fast_ema_period=fast_ema_period,
                    slow_ema_period=slow_ema_period,
                    chart_key_prefix=f"ema_all_{u}",
                )
    elif not refresh_all:
        st.info("Click **Load / refresh all scrips** to run and view all-scrip results.")

    st.divider()
    st.markdown("## Index scrips — same parameters")
    idx_cache = st.session_state.get("ema_llm_index_scrips_cache")
    if idx_cache and idx_cache.get("fp") != fp_now:
        st.warning("Index-scrip cache was built with different parameters. Click refresh to update.")
    refresh_idx = st.button("Load / refresh index scrips", key="ema_llm_index_scrips_refresh")
    if refresh_idx:
        rows = []
        with st.spinner(f"Running full EMA+LLM flow for {len(FO_DEFAULT_UNDERLYINGS)} index scrips..."):
            for u in FO_DEFAULT_UNDERLYINGS:
                out_u = run_ema_llm_backtest_one_day(
                    kite=kite,
                    nse_instruments=nse,
                    nfo_instruments=nfo,
                    underlying=u,
                    session_date=chosen_date,
                    chosen_interval=chosen_interval,
                    trend_ema_period=trend_ema_period,
                    fast_ema_period=fast_ema_period,
                    slow_ema_period=slow_ema_period,
                    target_pct=target_pct_ui / 100.0,
                    stop_pct=stop_pct_ui / 100.0,
                    use_vwap_filter=use_vwap_filter,
                    use_strong_candle=use_strong_candle,
                    strong_candle_min_body_frac=strong_body,
                    brokerage_per_trade=40.0,
                    sideways_window=sideways_window,
                    sideways_flat_slope_pct=sideways_flat_slope,
                    sideways_max_crosses=sideways_crosses,
                )
                rows.append({"underlying": u, "out": out_u})
        st.session_state["ema_llm_index_scrips_cache"] = {"fp": fp_now, "rows": rows}

    show_idx = st.session_state.get("ema_llm_index_scrips_cache")
    if show_idx and show_idx.get("fp") == fp_now and show_idx.get("rows"):
        df_sum, totals = _bulk_summary_rows(show_idx["rows"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Index signals", totals["signals"])
        c2.metric("Index executed", totals["executed"])
        c3.metric("Index skipped", totals["skipped"])
        c4.metric("Index net P/L", f"₹{totals['net_pl']:+,.2f}")
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
        st.markdown("### Index-scrip details")
        for row in show_idx["rows"]:
            u = row["underlying"]
            with st.expander(f"{u} details", expanded=False):
                _render_single_run_output(
                    row["out"],
                    trend_ema_period=trend_ema_period,
                    fast_ema_period=fast_ema_period,
                    slow_ema_period=slow_ema_period,
                    chart_key_prefix=f"ema_idx_{u}",
                )
    elif not refresh_idx:
        st.info("Click **Load / refresh index scrips** to run and view index-only results.")
