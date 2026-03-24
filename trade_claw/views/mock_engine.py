"""Mock AI engine HUD: telemetry from worker + live Kite charts/LTP."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_claw import mock_engine_telemetry
from trade_claw import mock_trade_analytics as mtd_an
from trade_claw import mock_trade_store
from trade_claw.constants import ENVELOPE_EMA_PERIOD, NFO_EXCHANGE, NSE_EXCHANGE
from trade_claw.market_data import candles_to_dataframe
from trade_claw.mock_market_signal import load_nifty_session_minute_df, mock_agent_envelope_pct, now_ist
from trade_claw.strategies import add_ma_envelope_line_traces
from trade_claw.task_runtime import MOCK_TRADES_DB_PATH


def _session_date_today_ist() -> date:
    return now_ist().date()


def _nifty_ltp(kite) -> float | None:
    try:
        r = kite.ltp(["NSE:NIFTY 50"]).get("NSE:NIFTY 50") or {}
        v = r.get("last_price")
        return float(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def _opt_ltp(kite, tradingsymbol: str | None) -> float | None:
    if not tradingsymbol:
        return None
    try:
        k = f"NFO:{tradingsymbol}"
        r = kite.ltp([k]).get(k) or {}
        v = r.get("last_price")
        return float(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def _opt_minute_df(kite, nfo_instruments: list, tradingsymbol: str | None, session_d: date):
    if not tradingsymbol or not nfo_instruments:
        return None
    token = None
    for i in nfo_instruments:
        if i.get("tradingsymbol") == tradingsymbol:
            token = i.get("instrument_token")
            break
    if token is None:
        return None
    from_dt = datetime(session_d.year, session_d.month, session_d.day, 9, 15, 0)
    to_dt = datetime(session_d.year, session_d.month, session_d.day, 15, 30, 0)
    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")
    try:
        candles = kite.historical_data(int(token), from_str, to_str, interval="minute")
    except Exception:  # noqa: BLE001
        return None
    df = candles_to_dataframe(candles)
    if df is None or df.empty:
        return None
    return df.sort_values("date").reset_index(drop=True)


def _mock_replay_add_option_hlines(fig: go.Figure, row) -> None:
    """Match Live tab option chart: entry / target / stop; plus exit when present."""
    ep = float(row["entry_price"]) if pd.notna(row.get("entry_price")) else 0.0
    tp = float(row["target"]) if pd.notna(row.get("target")) else 0.0
    sp = float(row["stop_loss"]) if pd.notna(row.get("stop_loss")) else 0.0
    xp = float(row["exit_price"]) if pd.notna(row.get("exit_price")) else 0.0
    if ep > 0:
        fig.add_hline(y=ep, line_dash="dot", line_color="cyan", annotation_text="Entry")
    if tp > ep:
        fig.add_hline(y=tp, line_dash="dot", line_color="lime", annotation_text="Target")
    if 0 < sp < ep:
        fig.add_hline(y=sp, line_dash="dot", line_color="tomato", annotation_text="Stop")
    if xp > 0:
        fig.add_hline(y=xp, line_dash="dash", line_color="orange", annotation_text="Exit")


def _render_mock_analytics() -> None:
    mock_trade_store.init_db()
    st.subheader("Multi-month analysis (`mock_trades`)")
    st.caption(
        "Ledger fields are the source of truth for long horizons. "
        "Timestamps are stored as **naive UTC** strings; the range below uses **IST calendar days** "
        "(Asia/Kolkata) converted to UTC for SQL bounds. "
        "Kite does not reliably provide old option intraday data — use **snapshots** (env `MOCK_ENGINE_SNAPSHOT_BARS`) for replay."
    )
    today = _session_date_today_ist()
    c0, c1 = st.columns(2)
    with c0:
        d0 = st.date_input("From (IST date)", value=today - timedelta(days=90), key="mock_an_from")
    with c1:
        d1 = st.date_input("To (IST date)", value=today, key="mock_an_to")
    if d0 > d1:
        st.error("From date must be on or before To date.")
        return

    table_closed_only = st.checkbox("Trade table: closed rows only", value=False, key="mock_an_tbl_closed")

    df_all = mtd_an.load_trades_for_analytics(d0, d1, status=None, limit=10_000)
    if df_all.empty:
        st.info("No trades in this IST date window.")
        return
    df_closed = mtd_an.closed_trades_df(df_all)
    n_open = int((df_all["status"].astype(str).str.upper() == "OPEN").sum())

    kp = mtd_an.kpis_closed(df_closed)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Closed trades", kp["closed_count"])
    with m2:
        st.metric("Win rate", f"{100 * kp['win_rate']:.1f}%" if kp["win_rate"] is not None else "—")
    with m3:
        st.metric("Total PnL (₹)", f"{kp['total_pnl']:,.2f}")
    with m4:
        st.metric("Avg PnL (₹)", f"{kp['avg_pnl']:,.2f}" if kp["avg_pnl"] is not None else "—")
    with m5:
        st.metric("Best / worst (₹)", f"{kp['best_trade'] or 0:,.0f} / {kp['worst_trade'] or 0:,.0f}")
    with m6:
        st.metric("OPEN in range", n_open)

    if not df_closed.empty:
        monthly = mtd_an.monthly_pnl_summary(df_closed)
        if not monthly.empty:
            fig_m = go.Figure(
                data=[
                    go.Bar(x=monthly["month"], y=monthly["total_pnl"], name="Monthly PnL"),
                ]
            )
            fig_m.update_layout(
                title="Realised PnL by month (exit in range, IST month)",
                template="plotly_dark",
                height=320,
            )
            st.plotly_chart(fig_m, use_container_width=True, key="mock_an_monthly")

        eq = mtd_an.equity_curve_by_exit(df_closed)
        if not eq.empty:
            fig_e = go.Figure(
                data=[
                    go.Scatter(
                        x=eq["exit_time_utc"],
                        y=eq["cum_pnl"],
                        mode="lines+markers",
                        name="Equity",
                    )
                ]
            )
            fig_e.update_layout(
                title="Cumulative realised PnL (by exit time, UTC)",
                template="plotly_dark",
                height=320,
            )
            st.plotly_chart(fig_e, use_container_width=True, key="mock_an_equity")

        br = mtd_an.direction_breakdown(df_closed)
        if not br.empty:
            st.markdown("**By direction (closed)**")
            st.dataframe(br, use_container_width=True, hide_index=True)

    disp = df_closed if table_closed_only else df_all
    if disp.empty:
        st.info("No rows for this filter (try unchecking “closed only”).")
        return

    _json_cols = [
        "entry_bars_json",
        "exit_bars_json",
        "entry_underlying_bars_json",
        "exit_underlying_bars_json",
    ]
    show = disp.drop(columns=_json_cols, errors="ignore")
    st.markdown("**Trades**")
    st.dataframe(show, use_container_width=True, hide_index=True)

    csv_df = disp.drop(columns=_json_cols, errors="ignore")
    st.download_button(
        label="Download CSV (filtered window)",
        data=mtd_an.trades_to_csv_bytes(csv_df),
        file_name=f"mock_trades_{d0}_{d1}.csv",
        mime="text/csv",
        key="mock_an_csv",
    )

    has_opt = (
        disp["entry_bars_json"].notna() | disp["exit_bars_json"].notna()
        if "entry_bars_json" in disp.columns
        else pd.Series(False, index=disp.index)
    )
    has_u = (
        disp["entry_underlying_bars_json"].notna() | disp["exit_underlying_bars_json"].notna()
        if "entry_underlying_bars_json" in disp.columns
        else pd.Series(False, index=disp.index)
    )
    snap_rows = disp[has_opt | has_u] if not disp.empty else pd.DataFrame()
    if not snap_rows.empty:
        with st.expander("Replay stored minute snapshots (entry/exit)", expanded=False):
            st.caption(
                "NIFTY envelope bands use **current** `MOCK_AGENT_ENVELOPE_PCT` (same as Live), not the value at trade time."
            )
            choices = [int(x) for x in snap_rows["trade_id"].tolist()]
            tid = st.selectbox("trade_id", choices, key="mock_an_snap_tid")
            leg = st.radio("Leg", ["entry", "exit"], horizontal=True, key="mock_an_snap_leg")
            row = snap_rows[snap_rows["trade_id"] == tid].iloc[0]
            raw_u = (
                row["entry_underlying_bars_json"]
                if leg == "entry"
                else row.get("exit_underlying_bars_json")
            )
            if raw_u is not None and not (isinstance(raw_u, float) and pd.isna(raw_u)):
                try:
                    ubars = json.loads(str(raw_u))
                    if ubars:
                        df_u = pd.DataFrame(ubars)
                        df_u["date"] = pd.to_datetime(df_u["date"])
                        fig_u = go.Figure(
                            data=[
                                go.Candlestick(
                                    x=df_u["date"],
                                    open=df_u["open"],
                                    high=df_u["high"],
                                    low=df_u["low"],
                                    close=df_u["close"],
                                    name="NIFTY 50",
                                )
                            ]
                        )
                        add_ma_envelope_line_traces(
                            fig_u,
                            df_u,
                            ema_period=ENVELOPE_EMA_PERIOD,
                            pct=mock_agent_envelope_pct(),
                        )
                        fig_u.update_layout(
                            title=(
                                f"Trade {tid} — NIFTY 50 ({leg}) stored snapshot "
                                f"({len(ubars)} bars) + EMA envelope"
                            ),
                            template="plotly_dark",
                            height=300,
                            xaxis_rangeslider_visible=False,
                        )
                        st.plotly_chart(fig_u, use_container_width=True, key="mock_an_snap_under")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    st.warning(f"Could not plot underlying snapshot: {e}")
            else:
                st.caption("No underlying snapshot for this leg (older trades or capture failed).")

            raw = row["entry_bars_json"] if leg == "entry" else row["exit_bars_json"]
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                st.caption("No option snapshot for this leg.")
            else:
                try:
                    bars = json.loads(str(raw))
                    if not bars:
                        st.caption("Empty option snapshot.")
                    else:
                        dfp = pd.DataFrame(bars)
                        dfp["date"] = pd.to_datetime(dfp["date"])
                        fig_s = go.Figure(
                            data=[
                                go.Candlestick(
                                    x=dfp["date"],
                                    open=dfp["open"],
                                    high=dfp["high"],
                                    low=dfp["low"],
                                    close=dfp["close"],
                                    name=f"{leg} snapshot",
                                )
                            ]
                        )
                        _mock_replay_add_option_hlines(fig_s, row)
                        fig_s.update_layout(
                            title=f"Trade {tid} — option premium ({leg}, {len(bars)} bars)",
                            template="plotly_dark",
                            height=300,
                            xaxis_rangeslider_visible=False,
                        )
                        st.plotly_chart(fig_s, use_container_width=True, key="mock_an_snap_chart")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    st.warning(f"Could not plot option snapshot: {e}")


def render_mock_engine(kite):
    st.title("Mock AI engine (Nifty options)")
    st.caption(
        "Celery Beat runs `scan_mock_market` during NSE hours; the worker writes **telemetry** (last scan + last graph) "
        "and **trades** to SQLite. This page merges that with **live** Kite quotes and intraday charts."
    )
    st.markdown(f"| DB path | `{MOCK_TRADES_DB_PATH}` |")

    if st.session_state.nse_instruments is None:
        with st.spinner("Loading NSE instruments..."):
            try:
                st.session_state.nse_instruments = kite.instruments(NSE_EXCHANGE)
            except Exception as e:  # noqa: BLE001
                st.error(f"NSE instruments: {e}")
                return
    if st.session_state.get("nfo_instruments") is None:
        with st.spinner("Loading NFO instruments..."):
            try:
                st.session_state.nfo_instruments = kite.instruments(NFO_EXCHANGE)
            except Exception as e:  # noqa: BLE001
                st.error(f"NFO instruments: {e}")
                return

    nse = st.session_state.nse_instruments
    nfo = st.session_state.nfo_instruments
    session_d = _session_date_today_ist()
    env_pct = mock_agent_envelope_pct()

    @st.fragment(run_every=timedelta(seconds=20))
    def _hud():
        mock_trade_store.init_db()
        mock_engine_telemetry.init_telemetry_table()
        snap = mock_engine_telemetry.read_snapshot()
        last_scan = snap.get("last_scan") or {}
        last_graph = snap.get("last_graph") or {}

        live_nifty = _nifty_ltp(kite)
        open_rows = mock_trade_store.list_open_trades()
        live_opt = _opt_ltp(kite, open_rows[0].instrument if open_rows else None)

        st.subheader("Live quotes & worker status")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("NIFTY 50 LTP (live)", f"₹{live_nifty:,.2f}" if live_nifty else "—")
        with c2:
            st.metric(
                "Option LTP (live)",
                f"₹{live_opt:,.2f}" if live_opt is not None else "—",
                help="Open mock position only",
            )
        with c3:
            st.metric(
                "Last worker scan (IST ISO)",
                (last_scan.get("ist") or "")[:19] if last_scan.get("ist") else "—",
            )
        with c4:
            st.metric("Scan skip / state", last_scan.get("skipped") or "ok")
        with c5:
            st.metric("Envelope ±", f"{100 * env_pct:.3f}%", help="MOCK_AGENT_ENVELOPE_PCT each side of EMA20")

        if open_rows:
            st.subheader("Current mock position")
            r = open_rows[0]
            u_pnl = None
            if live_opt is not None and r.entry_price:
                u_pnl = (live_opt - float(r.entry_price)) * int(r.quantity or 1)
            st.markdown(f"**Instrument** `{r.instrument or '—'}` · **Direction** {r.direction or '—'}")
            oc3, oc4, oc5, oc6 = st.columns(4)
            with oc3:
                st.metric("Entry (mock)", f"₹{float(r.entry_price or 0):,.2f}")
            with oc4:
                st.metric("Stop / Target", f"₹{float(r.stop_loss or 0):,.2f} / ₹{float(r.target or 0):,.2f}")
            with oc5:
                st.metric("Qty (units)", int(r.quantity or 0))
            with oc6:
                st.metric("Unrealised (indic.)", f"₹{u_pnl:,.2f}" if u_pnl is not None else "—")
            if r.llm_rationale:
                st.caption("LLM rationale at entry")
                st.write(r.llm_rationale)
        else:
            st.info("No **OPEN** mock trade. Worker may skip with `open_position` only after one is opened.")

        st.subheader("Nifty spot — session candles + agent envelope (EMA20)")
        st.caption(
            f"Same geometry as worker: EMA **{ENVELOPE_EMA_PERIOD}** ± **{100 * env_pct:.3f}%** per side."
        )
        df_u, err_u = load_nifty_session_minute_df(kite, nse, session_d)
        if df_u is not None and not df_u.empty and not err_u:
            fig_u = go.Figure()
            fig_u.add_trace(
                go.Candlestick(
                    x=df_u["date"],
                    open=df_u["open"],
                    high=df_u["high"],
                    low=df_u["low"],
                    close=df_u["close"],
                    name="NIFTY 50",
                )
            )
            add_ma_envelope_line_traces(fig_u, df_u, ema_period=ENVELOPE_EMA_PERIOD, pct=env_pct)
            fig_u.update_layout(
                title="NIFTY 50 (1m) + envelope",
                height=380,
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
            )
            st.plotly_chart(fig_u, use_container_width=True, key="mock_eng_spot")
        else:
            st.warning(err_u or "No underlying candles for today (holiday, pre-open, or API).")

        if open_rows and open_rows[0].instrument:
            st.subheader("Option premium (1m session)")
            df_o = _opt_minute_df(kite, nfo, open_rows[0].instrument, session_d)
            if df_o is not None and not df_o.empty:
                fig_o = go.Figure()
                fig_o.add_trace(
                    go.Candlestick(
                        x=df_o["date"],
                        open=df_o["open"],
                        high=df_o["high"],
                        low=df_o["low"],
                        close=df_o["close"],
                        name=open_rows[0].instrument,
                    )
                )
                ep = float(open_rows[0].entry_price or 0)
                tp = float(open_rows[0].target or 0)
                sp = float(open_rows[0].stop_loss or 0)
                if ep > 0:
                    fig_o.add_hline(y=ep, line_dash="dot", line_color="cyan", annotation_text="Entry")
                if tp > ep:
                    fig_o.add_hline(y=tp, line_dash="dot", line_color="lime", annotation_text="Target")
                if 0 < sp < ep:
                    fig_o.add_hline(y=sp, line_dash="dot", line_color="tomato", annotation_text="Stop")
                fig_o.update_layout(
                    title=f"{open_rows[0].instrument} — premium",
                    height=320,
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                )
                st.plotly_chart(fig_o, use_container_width=True, key="mock_eng_opt")
            else:
                st.caption("Could not load option minute series (symbol/expiry or API).")

        st.subheader("Last agent graph snapshot (from worker)")
        if not last_graph:
            st.caption("No graph telemetry yet — start **worker + beat** and wait for a scan inside 09:15–15:19 IST.")
        else:
            g1, g2, g3 = st.columns(3)
            with g1:
                st.markdown(f"**Direction** · {last_graph.get('direction') or '—'}")
                st.markdown(f"**Leg** · {last_graph.get('leg') or '—'}")
            with g2:
                sp = last_graph.get("spot")
                st.markdown(f"**Spot @ signal** · ₹{float(sp):,.2f}" if sp is not None else "**Spot** · —")
            with g3:
                st.markdown(f"**Trade id** · {last_graph.get('trade_id') or '—'}")
                if last_graph.get("error"):
                    st.error(last_graph["error"])
            if last_graph.get("signal_text") or last_graph.get("notes"):
                st.text_area(
                    "Signal / notes",
                    value=str(last_graph.get("signal_text") or last_graph.get("notes") or ""),
                    height=100,
                    disabled=True,
                    key="mock_sig_txt",
                )
            cands = last_graph.get("candidates")
            if cands:
                st.markdown("**Top candidates seen by LLM**")
                st.dataframe(pd.DataFrame(cands), use_container_width=True, hide_index=True)
            if last_graph.get("llm_tradingsymbol"):
                st.markdown("**LLM pick**")
                st.json(
                    {
                        "tradingsymbol": last_graph.get("llm_tradingsymbol"),
                        "stop_loss": last_graph.get("stop_loss"),
                        "target": last_graph.get("target"),
                        "rationale": last_graph.get("llm_rationale"),
                    }
                )

        st.subheader("Scan side-effects (last tick)")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.write(
                f"Exits on target/stop: **{last_scan.get('exits_sl_target', 0)}** · "
                f"15:20 square-off closes: **{last_scan.get('exits_square_off', 0)}**"
            )
        with ex2:
            if last_scan.get("graph"):
                st.json(last_scan["graph"])

        with st.expander("Raw `last_scan` (worker)", expanded=False):
            st.json(last_scan)

        st.subheader("Mock trade book")
        rows = mock_trade_store.list_recent_trades(limit=100)
        if not rows:
            st.caption("No rows in `mock_trades` yet.")
        else:
            data = []
            for r in rows:
                data.append(
                    {
                        "trade_id": r.trade_id,
                        "entry_time": r.entry_time,
                        "exit_time": r.exit_time,
                        "instrument": r.instrument,
                        "direction": r.direction,
                        "entry": r.entry_price,
                        "stop": r.stop_loss,
                        "target": r.target,
                        "status": r.status,
                        "exit": r.exit_price,
                        "PnL": r.realized_pnl,
                        "qty": r.quantity,
                    }
                )
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        chart_rows = mock_trade_store.trades_for_chart()
        if chart_rows:
            cum = []
            s = 0.0
            xs = []
            for cr in chart_rows:
                pnl = float(cr["realized_pnl"] or 0)
                s += pnl
                xs.append(cr["trade_id"])
                cum.append(s)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=cum,
                    mode="lines+markers",
                    name="Cumulative realised PnL",
                    line=dict(color="#00d4ff", width=2),
                )
            )
            fig.update_layout(
                title="Cumulative realised PnL (closed trades)",
                xaxis_title="trade_id",
                yaxis_title="₹",
                template="plotly_dark",
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True, key="mock_eng_cum")

    tab_live, tab_an = st.tabs(["Live dashboard", "Analytics"])
    with tab_live:
        _hud()
    with tab_an:
        _render_mock_analytics()
