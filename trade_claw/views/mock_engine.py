"""Mock AI engine HUD: telemetry from worker + live Kite charts/LTP."""

from __future__ import annotations

import base64
import json
import re
from collections import defaultdict
from pathlib import Path
from datetime import UTC, date, datetime, timedelta
from typing import NamedTuple
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from trade_claw import mock_engine_telemetry
from trade_claw.mock_engine_run import manual_close_mock_trade
from trade_claw import mock_trade_analytics as mtd_an
from trade_claw import mock_trade_store
from trade_claw.constants import (
    ENVELOPE_EMA_PERIOD,
    FO_INDEX_UNDERLYING_LABELS,
    NFO_EXCHANGE,
    NSE_EXCHANGE,
)
from trade_claw.env_trading_params import (
    mock_engine_equity_envelope_decimal_per_side,
    mock_engine_index_envelope_decimal_per_side,
)
from trade_claw.market_data import candles_to_dataframe
from trade_claw.mock_index_vision_trend import (
    VISION_INDEX_KEYS,
    build_index_vision_candlestick_figure,
)
from trade_claw.mock_market_signal import (
    load_index_session_interval_df,
    load_index_session_minute_df,
    mock_agent_envelope_pct_for_underlying,
    mock_engine_underlyings,
    nse_index_ltp_symbol,
    now_ist,
)
from trade_claw.strategies import _envelope_series, add_ma_envelope_line_traces
from trade_claw.task_runtime import MOCK_TRADES_DB_PATH

_IST = ZoneInfo("Asia/Kolkata")
_TREND_STATES = frozenset({"BULLISH", "BEARISH", "NEUTRAL"})


def _vision_trend_snapshot(last_scan: dict) -> dict[str, str | None]:
    out: dict[str, str | None] = {k: None for k in VISION_INDEX_KEYS}
    raw = last_scan.get("index_trends_3m")
    if not isinstance(raw, dict):
        return out
    for k in VISION_INDEX_KEYS:
        block = raw.get(k)
        if isinstance(block, dict):
            t = block.get("trend")
            if t in _TREND_STATES:
                out[k] = str(t)
    return out


def _mock_bell_wav_path() -> Path:
    return Path(__file__).resolve().parent.parent / "static" / "mock_bell.wav"


def _format_stored_vision_error(msg: str | None) -> str:
    """
    Telemetry keeps the last per-index vision error in SQLite. Older builds used Plotly+Kaleido
    for PNGs; that error text is misleading after switching to matplotlib+mplfinance.
    """
    if not msg:
        return ""
    low = str(msg).lower()
    if "kaleido" in low or "plotly_get_chrome" in low or "google chrome" in low:
        return (
            "This message was saved by an **older worker** that used Plotly static export (Kaleido/Chrome). "
            "**Restart the Celery worker** so it runs the current code (matplotlib + mplfinance, no Chrome). "
            "Ensure the worker env has `matplotlib` and `mplfinance` installed, then wait for the next vision cycle."
        )
    return str(msg).strip()


def _play_mock_bell(*, html_key: str) -> None:
    p = _mock_bell_wav_path()
    if not p.is_file():
        return
    try:
        b64 = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    except OSError:
        return
    components.html(
        f"""
        <audio id="{html_key}" preload="auto">
          <source src="data:audio/wav;base64,{b64}" type="audio/wav" />
        </audio>
        <script>
          (function() {{
            var a = document.getElementById("{html_key}");
            if (a) {{ a.volume = 1.0; a.play().catch(function(){{}}); }}
          }})();
        </script>
        """,
        height=0,
    )


def _utc_naive_sql_to_ist_str(ts: str | None) -> str | None:
    """DB `mock_trades` times are naive UTC strings; show Asia/Kolkata in the trade book."""
    if ts is None:
        return None
    raw = str(ts).strip()
    if not raw:
        return None
    head = raw[:19]
    try:
        dt_utc = datetime.strptime(head, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return raw
    return dt_utc.astimezone(_IST).strftime("%Y-%m-%d %H:%M:%S")


class _SpotAxis(NamedTuple):
    y0: float
    y1: float
    session_zoom: bool  # True when price-space upper/lower sit far off-chart vs session range


def _spot_chart_y_axis(
    df: pd.DataFrame,
    *,
    ema_period: int,
    pct: float,
    wide_vs_candles: float = 4.0,
    pad_ratio: float = 0.06,
) -> _SpotAxis | None:
    """Y-axis bounds; ``session_zoom`` when we zoom to OHLC and bands are off-scale in price view."""
    if df is None or df.empty:
        return None
    need = {"low", "high", "close"}
    if not need.issubset(df.columns):
        return None
    if len(df) < ema_period:
        return None
    candle_lo = float(df["low"].min())
    candle_hi = float(df["high"].max())
    mid = 0.5 * (candle_lo + candle_hi)
    candle_span = max(candle_hi - candle_lo, abs(mid) * 1e-6)
    _, upper, lower = _envelope_series(df, ema_period, pct)
    env_lo = float(lower.min())
    env_hi = float(upper.max())
    if not all(map(lambda x: x == x, (env_lo, env_hi))):  # filter NaN
        return None
    env_span = max(env_hi - env_lo, 1e-9)
    floor = max(candle_span, abs(mid) * 0.015)
    pad = max(candle_span * pad_ratio, abs(mid) * 0.0003)
    if env_span > wide_vs_candles * floor:
        return _SpotAxis(candle_lo - pad, candle_hi + pad, True)
    return _SpotAxis(
        min(candle_lo, env_lo) - pad,
        max(candle_hi, env_hi) + pad,
        False,
    )


def _mock_index_spot_figure(
    df: pd.DataFrame,
    idx_label: str,
    env_pct: float,
    *,
    ema_period: int = ENVELOPE_EMA_PERIOD,
) -> tuple[go.Figure, bool]:
    """Spot candles + EMA; when envelope is huge vs session, add a lower panel: % vs EMA with ±pct bands."""
    axis = _spot_chart_y_axis(df, ema_period=ema_period, pct=env_pct)
    if axis is None or len(df) < ema_period:
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=idx_label,
            )
        )
        add_ma_envelope_line_traces(fig, df, ema_period=ema_period, pct=env_pct)
        return fig, False

    y0, y1, session_zoom = axis
    pct_label = f"{100 * env_pct:.1f}%"
    band_pct = 100.0 * env_pct

    if session_zoom:
        center, upper, lower = _envelope_series(df, ema_period, env_pct)
        safe_c = center.replace(0, float("nan"))
        pct_vs = ((df["close"] / safe_c) - 1.0) * 100.0
        pct_vs = pct_vs.fillna(0.0)
        xs = df["date"]
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            row_heights=[0.66, 0.34],
            subplot_titles=(None, f"% from EMA — dashed lines at ±{pct_label} (same envelope as worker)"),
        )
        fig.add_trace(
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=idx_label,
            ),
            row=1,
            col=1,
        )
        add_ma_envelope_line_traces(
            fig,
            df,
            ema_period=ema_period,
            pct=env_pct,
            include_price_bands=False,
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[band_pct] * len(xs),
                mode="lines",
                name=f"Upper (+{pct_label})",
                line=dict(color="rgba(0, 220, 130, 0.9)", width=1.5, dash="dash"),
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[-band_pct] * len(xs),
                mode="lines",
                name=f"Lower (−{pct_label})",
                line=dict(color="rgba(255, 120, 100, 0.9)", width=1.5, dash="dash"),
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=pct_vs,
                mode="lines",
                name="% vs EMA",
                line=dict(color="#c4b5fd", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(range=[y0, y1], autorange=False, row=1, col=1)
        y2_hi = max(float(pct_vs.max()), band_pct, 1.0) * 1.12
        y2_lo = min(float(pct_vs.min()), -band_pct, -1.0) * 1.12
        fig.update_yaxes(
            range=[y2_lo, y2_hi],
            autorange=False,
            title_text="%",
            row=2,
            col=1,
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig, True

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=idx_label,
        )
    )
    add_ma_envelope_line_traces(fig, df, ema_period=ema_period, pct=env_pct)
    fig.update_layout(
        yaxis=dict(range=[y0, y1], autorange=False),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
    )
    return fig, False


def _session_date_today_ist() -> date:
    return now_ist().date()


def _nse_underlying_quote(kite, underlying_key: str) -> tuple[float | None, float | None]:
    """Spot quote for index key or equity symbol (NSE cash)."""
    sym = nse_index_ltp_symbol(underlying_key) or f"NSE:{underlying_key.upper().strip()}"
    return _nse_symbol_quote(kite, sym)


def _nse_symbol_quote(kite, nse_key: str) -> tuple[float | None, float | None]:
    try:
        q = kite.quote([nse_key]).get(nse_key) or {}
        lp = q.get("last_price")
        ohlc = q.get("ohlc") if isinstance(q.get("ohlc"), dict) else {}
        prev = ohlc.get("close")
        lpf = float(lp) if lp is not None else None
        prevf = float(prev) if prev is not None else None
        if lpf is not None and prevf is not None:
            return lpf, lpf - prevf
        return lpf, None
    except Exception:  # noqa: BLE001
        return None, None


def _safe_key_fragment(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s)[:48] or "x"


def _opt_ltp(kite, tradingsymbol: str | None) -> float | None:
    lp, _d = _opt_quote(kite, tradingsymbol)
    return lp


def _opt_quote(kite, tradingsymbol: str | None) -> tuple[float | None, float | None]:
    """(last_price, day_change vs quote ohlc.close). Delta may be None."""
    if not tradingsymbol:
        return None, None
    try:
        k = f"NFO:{tradingsymbol}"
        q = kite.quote([k]).get(k) or {}
        lp = q.get("last_price")
        ohlc = q.get("ohlc") if isinstance(q.get("ohlc"), dict) else {}
        prev = ohlc.get("close")
        lpf = float(lp) if lp is not None else None
        prevf = float(prev) if prev is not None else None
        if lpf is not None and prevf is not None:
            return lpf, lpf - prevf
        return lpf, None
    except Exception:  # noqa: BLE001
        return None, None


def _render_last_graph_run_panel(run: dict | None, *, key_prefix: str) -> None:
    """Worker telemetry + LLM inference for one underlying (from ``last_graph``)."""
    if not run:
        st.caption(
            "No worker row for this scrip yet — start **worker + beat** and wait for a scan inside 09:15–15:19 IST."
        )
        return
    if run.get("skipped"):
        st.info(f"**Skipped:** {run.get('skipped')}")
    if run.get("error"):
        st.error(str(run["error"]))
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(f"**Direction** · {run.get('direction') or '—'}")
        st.markdown(f"**Leg** · {run.get('leg') or '—'}")
    with g2:
        sp = run.get("spot")
        st.markdown(
            f"**Spot @ signal** · ₹{float(sp):,.2f}" if sp is not None else "**Spot** · —"
        )
    with g3:
        st.markdown(f"**Trade id** · {run.get('trade_id') or '—'}")
        if run.get("llm_skip_trade"):
            st.caption("LLM **declined** to open (proceed_with_trade=false)")
    sig_txt = run.get("signal_text")
    notes_txt = run.get("notes")
    if sig_txt:
        st.text_area(
            "Envelope / signal (worker)",
            value=str(sig_txt),
            height=120,
            disabled=True,
            key=f"mock_sig_txt_{key_prefix}",
        )
    if notes_txt and str(notes_txt).strip() != str(sig_txt or "").strip():
        st.text_area(
            "Additional graph notes",
            value=str(notes_txt),
            height=100,
            disabled=True,
            key=f"mock_notes_txt_{key_prefix}",
        )
    rat = run.get("llm_rationale")
    if rat:
        st.markdown("**LLM rationale**")
        st.info(str(rat))
    cands = run.get("candidates")
    if cands:
        st.markdown("**Candidates sent to LLM**")
        st.dataframe(pd.DataFrame(cands), use_container_width=True, hide_index=True)
    if run.get("llm_tradingsymbol") or run.get("llm_skip_trade"):
        st.markdown("**LLM decision (structured)**")
        st.json(
            {
                "llm_skip_trade": run.get("llm_skip_trade"),
                "tradingsymbol": run.get("llm_tradingsymbol"),
                "stop_loss": run.get("stop_loss"),
                "target": run.get("target"),
                "rationale": run.get("llm_rationale"),
            }
        )


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
        "Ledger fields are the source of truth for long horizons. The engine may hold **several OPEN rows at once** "
        "(**at most one per index** underlying). This tab lists **every** `trade_id` in the date range; **OPEN in range** "
        "counts all of them. **Replay** is per trade (pick `trade_id`). "
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

        ix_br = mtd_an.index_underlying_breakdown(df_closed)
        if not ix_br.empty:
            st.markdown("**By index underlying (closed)**")
            st.dataframe(ix_br, use_container_width=True, hide_index=True)

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
                "Index spot envelope bands use **current** per-underlying bandwidth "
                "(`MOCK_ENGINE_INDEX_ENVELOPE_PCT` / `MOCK_ENGINE_EQUITY_ENVELOPE_PCT` or `MOCK_AGENT_ENVELOPE_PCT`), "
                "not the value at trade time. Underlying series matches the trade’s **index** column when set."
            )
            choices = [int(x) for x in snap_rows["trade_id"].tolist()]
            tid = st.selectbox("trade_id", choices, key="mock_an_snap_tid")
            leg = st.radio("Leg", ["entry", "exit"], horizontal=True, key="mock_an_snap_leg")
            row = snap_rows[snap_rows["trade_id"] == tid].iloc[0]
            _iu = None
            if "index_underlying" in row.index:
                v = row["index_underlying"]
                if pd.notna(v) and str(v).strip():
                    _iu = str(v).strip().upper()
            snap_idx = _iu or "NIFTY"
            snap_idx_label = FO_INDEX_UNDERLYING_LABELS.get(snap_idx, snap_idx)
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
                        env_pct_snap = mock_agent_envelope_pct_for_underlying(snap_idx)
                        fig_u, _dual_u = _mock_index_spot_figure(
                            df_u,
                            snap_idx_label,
                            env_pct_snap,
                            ema_period=ENVELOPE_EMA_PERIOD,
                        )
                        fig_u.update_layout(
                            title=(
                                f"Trade {tid} — {snap_idx_label} ({leg}) stored snapshot "
                                f"({len(ubars)} bars) + EMA envelope"
                            ),
                            height=420 if _dual_u else 300,
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
    st.title("Mock AI engine (NIFTY · BANKNIFTY · selected equities)")
    st.caption(
        "Celery Beat runs `scan_mock_market` **every minute** (IST weekdays, ~09:15–15:19 entries; 15:20 square-off). "
        "Default scan: **NIFTY**, **BANKNIFTY**, then **MOCK_ENGINE_SCAN_EQUITY_SYMBOLS** in `constants.py` "
        "(override with comma-separated **`MOCK_ENGINE_UNDERLYINGS`**). "
        "Live tab **auto-refreshes every minute**. **At most one OPEN mock trade per underlying.**"
    )
    st.markdown(f"| DB path | `{MOCK_TRADES_DB_PATH}` |")
    _br1, _br2, _br3 = st.columns([1, 1, 2])
    with _br1:
        if st.button(
            "Refresh quotes & charts",
            key="mock_eng_manual_refresh",
            help="Reload this page: latest Kite LTP, intraday charts, and SQLite telemetry/trades.",
        ):
            st.rerun()
    with _br2:
        if st.button("F&O Options", key="mock_nav_fo_page"):
            st.switch_page("pages/F_Options.py")

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
    idx_env_pct = mock_engine_index_envelope_decimal_per_side()
    eq_env_pct = mock_engine_equity_envelope_decimal_per_side()

    @st.fragment(run_every=timedelta(minutes=1))
    def _hud():
        mock_trade_store.init_db()
        mock_engine_telemetry.init_telemetry_table()
        snap = mock_engine_telemetry.read_snapshot()
        last_scan = snap.get("last_scan") or {}
        last_graph = snap.get("last_graph") or {}

        sound_on = st.checkbox(
            "Bell sounds (trend reversal / new trade)",
            True,
            key="mock_engine_sound_on",
            help="Plays a short bell when NIFTY/BANKNIFTY vision trend changes or when a new mock trade_id appears. Browsers may require a prior click on the page.",
        )
        snap_now = _vision_trend_snapshot(last_scan)
        snap_prev = st.session_state.get("mock_last_vision_trends")
        if snap_prev is None:
            st.session_state.mock_last_vision_trends = dict(snap_now)
        else:
            rev = False
            for k in VISION_INDEX_KEYS:
                a, b = snap_prev.get(k), snap_now.get(k)
                if a in _TREND_STATES and b in _TREND_STATES and a != b:
                    rev = True
                    break
            if rev and sound_on:
                _seq = int(st.session_state.get("_mock_bell_seq", 0))
                _play_mock_bell(html_key=f"mock_bell_rev_{_seq}")
                st.session_state._mock_bell_seq = _seq + 1
            st.session_state.mock_last_vision_trends = dict(snap_now)

        mx = mock_trade_store.max_trade_id()
        pm = st.session_state.get("mock_last_max_trade_id")
        if pm is None:
            st.session_state.mock_last_max_trade_id = mx
        elif mx > int(pm):
            if sound_on:
                _seq = int(st.session_state.get("_mock_bell_seq", 0))
                _play_mock_bell(html_key=f"mock_bell_tid_{_seq}")
                st.session_state._mock_bell_seq = _seq + 1
            st.session_state.mock_last_max_trade_id = mx

        open_rows = mock_trade_store.list_open_trades()
        und = mock_engine_underlyings()
        open_by_u: dict[str, list] = defaultdict(list)
        for r in open_rows:
            ux = (r.index_underlying or "").strip().upper()
            if ux:
                open_by_u[ux].append(r)

        st.subheader("Live summary")
        st.caption(
            "Auto-refresh: **every minute** (Kite + SQLite). Worker **every minute** on Beat. "
            f"**{len(und)}** scrips in scan list."
        )

        live_inst = open_rows[0].instrument if len(open_rows) == 1 else None
        live_opt, live_d = _opt_quote(kite, live_inst)
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.metric(
                "OPEN mock legs",
                str(len(open_rows)),
                help="One open position per index underlying at most",
            )
            if len(open_rows) == 1:
                if live_opt is not None:
                    ep0 = float(open_rows[0].entry_price or 0)
                    leg_delta = (live_opt - ep0) if ep0 else None
                    if leg_delta is not None:
                        st.metric(
                            "Option LTP",
                            f"₹{live_opt:,.2f}",
                            delta=f"₹{leg_delta:+,.2f} vs entry",
                            delta_color="normal",
                        )
                    elif live_d is not None:
                        st.metric(
                            "Option LTP",
                            f"₹{live_opt:,.2f}",
                            delta=f"₹{live_d:+,.2f} day",
                            delta_color="normal",
                        )
                    else:
                        st.metric("Option LTP", f"₹{live_opt:,.2f}")
                else:
                    st.caption("Option LTP · —")
            elif open_rows:
                opt_rows = []
                for t in open_rows:
                    lp, _dd = _opt_quote(kite, t.instrument)
                    ep = float(t.entry_price or 0)
                    chg_e = (lp - ep) if lp is not None and ep else None
                    opt_rows.append(
                        {
                            "trade_id": t.trade_id,
                            "index": t.index_underlying or "—",
                            "instrument": t.instrument or "—",
                            "LTP": f"₹{lp:,.2f}" if lp is not None else "—",
                            "vs_entry": chg_e,
                        }
                    )
                df_leg = pd.DataFrame(opt_rows)

                def _leg_chg_color(s: pd.Series) -> list[str]:
                    out: list[str] = []
                    for v in s:
                        if v is None or (isinstance(v, float) and v != v):
                            out.append("")
                        elif v > 0:
                            out.append("color: #22c55e")
                        elif v < 0:
                            out.append("color: #ef4444")
                        else:
                            out.append("")
                    return out

                sty_leg = (
                    df_leg.style.format({"vs_entry": "₹{:+,.2f}"}, na_rep="—")
                    .apply(_leg_chg_color, subset=["vs_entry"])
                )
                st.dataframe(sty_leg, use_container_width=True, hide_index=True)
            if len(open_rows) > 8:
                st.caption("Many open legs — details also appear under each scrip below.")
        with w2:
            st.metric(
                "Last worker scan (IST ISO)",
                (last_scan.get("ist") or "")[:19] if last_scan.get("ist") else "—",
            )
        with w3:
            st.metric("Scan skip / state", last_scan.get("skipped") or "ok")
        with w4:
            if abs(idx_env_pct - eq_env_pct) < 1e-15:
                st.metric(
                    "Envelope ±",
                    f"{100 * idx_env_pct:.3f}%",
                    help="All scrips: MOCK_AGENT_ENVELOPE_PCT, or set MOCK_ENGINE_INDEX_ENVELOPE_PCT / MOCK_ENGINE_EQUITY_ENVELOPE_PCT per kind.",
                )
            else:
                st.metric(
                    "Envelope ±",
                    f"idx {100 * idx_env_pct:.2f}% · eq {100 * eq_env_pct:.2f}%",
                    help="Index: MOCK_ENGINE_INDEX_ENVELOPE_PCT; equity: MOCK_ENGINE_EQUITY_ENVELOPE_PCT; unset kinds fall back to MOCK_AGENT_ENVELOPE_PCT.",
                )

        if not open_rows:
            st.info(
                "No **OPEN** mock trades. The worker opens at most **one leg per underlying** when a breakout fires and that underlying is flat."
            )

        st.subheader("Index vision trend (Claude Haiku 4.5 · 1m chart)")
        st.caption(
            "The worker builds a **1-minute** session candlestick chart for each index, sends it to **Claude Haiku 4.5** "
            "(**MOCK_INDEX_VISION_MODEL**, default `claude-haiku-4-5-20251001`), and refreshes that call every **MOCK_INDEX_VISION_INTERVAL_MIN** minutes (default 3). "
            "Telemetry field `index_trends_3m` is the stored snapshot name; data is **1m** candles. "
            "**Vision rationale** is saved only when that worker step runs successfully (this page does not call Claude). "
            "Use a running **Celery worker + Beat** with **ANTHROPIC_API_KEY** in the worker environment, then wait for a refresh or the throttle interval."
        )
        vis_as_of = last_scan.get("index_vision_as_of_ist")
        vis_model = last_scan.get("index_vision_model") or "—"
        vis_dis = last_scan.get("index_vision_disabled")
        vis_ref = last_scan.get("index_vision_refreshed")
        st.caption(
            f"As-of: **{vis_as_of or '—'}** · model: `{vis_model}` · vision disabled: **{vis_dis}** · "
            f"refreshed this worker tick: **{vis_ref}**"
        )
        trends = last_scan.get("index_trends_3m")
        ic1, ic2 = st.columns(2)
        for ix, key in enumerate(VISION_INDEX_KEYS):
            with ic1 if ix == 0 else ic2:
                trec = (trends or {}).get(key) if isinstance(trends, dict) else None
                tr, terr, t_rat = None, None, None
                if isinstance(trec, dict):
                    tr = trec.get("trend")
                    terr = trec.get("error")
                    t_rat = trec.get("rationale")
                label = FO_INDEX_UNDERLYING_LABELS.get(key, key)
                if vis_dis:
                    badge = "— (vision disabled)"
                elif tr in ("BULLISH", "BEARISH", "NEUTRAL"):
                    badge = str(tr)
                else:
                    badge = "—"
                st.markdown(f"**{label}** · trend: **{badge}**")
                if isinstance(t_rat, str) and t_rat.strip():
                    st.info(f"**Vision rationale:** {t_rat.strip()}")
                elif tr in ("BULLISH", "BEARISH", "NEUTRAL") and not (
                    isinstance(t_rat, str) and t_rat.strip()
                ):
                    st.caption(
                        "No rationale text in the last snapshot (Claude must return JSON including a "
                        "`rationale` field, or the response was parsed from trend word only)."
                    )
                if terr and not tr:
                    raw_e = str(terr)
                    low_e = raw_e.lower()
                    if "kaleido" in low_e or "plotly_get_chrome" in low_e or "google chrome" in low_e:
                        st.info(_format_stored_vision_error(raw_e))
                    else:
                        st.caption(raw_e)
                df_1, err_1 = load_index_session_interval_df(
                    kite, nse, session_d, key, "minute"
                )
                if df_1 is None or err_1 or df_1.empty:
                    st.warning(err_1 or "No 1-minute data for this index.")
                else:
                    fig3 = build_index_vision_candlestick_figure(df_1, key)
                    fig3.update_layout(title=f"{label} — 1m · vision trend: {badge}")
                    st.plotly_chart(fig3, use_container_width=True, key=f"mock_idx1m_{key}")

        st.subheader("Scrips — two columns")
        st.caption(
            f"EMA **{ENVELOPE_EMA_PERIOD}** envelope on spot (1m): **index** ±{100 * idx_env_pct:.3f}% · **equity** ±{100 * eq_env_pct:.3f}% (each side). "
            "Each cell: quote, spot chart, last worker/LLM telemetry, and any **open** position for that underlying."
        )
        pu_map = last_graph.get("per_underlying") if isinstance(last_graph, dict) else None
        runs_list = last_graph.get("runs") if isinstance(last_graph.get("runs"), list) else []

        def _resolve_run(under: str) -> dict | None:
            uu = under.strip().upper()
            if isinstance(pu_map, dict):
                r0 = pu_map.get(uu) or pu_map.get(under)
                if r0:
                    return r0
            for rr in runs_list:
                if str(rr.get("underlying") or "").strip().upper() == uu:
                    return rr
            return None

        def _render_open_trade_block(r, *, chart_key: str) -> None:
            leg_ltp = _opt_ltp(kite, r.instrument)
            u_pnl = None
            if leg_ltp is not None and r.entry_price:
                u_pnl = (leg_ltp - float(r.entry_price)) * int(r.quantity or 1)
            st.markdown(
                f"**trade_id {r.trade_id}** · `{r.instrument or '—'}` · **{r.direction or '—'}**"
            )
            oc3, oc4, oc5, oc6, oc7 = st.columns(5)
            with oc3:
                st.metric("Entry", f"₹{float(r.entry_price or 0):,.2f}")
            with oc4:
                st.metric(
                    "Stop / Target",
                    f"₹{float(r.stop_loss or 0):,.2f} / ₹{float(r.target or 0):,.2f}",
                )
            with oc5:
                st.metric("Qty", int(r.quantity or 0))
            with oc6:
                ep_r = float(r.entry_price or 0)
                if leg_ltp is not None and ep_r:
                    st.metric(
                        "Option LTP",
                        f"₹{leg_ltp:,.2f}",
                        delta=f"₹{leg_ltp - ep_r:+,.2f} vs entry",
                        delta_color="normal",
                    )
                elif leg_ltp is not None:
                    st.metric("Option LTP", f"₹{leg_ltp:,.2f}")
                else:
                    st.metric("Option LTP", "—")
            with oc7:
                st.metric(
                    "Unrealised",
                    f"₹{u_pnl:,.2f}" if u_pnl is not None else "—",
                )
            if r.llm_rationale:
                st.caption("LLM rationale at entry")
                st.write(r.llm_rationale)
            if r.instrument:
                df_o = _opt_minute_df(kite, nfo, r.instrument, session_d)
                if df_o is not None and not df_o.empty:
                    fig_o = go.Figure()
                    fig_o.add_trace(
                        go.Candlestick(
                            x=df_o["date"],
                            open=df_o["open"],
                            high=df_o["high"],
                            low=df_o["low"],
                            close=df_o["close"],
                            name=r.instrument,
                        )
                    )
                    ep = float(r.entry_price or 0)
                    tp = float(r.target or 0)
                    sp = float(r.stop_loss or 0)
                    if ep > 0:
                        fig_o.add_hline(
                            y=ep, line_dash="dot", line_color="cyan", annotation_text="Entry"
                        )
                    if tp > ep:
                        fig_o.add_hline(
                            y=tp, line_dash="dot", line_color="lime", annotation_text="Target"
                        )
                    if 0 < sp < ep:
                        fig_o.add_hline(
                            y=sp, line_dash="dot", line_color="tomato", annotation_text="Stop"
                        )
                    fig_o.update_layout(
                        title=f"{r.instrument} — premium (1m)",
                        height=260,
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_o, use_container_width=True, key=chart_key)
                else:
                    st.caption(f"No minute series for `{r.instrument}`.")

        for row_start in range(0, len(und), 2):
            col_a, col_b = st.columns(2)
            pair = und[row_start : row_start + 2]
            for ci, u in enumerate(pair):
                col = col_a if ci == 0 else col_b
                kf = _safe_key_fragment(u)
                with col:
                    idx_label = FO_INDEX_UNDERLYING_LABELS.get(u, u)
                    st.markdown(f"##### {idx_label} · `{u}`")
                    _lv, _d = _nse_underlying_quote(kite, u)
                    _fmt = f"₹{_lv:,.2f}" if _lv is not None else "—"
                    if _d is not None:
                        st.metric("Spot LTP", _fmt, delta=f"₹{_d:+,.2f}", delta_color="normal")
                    else:
                        st.metric("Spot LTP", _fmt)

                    env_pct = mock_agent_envelope_pct_for_underlying(u)
                    df_u, err_u = load_index_session_minute_df(kite, nse, session_d, u)
                    if df_u is not None and not df_u.empty and not err_u:
                        center, upper, lower = _envelope_series(
                            df_u, ENVELOPE_EMA_PERIOD, env_pct
                        )
                        if len(df_u) >= ENVELOPE_EMA_PERIOD:
                            ema_v = float(center.iloc[-1])
                            up_v = float(upper.iloc[-1])
                            lo_v = float(lower.iloc[-1])
                            if ema_v == ema_v and up_v == up_v and lo_v == lo_v:
                                spot_last = float(df_u["close"].iloc[-1])
                                m1, m2, m3 = st.columns(3)
                                with m1:
                                    d_ema = spot_last - ema_v
                                    st.metric(
                                        "EMA20",
                                        f"₹{ema_v:,.2f}",
                                        delta=f"₹{d_ema:+,.2f}",
                                        delta_color="normal",
                                    )
                                with m2:
                                    d_u = spot_last - up_v
                                    st.metric(
                                        f"Upper +{100 * env_pct:.1f}%",
                                        f"₹{up_v:,.2f}",
                                        delta=f"₹{d_u:+,.2f}",
                                        delta_color="normal",
                                    )
                                with m3:
                                    d_l = spot_last - lo_v
                                    st.metric(
                                        f"Lower −{100 * env_pct:.1f}%",
                                        f"₹{lo_v:,.2f}",
                                        delta=f"₹{d_l:+,.2f}",
                                        delta_color="normal",
                                    )
                        fig_u, dual_panel = _mock_index_spot_figure(
                            df_u, idx_label, env_pct, ema_period=ENVELOPE_EMA_PERIOD
                        )
                        fig_u.update_layout(
                            title=f"{u} spot (1m)",
                            height=340 if dual_panel else 280,
                        )
                        st.plotly_chart(
                            fig_u,
                            use_container_width=True,
                            key=f"mock_eng_spot_{kf}_{row_start}_{ci}",
                        )
                    else:
                        st.warning(
                            f"{idx_label}: {err_u or 'No spot candles today.'}"
                        )

                    st.markdown("**Worker / LLM (last tick)**")
                    run_u = _resolve_run(u)
                    _render_last_graph_run_panel(
                        run_u, key_prefix=f"sg_{kf}_{row_start}_{ci}"
                    )

                    pos_here = open_by_u.get(u, [])
                    if pos_here:
                        st.markdown("**Open mock trade(s)**")
                        for pi, r in enumerate(pos_here):
                            _render_open_trade_block(
                                r,
                                chart_key=f"mock_eng_opt_{kf}_{row_start}_{ci}_{pi}_{r.trade_id}",
                            )
                            if pi < len(pos_here) - 1:
                                st.divider()

        if last_graph and not isinstance(last_graph.get("runs"), list):
            st.subheader("Last agent snapshot (legacy format)")
            _render_last_graph_run_panel(dict(last_graph), key_prefix="legacy_fmt")

        st.subheader("Scan side-effects (last tick)")
        st.write(
            f"Exits on target/stop: **{last_scan.get('exits_sl_target', 0)}** · "
            f"15:20 square-off closes: **{last_scan.get('exits_square_off', 0)}**"
        )
        if isinstance(last_graph.get("runs"), list) and last_graph.get("tick_ist"):
            st.caption(
                f"Last graph tick (IST): **{str(last_graph.get('tick_ist', ''))[:19]}** · "
                "`skipped: open_position` means that index already had an OPEN leg."
            )

        st.subheader("Mock trade book")
        if open_rows:
            st.markdown("**Manual exit (OPEN legs)**")
            st.caption(
                "Uses **option LTP minus mock slippage** (same fill logic as automatic target/stop/square-off). "
                "Requires a working Kite session and NFO quote for the instrument."
            )
            for r in open_rows:
                uix = (r.index_underlying or "—").strip() or "—"
                inst = r.instrument or "—"
                em1, em2, em3 = st.columns([1, 4, 1])
                with em1:
                    st.write(f"`{r.trade_id}`")
                with em2:
                    st.caption(
                        f"{uix} · `{inst}` · entry ₹{float(r.entry_price or 0):,.2f} · qty {int(r.quantity or 0)}"
                    )
                with em3:
                    if st.button(
                        "Exit",
                        key=f"mock_manual_exit_{r.trade_id}",
                        help="Close this OPEN mock leg at LTP (human intervention)",
                    ):
                        ok, msg = manual_close_mock_trade(kite, r.trade_id)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                        st.rerun()
        rows = mock_trade_store.list_recent_trades(limit=100)
        if not rows:
            st.caption("No rows in `mock_trades` yet.")
        else:
            data = []
            for r in rows:
                try:
                    pnl_f = float(r.realized_pnl) if r.realized_pnl is not None else None
                except (TypeError, ValueError):
                    pnl_f = None
                if pnl_f is not None and pnl_f != pnl_f:
                    pnl_f = None
                ltp = _opt_ltp(kite, r.instrument)
                unreal_f: float | None = None
                if (r.status or "").strip().upper() == "OPEN":
                    ep = float(r.entry_price or 0.0)
                    qty = int(r.quantity or 0)
                    if ltp is not None and ep > 0 and qty > 0:
                        unreal_f = (ltp - ep) * qty
                data.append(
                    {
                        "trade_id": r.trade_id,
                        "entry_time": _utc_naive_sql_to_ist_str(r.entry_time),
                        "exit_time": _utc_naive_sql_to_ist_str(r.exit_time),
                        "instrument": r.instrument,
                        "index": r.index_underlying,
                        "direction": r.direction,
                        "entry": r.entry_price,
                        "LTP": ltp,
                        "stop": r.stop_loss,
                        "target": r.target,
                        "status": r.status,
                        "exit": r.exit_price,
                        "Unreal ₹": unreal_f,
                        "PnL": pnl_f,
                        "qty": r.quantity,
                    }
                )
            df_book = pd.DataFrame(data)

            def _pnl_color(s: pd.Series) -> list[str]:
                out: list[str] = []
                for v in s:
                    if v is None or (isinstance(v, float) and v != v):
                        out.append("")
                    elif v > 0:
                        out.append("color: #22c55e")
                    elif v < 0:
                        out.append("color: #ef4444")
                    else:
                        out.append("")
                return out

            st.caption(
                "**entry_time / exit_time** = **IST** (Asia/Kolkata), converted from UTC stored in the DB. "
                "**LTP** = live option last price from Kite (when available). "
                "**Unreal ₹** = (LTP − entry) × qty for **OPEN** rows. "
                "**PnL** = realised on **CLOSED** rows."
            )
            st.dataframe(
                df_book.style.format(
                    {"LTP": "₹{:,.2f}", "Unreal ₹": "₹{:+,.2f}", "PnL": "₹{:,.2f}"},
                    na_rep="—",
                )
                .apply(_pnl_color, subset=["Unreal ₹"])
                .apply(_pnl_color, subset=["PnL"]),
                use_container_width=True,
                hide_index=True,
            )

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
