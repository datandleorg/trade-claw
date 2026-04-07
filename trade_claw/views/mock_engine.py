"""Mock AI engine HUD: telemetry from worker + live Kite charts/LTP."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import NamedTuple
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from trade_claw import mock_engine_telemetry
from trade_claw import mock_trade_analytics as mtd_an
from trade_claw import mock_trade_store
from trade_claw.constants import (
    ENVELOPE_EMA_PERIOD,
    FO_INDEX_UNDERLYING_LABELS,
    NFO_EXCHANGE,
    NSE_EXCHANGE,
)
from trade_claw.market_data import candles_to_dataframe
from trade_claw.mock_market_signal import (
    load_index_session_minute_df,
    mock_agent_envelope_pct,
    mock_engine_underlyings,
    nse_index_ltp_symbol,
    now_ist,
)
from trade_claw.plotly_ohlc import (
    add_candlestick_trace,
    add_volume_bar_trace,
    create_ohlc_volume_figure,
    finalize_ohlc_volume_figure,
)
from trade_claw.pl_style import pl_title_color
from trade_claw.strategies import _envelope_series, add_ma_envelope_line_traces
from trade_claw.task_runtime import MOCK_TRADES_DB_PATH

_IST = ZoneInfo("Asia/Kolkata")


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
    """Spot candles + EMA; when envelope is huge vs session, middle panel: % vs EMA; volume on bottom when data exists."""
    _ROW_PRICE = 1
    _ROW_PCT = 2
    _ROW_VOL = 3

    axis = _spot_chart_y_axis(df, ema_period=ema_period, pct=env_pct)
    if axis is None or len(df) < ema_period:
        fig, pr, vr = create_ohlc_volume_figure(df)
        add_candlestick_trace(fig, df, name=idx_label, price_row=pr, volume_row=vr)
        add_ma_envelope_line_traces(
            fig,
            df,
            ema_period=ema_period,
            pct=env_pct,
            row=pr if vr is not None else None,
            col=1 if vr is not None else None,
        )
        if vr is not None:
            add_volume_bar_trace(fig, df, volume_row=vr)
        finalize_ohlc_volume_figure(fig, height=400, template="plotly_dark")
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
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.48, 0.32, 0.20],
            subplot_titles=(
                None,
                f"% from EMA — dashed lines at ±{pct_label} (same envelope as worker)",
                None,
            ),
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
            row=_ROW_PRICE,
            col=1,
        )
        add_ma_envelope_line_traces(
            fig,
            df,
            ema_period=ema_period,
            pct=env_pct,
            include_price_bands=False,
            row=_ROW_PRICE,
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
            row=_ROW_PCT,
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
            row=_ROW_PCT,
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
            row=_ROW_PCT,
            col=1,
        )
        add_volume_bar_trace(fig, df, volume_row=_ROW_VOL)
        fig.update_yaxes(range=[y0, y1], autorange=False, row=_ROW_PRICE, col=1)
        y2_hi = max(float(pct_vs.max()), band_pct, 1.0) * 1.12
        y2_lo = min(float(pct_vs.min()), -band_pct, -1.0) * 1.12
        fig.update_yaxes(
            range=[y2_lo, y2_hi],
            autorange=False,
            title_text="%",
            row=_ROW_PCT,
            col=1,
        )
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(
            template="plotly_dark",
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig, True

    fig, pr, vr = create_ohlc_volume_figure(df)
    add_candlestick_trace(fig, df, name=idx_label, price_row=pr, volume_row=vr)
    add_ma_envelope_line_traces(
        fig,
        df,
        ema_period=ema_period,
        pct=env_pct,
        row=pr if vr is not None else None,
        col=1 if vr is not None else None,
    )
    if vr is not None:
        add_volume_bar_trace(fig, df, volume_row=vr)
    finalize_ohlc_volume_figure(fig, height=400, template="plotly_dark")
    if vr is not None:
        fig.update_yaxes(range=[y0, y1], autorange=False, row=1, col=1)
    else:
        fig.update_layout(yaxis=dict(range=[y0, y1], autorange=False))
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


def _pct_change_vs_base(delta: float | None, base: float | None) -> float | None:
    """Return ``100 * delta / base`` when base is valid (e.g. previous close, entry premium)."""
    if delta is None or base is None:
        return None
    b = float(base)
    if abs(b) < 1e-12 or b != b:
        return None
    return 100.0 * float(delta) / b


def _metric_inr_and_pct_desc(
    delta_inr: float | None,
    *,
    pct_base: float | None,
) -> tuple[float | None, str | None]:
    """Rounded INR delta for ``st.metric`` and optional ``delta_description`` for % vs base."""
    if delta_inr is None:
        return None, None
    d_round = round(float(delta_inr), 2)
    pct = _pct_change_vs_base(delta_inr, pct_base)
    desc = f"{round(pct, 2):+.2f}%" if pct is not None else None
    return d_round, desc


def _safe_key_fragment(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s)[:48] or "x"


def _mock_engine_trade_sound_html(steps: list[str]) -> str:
    """Web Audio: two-tone bell (detection) + square-wave buzzer (execution). ``steps`` = kinds in order."""
    payload = json.dumps(steps)
    return f"""
<div style="height:0;width:0;overflow:hidden" aria-hidden="true">
<script>
(function () {{
  const steps = {payload};
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx || !steps.length) return;
  const ctx = new Ctx();
  function resume() {{
    if (ctx.state === "suspended") ctx.resume();
  }}
  function toneBurst(t0, freq, dur, gainHi, wave) {{
    resume();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = wave;
    o.frequency.value = freq;
    o.connect(g);
    g.connect(ctx.destination);
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(gainHi, t0 + 0.02);
    g.gain.exponentialRampToValueAtTime(0.01, t0 + dur);
    o.start(t0);
    o.stop(t0 + dur + 0.01);
  }}
  function playDetection(t0) {{
    toneBurst(t0, 880, 0.2, 0.2, "sine");
    toneBurst(t0 + 0.16, 1174.66, 0.22, 0.18, "sine");
  }}
  function playExecution(t0) {{
    toneBurst(t0, 220, 0.32, 0.42, "square");
    toneBurst(t0 + 0.38, 165, 0.22, 0.32, "square");
  }}
  let nextT = ctx.currentTime + 0.06;
  for (let i = 0; i < steps.length; i++) {{
    const k = steps[i];
    if (k === "detection") {{
      playDetection(nextT);
      nextT += 0.52;
    }} else if (k === "execution") {{
      playExecution(nextT);
      nextT += 0.72;
    }}
  }}
}})();
</script>
</div>
"""


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
    if run.get("llm_tradingsymbol"):
        st.markdown("**LLM decision (structured)**")
        st.json(
            {
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


def _trade_row_get(row, key: str, default=None):
    """Support analytics ``pd.Series`` rows and Live ``MockTradeRow`` dataclass instances."""
    if row is None:
        return default
    if isinstance(row, pd.Series):
        if key not in row.index:
            return default
        v = row.loc[key]
        if pd.isna(v):
            return default
        return v
    return getattr(row, key, default)


def _bars_json_to_df(raw) -> pd.DataFrame | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        bars = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return None
    if not bars:
        return None
    dfp = pd.DataFrame(bars)
    if dfp.empty or "date" not in dfp.columns:
        return None
    dfp = dfp.copy()
    dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
    dfp = dfp.dropna(subset=["date"])
    if dfp.empty:
        return None
    return dfp.sort_values("date").reset_index(drop=True)


def _merge_ohlc_snapshots(
    df_a: pd.DataFrame | None,
    df_b: pd.DataFrame | None,
) -> pd.DataFrame | None:
    frames = [f for f in (df_a, df_b) if f is not None and not f.empty]
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last")
    return merged.sort_values("date").reset_index(drop=True)


def _bar_times_utc_compare(dfp: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(dfp["date"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(_IST, ambiguous="infer", nonexistent="shift_forward")
    return ts.dt.tz_convert("UTC")


def _event_utc_parse(raw) -> pd.Timestamp | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    t = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(t):
        return None
    return t


def _nearest_bar_idx(dfp: pd.DataFrame, event_raw) -> int | None:
    if dfp is None or dfp.empty:
        return None
    ev = _event_utc_parse(event_raw)
    if ev is None:
        return None
    bar_t = _bar_times_utc_compare(dfp)
    if bar_t.isna().all():
        return None
    delta = (bar_t - ev).abs()
    return int(delta.argmin())


def _mock_option_leg_label(row) -> str:
    inst = str(_trade_row_get(row, "instrument") or "")
    d = str(_trade_row_get(row, "direction") or "").upper()
    if "CE" in inst.upper() or d == "BUY":
        return "Long CE"
    if "PE" in inst.upper() or d == "SELL":
        return "Long PE"
    return d or "Option"


def _mock_option_strike_display(inst: str) -> str | None:
    """Last 4–8 digit run before CE/PE suffix (typical NFO tradingsymbol)."""
    m = re.search(r"(\d{4,8})(CE|PE)\s*$", inst.upper())
    return m.group(1) if m else None


def _add_fo_style_option_markers_and_hlines(
    fig: go.Figure,
    dfp: pd.DataFrame,
    trade,
    *,
    entry_idx: int | None,
    exit_idx: int | None,
    price_row: int | None = None,
    price_col: int = 1,
) -> None:
    """F&O Options–style: entry/exit markers + dashed target/stop (no duplicate entry/exit hlines)."""
    ddt = dfp["date"]
    _sub = {"row": price_row, "col": price_col} if price_row is not None else {}
    _ep = _trade_row_get(trade, "entry_price")
    ep = float(_ep) if _ep is not None and pd.notna(_ep) else 0.0
    _xp = _trade_row_get(trade, "exit_price")
    xp = float(_xp) if _xp is not None and pd.notna(_xp) else 0.0
    if entry_idx is not None and 0 <= entry_idx < len(ddt) and ep > 0:
        fig.add_trace(
            go.Scatter(
                x=[ddt.iloc[entry_idx]],
                y=[ep],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="cyan",
                    line=dict(width=1, color="black"),
                ),
                name="Entry",
            ),
            **_sub,
        )
    st_cl = str(_trade_row_get(trade, "status") or "").upper()
    if (
        exit_idx is not None
        and 0 <= exit_idx < len(ddt)
        and xp > 0
        and st_cl == "CLOSED"
    ):
        fig.add_trace(
            go.Scatter(
                x=[ddt.iloc[exit_idx]],
                y=[xp],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=10,
                    color="gold",
                    line=dict(width=1, color="orange"),
                ),
                name="Exit",
            ),
            **_sub,
        )
    _tp = _trade_row_get(trade, "target")
    tp = float(_tp) if _tp is not None and pd.notna(_tp) else 0.0
    _sp = _trade_row_get(trade, "stop_loss")
    sp = float(_sp) if _sp is not None and pd.notna(_sp) else 0.0
    if tp > ep and ep > 0:
        fig.add_hline(
            y=tp,
            line_dash="dash",
            line_color="rgba(34,197,94,0.85)",
            annotation_text="Target",
            annotation_position="right",
            **_sub,
        )
    if 0 < sp < ep:
        fig.add_hline(
            y=sp,
            line_dash="dash",
            line_color="rgba(239,68,68,0.85)",
            annotation_text="Stop",
            annotation_position="right",
            **_sub,
        )


def _mock_analytics_replay_option_title(row, tid: int) -> tuple[str, str]:
    leg = _mock_option_leg_label(row)
    inst = str(_trade_row_get(row, "instrument") or "Option")
    sk = _mock_option_strike_display(inst)
    leg_strike = f"{leg} @{float(sk):,.0f}" if sk else leg
    st = str(_trade_row_get(row, "status") or "").upper()
    pnl = pd.to_numeric(_trade_row_get(row, "realized_pnl"), errors="coerce")
    if st == "OPEN":
        title = f"{leg_strike} · `{inst}` · trade {tid} · OPEN"
        col = "#94a3b8"
    else:
        pnl_v = float(pnl) if pd.notna(pnl) else 0.0
        title = f"{leg_strike} · `{inst}` · Net ₹{pnl_v:+,.2f} · trade {tid} · CLOSED"
        col = pl_title_color(pnl_v)
    return title, col


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
        with st.expander("Replay stored minute snapshots (merged entry + exit)", expanded=False):
            st.caption(
                "Option and underlying charts **merge** entry and exit snapshots on one time axis (dedupe by bar time). "
                "Entry/exit markers match **F&O Options** style. Index envelope uses **current** `MOCK_AGENT_ENVELOPE_PCT`. "
                "Requires env **`MOCK_ENGINE_SNAPSHOT_BARS`** > 0 when trades are captured."
            )
            choices = [int(x) for x in snap_rows["trade_id"].tolist()]
            tid = st.selectbox("trade_id", choices, key="mock_an_snap_tid")
            row = snap_rows[snap_rows["trade_id"] == tid].iloc[0]
            _iu = None
            if "index_underlying" in row.index:
                v = row["index_underlying"]
                if pd.notna(v) and str(v).strip():
                    _iu = str(v).strip().upper()
            snap_idx = _iu or "NIFTY"
            snap_idx_label = FO_INDEX_UNDERLYING_LABELS.get(snap_idx, snap_idx)

            df_ue = _bars_json_to_df(row.get("entry_underlying_bars_json"))
            df_ux = _bars_json_to_df(row.get("exit_underlying_bars_json"))
            df_um = _merge_ohlc_snapshots(df_ue, df_ux)
            if df_um is not None and not df_um.empty:
                try:
                    env_pct_snap = mock_agent_envelope_pct()
                    fig_u, _dual_u = _mock_index_spot_figure(
                        df_um,
                        snap_idx_label,
                        env_pct_snap,
                        ema_period=ENVELOPE_EMA_PERIOD,
                    )
                    fig_u.update_layout(
                        title=(
                            f"Trade {tid} — {snap_idx_label} spot (merged, {len(df_um)} bars) + EMA envelope"
                        ),
                        height=540 if _dual_u else 420,
                    )
                    st.plotly_chart(fig_u, use_container_width=True, key="mock_an_snap_under")
                except (KeyError, ValueError) as e:
                    st.warning(f"Could not plot underlying snapshot: {e}")
            else:
                st.caption("No underlying snapshot (older trades or capture failed).")

            rat = _trade_row_get(row, "llm_rationale")
            if rat is not None and str(rat).strip():
                st.markdown("**LLM rationale (at entry)**")
                st.info(str(rat).strip())

            df_oe = _bars_json_to_df(row.get("entry_bars_json"))
            df_ox = _bars_json_to_df(row.get("exit_bars_json"))
            dfp = _merge_ohlc_snapshots(df_oe, df_ox)
            if dfp is None or dfp.empty:
                st.caption("No option snapshot (enable `MOCK_ENGINE_SNAPSHOT_BARS` for new trades).")
            else:
                try:
                    need = {"open", "high", "low", "close"}
                    if not need.issubset(dfp.columns):
                        st.caption("Option snapshot JSON missing OHLC columns.")
                    else:
                        ent_i = _nearest_bar_idx(dfp, row.get("entry_time"))
                        ex_i = _nearest_bar_idx(dfp, row.get("exit_time"))
                        opt_name = str(row.get("instrument") or "Option")
                        fig_s, pr_s, vr_s = create_ohlc_volume_figure(dfp)
                        add_candlestick_trace(
                            fig_s, dfp, name=opt_name[:48], price_row=pr_s, volume_row=vr_s
                        )
                        if vr_s is not None:
                            add_volume_bar_trace(fig_s, dfp, volume_row=vr_s)
                        _add_fo_style_option_markers_and_hlines(
                            fig_s,
                            dfp,
                            row,
                            entry_idx=ent_i,
                            exit_idx=ex_i,
                            price_row=pr_s if vr_s is not None else None,
                        )
                        t_title, t_col = _mock_analytics_replay_option_title(row, tid)
                        finalize_ohlc_volume_figure(fig_s, height=400, template="plotly_dark")
                        fig_s.update_layout(
                            title=dict(text=t_title, font=dict(color=t_col, size=13)),
                        )
                        st.plotly_chart(fig_s, use_container_width=True, key="mock_an_snap_chart")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    st.warning(f"Could not plot option snapshot: {e}")


def render_mock_engine(kite):
    st.title("Mock AI engine (indices + Nifty 50)")
    st.caption(
        "Celery Beat runs `scan_mock_market` **every minute** (IST weekdays, ~09:15–15:19 entries; 15:20 square-off). "
        "Scan list = **`FO_UNDERLYING_OPTIONS`** in `constants.py` (same as F&O Options); optional env **`MOCK_ENGINE_UNDERLYINGS`** comma subset. "
        "Live tab **auto-refreshes every 10 seconds**. **At most one OPEN mock trade per underlying.**"
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
    env_pct = mock_agent_envelope_pct()

    @st.fragment(run_every=timedelta(minutes=1))
    def _hud():
        mock_trade_store.init_db()
        mock_engine_telemetry.init_telemetry_table()
        snap = mock_engine_telemetry.read_snapshot()
        last_scan = snap.get("last_scan") or {}
        last_graph = snap.get("last_graph") or {}

        if "mock_engine_sound_seen_ids" not in st.session_state:
            st.session_state.mock_engine_sound_seen_ids = set()

        _sc1, _sc2 = st.columns([1, 2])
        with _sc1:
            bells_on = st.checkbox(
                "Trade bells (detection + execution)",
                value=True,
                key="mock_eng_trade_bells",
                help="Bell tones when the worker sees an envelope breakout; buzzer when a mock trade is inserted. "
                "Some browsers mute Web Audio until you interact with the tab.",
            )

        raw_sound = last_scan.get("sound_alerts")
        if not isinstance(raw_sound, list):
            raw_sound = []
        seen_ids: set = st.session_state.mock_engine_sound_seen_ids
        play_kinds: list[str] = []
        for item in raw_sound:
            if not isinstance(item, dict):
                continue
            aid = item.get("id")
            kind = item.get("kind")
            if not aid or kind not in ("detection", "execution"):
                continue
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            play_kinds.append(str(kind))
        if bells_on and play_kinds:
            components.html(_mock_engine_trade_sound_html(play_kinds), height=0)

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
                        d_r, pct_s = _metric_inr_and_pct_desc(leg_delta, pct_base=ep0)
                        st.metric(
                            "Option LTP",
                            f"₹{live_opt:,.2f}",
                            delta=d_r,
                            delta_color="normal",
                            delta_description=pct_s,
                            help="Premium change vs entry (₹ and % of entry).",
                        )
                    elif live_d is not None:
                        prev_o = (live_opt - live_d) if live_opt is not None else None
                        d_r, pct_s = _metric_inr_and_pct_desc(live_d, pct_base=prev_o)
                        st.metric(
                            "Option LTP",
                            f"₹{live_opt:,.2f}",
                            delta=d_r,
                            delta_color="normal",
                            delta_description=pct_s,
                            help="Change vs option previous close (₹ and %).",
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
                    pct_e = _pct_change_vs_base(chg_e, ep) if chg_e is not None else None
                    opt_rows.append(
                        {
                            "trade_id": t.trade_id,
                            "index": t.index_underlying or "—",
                            "instrument": t.instrument or "—",
                            "LTP": f"₹{lp:,.2f}" if lp is not None else "—",
                            "vs_entry": round(chg_e, 2) if chg_e is not None else None,
                            "vs_entry_%": round(pct_e, 2) if pct_e is not None else None,
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
                    df_leg.style.format(
                        {"vs_entry": "₹{:+,.2f}", "vs_entry_%": "{:+.2f}%"},
                        na_rep="—",
                    )
                    .apply(_leg_chg_color, subset=["vs_entry"])
                    .apply(_leg_chg_color, subset=["vs_entry_%"])
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
            st.metric(
                "Envelope ±",
                f"{100 * env_pct:.3f}%",
                help="From MOCK_AGENT_ENVELOPE_PCT (trade_claw.env_trading_params.fno_envelope_decimal_per_side)",
            )

        if not open_rows:
            st.info(
                "No **OPEN** mock trades. The worker opens at most **one leg per underlying** when a breakout fires and that underlying is flat."
            )

        st.subheader("Scrips — two columns")
        st.caption(
            f"EMA **{ENVELOPE_EMA_PERIOD}** ± **{100 * env_pct:.3f}%** envelope on spot (1m). "
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
                    dv = leg_ltp - ep_r
                    d_r, pct_s = _metric_inr_and_pct_desc(dv, pct_base=ep_r)
                    st.metric(
                        "Option LTP",
                        f"₹{leg_ltp:,.2f}",
                        delta=d_r,
                        delta_color="normal",
                        delta_description=pct_s,
                        help="Premium change vs entry (₹ and % of entry).",
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
                    fig_o, pr_o, vr_o = create_ohlc_volume_figure(df_o)
                    add_candlestick_trace(
                        fig_o, df_o, name=r.instrument, price_row=pr_o, volume_row=vr_o
                    )
                    if vr_o is not None:
                        add_volume_bar_trace(fig_o, df_o, volume_row=vr_o)
                    ent_i = _nearest_bar_idx(df_o, r.entry_time)
                    _add_fo_style_option_markers_and_hlines(
                        fig_o,
                        df_o,
                        r,
                        entry_idx=ent_i,
                        exit_idx=None,
                        price_row=pr_o if vr_o is not None else None,
                    )
                    leg = _mock_option_leg_label(r)
                    inst = r.instrument or "Option"
                    sk = _mock_option_strike_display(inst)
                    leg_strike = f"{leg} @{float(sk):,.0f}" if sk else leg
                    up = float(u_pnl) if u_pnl is not None else 0.0
                    live_title = f"{leg_strike} · `{inst}` · Unrealised ₹{up:+,.2f} · OPEN"
                    finalize_ohlc_volume_figure(fig_o, height=380, template="plotly_dark")
                    fig_o.update_layout(
                        title=dict(
                            text=live_title,
                            font=dict(color=pl_title_color(up), size=13),
                        ),
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
                    if _d is not None and _lv is not None:
                        prev_s = _lv - _d
                        d_r, pct_s = _metric_inr_and_pct_desc(_d, pct_base=prev_s)
                        st.metric(
                            "Spot LTP",
                            _fmt,
                            delta=d_r,
                            delta_color="normal",
                            delta_description=pct_s,
                            help="Change vs previous close (₹ and %).",
                        )
                    else:
                        st.metric("Spot LTP", _fmt)

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
                                    d_r, pct_s = _metric_inr_and_pct_desc(d_ema, pct_base=ema_v)
                                    st.metric(
                                        "EMA20",
                                        f"₹{ema_v:,.2f}",
                                        delta=d_r,
                                        delta_color="off",
                                        delta_description=pct_s,
                                        help="Spot last close minus EMA20 (₹ and % of EMA).",
                                    )
                                with m2:
                                    d_u = spot_last - up_v
                                    d_r, pct_s = _metric_inr_and_pct_desc(d_u, pct_base=up_v)
                                    st.metric(
                                        f"Upper +{100 * env_pct:.1f}%",
                                        f"₹{up_v:,.2f}",
                                        delta=d_r,
                                        delta_color="off",
                                        delta_description=pct_s,
                                        help="Spot minus upper band (₹ and % of band level).",
                                    )
                                with m3:
                                    d_l = spot_last - lo_v
                                    d_r, pct_s = _metric_inr_and_pct_desc(d_l, pct_base=lo_v)
                                    st.metric(
                                        f"Lower −{100 * env_pct:.1f}%",
                                        f"₹{lo_v:,.2f}",
                                        delta=d_r,
                                        delta_color="off",
                                        delta_description=pct_s,
                                        help="Spot minus lower band (₹ and % of band level).",
                                    )
                        fig_u, dual_panel = _mock_index_spot_figure(
                            df_u, idx_label, env_pct, ema_period=ENVELOPE_EMA_PERIOD
                        )
                        fig_u.update_layout(
                            title=f"{u} spot (1m)",
                            height=520 if dual_panel else 400,
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
        rows = mock_trade_store.list_recent_trades(limit=100)
        if not rows:
            st.caption("No rows in `mock_trades` yet.")
        else:
            ltp_cache: dict[str, float | None] = {}

            def _cached_opt_ltp(sym: str | None) -> float | None:
                if not sym:
                    return None
                if sym not in ltp_cache:
                    ltp_cache[sym] = _opt_ltp(kite, sym)
                return ltp_cache[sym]

            data = []
            for r in rows:
                try:
                    pnl_f = float(r.realized_pnl) if r.realized_pnl is not None else None
                except (TypeError, ValueError):
                    pnl_f = None
                if pnl_f is not None and pnl_f != pnl_f:
                    pnl_f = None
                st_open = (r.status or "").strip().upper() == "OPEN"
                ltp_v: float | None = None
                unreal_v: float | None = None
                if st_open:
                    ltp_v = _cached_opt_ltp(r.instrument)
                    ep = float(r.entry_price or 0)
                    q = int(r.quantity or 1)
                    if ltp_v is not None and ep > 0 and q > 0:
                        unreal_v = (ltp_v - ep) * q
                data.append(
                    {
                        "trade_id": r.trade_id,
                        "entry_time": r.entry_time,
                        "exit_time": r.exit_time,
                        "instrument": r.instrument,
                        "index": r.index_underlying,
                        "direction": r.direction,
                        "entry": r.entry_price,
                        "stop": r.stop_loss,
                        "target": r.target,
                        "status": r.status,
                        "exit": r.exit_price,
                        "LTP": ltp_v,
                        "PnL": pnl_f if not st_open else None,
                        "Unrealised PnL": unreal_v,
                        "qty": r.quantity,
                    }
                )

            total_real_portfolio = mock_trade_store.sum_realized_pnl_closed()
            total_unreal_portfolio = 0.0
            for ot in mock_trade_store.list_open_trades():
                lp = _cached_opt_ltp(ot.instrument)
                ep = float(ot.entry_price or 0)
                q = int(ot.quantity or 1)
                if lp is not None and ep > 0 and q > 0:
                    total_unreal_portfolio += (lp - ep) * q

            summary_row = {
                "trade_id": "Total (portfolio)",
                "entry_time": "",
                "exit_time": "",
                "instrument": "",
                "index": "",
                "direction": "",
                "entry": None,
                "stop": None,
                "target": None,
                "status": "",
                "exit": None,
                "LTP": None,
                "PnL": total_real_portfolio,
                "Unrealised PnL": total_unreal_portfolio,
                "qty": None,
            }
            df_book = pd.concat(
                [pd.DataFrame(data), pd.DataFrame([summary_row])],
                ignore_index=True,
            )

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
                "**LTP** / **Unrealised PnL** for **OPEN** legs (live quote). "
                "**PnL** = realised on **CLOSED**. Last row: all-time realised (closed) + current unrealised (all open legs)."
            )
            fmt = {
                "entry": "₹{:,.2f}",
                "stop": "₹{:,.2f}",
                "target": "₹{:,.2f}",
                "exit": "₹{:,.2f}",
                "LTP": "₹{:,.2f}",
                "PnL": "₹{:,.2f}",
                "Unrealised PnL": "₹{:,.2f}",
            }
            sty_book = (
                df_book.style.format(fmt, na_rep="—")
                .apply(_pnl_color, subset=["PnL"])
                .apply(_pnl_color, subset=["Unrealised PnL"])
            )
            st.dataframe(sty_book, use_container_width=True, hide_index=True)

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
