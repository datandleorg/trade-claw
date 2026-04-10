"""Dedicated UI for autonomous LLM BANKNIFTY mock engine."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_claw import mock_trade_store
from trade_claw.constants import ENVELOPE_EMA_PERIOD
from trade_claw.env_trading_params import mock_engine_envelope_decimal_per_side, mock_llm_prompt_log_dir
from trade_claw.market_data import candles_to_dataframe
from trade_claw.mock_llm_flow_log import list_flow_dirs_in_date_range, read_json_if_exists
from trade_claw.mock_market_signal import load_index_session_minute_df, now_ist
from trade_claw.plotly_ohlc import (
    add_candlestick_trace,
    add_volume_bar_trace,
    create_ohlc_volume_figure,
    finalize_ohlc_volume_figure,
)
from trade_claw.strategies import add_ma_envelope_line_traces

_UNDERLYING = "BANKNIFTY"
_IST = ZoneInfo("Asia/Kolkata")
_UTC = ZoneInfo("UTC")


def _vwap_series(df: pd.DataFrame) -> pd.Series:
    vol = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum = vol.cumsum().replace(0.0, pd.NA)
    out = ((tp * vol).cumsum() / cum).bfill()
    return out.fillna(pd.to_numeric(df["close"], errors="coerce"))


def _render_banknifty_live_chart(kite, nse_instruments) -> None:
    session_d = now_ist().date()
    df_u, err = load_index_session_minute_df(kite, nse_instruments, session_d, _UNDERLYING)
    if df_u is None or df_u.empty or err:
        st.warning(err or "No BANKNIFTY candles for session.")
        return
    fig, pr, vr = create_ohlc_volume_figure(df_u)
    add_candlestick_trace(fig, df_u, name=_UNDERLYING, price_row=pr, volume_row=vr)
    env_pct = mock_engine_envelope_decimal_per_side(_UNDERLYING)
    add_ma_envelope_line_traces(
        fig,
        df_u,
        ema_period=ENVELOPE_EMA_PERIOD,
        pct=env_pct,
        row=pr if vr is not None else None,
        col=1 if vr is not None else None,
    )
    close = pd.to_numeric(df_u["close"], errors="coerce")
    for span, color in ((9, "#38bdf8"), (21, "#a78bfa")):
        if len(df_u) >= span:
            ema = close.ewm(span=span, adjust=False).mean()
            kw = {"row": pr, "col": 1} if vr is not None else {}
            fig.add_trace(
                go.Scatter(
                    x=df_u["date"],
                    y=ema,
                    mode="lines",
                    name=f"EMA {span}",
                    line=dict(width=1.2, color=color),
                ),
                **kw,
            )
    vw = _vwap_series(df_u)
    kw_v = {"row": pr, "col": 1} if vr is not None else {}
    fig.add_trace(
        go.Scatter(
            x=df_u["date"],
            y=vw,
            mode="lines",
            name="VWAP",
            line=dict(color="#fbbf24", width=1.5, dash="dot"),
        ),
        **kw_v,
    )
    if vr is not None:
        add_volume_bar_trace(fig, df_u, volume_row=vr)
    finalize_ohlc_volume_figure(fig, height=480 if vr is not None else 420, template="plotly_dark")
    fig.update_layout(
        title=f"{_UNDERLYING} spot (1m) — EMA{ENVELOPE_EMA_PERIOD} ±{100 * env_pct:.2f}% envelope (same geometry as vision chart) + EMA9/21 + VWAP",
    )
    st.plotly_chart(fig, use_container_width=True, key="llm_bn_live_chart")


def _option_minute_df(kite, nfo_instruments: list, tradingsymbol: str | None, session_d: date):
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
    try:
        candles = kite.historical_data(
            int(token),
            from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            to_dt.strftime("%Y-%m-%d %H:%M:%S"),
            interval="minute",
        )
    except Exception:  # noqa: BLE001
        return None
    df = candles_to_dataframe(candles)
    if df is None or df.empty:
        return None
    return df.sort_values("date").reset_index(drop=True)


def _nearest_bar_x_for_entry_ist(df: pd.DataFrame, entry_time_raw: str | None):
    if not entry_time_raw or df is None or df.empty:
        return None
    try:
        dt_utc = datetime.strptime(str(entry_time_raw)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=_UTC)
    except ValueError:
        return None
    entry_ist = dt_utc.astimezone(_IST)
    bar_t = pd.to_datetime(df["date"], errors="coerce")
    if bar_t.dt.tz is None:
        bar_t = bar_t.dt.tz_localize(_IST, ambiguous="infer", nonexistent="shift_forward")
    entry_ts = pd.Timestamp(entry_ist)
    diff = (bar_t - entry_ts).abs()
    i = int(diff.to_numpy().argmin())
    return df["date"].iloc[i]


def _x_as_python_datetime(x_raw) -> datetime | None:
    if x_raw is None or (isinstance(x_raw, float) and pd.isna(x_raw)):
        return None
    ts = pd.Timestamp(x_raw)
    # Align with typical Kite candlestick axis (naive wall-clock times).
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_IST).tz_localize(None)
    return ts.to_pydatetime()


def _add_entry_minute_vline(fig, x_entry, *, sub: dict) -> None:
    """Avoid fig.add_vline with pandas Timestamp — Plotly may sum x values and crash."""
    xv = _x_as_python_datetime(x_entry)
    if xv is None:
        return
    kw = dict(
        type="line",
        x0=xv,
        x1=xv,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dot", width=1, color="rgba(255,255,255,0.45)"),
        layer="above",
    )
    if sub:
        fig.add_shape(**kw, **sub)
    else:
        fig.add_shape(**kw, xref="x", yref="paper")


def _render_open_trade_option_chart(kite, nfo_instruments: list, row: mock_trade_store.MockTradeRow) -> None:
    tradingsymbol = row.instrument or ""
    trade_id = row.trade_id
    df_o = _option_minute_df(kite, nfo_instruments, tradingsymbol, now_ist().date())
    if df_o is None or df_o.empty:
        st.caption(f"No minute option candles for `{tradingsymbol}`.")
        return
    fig, pr, vr = create_ohlc_volume_figure(df_o)
    add_candlestick_trace(fig, df_o, name=tradingsymbol, price_row=pr, volume_row=vr)
    sub = {"row": pr, "col": 1} if vr is not None else {}
    close = pd.to_numeric(df_o["close"], errors="coerce")
    for span, color in ((9, "#38bdf8"), (21, "#a78bfa"), (50, "#fb923c")):
        if len(df_o) >= span:
            ema = close.ewm(span=span, adjust=False).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_o["date"],
                    y=ema,
                    mode="lines",
                    name=f"EMA {span}",
                    line=dict(width=1.1, color=color),
                ),
                **sub,
            )
    vw = _vwap_series(df_o)
    fig.add_trace(
        go.Scatter(
            x=df_o["date"],
            y=vw,
            mode="lines",
            name="VWAP",
            line=dict(color="#fbbf24", width=1.4, dash="dot"),
        ),
        **sub,
    )
    ep = float(row.entry_price or 0.0)
    stp = float(row.stop_loss or 0.0)
    tgt = float(row.target or 0.0)
    if ep > 0:
        fig.add_hline(y=ep, line_dash="solid", line_color="cyan", annotation_text="Entry", **sub)
    if stp > 0 and stp < ep:
        fig.add_hline(y=stp, line_dash="dash", line_color="rgba(239,68,68,0.9)", annotation_text="Stop", **sub)
    if tgt > ep > 0:
        fig.add_hline(y=tgt, line_dash="dash", line_color="rgba(34,197,94,0.9)", annotation_text="Target", **sub)
    x_entry = _nearest_bar_x_for_entry_ist(df_o, row.entry_time)
    _add_entry_minute_vline(fig, x_entry, sub=sub)
    if vr is not None:
        add_volume_bar_trace(fig, df_o, volume_row=vr)
    finalize_ohlc_volume_figure(fig, height=420 if vr is not None else 380, template="plotly_dark")
    fig.update_layout(
        title=f"Trade {trade_id} · {tradingsymbol} (1m) — EMA9/21/50, VWAP, entry/stop/target, entry bar",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"llm_bn_open_opt_{trade_id}")


def _opt_ltp(kite, tradingsymbol: str | None) -> float | None:
    if not tradingsymbol:
        return None
    try:
        row = kite.quote([f"NFO:{tradingsymbol}"]).get(f"NFO:{tradingsymbol}") or {}
        v = row.get("last_price")
        return float(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def _banknifty_spot_quote(kite) -> float | None:
    try:
        row = kite.quote(["NSE:NIFTY BANK"]).get("NSE:NIFTY BANK") or {}
        v = row.get("last_price")
        return float(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def _entry_time_ist_label(entry_time_raw: str | None) -> str:
    """DB stores entry_time as naive UTC; display Asia/Kolkata."""
    if not entry_time_raw:
        return "—"
    try:
        dt_utc = datetime.strptime(str(entry_time_raw)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=_UTC)
        return dt_utc.astimezone(_IST).strftime("%Y-%m-%d %H:%M IST")
    except ValueError:
        return str(entry_time_raw)


def _find_flow_for_trade_id(trade_id: int, root: str | None) -> str | None:
    if not root:
        return None
    try:
        flows = list_flow_dirs_in_date_range(Path(root), date.today() - timedelta(days=30), date.today())
    except OSError:
        return None
    for fp in flows:
        out = read_json_if_exists(fp / "outcome.json")
        if isinstance(out, dict) and out.get("trade_id") == trade_id:
            return f"{fp.parent.parent.name}/{fp.parent.name}/{fp.name}"
    return None


def _render_llm_observability() -> None:
    st.subheader("Run observability")
    root = mock_llm_prompt_log_dir()
    if not root:
        st.info("Set `MOCK_LLM_PROMPT_LOG_DIR` to view run artifacts.")
        return

    c0, c1 = st.columns(2)
    with c0:
        d0 = st.date_input("From session date", value=date.today() - timedelta(days=7), key="llm_mock_d0")
    with c1:
        d1 = st.date_input("To session date", value=date.today(), key="llm_mock_d1")
    if d0 > d1:
        st.warning("From date must be before or equal to To date.")
        return

    flows = list_flow_dirs_in_date_range(Path(root), d0, d1)
    if not flows:
        st.info("No run artifacts found for selected range.")
        return

    def _label(i: int) -> str:
        p = flows[i]
        return f"{p.parent.parent.name}/{p.parent.name}/{p.name}"

    idx = st.selectbox("Run", range(len(flows)), format_func=_label, key="llm_mock_flow_pick")
    fp = flows[int(idx)]

    sup_int = read_json_if_exists(fp / "supervisor" / "intent.json")
    sup_dec = read_json_if_exists(fp / "supervisor" / "decision.json")
    sup_cands = read_json_if_exists(fp / "supervisor" / "candidates.json")
    vis_json = read_json_if_exists(fp / "vision" / "analysis.json")
    out = read_json_if_exists(fp / "outcome.json")
    vis_chart = fp / "vision" / "chart.png"

    lc, rc = st.columns(2)
    with lc:
        st.markdown("**Supervisor intent**")
        st.json(sup_int if sup_int is not None else {"info": "No intent.json"})
        st.markdown("**Supervisor decision**")
        st.json(sup_dec if sup_dec is not None else {"info": "No decision.json"})
        st.markdown("**Outcome**")
        st.json(out if out is not None else {"info": "No outcome.json"})
    with rc:
        st.markdown("**Chart sent to vision model**")
        if vis_chart.is_file():
            st.image(str(vis_chart), caption="vision/chart.png")
        else:
            st.caption("No vision chart for this run.")
        st.markdown("**Vision analysis output**")
        st.json(vis_json if vis_json is not None else {"info": "No analysis.json"})

    if sup_cands is not None:
        st.markdown("**Candidates passed to supervisor**")
        st.json(sup_cands)


def render_llm_mock_engine(kite) -> None:
    st.title("LLM Mock Engine (BANKNIFTY only)")
    st.caption(
        "Standalone autonomous mock engine. Every minute: supervisor (gpt-5-mini) decides SEARCH/MANAGE, "
        "optionally calls vision tool (Claude Haiku), and executes HOLD/ENTER/EXIT in mock mode."
    )
    st.markdown(f"- Underlying: `{_UNDERLYING}`")
    st.markdown(f"- EMA reference period: `{ENVELOPE_EMA_PERIOD}`")

    @st.fragment(run_every=timedelta(seconds=10))
    def _auto_refresh_panel() -> None:
        st.caption("Auto-refresh: every 10 seconds.")
        if st.session_state.nse_instruments is None:
            with st.spinner("Loading NSE instruments..."):
                st.session_state.nse_instruments = kite.instruments("NSE")
        if st.session_state.get("nfo_instruments") is None:
            with st.spinner("Loading NFO instruments..."):
                st.session_state.nfo_instruments = kite.instruments("NFO")

        _render_banknifty_live_chart(kite, st.session_state.nse_instruments)

        open_rows = [r for r in mock_trade_store.list_open_trades() if (r.index_underlying or "").upper() == _UNDERLYING]
        st.subheader("Live quotes")
        q1, q2 = st.columns(2)
        with q1:
            bn_spot = _banknifty_spot_quote(kite)
            st.metric("BANKNIFTY spot", f"₹{bn_spot:,.2f}" if bn_spot is not None else "—")
        with q2:
            if not open_rows:
                st.caption("No open BANKNIFTY option quotes.")
            else:
                q_rows = []
                for r in open_rows:
                    ltp = _opt_ltp(kite, r.instrument)
                    q_rows.append(
                        {
                            "trade_id": r.trade_id,
                            "instrument": r.instrument or "—",
                            "option_ltp": f"₹{ltp:,.2f}" if ltp is not None else "—",
                        }
                    )
                st.dataframe(q_rows, use_container_width=True, hide_index=True)

        st.subheader("Open BANKNIFTY mock positions")
        if not open_rows:
            st.caption("No open BANKNIFTY trades.")
        else:
            root = mock_llm_prompt_log_dir()
            table_rows = []
            for r in open_rows:
                ltp = _opt_ltp(kite, r.instrument)
                ep = float(r.entry_price or 0.0)
                qty = int(r.quantity or 0)
                unreal = ((ltp - ep) * qty) if (ltp is not None and ep > 0 and qty > 0) else None
                unreal_pct = ((ltp - ep) * 100.0 / ep) if (ltp is not None and ep > 0) else None
                flow_label = _find_flow_for_trade_id(r.trade_id, root)
                table_rows.append(
                    {
                        "trade_id": r.trade_id,
                        "instrument": r.instrument,
                        "entry_time": r.entry_time,
                        "entry_minute_ist": _entry_time_ist_label(r.entry_time),
                        "entry_price": r.entry_price,
                        "ltp": ltp,
                        "unrealised_pnl": round(unreal, 2) if unreal is not None else None,
                        "unrealised_pct": round(unreal_pct, 2) if unreal_pct is not None else None,
                        "stop_loss": r.stop_loss,
                        "target": r.target,
                        "quantity": r.quantity,
                        "status": r.status,
                        "run_folder": flow_label or "not_found",
                    }
                )
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
            for r in open_rows:
                with st.expander(f"Detailed rationale · trade {r.trade_id}", expanded=False):
                    st.markdown(f"**Trade time (IST):** `{_entry_time_ist_label(r.entry_time)}`")
                    st.markdown(
                        f"**Run folder:** `{_find_flow_for_trade_id(r.trade_id, root) or 'Not found in last 30 days'}`"
                    )
                    st.markdown("**Detailed rationale**")
                    st.info((r.llm_rationale or "No rationale saved on trade row.").strip())
            st.markdown("**Open trade option charts**")
            for r in open_rows:
                if r.instrument:
                    _render_open_trade_option_chart(kite, st.session_state.nfo_instruments, r)

    _auto_refresh_panel()
    _render_llm_observability()

    st.subheader("BANKNIFTY positions (open + closed)")
    all_rows = [
        r
        for r in mock_trade_store.list_recent_trades(limit=400)
        if (r.index_underlying or "").upper() == _UNDERLYING
    ]
    if not all_rows:
        st.caption("No BANKNIFTY trades yet.")
    else:
        out_rows = []
        for r in all_rows:
            stt = (r.status or "").upper()
            ltp = _opt_ltp(kite, r.instrument) if stt == "OPEN" else None
            ep = float(r.entry_price or 0.0)
            qty = int(r.quantity or 0)
            unreal = ((ltp - ep) * qty) if (ltp is not None and ep > 0 and qty > 0) else None
            out_rows.append(
                {
                    "trade_id": r.trade_id,
                    "status": stt,
                    "instrument": r.instrument,
                    "entry_time_ist": _entry_time_ist_label(r.entry_time),
                    "exit_time_ist": _entry_time_ist_label(r.exit_time),
                    "entry_price": r.entry_price,
                    "ltp": ltp,
                    "exit_price": r.exit_price,
                    "quantity": r.quantity,
                    "stop_loss": r.stop_loss,
                    "target": r.target,
                    "unrealised_pnl": round(unreal, 2) if unreal is not None else None,
                    "realized_pnl": r.realized_pnl,
                }
            )
        st.dataframe(
            out_rows,
            use_container_width=True,
            hide_index=True,
        )
