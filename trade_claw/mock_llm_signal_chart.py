"""Static PNG chart for mock-engine LLM context (1m session + EMA envelope + signal marker)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from trade_claw.constants import ENVELOPE_EMA_PERIOD
from trade_claw.strategies import add_ma_envelope_line_traces

_log = logging.getLogger(__name__)
_kaleido_issue_logged: list[bool] = [False]


def _log_kaleido_once(msg: str, *args: object) -> None:
    if not _kaleido_issue_logged[0]:
        _log.warning(msg, *args)
        _kaleido_issue_logged[0] = True


def _signal_bar_x_for_vline(df: pd.DataFrame, signal_bar_time: str | None) -> Any:
    if df is None or df.empty:
        return None
    if not (signal_bar_time or "").strip():
        return df["date"].iloc[-1]
    t = pd.to_datetime(str(signal_bar_time).strip(), errors="coerce")
    if pd.isna(t):
        return df["date"].iloc[-1]
    dts = pd.to_datetime(df["date"])
    j = int((dts - t).abs().argmin())
    return df["date"].iloc[j]


def underlying_session_chart_png_bytes(
    df: pd.DataFrame,
    *,
    envelope_pct: float,
    signal_bar_time: str | None,
    underlying_label: str,
    direction: str,
    ema_period: int = ENVELOPE_EMA_PERIOD,
    width: int = 1200,
    height: int = 600,
    assessment_mode: bool = False,
) -> bytes | None:
    """
    Build a single-panel candlestick + EMA envelope (same geometry as ``envelope_breakout_on_last_bar``)
    and return PNG bytes via Kaleido, or ``None`` on failure.

    ``assessment_mode=True``: neutral title for vision breakout review (yellow line = latest bar when
    ``signal_bar_time`` is unset).
    """
    if df is None or df.empty or len(df) < ema_period:
        return None

    try:
        import importlib.util

        if importlib.util.find_spec("kaleido") is None:
            raise ImportError("kaleido")
    except ImportError:
        _log_kaleido_once(
            "MOCK_LLM_ATTACH_UNDERLYING_CHART is on but kaleido is not installed; "
            "chart attachment skipped (pip/uv: kaleido)."
        )
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Spot",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )
    add_ma_envelope_line_traces(fig, df, ema_period=ema_period, pct=envelope_pct)

    vx = _signal_bar_x_for_vline(df, signal_bar_time)
    if vx is not None:
        # Avoid add_vline with mixed candlestick x types (plotly shapeannotation can error).
        vx_plot = pd.Timestamp(vx).to_pydatetime()
        fig.add_shape(
            type="line",
            x0=vx_plot,
            x1=vx_plot,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="rgba(255, 230, 100, 0.95)", width=2),
        )
        fig.add_annotation(
            x=vx_plot,
            y=1,
            yref="paper",
            text="signal bar",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="rgba(255, 230, 100, 0.95)", size=11),
        )

    if assessment_mode:
        title = (
            f"{underlying_label} — 1m session (IST) · EMA {ema_period} ±{100 * envelope_pct:.2f}% · "
            "breakout assessment (yellow line = reference bar)"
        )
    else:
        leg = "CE" if (direction or "").upper().startswith("BULL") else "PE"
        title = (
            f"{underlying_label} — 1m session (IST) · EMA {ema_period} ±{100 * envelope_pct:.2f}% · "
            f"{direction} ({leg})"
        )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        width=width,
        height=height,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=30, t=60, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    try:
        return fig.to_image(format="png", width=width, height=height, engine="kaleido")
    except Exception as e:  # noqa: BLE001
        _log_kaleido_once("Kaleido PNG export failed (chart attachment skipped): %s", e)
        return None
