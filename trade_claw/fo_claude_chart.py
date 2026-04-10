"""Session chart PNG with tool-driven TA overlays for Claude vision (Plotly + Kaleido)."""

from __future__ import annotations

import base64
import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go

_log = logging.getLogger(__name__)


def _vwap_series(df: pd.DataFrame) -> pd.Series:
    vol = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum = vol.cumsum().replace(0.0, pd.NA)
    out = (tp * vol).cumsum() / cum
    return out.bfill().fillna(df["close"])


def merge_overlay_spec(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    out = dict(base) if base else {}
    for k, v in (delta or {}).items():
        if k == "ema_periods" and isinstance(v, list):
            out["ema_periods"] = [int(x) for x in v if int(x) > 0][:12]
        elif k == "hlines" and isinstance(v, list):
            out["hlines"] = v[:24]
        elif k == "annotations" and isinstance(v, list):
            out["annotations"] = v[:24]
        else:
            out[k] = v
    return out


def render_session_chart_png(
    df: pd.DataFrame,
    *,
    underlying_label: str,
    overlay_spec: dict[str, Any] | None,
    width: int = 1200,
    height: int = 640,
) -> bytes | None:
    """
    Candlesticks + optional EMAs, VWAP, horizontal lines, vertical line at bar index, annotations.
    Returns PNG bytes or None if Kaleido unavailable / empty df.
    """
    if df is None or df.empty or len(df) < 3:
        return None
    try:
        import importlib.util

        if importlib.util.find_spec("kaleido") is None:
            raise ImportError("kaleido")
    except ImportError:
        _log.warning("fo_claude_chart: kaleido not installed; PNG skipped")
        return None

    spec = overlay_spec or {}
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
    close = pd.to_numeric(df["close"], errors="coerce")
    x = df["date"]

    ema_periods = spec.get("ema_periods") or [9, 21, 50]
    if isinstance(ema_periods, list):
        colors = ["#38bdf8", "#fbbf24", "#4ade80", "#c084fc", "#f472b6"]
        for i, p in enumerate(ema_periods[:8]):
            try:
                period = int(p)
            except (TypeError, ValueError):
                continue
            if period < 2 or len(df) < period:
                continue
            ema = close.ewm(span=period, adjust=False).mean()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ema,
                    mode="lines",
                    name=f"EMA {period}",
                    line=dict(width=2, color=colors[i % len(colors)]),
                )
            )

    if spec.get("show_vwap", True):
        vw = _vwap_series(df)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=vw,
                mode="lines",
                name="VWAP",
                line=dict(width=1.5, color="rgba(168,85,247,0.9)", dash="dot"),
            )
        )

    for hl in spec.get("hlines") or []:
        try:
            y = float(hl.get("y"))
        except (TypeError, ValueError, AttributeError):
            continue
        fig.add_hline(y=y, line_dash="dash", line_color="rgba(250,204,21,0.7)", annotation_text=str(hl.get("label", ""))[:32])

    vbi = spec.get("vline_bar_idx")
    try:
        vbi = int(vbi) if vbi is not None else None
    except (TypeError, ValueError):
        vbi = None
    if vbi is not None and 0 <= vbi < len(df):
        xv = df["date"].iloc[vbi]
        fig.add_vline(x=xv, line_color="rgba(250,204,21,0.9)", line_width=2, annotation_text="trigger")

    for ann in spec.get("annotations") or []:
        try:
            bi = int(ann.get("bar_idx", -1))
            y = float(ann.get("y", 0))
            txt = str(ann.get("text", ""))[:80]
        except (TypeError, ValueError):
            continue
        if 0 <= bi < len(df):
            fig.add_annotation(
                x=df["date"].iloc[bi],
                y=y,
                text=txt,
                showarrow=True,
                arrowhead=2,
                font=dict(size=11, color="#e2e8f0"),
                bgcolor="rgba(15,23,42,0.75)",
            )

    fig.update_layout(
        title=dict(text=f"{underlying_label} — session (tool overlays)", font=dict(size=14)),
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=height,
        width=width,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(rangeslider_visible=False)

    try:
        return fig.to_image(format="png", engine="kaleido", scale=1)
    except Exception as e:
        _log.warning("fo_claude_chart: to_image failed: %s", e)
        return None


def png_to_data_url(png: bytes) -> str:
    b64 = base64.standard_b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"
