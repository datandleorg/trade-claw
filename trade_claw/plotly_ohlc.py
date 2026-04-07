"""Plotly OHLC figures with optional volume panel (shared x-axis)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OHLC_VOLUME_ROW_HEIGHTS = (0.72, 0.28)
OHLC_VOLUME_VERTICAL_SPACING = 0.08
_DEFAULT_OHLC_VOL_HEIGHT = 420


def df_has_volume_column(df: pd.DataFrame | None) -> bool:
    """True if ``volume`` exists and has at least one non-null value."""
    if df is None or df.empty or "volume" not in df.columns:
        return False
    v = pd.to_numeric(df["volume"], errors="coerce")
    return bool(v.notna().any())


def create_ohlc_volume_figure(
    df: pd.DataFrame,
    *,
    include_volume: bool | None = None,
    row_heights: tuple[float, float] = OHLC_VOLUME_ROW_HEIGHTS,
    vertical_spacing: float = OHLC_VOLUME_VERTICAL_SPACING,
) -> tuple[go.Figure, int, int | None]:
    """
    Build either a plain ``go.Figure()`` or a 2-row subplot (price + volume).

    ``include_volume``: ``True`` = always two rows (volume 0 if column missing); ``False`` = one row;
    ``None`` = two rows only if :func:`df_has_volume_column` is true.

    Returns ``(fig, price_row, volume_row)`` where ``volume_row`` is ``None`` when single-panel.
    """
    if include_volume is None:
        include_volume = df_has_volume_column(df)
    if not include_volume:
        return go.Figure(), 1, None
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        row_heights=list(row_heights),
    )
    return fig, 1, 2


def add_candlestick_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    name: str,
    price_row: int,
    volume_row: int | None,
    **kwargs,
) -> None:
    """Add OHLC candlesticks on the price row (or default axis if single-panel)."""
    tr = go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=name,
        **kwargs,
    )
    if volume_row is not None:
        fig.add_trace(tr, row=price_row, col=1)
    else:
        fig.add_trace(tr)


def add_volume_bar_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    volume_row: int,
) -> None:
    """Bar chart of volume under the price panel (green/red vs open)."""
    x = df["date"]
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        vol = pd.Series(0.0, index=df.index)
    o = pd.to_numeric(df["open"], errors="coerce").fillna(0.0)
    c = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    green = "rgba(34,197,94,0.55)"
    red = "rgba(239,68,68,0.55)"
    colors = [green if float(c.iloc[i]) >= float(o.iloc[i]) else red for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=x,
            y=vol,
            name="Volume",
            marker=dict(color=colors, line=dict(width=0)),
            showlegend=False,
        ),
        row=volume_row,
        col=1,
    )
    fig.update_yaxes(title_text="Vol", row=volume_row, col=1)


def finalize_ohlc_volume_figure(
    fig: go.Figure,
    *,
    height: int = _DEFAULT_OHLC_VOL_HEIGHT,
    template: str | None = None,
    legend_orientation: str = "h",
) -> None:
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        height=height,
        legend=dict(orientation=legend_orientation, yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if template:
        fig.update_layout(template=template)
