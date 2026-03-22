"""Green / red styling for P&L in Streamlit and pandas Styler."""
from __future__ import annotations

import pandas as pd

# Accessible greens / reds (good contrast on light bg)
PL_POS = "#15803d"
PL_NEG = "#b91c1c"


def pl_styler_cell(val):
    """Return CSS for a single cell (pandas Styler .map)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return f"color: {PL_POS}; font-weight: 600"
    if v < 0:
        return f"color: {PL_NEG}; font-weight: 600"
    return "color: #64748b; font-weight: 500"


def style_pl_dataframe(styler, *columns: str):
    """Apply green/red to named numeric columns if present."""
    for c in columns:
        if c in styler.data.columns:
            styler = styler.map(pl_styler_cell, subset=[c])
    return styler


def pl_markdown(amount: float) -> str:
    """Streamlit markdown :green[] / :red[] for signed rupee amounts."""
    s = f"₹{amount:+,.2f}"
    if amount > 0:
        return f":green[{s}]"
    if amount < 0:
        return f":red[{s}]"
    return f"₹{amount:+,.2f}"


def pl_title_color(amount: float) -> str:
    """Hex color for Plotly title font."""
    if amount > 0:
        return PL_POS
    if amount < 0:
        return PL_NEG
    return "#1e293b"
