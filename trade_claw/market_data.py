"""Kite historical helpers."""
import pandas as pd


def get_instrument_token(symbol, instruments):
    if not instruments:
        return None
    match = next((i for i in instruments if i.get("tradingsymbol") == symbol), None)
    return match["instrument_token"] if match else None


def candles_to_dataframe(candles):
    if not candles:
        return pd.DataFrame()
    first = candles[0]
    if isinstance(first, dict):
        return pd.DataFrame(candles).rename(columns={"date": "date"} if "date" in (first or {}) else {})
    df = pd.DataFrame(
        candles,
        columns=["date", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df
