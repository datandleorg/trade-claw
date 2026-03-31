"""App-wide constants (symbols, intervals, strategy config)."""
import os

# Project root (parent of trade_claw package)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_default_session = os.path.join(_ROOT, ".kite_session.json")
# Docker: set KITE_SESSION_FILE=/app/data/.kite_session.json so Streamlit + Celery share the same login.
SESSION_FILE = os.path.expanduser(os.environ.get("KITE_SESSION_FILE", _default_session))

NSE_EXCHANGE = "NSE"
NFO_EXCHANGE = "NFO"

# F&O mock target/stop defaults: use trade_claw.env_trading_params (MOCK_ENGINE_* / FO_OPTION_* at call time).
# NSE index F&O underlyings (keys must match Kite NFO `name` / resolver in fo_support)
FO_INDEX_UNDERLYING_KEYS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]
# Human labels for UI (NIFTYNXT50 = Nifty Next 50; NFO may list as NIFTYJR or NIFTYNXT50)
FO_INDEX_UNDERLYING_LABELS = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "BANK NIFTY",
    "FINNIFTY": "FINNIFTY (Nifty Financial Services)",
    "MIDCPNIFTY": "MIDCP NIFTY (Nifty Midcap Select)",
    "NIFTYNXT50": "NIFTY NEXT 50",
}
FO_DEFAULT_UNDERLYINGS = list(FO_INDEX_UNDERLYING_KEYS)
# Summary metrics: Target or Stop = realised; EOD = held to close
FO_CLOSED_AT_REALISED = ("Target", "Stop")
# F&O UI: envelope distance each side as % of EMA (upper = EMA×(1+pct), lower = EMA×(1−pct))
FO_ENVELOPE_BANDWIDTH_MIN_PCT = 0.0
# Max 30% allows mock-agent default (25% / side) to fit the slider when synced from ``MOCK_AGENT_ENVELOPE_PCT``.
FO_ENVELOPE_BANDWIDTH_MAX_PCT = 30.0
FO_ENVELOPE_BANDWIDTH_STEP = 0.05
# F&O page strategy dropdown (underlying signal → long CE / long PE)
FO_STRATEGY_ENVELOPE = "MA envelope (EMA ± bandwidth)"
FO_STRATEGY_MA_CROSS = "MA (EMA crossover)"
FO_STRATEGY_OPTIONS = [FO_STRATEGY_ENVELOPE, FO_STRATEGY_MA_CROSS]
# Mock costs: ₹ per **lot** for full **round trip** (buy + sell); tune to your broker / tax assumptions
FO_BROKERAGE_PER_LOT_RT_DEFAULT = 40.0
FO_TAXES_PER_LOT_RT_DEFAULT = 35.0
# F&O strike vs spot (steps apply to the traded leg: CE shifts up for +OTM, PE shifts down for +OTM)
FO_STRIKE_POLICY_LABELS = [
    "ATM — nearest strike to spot (default)",
    "OTM +1 — one step out-of-the-money",
    "OTM +2 — two steps OTM",
    "ITM −1 — one step in-the-money",
    "ITM −2 — two steps ITM",
]
FO_STRIKE_POLICY_STEPS = [0, 1, 2, -1, -2]
# When the first NFO row for nearest expiry returns no minute candles, try more expiries / duplicate symbols
FO_MAX_EXPIRY_CANDLE_FALLBACKS = 12
DEFAULT_INTERVALS = ["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute", "day"]
# Default candle interval for intraday selectboxes (F&O, intraday home, reports day mode)
DEFAULT_INTRADAY_INTERVAL = "minute"

NIFTY50_SYMBOLS = [
    "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
    "BPCL", "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JSWSTEEL", "KOTAKBANK",
    "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SUNPHARMA", "TATAMOTORS", "TCS", "TATACONSUM", "TATASTEEL",
    "TECHM", "TITAN", "UPL", "WIPRO", "APOLLOHOSP", "BEL", "ULTRACEMCO", "LTIM", "ADANIENT",
]
FO_UNDERLYING_OPTIONS = list(FO_INDEX_UNDERLYING_KEYS) + NIFTY50_SYMBOLS
NIFTY50_TOP10 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
]
NIFTY50_DEFAULT_SELECTED = [
    "HDFCBANK", 
    "RELIANCE", "ULTRACEMCO",
    "TCS", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
]

ALL10_STRATEGY_OPTIONS = [
    "All strategies",
    "ORB",
    "VWAP",
    "RSI",
    "Flag",
    "MA (EMA crossover)",
    "MA envelope (0.5% bands)",
]

MA_EMA_FAST = 9
MA_EMA_SLOW = 20
ENVELOPE_EMA_PERIOD = 20
# Default intraday/strategy envelope when ``INTRADAY_ENVELOPE_DECIMAL`` is unset (see env_trading_params.intraday_envelope_decimal).
ENVELOPE_PCT = 0.0030

CLOSED_AT_REALISED = ("Target", "Stop", "Opposite envelope")

INTERVAL_MINUTES = {
    "minute": 1,
    "3minute": 3,
    "5minute": 5,
    "10minute": 10,
    "15minute": 15,
    "30minute": 30,
    "60minute": 60,
}

ALLOCATED_AMOUNT = 10_000
REPORTS_PAGE_SIZE = 25
REPORTS_MIN_BARS_PER_DAY = 5

# --- Index ETF / fund — Institutional floor (buy-only, daily SMAs) ---
INSTITUTIONAL_SMA_SHORT = 50
INSTITUTIONAL_SMA_LONG = 200
INSTITUTIONAL_STANDARD_BUY = 10_000
INSTITUTIONAL_AGGRESSIVE_BUY = 20_000
INSTITUTIONAL_EXTENDED_ABOVE_PCT = 10.0  # % above SMA200 → "standard" / avoid chasing

# Liquid NSE index / sector ETFs (verify symbols in Kite instrument master; edit as needed)
NSE_INDEX_ETF_SYMBOLS = [
    "NIFTYBEES",
    "SETFNIFTY",
    "SENSEXIETF",
    "JUNIORBEES",
    "BANKBEES",
    "NV20BEES",
    "MONIFTY50",
    "UTINIFTY",
    "CPSEETF",
    "ITBEES",
]
