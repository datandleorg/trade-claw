"""
Streamlit app: Kite Connect – Nifty 50 top 10 dashboard and historical prices (NSE).
Loads KITE_API_KEY and KITE_API_SECRET from .env (or env / Streamlit secrets).
After login, landing shows top 10 Nifty 50 stocks; click a stock for historical chart.
Session is persisted to a file so refresh keeps you logged in.
Run: uv run streamlit run app.py
"""
from datetime import datetime, timedelta
import json
import os

from dotenv import load_dotenv

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from kiteconnect import KiteConnect

load_dotenv()

# --- Constants ---
NSE_EXCHANGE = "NSE"
DEFAULT_INTERVALS = ["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute", "day"]
NIFTY50_TOP10 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
]
SESSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".kite_session.json")


def get_kite_credentials():
    """API key and secret from .env / env / Streamlit secrets."""
    api_key = os.environ.get("KITE_API_KEY") or (st.secrets.get("KITE_API_KEY") if st.secrets else None)
    api_secret = os.environ.get("KITE_API_SECRET") or (st.secrets.get("KITE_API_SECRET") if st.secrets else None)
    return api_key, api_secret


def init_session_state():
    if "kite" not in st.session_state:
        st.session_state.kite = None
    if "access_token" not in st.session_state:
        st.session_state.access_token = None
    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = None
    if "nse_instruments" not in st.session_state:
        st.session_state.nse_instruments = None
    if "view" not in st.session_state:
        st.session_state.view = "all10"  # "all10" = home, "dashboard" = stock list


def save_session_to_file(api_key, access_token):
    """Persist session so it survives browser refresh."""
    try:
        with open(SESSION_FILE, "w") as f:
            json.dump({"api_key": api_key, "access_token": access_token}, f)
    except Exception:
        pass


def load_session_from_file(api_key):
    """Restore session from file if it exists and api_key matches. Returns (kite, access_token) or None."""
    if not os.path.isfile(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, "r") as f:
            data = json.load(f)
        if data.get("api_key") != api_key:
            return None
        token = data.get("access_token")
        if not token:
            return None
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(token)
        return kite, token
    except Exception:
        return None


def clear_session_file():
    """Remove persisted session file."""
    try:
        if os.path.isfile(SESSION_FILE):
            os.remove(SESSION_FILE)
    except Exception:
        pass


def get_instrument_token(symbol, instruments):
    """Resolve instrument_token for an NSE symbol from instruments list."""
    if not instruments:
        return None
    match = next((i for i in instruments if i.get("tradingsymbol") == symbol), None)
    return match["instrument_token"] if match else None


def candles_to_dataframe(candles):
    """Convert Kite historical_data response to DataFrame. Handles list of lists or list of dicts."""
    if not candles:
        return pd.DataFrame()
    first = candles[0]
    if isinstance(first, dict):
        return pd.DataFrame(candles).rename(columns={"date": "date"} if "date" in (first or {}) else {})
    # List of [timestamp, open, high, low, close, volume]
    df = pd.DataFrame(
        candles,
        columns=["date", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


# --- Strategy applicability (ORB, VWAP, RSI, Flag) ---
INTERVAL_MINUTES = {
    "minute": 1,
    "3minute": 3,
    "5minute": 5,
    "10minute": 10,
    "15minute": 15,
    "30minute": 30,
    "60minute": 60,
}


def _orb_analysis(df, opening_range_minutes=15, interval_minutes=5):
    """Return (applicable, explanation, signal_dict with signal/target/stop/entry_bar_idx)."""
    empty_sig = {"signal": None, "target": None, "stop": None, "entry_bar_idx": None}
    if len(df) < 2:
        return False, "", empty_sig
    n_bars = max(1, (opening_range_minutes + interval_minutes - 1) // interval_minutes)
    n_bars = min(n_bars, len(df) - 1)
    or_high = df["high"].iloc[:n_bars].max()
    or_low = df["low"].iloc[:n_bars].min()
    or_range = or_high - or_low
    rest = df.iloc[n_bars:]
    break_above = (rest["high"] > or_high).any()
    break_below = (rest["low"] < or_low).any()
    applicable = break_above or break_below
    parts = [f"OR High = ₹{or_high:,.2f}, OR Low = ₹{or_low:,.2f} (first {n_bars} bars)."]
    sig = empty_sig.copy()
    if break_above and break_below:
        pos_above = (rest["high"] > or_high).values.argmax()
        pos_below = (rest["low"] < or_low).values.argmax()
        if pos_above <= pos_below:
            entry_bar_idx = n_bars + pos_above
            sig = {"signal": "BUY", "target": or_high + or_range, "stop": or_low, "entry_bar_idx": entry_bar_idx}
            parts.append(f" Broke above first at bar {n_bars + pos_above + 1}.")
        else:
            entry_bar_idx = n_bars + pos_below
            sig = {"signal": "SELL", "target": or_low - or_range, "stop": or_high, "entry_bar_idx": entry_bar_idx}
            parts.append(f" Broke below first at bar {n_bars + pos_below + 1}.")
    elif break_above:
        pos = (rest["high"] > or_high).values.argmax()
        bar_num = n_bars + pos + 1
        entry_bar_idx = n_bars + pos
        row = rest.iloc[pos]
        parts.append(f" Broke above at bar {bar_num} (high ₹{row['high']:,.2f}).")
        sig = {"signal": "BUY", "target": or_high + or_range, "stop": or_low, "entry_bar_idx": entry_bar_idx}
    elif break_below:
        pos = (rest["low"] < or_low).values.argmax()
        bar_num = n_bars + pos + 1
        entry_bar_idx = n_bars + pos
        row = rest.iloc[pos]
        parts.append(f" Broke below at bar {bar_num} (low ₹{row['low']:,.2f}).")
        sig = {"signal": "SELL", "target": or_low - or_range, "stop": or_high, "entry_bar_idx": entry_bar_idx}
    return applicable, "".join(parts), sig


def _vwap_analysis(df):
    """Return (applicable, explanation, signal_dict)."""
    empty_sig = {"signal": None, "target": None, "stop": None, "entry_bar_idx": None}
    if len(df) < 2:
        return False, "", empty_sig
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap_series = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    vwap_last = vwap_series.iloc[-1]
    diff = df["close"].values - vwap_series.values
    crosses = 0
    for i in range(1, len(diff)):
        if (diff[i - 1] >= 0 and diff[i] < 0) or (diff[i - 1] <= 0 and diff[i] > 0):
            crosses += 1
    applicable = crosses > 0
    last_close = df["close"].iloc[-1]
    day_high = df["high"].max()
    day_low = df["low"].min()
    side = "above" if last_close >= vwap_last else "below"
    text = f"VWAP(day) = ₹{vwap_last:,.2f}. Price crossed VWAP {crosses} time(s). Last close ₹{last_close:,.2f} ({side} VWAP)."
    entry_bar_idx = len(df) - 1
    if last_close >= vwap_last:
        sig = {"signal": "BUY", "target": day_high, "stop": vwap_last, "entry_bar_idx": entry_bar_idx}
    else:
        sig = {"signal": "SELL", "target": day_low, "stop": vwap_last, "entry_bar_idx": entry_bar_idx}
    return applicable, text, sig


def _rsi_analysis(df, period=14, overbought=70, oversold=30):
    """Return (applicable, explanation, signal_dict)."""
    empty_sig = {"signal": None, "target": None, "stop": None, "entry_bar_idx": None}
    if len(df) <= period:
        return False, "", empty_sig
    close = df["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_loss.where(avg_loss != 0, 1e-10).rdiv(avg_gain)
    rsi = 100 - (100 / (1 + rs))
    rsi_max = rsi.max()
    rsi_min = rsi.min()
    applicable = (rsi >= overbought).any() or (rsi <= oversold).any()
    parts = [f"RSI(14) range: {rsi_min:.1f}–{rsi_max:.1f}. Last RSI = {rsi.iloc[-1]:.1f}."]
    sig = empty_sig.copy()
    pos_ob = (rsi >= overbought).values.argmax() if (rsi >= overbought).any() else len(rsi)
    pos_os = (rsi <= oversold).values.argmax() if (rsi <= oversold).any() else len(rsi)
    if (rsi <= oversold).any() and (not (rsi >= overbought).any() or pos_os >= pos_ob):
        pos = pos_os
        bar = df.iloc[pos]
        parts.append(f" Oversold (≤{oversold}) at bar {pos + 1} (RSI {rsi.iloc[pos]:.1f}).")
        sig = {"signal": "BUY", "target": bar["high"] + (bar["high"] - bar["low"]), "stop": bar["low"], "entry_bar_idx": pos}
    elif (rsi >= overbought).any():
        pos = pos_ob
        bar = df.iloc[pos]
        parts.append(f" Overbought (≥{overbought}) at bar {pos + 1} (RSI {rsi.iloc[pos]:.1f}).")
        sig = {"signal": "SELL", "target": bar["low"] - (bar["high"] - bar["low"]), "stop": bar["high"], "entry_bar_idx": pos}
    return applicable, "".join(parts), sig


def _flag_analysis(df, bars_recent=10, bars_prior=20):
    """Return (applicable, explanation, signal_dict)."""
    empty_sig = {"signal": None, "target": None, "stop": None, "entry_bar_idx": None}
    if len(df) < bars_recent:
        return len(df) >= 10, "Enough bars to spot flagpole + consolidation visually.", empty_sig
    recent = df.iloc[-bars_recent:]
    r_high = recent["high"].max()
    r_low = recent["low"].min()
    range_recent = r_high - r_low
    last_close = df["close"].iloc[-1]
    mid = (r_high + r_low) / 2
    entry_bar_idx = len(df) - 1
    if len(df) >= bars_recent + bars_prior:
        prior = df.iloc[-(bars_recent + bars_prior):-bars_recent]
        range_prior = prior["high"].max() - prior["low"].min()
        pct = (100 * range_recent / range_prior) if range_prior > 0 else 0
        text = f"Last {bars_recent} bars range = ₹{range_recent:,.2f}; prior {bars_prior} bars range = ₹{range_prior:,.2f} ({pct:.0f}%). Tight consolidation suggests flag."
    else:
        text = "Enough bars to spot flagpole + consolidation visually."
    if last_close >= mid:
        sig = {"signal": "BUY", "target": r_high + range_recent, "stop": r_low, "entry_bar_idx": entry_bar_idx}
    else:
        sig = {"signal": "SELL", "target": r_low - range_recent, "stop": r_high, "entry_bar_idx": entry_bar_idx}
    return True, text, sig


def _simulate_trade_close(df, entry_bar_idx, entry_price, target, stop, signal, qty=1):
    """Simulate bars after entry: close trade when target or stop is hit first. Return (closed_at, exit_price, pl)."""
    if entry_bar_idx is None or entry_bar_idx >= len(df) - 1:
        exit_price = df["close"].iloc[-1]
        if signal == "BUY":
            pl = (exit_price - entry_price) * qty
        else:
            pl = (entry_price - exit_price) * qty
        return "EOD", exit_price, pl
    for i in range(entry_bar_idx + 1, len(df)):
        row = df.iloc[i]
        high, low = row["high"], row["low"]
        if signal == "BUY":
            if high >= target and low <= stop:
                exit_price = target
                pl = (exit_price - entry_price) * qty
                return "Target", exit_price, pl
            if high >= target:
                exit_price = target
                pl = (exit_price - entry_price) * qty
                return "Target", exit_price, pl
            if low <= stop:
                exit_price = stop
                pl = (exit_price - entry_price) * qty
                return "Stop", exit_price, pl
        else:
            if low <= target and high >= stop:
                exit_price = target
                pl = (entry_price - exit_price) * qty
                return "Target", exit_price, pl
            if low <= target:
                exit_price = target
                pl = (entry_price - exit_price) * qty
                return "Target", exit_price, pl
            if high >= stop:
                exit_price = stop
                pl = (entry_price - exit_price) * qty
                return "Stop", exit_price, pl
    exit_price = df["close"].iloc[-1]
    if signal == "BUY":
        pl = (exit_price - entry_price) * qty
    else:
        pl = (entry_price - exit_price) * qty
    return "EOD", exit_price, pl


def get_applicable_strategies(df, interval_str="5minute"):
    """Return list of strategy names applicable for this OHLCV DataFrame."""
    strategies = []
    if df.empty or len(df) < 2:
        return strategies
    if interval_str in INTERVAL_MINUTES:
        interval_min = INTERVAL_MINUTES[interval_str]
        if _orb_analysis(df, 15, interval_min)[0]:
            strategies.append("ORB")
    if _vwap_analysis(df)[0]:
        strategies.append("VWAP")
    if _rsi_analysis(df)[0]:
        strategies.append("RSI")
    if _flag_analysis(df)[0]:
        strategies.append("Flag")
    return strategies


def get_strategy_analyses(df, interval_str="5minute"):
    """Return list of (strategy_name, explanation_str, signal_dict) for each applicable strategy."""
    out = []
    if df.empty or len(df) < 2:
        return out
    if interval_str in INTERVAL_MINUTES:
        interval_min = INTERVAL_MINUTES[interval_str]
        ok, text, sig = _orb_analysis(df, 15, interval_min)
        if ok:
            out.append(("ORB", text, sig))
    ok, text, sig = _vwap_analysis(df)
    if ok:
        out.append(("VWAP", text, sig))
    ok, text, sig = _rsi_analysis(df)
    if ok:
        out.append(("RSI", text, sig))
    ok, text, sig = _flag_analysis(df)
    if ok:
        out.append(("Flag", text, sig))
    return out


def main():
    st.set_page_config(page_title="Nifty 50 – Kite historical", layout="wide")
    init_session_state()

    api_key, api_secret = get_kite_credentials()
    if not api_key or not api_secret:
        st.error(
            "Set KITE_API_KEY and KITE_API_SECRET in environment or in Streamlit secrets "
            "(e.g. .streamlit/secrets.toml)."
        )
        return

    # --- Restore session from file (survives refresh) ---
    if not st.session_state.access_token:
        restored = load_session_from_file(api_key)
        if restored:
            kite, token = restored
            st.session_state.kite = kite
            st.session_state.access_token = token
            st.rerun()

    # --- Auto-read request_token from redirect URL (no manual paste) ---
    if not st.session_state.access_token:
        raw = st.query_params.get("request_token")
        request_token_from_url = raw[0] if isinstance(raw, list) else raw
        if request_token_from_url and isinstance(request_token_from_url, str):
            request_token_from_url = request_token_from_url.strip()
            if request_token_from_url:
                try:
                    kite = KiteConnect(api_key=api_key)
                    data = kite.generate_session(request_token_from_url, api_secret=api_secret)
                    access_token = data["access_token"]
                    kite.set_access_token(access_token)
                    st.session_state.kite = kite
                    st.session_state.access_token = access_token
                    save_session_to_file(api_key, access_token)
                    # Clear URL so token is not in address bar and refresh does not reuse it
                    st.query_params.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not complete login: {e}")

    # --- Sidebar: Auth ---
    with st.sidebar:
        st.subheader("Kite Connect login")
        if st.session_state.access_token:
            st.success("Session active")
            if st.button("Clear session"):
                clear_session_file()
                st.session_state.access_token = None
                st.session_state.kite = None
                st.session_state.selected_symbol = None
                st.session_state.nse_instruments = None
                st.session_state.view = "all10"
                st.rerun()
        else:
            try:
                kite = KiteConnect(api_key=api_key)
                login_url = kite.login_url()
                st.markdown(f"[Login with Kite]({login_url})")
                st.caption("After login you’ll be redirected back and signed in automatically.")
            except Exception as e:
                st.error(str(e))

        if not st.session_state.access_token:
            st.stop()

    kite = st.session_state.kite

    # --- All 10 stocks for one date (date picker at top) ---
    if st.session_state.view == "all10":
        if st.session_state.nse_instruments is None:
            with st.spinner("Loading NSE instruments..."):
                try:
                    st.session_state.nse_instruments = kite.instruments(NSE_EXCHANGE)
                except Exception as e:
                    st.error(f"Failed to load instruments: {e}")
                    st.stop()
        instruments = st.session_state.nse_instruments
        symbol_to_name = {i["tradingsymbol"]: i.get("name", i["tradingsymbol"]) for i in instruments}

        st.title("All 10 stocks – single day")
        intraday_intervals = [i for i in DEFAULT_INTERVALS if i != "day"]
        chosen_date = st.date_input(
            "Date",
            value=datetime(2026, 3, 16).date(),
            key="all10_date",
        )
        chosen_interval = st.selectbox(
            "Minute interval",
            options=intraday_intervals,
            index=intraday_intervals.index("5minute"),
            key="all10_interval",
        )
        st.caption("Changing date or interval reloads data and reruns strategy analysis for all 10 stocks.")
        if st.button("Go to stock list", key="all10_back"):
            st.session_state.view = "dashboard"
            st.rerun()

        from_dt = datetime(chosen_date.year, chosen_date.month, chosen_date.day, 9, 15, 0)
        to_dt = datetime(chosen_date.year, chosen_date.month, chosen_date.day, 15, 30, 0)
        from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
        to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")

        for symbol in NIFTY50_TOP10:
            name = symbol_to_name.get(symbol, symbol)
            token = get_instrument_token(symbol, instruments)
            if token is None:
                st.warning(f"{symbol} – instrument not found.")
                continue
            try:
                candles = kite.historical_data(token, from_str, to_str, interval=chosen_interval)
            except Exception as e:
                st.error(f"{symbol} – {e}")
                continue
            df = candles_to_dataframe(candles)
            if df.empty:
                st.caption(f"**{symbol}** — {name}: No data for {chosen_date}")
                continue
            strategies = get_applicable_strategies(df, chosen_interval)
            strategy_label = ", ".join(strategies) if strategies else "—"
            with st.expander(f"**{symbol}** — {name} · {strategy_label}", expanded=True):
                last_close = df["close"].iloc[-1]
                day_high = df["high"].max()
                day_low = df["low"].min()
                day_vol = df["volume"].sum()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Close", f"₹{last_close:,.2f}")
                c2.metric("High", f"₹{day_high:,.2f}")
                c3.metric("Low", f"₹{day_low:,.2f}")
                c4.metric("Volume", f"{day_vol:,.0f}")
                col_chart, col_analysis = st.columns([2, 1])
                with col_chart:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Candlestick(
                            x=df["date"],
                            open=df["open"],
                            high=df["high"],
                            low=df["low"],
                            close=df["close"],
                            name="OHLC",
                        )
                    )
                    fig.update_layout(
                        title=f"{symbol} – {chosen_interval}",
                        height=280,
                        xaxis_rangeslider_visible=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col_analysis:
                    st.markdown("**Why (numbers)**")
                    analyses = get_strategy_analyses(df, chosen_interval)
                    MOCK_QTY = 1
                    trade_rows = []
                    if analyses:
                        for sname, text, sig in analyses:
                            st.markdown(f"**{sname}**")
                            st.caption(text)
                            if sig.get("signal") and sig.get("target") is not None and sig.get("stop") is not None:
                                entry_bar_idx = sig.get("entry_bar_idx")
                                if entry_bar_idx is not None and 0 <= entry_bar_idx < len(df):
                                    entry = float(df.iloc[entry_bar_idx]["close"])
                                else:
                                    entry = last_close
                                target = sig["target"]
                                stop = sig["stop"]
                                closed_at, exit_price, pl = _simulate_trade_close(
                                    df, entry_bar_idx, entry, target, stop, sig["signal"], MOCK_QTY
                                )
                                value = entry * MOCK_QTY
                                trade_rows.append({
                                    "Strategy": sname,
                                    "Signal": sig["signal"],
                                    "Entry": entry,
                                    "Closed at": closed_at,
                                    "Exit": exit_price,
                                    "P/L": pl,
                                    "Value": value,
                                })
                                if sig["signal"] == "SELL":
                                    st.error(f"**SELL** · Target ₹{target:,.2f} · Stop ₹{stop:,.2f}")
                                else:
                                    st.success(f"**BUY** · Target ₹{target:,.2f} · Stop ₹{stop:,.2f}")
                                st.caption(f"Trade closed at **{closed_at}** · Exit ₹{exit_price:,.2f} · P/L: ₹{pl:+,.2f}")
                    if trade_rows:
                        st.markdown("**Mock trade record**")
                        trade_df = pd.DataFrame(trade_rows)
                        st.dataframe(
                            trade_df.style.format({
                                "Entry": "₹{:,.2f}", "Exit": "₹{:,.2f}", "P/L": "₹{:+,.2f}", "Value": "₹{:,.2f}",
                            }, na_rep="—"),
                            use_container_width=True,
                            hide_index=True,
                        )
                        total_value = sum(r["Value"] for r in trade_rows)
                        total_pl = sum(r["P/L"] for r in trade_rows)
                        st.caption(f"**Total value:** ₹{total_value:,.2f} · **Total P/L:** ₹{total_pl:+,.2f} (trade closed when target or stop hit).")
                    if not analyses:
                        st.caption("No strategy met the criteria for this data.")
        return

    # --- Dashboard: top 10 Nifty 50 ---
    if st.session_state.selected_symbol is None:
        if st.session_state.nse_instruments is None:
            with st.spinner("Loading NSE instruments..."):
                try:
                    st.session_state.nse_instruments = kite.instruments(NSE_EXCHANGE)
                except Exception as e:
                    st.error(f"Failed to load instruments: {e}")
                    st.stop()
        instruments = st.session_state.nse_instruments
        symbol_to_name = {i["tradingsymbol"]: i.get("name", i["tradingsymbol"]) for i in instruments}
        st.title("Nifty 50 – Top 10")
        st.caption("Click a stock to view historical prices and chart.")
        if st.button("Home (all 10 for a date)"):
            st.session_state.view = "all10"
            st.rerun()
        for symbol in NIFTY50_TOP10:
            name = symbol_to_name.get(symbol, symbol)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{symbol}** — {name}")
            with col2:
                if st.button("View", key=f"btn_{symbol}"):
                    st.session_state.selected_symbol = symbol
                    st.rerun()
        return

    # --- Stock page: date range in sidebar only when a stock is selected ---
    with st.sidebar:
        st.divider()
        st.subheader("Data range")
        interval = st.selectbox("Interval", DEFAULT_INTERVALS, index=DEFAULT_INTERVALS.index("day"))
        from_d = st.date_input("From", value=datetime.now().date() - timedelta(days=365))
        to_d = st.date_input("To", value=datetime.now().date())
        if from_d > to_d:
            st.warning("From must be before To.")
            st.stop()

    selected_symbol = st.session_state.selected_symbol
    if st.session_state.nse_instruments is None:
        with st.spinner("Loading NSE instruments..."):
            try:
                st.session_state.nse_instruments = kite.instruments(NSE_EXCHANGE)
            except Exception as e:
                st.error(f"Failed to load instruments: {e}")
                st.stop()
    instrument_token = get_instrument_token(selected_symbol, st.session_state.nse_instruments)
    if instrument_token is None:
        st.error(f"Symbol {selected_symbol} not found on NSE.")
        if st.button("Back to dashboard"):
            st.session_state.selected_symbol = None
            st.rerun()
        return

    if st.button("Back to dashboard"):
        st.session_state.selected_symbol = None
        st.rerun()

    # --- Build from/to datetime for API ---
    if interval == "day":
        from_dt = datetime(from_d.year, from_d.month, from_d.day, 0, 0, 0)
        to_dt = datetime(to_d.year, to_d.month, to_d.day, 23, 59, 59)
    else:
        from_dt = datetime(from_d.year, from_d.month, from_d.day, 9, 15, 0)
        to_dt = datetime(to_d.year, to_d.month, to_d.day, 15, 30, 0)

    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")

    # --- Fetch historical data ---
    try:
        candles = kite.historical_data(instrument_token, from_str, to_str, interval=interval)
    except Exception as e:
        st.error(f"Historical data error: {e}")
        st.stop()

    df = candles_to_dataframe(candles)
    if df.empty:
        st.warning("No data returned for the selected range.")
        return

    stock_strategies = get_applicable_strategies(df, interval)
    strategy_label = ", ".join(stock_strategies) if stock_strategies else "—"
    st.caption(f"Suggested strategies: {strategy_label}")

    # --- Summary metrics ---
    st.subheader(f"{selected_symbol} (NSE) – {interval}")
    last_close = df["close"].iloc[-1]
    period_high = df["high"].max()
    period_low = df["low"].min()
    total_vol = df["volume"].sum()
    n_candles = len(df)
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Last close", f"₹{last_close:,.2f}")
    col2.metric("Period high", f"₹{period_high:,.2f}")
    col3.metric("Period low", f"₹{period_low:,.2f}")
    col4.metric("Total volume", f"{total_vol:,.0f}")
    col5.metric("Candles", n_candles)
    st.caption(f"Date range: {date_range}")

    # --- Plotly chart ---
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        )
    )
    fig.update_layout(
        title=f"{selected_symbol} – {interval}",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        xaxis_rangeslider_visible=False,
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Optional: recent data table ---
    with st.expander("Recent candles"):
        st.dataframe(
            df.tail(50).sort_values("date", ascending=False).style.format(
                {"open": "{:,.2f}", "high": "{:,.2f}", "low": "{:,.2f}", "close": "{:,.2f}", "volume": "{:,.0f}"}
            ),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
