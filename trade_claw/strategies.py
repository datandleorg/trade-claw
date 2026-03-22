"""ORB, VWAP, RSI, Flag, MA crossover, MA envelope + trade simulation + Plotly overlays."""
import plotly.graph_objects as go

from trade_claw.constants import (
    ENVELOPE_EMA_PERIOD,
    ENVELOPE_PCT,
    INTERVAL_MINUTES,
    MA_EMA_FAST,
    MA_EMA_SLOW,
)


def _orb_analysis(df, opening_range_minutes=15, interval_minutes=5):
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


def _ma_ema_crossover_analysis(df, fast_period=MA_EMA_FAST, slow_period=MA_EMA_SLOW):
    empty_sig = {"signal": None, "target": None, "stop": None, "entry_bar_idx": None}
    min_bars = max(fast_period, slow_period) + 2
    if len(df) < min_bars:
        return False, f"Need at least {min_bars} bars for EMA({fast_period}/{slow_period}) crossover.", empty_sig
    close = df["close"]
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    entry_bar_idx = None
    signal = None
    start = max(fast_period, slow_period)
    for i in range(start, len(df)):
        f_prev, f_now = float(ema_fast.iloc[i - 1]), float(ema_fast.iloc[i])
        s_prev, s_now = float(ema_slow.iloc[i - 1]), float(ema_slow.iloc[i])
        if f_prev <= s_prev and f_now > s_now:
            entry_bar_idx = i
            signal = "BUY"
            break
        if f_prev >= s_prev and f_now < s_now:
            entry_bar_idx = i
            signal = "SELL"
            break
    if entry_bar_idx is None:
        text = (
            f"EMA {fast_period}/{slow_period}: no golden or death cross in this session. "
            f"Last fast ₹{ema_fast.iloc[-1]:,.2f}, slow ₹{ema_slow.iloc[-1]:,.2f} vs close ₹{close.iloc[-1]:,.2f}."
        )
        return False, text, empty_sig
    day_high = df["high"].max()
    day_low = df["low"].min()
    es = float(ema_slow.iloc[entry_bar_idx])
    el = float(df.iloc[entry_bar_idx]["low"])
    eh = float(df.iloc[entry_bar_idx]["high"])
    if signal == "BUY":
        stop = min(es, el)
        sig = {"signal": "BUY", "target": day_high, "stop": stop, "entry_bar_idx": entry_bar_idx}
        cross = "Golden cross"
    else:
        stop = max(es, eh)
        sig = {"signal": "SELL", "target": day_low, "stop": stop, "entry_bar_idx": entry_bar_idx}
        cross = "Death cross"
    text = (
        f"EMA {fast_period}/{slow_period}: {cross} at bar {entry_bar_idx + 1}. "
        f"Fast ₹{ema_fast.iloc[entry_bar_idx]:,.2f} crossed slow ₹{ema_slow.iloc[entry_bar_idx]:,.2f}. "
        f"Last: fast ₹{ema_fast.iloc[-1]:,.2f}, slow ₹{ema_slow.iloc[-1]:,.2f}."
    )
    return True, text, sig


def add_ma_ema_line_traces(fig, df, fast_period=MA_EMA_FAST, slow_period=MA_EMA_SLOW):
    if df is None or df.empty or len(df) < slow_period:
        return
    close = df["close"]
    x = df["date"]
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ema_fast,
            mode="lines",
            name=f"EMA {fast_period}",
            line=dict(color="#00d4ff", width=2),
            opacity=0.95,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ema_slow,
            mode="lines",
            name=f"EMA {slow_period}",
            line=dict(color="#ff9f1a", width=2),
            opacity=0.95,
        )
    )


def _envelope_series(df, ema_period=ENVELOPE_EMA_PERIOD, pct=ENVELOPE_PCT):
    close = df["close"]
    center = close.ewm(span=ema_period, adjust=False).mean()
    upper = center * (1 + pct)
    lower = center * (1 - pct)
    return center, upper, lower


def _ma_envelope_analysis(df, ema_period=ENVELOPE_EMA_PERIOD, pct=ENVELOPE_PCT):
    empty_sig = {"signal": None, "target": None, "stop": None, "entry_bar_idx": None}
    min_bars = ema_period + 2
    if len(df) < min_bars:
        return False, f"Need at least {min_bars} bars for EMA({ema_period}) envelope.", empty_sig
    close = df["close"]
    center, upper, lower = _envelope_series(df, ema_period, pct)
    events = []
    for i in range(ema_period, len(df)):
        c, c_prev = float(close.iloc[i]), float(close.iloc[i - 1])
        u, u_prev = float(upper.iloc[i]), float(upper.iloc[i - 1])
        lo, lo_prev = float(lower.iloc[i]), float(lower.iloc[i - 1])
        if c > u and c_prev <= u_prev:
            events.append((i, "BUY"))
        if c < lo and c_prev >= lo_prev:
            events.append((i, "SELL"))
    if not events:
        text = (
            f"EMA({ema_period}) ±{100 * pct:.1f}% envelope: no breakout in session. "
            f"Last close ₹{close.iloc[-1]:,.2f}, upper ₹{upper.iloc[-1]:,.2f}, lower ₹{lower.iloc[-1]:,.2f}."
        )
        return False, text, empty_sig
    events.sort(key=lambda x: (x[0], 0 if x[1] == "BUY" else 1))
    entry_bar_idx, signal = events[0]
    ce = float(center.iloc[entry_bar_idx])
    ue = float(upper.iloc[entry_bar_idx])
    le = float(lower.iloc[entry_bar_idx])
    sig = {
        "signal": signal,
        "target": ue,
        "stop": le,
        "entry_bar_idx": entry_bar_idx,
        "envelope": True,
        "ema_period": ema_period,
        "pct": pct,
        "center_at_entry": ce,
        "upper_at_entry": ue,
        "lower_at_entry": le,
    }
    side = "above upper" if signal == "BUY" else "below lower"
    text = (
        f"EMA({ema_period}) ±{100 * pct:.1f}% envelope: first break {side} at bar {entry_bar_idx + 1}. "
        f"Center ₹{ce:,.2f}, upper ₹{ue:,.2f}, lower ₹{le:,.2f}. "
        f"Exit rule: opposite band cross (long → lower, short → upper)."
    )
    return True, text, sig


def simulate_envelope_trade_close(
    df, entry_bar_idx, entry_price, signal, qty, ema_period=ENVELOPE_EMA_PERIOD, pct=ENVELOPE_PCT
):
    last_idx = len(df) - 1
    close = df["close"]
    _, upper, lower = _envelope_series(df, ema_period, pct)
    if entry_bar_idx is None or entry_bar_idx >= last_idx:
        exit_price = float(close.iloc[-1])
        if signal == "BUY":
            pl = (exit_price - entry_price) * qty
        else:
            pl = (entry_price - exit_price) * qty
        return "EOD", exit_price, pl, last_idx
    for i in range(entry_bar_idx + 1, len(df)):
        c, c_prev = float(close.iloc[i]), float(close.iloc[i - 1])
        u, u_prev = float(upper.iloc[i]), float(upper.iloc[i - 1])
        lo, lo_prev = float(lower.iloc[i]), float(lower.iloc[i - 1])
        if signal == "BUY":
            if c < lo and c_prev >= lo_prev:
                exit_price = c
                pl = (exit_price - entry_price) * qty
                return "Opposite envelope", exit_price, pl, i
        else:
            if c > u and c_prev <= u_prev:
                exit_price = c
                pl = (entry_price - exit_price) * qty
                return "Opposite envelope", exit_price, pl, i
    exit_price = float(close.iloc[-1])
    if signal == "BUY":
        pl = (exit_price - entry_price) * qty
    else:
        pl = (entry_price - exit_price) * qty
    return "EOD", exit_price, pl, last_idx


def add_ma_envelope_line_traces(fig, df, ema_period=ENVELOPE_EMA_PERIOD, pct=ENVELOPE_PCT):
    if df is None or df.empty or len(df) < ema_period:
        return
    x = df["date"]
    center, upper, lower = _envelope_series(df, ema_period, pct)
    pct_label = f"{100 * pct:.1f}%"
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            name=f"Upper (+{pct_label})",
            line=dict(color="rgba(0, 200, 100, 0.85)", width=1.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            name=f"Lower (−{pct_label})",
            line=dict(color="rgba(255, 80, 80, 0.85)", width=1.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=center,
            mode="lines",
            name=f"EMA {ema_period}",
            line=dict(color="#ff9f1a", width=2),
            opacity=0.95,
        )
    )


def simulate_trade_close(df, entry_bar_idx, entry_price, target, stop, signal, qty=1):
    last_idx = len(df) - 1
    if entry_bar_idx is None or entry_bar_idx >= last_idx:
        exit_price = df["close"].iloc[-1]
        if signal == "BUY":
            pl = (exit_price - entry_price) * qty
        else:
            pl = (entry_price - exit_price) * qty
        return "EOD", exit_price, pl, last_idx
    for i in range(entry_bar_idx + 1, len(df)):
        row = df.iloc[i]
        high, low = row["high"], row["low"]
        if signal == "BUY":
            if high >= target and low <= stop:
                exit_price = target
                pl = (exit_price - entry_price) * qty
                return "Target", exit_price, pl, i
            if high >= target:
                exit_price = target
                pl = (exit_price - entry_price) * qty
                return "Target", exit_price, pl, i
            if low <= stop:
                exit_price = stop
                pl = (exit_price - entry_price) * qty
                return "Stop", exit_price, pl, i
        else:
            if low <= target and high >= stop:
                exit_price = target
                pl = (entry_price - exit_price) * qty
                return "Target", exit_price, pl, i
            if low <= target:
                exit_price = target
                pl = (entry_price - exit_price) * qty
                return "Target", exit_price, pl, i
            if high >= stop:
                exit_price = stop
                pl = (entry_price - exit_price) * qty
                return "Stop", exit_price, pl, i
    exit_price = df["close"].iloc[-1]
    if signal == "BUY":
        pl = (exit_price - entry_price) * qty
    else:
        pl = (entry_price - exit_price) * qty
    return "EOD", exit_price, pl, last_idx


def get_applicable_strategies(df, interval_str="5minute"):
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
    if _ma_ema_crossover_analysis(df)[0]:
        strategies.append("MA")
    if _ma_envelope_analysis(df)[0]:
        strategies.append("Envelope")
    return strategies


def get_strategy_analyses(df, interval_str="5minute"):
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
    ok, text, sig = _ma_ema_crossover_analysis(df)
    if ok:
        out.append(("MA", text, sig))
    ok, text, sig = _ma_envelope_analysis(df)
    if ok:
        out.append(("Envelope", text, sig))
    return out


def strategy_key_from_ui_choice(strategy_choice: str) -> str | None:
    if strategy_choice == "All strategies":
        return None
    if strategy_choice == "MA (EMA crossover)":
        return "MA"
    if strategy_choice == "MA envelope (0.5% bands)":
        return "Envelope"
    return strategy_choice


def filter_analyses_by_strategy_choice(analyses, strategies, strategy_choice: str):
    key = strategy_key_from_ui_choice(strategy_choice)
    if key is None:
        return analyses, strategies
    analyses_f = [t for t in analyses if t[0] == key]
    strategies_f = [s for s in strategies if s == key]
    return analyses_f, strategies_f
