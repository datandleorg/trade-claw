"""Single-stock historical chart and suggested strategies."""
from datetime import datetime, timedelta

import plotly.graph_objects as go
import streamlit as st

from trade_claw.constants import DEFAULT_INTERVALS, NSE_EXCHANGE
from trade_claw.market_data import candles_to_dataframe, get_instrument_token
from trade_claw.plotly_ohlc import (
    add_candlestick_trace,
    add_volume_bar_trace,
    create_ohlc_volume_figure,
    finalize_ohlc_volume_figure,
)
from trade_claw.strategies import get_applicable_strategies


def render_stock_detail(kite, selected_symbol: str):
    with st.sidebar:
        st.divider()
        st.subheader("Data range")
        interval = st.selectbox("Interval", DEFAULT_INTERVALS, index=DEFAULT_INTERVALS.index("day"))
        from_d = st.date_input("From", value=datetime.now().date() - timedelta(days=365))
        to_d = st.date_input("To", value=datetime.now().date())
        if from_d > to_d:
            st.warning("From must be before To.")
            st.stop()

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

    if interval == "day":
        from_dt = datetime(from_d.year, from_d.month, from_d.day, 0, 0, 0)
        to_dt = datetime(to_d.year, to_d.month, to_d.day, 23, 59, 59)
    else:
        from_dt = datetime(from_d.year, from_d.month, from_d.day, 9, 15, 0)
        to_dt = datetime(to_d.year, to_d.month, to_d.day, 15, 30, 0)

    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")

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

    fig, pr, vr = create_ohlc_volume_figure(df)
    add_candlestick_trace(fig, df, name="OHLC", price_row=pr, volume_row=vr)
    if vr is not None:
        add_volume_bar_trace(fig, df, volume_row=vr)
    finalize_ohlc_volume_figure(fig, height=560)
    fig.update_layout(title=f"{selected_symbol} – {interval}", xaxis_title="Date")
    if vr is not None:
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    else:
        fig.update_layout(yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Recent candles"):
        st.dataframe(
            df.tail(50).sort_values("date", ascending=False).style.format(
                {"open": "{:,.2f}", "high": "{:,.2f}", "low": "{:,.2f}", "close": "{:,.2f}", "volume": "{:,.0f}"}
            ),
            use_container_width=True,
        )
