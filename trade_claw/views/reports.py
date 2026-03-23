"""Date-range P/L report: per stock, per trade, totals with pagination."""
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from trade_claw.constants import (
    ALL10_STRATEGY_OPTIONS,
    CLOSED_AT_REALISED,
    DEFAULT_INTERVALS,
    DEFAULT_INTRADAY_INTERVAL,
    NIFTY50_DEFAULT_SELECTED,
    NIFTY50_SYMBOLS,
    NSE_EXCHANGE,
    REPORTS_PAGE_SIZE,
)
from trade_claw.market_data import candles_to_dataframe, get_instrument_token
from trade_claw.pl_style import pl_markdown, style_pl_dataframe
from trade_claw.trade_engine import (
    build_trade_rows_for_df,
    filter_trade_rows_by_view,
    split_dataframe_by_trading_day,
)


def render_reports(kite):
    if st.session_state.nse_instruments is None:
        with st.spinner("Loading NSE instruments..."):
            try:
                st.session_state.nse_instruments = kite.instruments(NSE_EXCHANGE)
            except Exception as e:
                st.error(f"Failed to load instruments: {e}")
                st.stop()
    instruments = st.session_state.nse_instruments
    symbol_to_name = {i["tradingsymbol"]: i.get("name", i["tradingsymbol"]) for i in instruments}

    st.title("Reports – strategy P/L over a range")
    nav1, nav2, nav3, nav4, nav5 = st.columns(5)
    with nav1:
        if st.button("Home (single day)", key="rep_nav_all10"):
            st.session_state.view = "all10"
            st.rerun()
    with nav2:
        if st.button("Stock list", key="rep_nav_dash"):
            st.session_state.view = "dashboard"
            st.session_state.selected_symbol = None
            st.rerun()
    with nav3:
        if st.button("Index ETFs", key="rep_nav_etf"):
            st.session_state.view = "index_etfs"
            st.session_state.selected_symbol = None
            st.rerun()
    with nav4:
        if st.button("F&O Options", key="rep_nav_fo"):
            st.session_state.view = "fo_options"
            st.session_state.selected_symbol = None
            st.rerun()
    with nav5:
        if st.button("F&O Agent", key="rep_nav_fo_agent"):
            st.session_state.view = "fo_agent"
            st.session_state.selected_symbol = None
            st.rerun()

    today = datetime.now().date()
    default_from = today - timedelta(days=7)
    col_a, col_b = st.columns(2)
    with col_a:
        from_d = st.date_input("From date", value=default_from, key="rep_from_d")
    with col_b:
        to_d = st.date_input("To date", value=today, key="rep_to_d")

    if from_d > to_d:
        st.warning("From date must be on or before To date.")
        st.stop()

    trade_view = st.radio(
        "Trade direction",
        options=["All trades", "Long only", "Short only"],
        index=1,
        horizontal=True,
        key="rep_trade_view",
    )
    strategy_for_scrip = st.selectbox(
        "Strategy",
        options=ALL10_STRATEGY_OPTIONS,
        index=ALL10_STRATEGY_OPTIONS.index("MA (EMA crossover)"),
        key="rep_strategy",
    )
    intraday_intervals = [i for i in DEFAULT_INTERVALS if i != "day"]
    _int_idx = (
        intraday_intervals.index(DEFAULT_INTRADAY_INTERVAL)
        if DEFAULT_INTRADAY_INTERVAL in intraday_intervals
        else 0
    )
    chosen_interval = st.selectbox(
        "Minute interval",
        options=intraday_intervals,
        index=_int_idx,
        key="rep_interval",
    )
    selected_stocks = st.multiselect(
        "Stocks",
        options=NIFTY50_SYMBOLS,
        default=NIFTY50_DEFAULT_SELECTED,
        key="rep_stocks",
    )
    symbols_to_load = selected_stocks if selected_stocks else NIFTY50_DEFAULT_SELECTED

    run = st.button("Generate report", type="primary", key="rep_run")

    if run:
        from_dt = datetime(from_d.year, from_d.month, from_d.day, 9, 15, 0)
        to_dt = datetime(to_d.year, to_d.month, to_d.day, 15, 30, 0)
        from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
        to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")

        all_detail_rows: list[dict] = []

        with st.spinner(f"Fetching {chosen_interval} data for {len(symbols_to_load)} symbol(s)…"):
            for symbol in symbols_to_load:
                token = get_instrument_token(symbol, instruments)
                if token is None:
                    continue
                try:
                    candles = kite.historical_data(token, from_str, to_str, interval=chosen_interval)
                except Exception as e:
                    st.warning(f"{symbol}: historical_data failed ({e})")
                    continue
                df = candles_to_dataframe(candles)
                if df.empty:
                    continue
                name = symbol_to_name.get(symbol, symbol)
                for session_date, df_day in split_dataframe_by_trading_day(df):
                    raw = build_trade_rows_for_df(df_day, chosen_interval, strategy_for_scrip)
                    for t in filter_trade_rows_by_view(raw, trade_view):
                        all_detail_rows.append({
                            "Date": session_date,
                            "Symbol": symbol,
                            "Name": name,
                            "Strategy": t["Strategy"],
                            "Signal": t["Signal"],
                            "Qty": t["Qty"],
                            "Entry": t["Entry"],
                            "Exit": t["Exit"],
                            "Closed at": t["Closed at"],
                            "P/L": t["P/L"],
                            "Value": t["Value"],
                        })

        if not all_detail_rows:
            st.session_state.rep_cache_detail = None
            st.warning("No trades matched your filters for this range. Try other dates, stocks, or strategy.")
        else:
            st.session_state.rep_cache_detail = pd.DataFrame(all_detail_rows)
            st.session_state.rep_page = 1
        st.rerun()

    detail_df = st.session_state.get("rep_cache_detail")

    if detail_df is None or (isinstance(detail_df, pd.DataFrame) and detail_df.empty):
        st.info(
            "Choose date range, stocks, strategy, and interval, then click **Generate report**. "
            "Each trading day is analysed separately (intraday session per day). "
            "Use **Page** below after generating to browse all trades."
        )
        return

    total_trades = len(detail_df)
    total_pl = float(detail_df["P/L"].sum())
    realised_pl = float(
        detail_df.loc[detail_df["Closed at"].isin(CLOSED_AT_REALISED), "P/L"].sum()
    )
    unrealised_pl = float(detail_df.loc[detail_df["Closed at"] == "EOD", "P/L"].sum())
    total_value = float(detail_df["Value"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total trades", total_trades)
    with c2:
        st.markdown(f"**Total P/L**  \n{pl_markdown(total_pl)}")
    with c3:
        st.markdown(f"**Realised P/L**  \n{pl_markdown(realised_pl)}")
    with c4:
        st.markdown(f"**Unrealised P/L**  \n{pl_markdown(unrealised_pl)}")
    c5.metric("Total traded value", f"₹{total_value:,.2f}")

    st.subheader("P/L by stock")
    by_sym = (
        detail_df.groupby(["Symbol", "Name"], as_index=False)
        .agg(Trades=("P/L", "count"), Total_P_L=("P/L", "sum"), Total_Value=("Value", "sum"))
        .sort_values("Total_P_L", ascending=False)
    )
    _by_styled = style_pl_dataframe(
        by_sym.style.format({"Total_P_L": "₹{:+,.2f}", "Total_Value": "₹{:,.2f}"}),
        "Total_P_L",
    )
    st.dataframe(_by_styled, use_container_width=True, hide_index=True)

    st.subheader("All trades (paginated)")
    n = len(detail_df)
    n_pages = max(1, (n + REPORTS_PAGE_SIZE - 1) // REPORTS_PAGE_SIZE)
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1, key="rep_page")
    start = (int(page) - 1) * REPORTS_PAGE_SIZE
    end = min(start + REPORTS_PAGE_SIZE, n)
    st.caption(f"Showing {start + 1}–{end} of {n} trades (page size {REPORTS_PAGE_SIZE}).")

    page_df = detail_df.iloc[start:end].copy()
    display_df = page_df[
        ["Date", "Symbol", "Strategy", "Signal", "Qty", "Entry", "Exit", "Closed at", "P/L", "Value"]
    ]
    _page_styled = style_pl_dataframe(
        display_df.style.format({
            "Qty": "{:,.0f}",
            "Entry": "₹{:,.2f}",
            "Exit": "₹{:,.2f}",
            "P/L": "₹{:+,.2f}",
            "Value": "₹{:,.2f}",
        }),
        "P/L",
    )
    st.dataframe(_page_styled, use_container_width=True, hide_index=True)


def seed_reports_from_home(
    chosen_date,
    selected_stocks: list | None,
    strategy_for_scrip: str,
    chosen_interval: str,
    trade_view: str,
):
    """Copy single-day home settings into session_state for the reports page widgets."""
    st.session_state.rep_from_d = chosen_date
    st.session_state.rep_to_d = chosen_date
    st.session_state.rep_stocks = list(selected_stocks) if selected_stocks else list(NIFTY50_DEFAULT_SELECTED)
    if strategy_for_scrip in ALL10_STRATEGY_OPTIONS:
        st.session_state.rep_strategy = strategy_for_scrip
    st.session_state.rep_interval = chosen_interval
    st.session_state.rep_trade_view = trade_view
    st.session_state.rep_cache_detail = None
    st.session_state.view = "reports"
