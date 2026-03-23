"""
Streamlit app: Kite Connect – Nifty 50 dashboard and historical prices (NSE).
Run: uv run streamlit run app.py
"""
from dotenv import load_dotenv
import streamlit as st
from kiteconnect import KiteConnect

from trade_claw.kite_session import (
    clear_session_file,
    get_kite_credentials,
    init_session_state,
    load_session_from_file,
    save_session_to_file,
)
from trade_claw.views.all_ten import render_all_ten
from trade_claw.views.dashboard import render_dashboard
from trade_claw.views.fo_agent_options import render_fo_agent_options
from trade_claw.views.fo_options import render_fo_options
from trade_claw.views.fo_options_snapshots_report import render_fo_options_snapshots_report
from trade_claw.views.index_etfs import render_index_etfs
from trade_claw.views.reports import render_reports
from trade_claw.views.stock_detail import render_stock_detail

load_dotenv()


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

    if not st.session_state.access_token:
        restored = load_session_from_file(api_key)
        if restored:
            kite, token = restored
            st.session_state.kite = kite
            st.session_state.access_token = token
            st.rerun()

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
                    st.query_params.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not complete login: {e}")

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
                st.session_state.pop("etf_cache_rows", None)
                st.session_state.pop("etf_cache_charts", None)
                st.session_state.pop("nfo_instruments", None)
                st.session_state.pop("fo_agent_last_result", None)
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

        st.divider()
        st.caption("Navigate")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Intraday home", use_container_width=True, key="nav_all10"):
                st.session_state.view = "all10"
                st.session_state.selected_symbol = None
                st.rerun()
        with c2:
            if st.button("Top 10 list", use_container_width=True, key="nav_dash"):
                st.session_state.view = "dashboard"
                st.session_state.selected_symbol = None
                st.rerun()
        c3, c4 = st.columns(2)
        with c3:
            if st.button("Reports", use_container_width=True, key="nav_rep"):
                st.session_state.view = "reports"
                st.session_state.selected_symbol = None
                st.rerun()
        with c4:
            if st.button("Index ETFs", use_container_width=True, key="nav_etf"):
                st.session_state.view = "index_etfs"
                st.session_state.selected_symbol = None
                st.rerun()
        c5, c6 = st.columns(2)
        with c5:
            if st.button("F&O Options", use_container_width=True, key="nav_fo"):
                st.session_state.view = "fo_options"
                st.session_state.selected_symbol = None
                st.rerun()
        with c6:
            if st.button("F&O Agent", use_container_width=True, key="nav_fo_agent"):
                st.session_state.view = "fo_agent"
                st.session_state.selected_symbol = None
                st.rerun()
        if st.button("F&O snapshots report", use_container_width=True, key="nav_fo_snap"):
            st.session_state.view = "fo_snapshots"
            st.session_state.selected_symbol = None
            st.rerun()

    kite = st.session_state.kite

    if st.session_state.view == "all10":
        render_all_ten(kite)
        return

    if st.session_state.view == "reports":
        render_reports(kite)
        return

    if st.session_state.view == "index_etfs":
        render_index_etfs(kite)
        return

    if st.session_state.view == "fo_options":
        render_fo_options(kite)
        return

    if st.session_state.view == "fo_agent":
        render_fo_agent_options(kite)
        return

    if st.session_state.view == "fo_snapshots":
        render_fo_options_snapshots_report(kite)
        return

    if st.session_state.selected_symbol is None:
        render_dashboard(kite)
        return

    render_stock_detail(kite, st.session_state.selected_symbol)


if __name__ == "__main__":
    main()
