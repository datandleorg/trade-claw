"""Shared Streamlit login sidebar and Kite session for multipage app."""

from __future__ import annotations

import streamlit as st
from kiteconnect import KiteConnect

from trade_claw.kite_session import (
    clear_session_file,
    get_kite_credentials,
    init_session_state,
    load_session_from_file,
    save_session_to_file,
)


def page_config() -> None:
    st.set_page_config(page_title="Trade Claw", layout="wide")


def ensure_kite_session() -> KiteConnect:
    """
    Sidebar login, OAuth callback, persisted session. Stops the app if not logged in.
    Returns the authenticated KiteConnect instance.
    """
    init_session_state()

    api_key, api_secret = get_kite_credentials()
    if not api_key or not api_secret:
        st.error(
            "Set KITE_API_KEY and KITE_API_SECRET in environment or in Streamlit secrets "
            "(e.g. .streamlit/secrets.toml)."
        )
        st.stop()

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
                except Exception as e:  # noqa: BLE001
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
                st.session_state.pop("view", None)
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
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

        if not st.session_state.access_token:
            st.stop()

    return st.session_state.kite
