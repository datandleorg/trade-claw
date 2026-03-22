"""Kite credentials, Streamlit session state, persisted login."""
import json
import os

import streamlit as st
from kiteconnect import KiteConnect

from trade_claw.constants import SESSION_FILE


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
        st.session_state.view = "all10"


def save_session_to_file(api_key, access_token):
    try:
        with open(SESSION_FILE, "w") as f:
            json.dump({"api_key": api_key, "access_token": access_token}, f)
    except Exception:
        pass


def load_session_from_file(api_key):
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
    try:
        if os.path.isfile(SESSION_FILE):
            os.remove(SESSION_FILE)
    except Exception:
        pass
