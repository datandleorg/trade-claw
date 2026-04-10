"""EMA + LLM options backtest page route."""
from dotenv import load_dotenv

load_dotenv()

from trade_claw.streamlit_kite_shell import ensure_kite_session, page_config
from trade_claw.views.ema_llm_options import render_ema_llm_options

page_config()
kite = ensure_kite_session()
render_ema_llm_options(kite)
