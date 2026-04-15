"""F&O Options — multipage route (sidebar)."""
from dotenv import load_dotenv

load_dotenv()

from trade_claw.streamlit_kite_shell import ensure_kite_session, page_config
from trade_claw.views.fo_options import render_fo_options

page_config()
kite = ensure_kite_session()
render_fo_options(kite)
