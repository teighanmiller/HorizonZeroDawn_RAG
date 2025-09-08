import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

dashboard_page = st.Page("dashboard.py", title="Dashboard")
chat_page = st.Page("chat_ui.py", title="GAIA")

pg = st.navigation([chat_page, dashboard_page])

st.set_page_config(page_title="Chat")
pg.run()
