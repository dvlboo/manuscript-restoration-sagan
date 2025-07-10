import streamlit as st

pages = {
  "Menu": [
    st.Page("pages/home.py", title="🏠 Homepage"),
    st.Page("pages/restore.py", title="💥 Restore"),
  ]
}

pg = st.navigation(pages)
pg.run()