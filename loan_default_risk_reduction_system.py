import streamlit as st

st.set_page_config(
    page_title="Acme Loan Default Risk",
    page_icon="👋",
)

st.write("# Welcome to Acme Loan Default Risk smart system! 👋")

st.sidebar.success("Select a page above")

st.markdown(
    """
    ### Background
    AI-driven solution to reduce loan defaults. The goal is to use advanced machine learning techniques to accurately predict the likelihood of loan defaults, aiding in informed lending decisions.

    ### Objective
    Build a robust ML model predicting loan default likelihood, incorporating various data sources to improve loan approval processes and reduce default rates.
"""
)