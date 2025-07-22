#!/usr/bin/env python3
"""
Streamlit app to fetch and display Oanda instrument historical data as Plotly candlesticks.

Allows selection of instrument, start/end dates, and granularity (default daily).
Reads credentials from 'oanda.cfg' in the same directory; credentials are never hardcoded.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import tpqoa


def main():
    st.title("Oanda Historical Data Viewer")

    # Initialize Oanda API client and fetch instruments
    oanda = tpqoa.tpqoa("oanda.cfg")
    instruments = oanda.get_instruments()
    instrument_display = [f"{dn} ({n})" for dn, n in instruments]
    choice = st.sidebar.selectbox("Instrument", instrument_display)
    instrument = choice.split("(")[-1].strip(")")

    # Date inputs: default last 3 months to today (UTC date)
    today = pd.Timestamp.utcnow().date()
    default_start = (pd.Timestamp.utcnow() - pd.DateOffset(months=3)).date()
    start_date = st.sidebar.date_input("Start date", value=default_start)
    end_date = st.sidebar.date_input("End date", value=today)

    # Granularity selection
    gran_options = ["D", "H1", "H4", "M30", "M15", "M5", "M1"]
    gran = st.sidebar.selectbox("Granularity", gran_options, index=0)

    # Button to fetch data and store in session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if st.sidebar.button("Fetch data"):
        with st.spinner("Fetching data…"):
            df = oanda.get_history(
                instrument, start_date, end_date, granularity=gran, price="M"
            )
        if df.empty:
            st.warning("No data found for this range.")
            st.session_state.df = None
        else:
            st.session_state.df = df

    # If data is loaded, display charts and optional table
    if st.session_state.df is not None:
        df = st.session_state.df

        # Main candlestick chart
        fig = go.Figure(
            data=[go.Candlestick(
                x=df.index, open=df['o'], high=df['h'], low=df['l'], close=df['c']
            )]
        )
        fig.update_layout(
            title=f"{instrument} {gran} mid-price",
            xaxis_title="Time", yaxis_title="Price"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Comparison with up to two other instruments
        others = [f"{dn} ({n})" for dn, n in instruments if n != instrument]
        compare = st.sidebar.multiselect(
            "Compare closes with (max 2)", others,
            help="Select up to two instruments"
        )[:2]
        if compare:
            comp_df = pd.DataFrame({instrument: df['c']})
            for sel in compare:
                sym = sel.split("(")[-1].strip(")")
                df2 = oanda.get_history(sym, start_date, end_date, granularity=gran, price="M")
                comp_df[sym] = df2['c']
            # Optionally normalize series to 1 at first point
            norm = st.sidebar.radio("Normalize comparison series?", ("No", "Yes"), index=0)
            if norm == "Yes":
                comp_df = comp_df / comp_df.iloc[0]
            fig2 = go.Figure()
            for col in comp_df.columns:
                fig2.add_trace(go.Scatter(x=comp_df.index, y=comp_df[col], mode='lines', name=col))
            fig2.update_layout(
                title=("Normalized Closing Price Comparison" if norm == "Yes" else "Closing Price Comparison"),
                xaxis_title="Time", yaxis_title=("Normalized Price" if norm == "Yes" else "Price")
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Optional data table
        show_tbl = st.sidebar.radio("Show data table?", ("No", "Yes"), index=0)
        if show_tbl == "Yes":
            st.dataframe(df)


if __name__ == "__main__":
    main()
