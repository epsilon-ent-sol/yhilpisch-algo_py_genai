#!/usr/bin/env python3
"""
Fetches the last 3 months of hourly mid-price data for Nasdaq 100 from Oanda.

This script reads Oanda API credentials from the 'oanda.cfg' file in the same directory.
Ensure that 'oanda.cfg' is properly configured but do not hardcode credentials here.
"""

import tpqoa
import pandas as pd


def main():
    # Instantiate the Oanda API client
    oanda = tpqoa.tpqoa("oanda.cfg")

    # Define time range: last 3 months until now (UTC)
    end = pd.Timestamp.utcnow()
    start = end - pd.DateOffset(months=3)

    # Retrieve historical data: instrument NAS100_USD, hourly, mid price
    data = oanda.get_history(
        "NAS100_USD", start, end, granularity="H1", price="M"
    )

    # Print the retrieved DataFrame
    print(data)


if __name__ == "__main__":
    main()
