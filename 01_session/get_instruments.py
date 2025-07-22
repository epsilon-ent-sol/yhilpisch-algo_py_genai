#!/usr/bin/env python3
"""
Retrieves and prints available instruments from Oanda using the tpqoa wrapper.

This script reads Oanda API credentials from the 'oanda.cfg' file in the same directory.
Ensure that 'oanda.cfg' is properly configured but do not hardcode credentials here.
"""

import tpqoa


def main():
    # Instantiate the Oanda API client using credentials from oanda.cfg
    oanda = tpqoa.tpqoa("oanda.cfg")
    instruments = oanda.get_instruments()
    print("Available instruments (display name, name):")
    for display_name, name in instruments:
        print(f"{display_name}: {name}")


if __name__ == "__main__":
    main()
