#!/usr/bin/env python3
"""Minimal script to retrieve and print all instruments for an Oanda account using tpqoa."""

from tpqoa import tpqoa


def main():
    # Initialize tpqoa with config file (oanda.cfg should contain your credentials)
    oanda = tpqoa('oanda.cfg')
    instruments = oanda.get_instruments()
    for display_name, name in instruments:
        print(f"{display_name}: {name}")


if __name__ == '__main__':
    main()
