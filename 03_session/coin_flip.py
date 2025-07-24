#!/usr/bin/env python3
"""
Simple coin-flip strategy: every 10 seconds alternate long/short 1 unit DE30_EUR, stop after 6 trades.
"""

import time
import argparse

from tpqoa import tpqoa


def main():
    parser = argparse.ArgumentParser(
        description='Coin-flip market-order test for DE30_EUR'
    )
    parser.add_argument('--config',     default='oanda.cfg',
                        help='path to OANDA config file')
    parser.add_argument('--instrument', default='DE30_EUR',
                        help='instrument to trade')
    parser.add_argument('--trades',     type=int, default=6,
                        help='total number of market orders')
    parser.add_argument('--interval',   type=int, default=10,
                        help='seconds between orders')
    args = parser.parse_args()

    api = tpqoa(args.config)
    net_pos = 0
    for i in range(1, args.trades + 1):
        # flip position: first go long 1, then flip net position each trade
        units = -2 * net_pos if net_pos != 0 else 1
        net_pos += units
        side = 'long' if units > 0 else 'short'
        print(f"[{i}/{args.trades}] Placing {side} {abs(units)} units of {args.instrument}")
        api.create_order(args.instrument, units)
        if i < args.trades:
            time.sleep(args.interval)

    # flatten final position if any
    if net_pos != 0:
        closing = -net_pos
        side = 'sell' if closing < 0 else 'buy'
        print(f"Closing out final position: {side} {abs(closing)} units")
        api.create_order(args.instrument, closing)


if __name__ == '__main__':
    main()
