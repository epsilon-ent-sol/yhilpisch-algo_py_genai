#!/usr/bin/env python3
"""
Live mean-reversion trading using 3-second bars and tpqoa streaming.

Fixed strategy parameters:
  bb_window=10, bb_std_dev=2, rsi_period=14,
  ema_period=10, atr_period=14
"""

import argparse

import numpy as np
import pandas as pd

from tpqoa import tpqoa


def main():
    p = argparse.ArgumentParser(description='Live MR trading (3s bars)')
    p.add_argument('--config',     default='oanda.cfg',
                   help='OANDA config file')
    p.add_argument('--instrument', default='DE30_EUR',
                   help='instrument to trade')
    args = p.parse_args()

    api = tpqoa(args.config)
    # Strategy params
    bb_w, bb_sd = 10, 2
    rsi_p, ema_p, atr_p = 14, 10, 14

    # Tick buffer and bar storage
    ticks = []
    bars = []
    start = None
    position = 0

    def on_tick(inst, t, bid, ask):
        nonlocal ticks, bars, start, position
        # collect mid-price ticks
        dt = pd.to_datetime(t)
        price = 0.5 * (bid + ask)
        if start is None:
            start = dt
        ticks.append(price)
        # build bar every 3 seconds
        if (dt - start).total_seconds() >= 3:
            o = ticks[0]
            h = max(ticks)
            l = min(ticks)
            c = ticks[-1]
            bars.append({'time': start, 'o': o, 'h': h, 'l': l, 'c': c})
            # show progress: number of bars collected
            print(f"\rBars collected: {len(bars)}", end='', flush=True)
            ticks = []
            start = dt
            df = pd.DataFrame(bars).set_index('time')
            if len(df) >= bb_w:
                # compute indicators
                df['mb'] = df['c'].rolling(bb_w).mean()
                df['std'] = df['c'].rolling(bb_w).std()
                df['ub'] = df['mb'] + bb_sd * df['std']
                df['lb'] = df['mb'] - bb_sd * df['std']
                delta = df['c'].diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                roll_up = up.ewm(com=rsi_p-1, adjust=False).mean()
                roll_down = down.ewm(com=rsi_p-1, adjust=False).mean()
                rs = roll_up / roll_down
                df['rsi'] = 100 - 100 / (1 + rs)
                df['ema'] = df['c'].ewm(span=ema_p, adjust=False).mean()
                tr1 = df['h'] - df['l']
                tr2 = (df['h'] - df['c'].shift()).abs()
                tr3 = (df['l'] - df['c'].shift()).abs()
                df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(atr_p).mean()
                last = df.iloc[-1]
                price = last['c']
                # entry logic: simple one position at a time
                if position == 0:
                    if price < last['lb'] and last['rsi'] < 30:
                        api.create_order(args.instrument, 1)
                        position = 1
                        print(f"{last.name} LONG @ {price:.2f}")
                    elif price > last['ub'] and last['rsi'] > 70:
                        api.create_order(args.instrument, -1)
                        position = -1
                        print(f"{last.name} SHORT @ {price:.2f}")
                # exit logic: simple mean reversion to middle band
                elif position == 1 and price >= last['mb']:
                    api.create_order(args.instrument, -1)
                    position = 0
                    print(f"{last.name} EXIT LONG @ {price:.2f}")
                elif position == -1 and price <= last['mb']:
                    api.create_order(args.instrument, 1)
                    position = 0
                    print(f"{last.name} EXIT SHORT @ {price:.2f}")

    # start streaming ticks
    print("Collecting 3-second bars...", end='\n')
    api.stream_data(args.instrument, callback=on_tick)


if __name__ == '__main__':
    main()
