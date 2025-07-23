#!/usr/bin/env python3
"""
Backtest intraday mean-reversion strategy on DE30_EUR using tpqoa.
Refactored as a class; uses timezone-aware datetime.now(timezone.utc).
"""

import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from tpqoa import tpqoa


class MeanRevBacktester:
    """Intraday mean-reversion backtester for DE30_EUR."""

    def __init__(self,
                 config='oanda.cfg', instrument='DE30_EUR',
                 granularity='M10', price='M', equity=100000.0):
        self.oanda = tpqoa(config)
        self.instrument = instrument
        self.granularity = granularity
        self.price = price
        self.initial_equity = equity
        self.end = datetime.now(timezone.utc)
        self.start = self.end - relativedelta(months=1)

    def fetch_data(self):
        df = self.oanda.get_history(
            self.instrument, self.start, self.end,
            self.granularity, self.price
        )
        if df.empty:
            print('No data retrieved.')
            sys.exit(1)
        return df

    def compute_indicators(self, df):
        # Bollinger Bands
        df['mb'] = df['c'].rolling(20).mean()
        df['std'] = df['c'].rolling(20).std()
        df['ub'] = df['mb'] + 2 * df['std']
        df['lb'] = df['mb'] - 2 * df['std']
        # RSI(14)
        delta = df['c'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(com=14 - 1, adjust=False).mean()
        roll_down = down.ewm(com=14 - 1, adjust=False).mean()
        rs = roll_up / roll_down
        df['rsi'] = 100 - 100 / (1 + rs)
        # EMA(20)
        df['ema'] = df['c'].ewm(span=20, adjust=False).mean()
        # ATR(14)
        tr1 = df['h'] - df['l']
        tr2 = (df['h'] - df['c'].shift()).abs()
        tr3 = (df['l'] - df['c'].shift()).abs()
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        # Range & avg range(20)
        df['range'] = df['h'] - df['l']
        df['avg_range20'] = df['range'].rolling(20).mean()
        return df.dropna()

    def run_backtest(self, df):
        equity = self.initial_equity
        trades = []
        i, n = 0, len(df)
        max_concurrent = 3

        while i < n:
            row = df.iloc[i]
            time = df.index[i]
            # momentum filter: skip if last 3 bars >2×avg_range20
            recent = df['range'].iloc[max(i - 3, 0):i]
            if recent.gt(2 * df['avg_range20'].iloc[max(i - 3, 0):i]).any():
                i += 1
                continue
            long_entry = (
                row['c'] < row['lb'] and row['rsi'] < 30
                and row['c'] >= row['ema'] - 2 * row['atr']
            )
            short_entry = (
                row['c'] > row['ub'] and row['rsi'] > 70
                and row['c'] <= row['ema'] + 2 * row['atr']
            )
            # Long
            if long_entry and len(trades) < max_concurrent:
                entry = row['c']
                stop = entry - 1.5 * row['atr']
                target = row['mb']
                risk = equity * 0.01
                units = int(risk / (entry - stop)) if entry > stop else 0
                cutoff = time + timedelta(hours=1)
                for j in range(i + 1, n):
                    r = df.iloc[j]; t = df.index[j]
                    if r['c'] <= stop or r['c'] >= target or 45 <= r['rsi'] <= 55 or t >= cutoff:
                        pnl = (r['c'] - entry) * units
                        equity += pnl
                        trades.append({
                            'entry_time': time, 'exit_time': t,
                            'direction': 'long', 'entry': entry,
                            'exit': r['c'], 'units': units,
                            'pnl': pnl, 'equity': equity
                        })
                        i = j
                        break
            # Short
            elif short_entry and len(trades) < max_concurrent:
                entry = row['c']
                stop = entry + 1.5 * row['atr']
                target = row['mb']
                risk = equity * 0.01
                units = int(risk / (stop - entry)) if stop > entry else 0
                cutoff = time + timedelta(hours=1)
                for j in range(i + 1, n):
                    r = df.iloc[j]; t = df.index[j]
                    if r['c'] >= stop or r['c'] <= target or 45 <= r['rsi'] <= 55 or t >= cutoff:
                        pnl = (entry - r['c']) * units
                        equity += pnl
                        trades.append({
                            'entry_time': time, 'exit_time': t,
                            'direction': 'short', 'entry': entry,
                            'exit': r['c'], 'units': units,
                            'pnl': pnl, 'equity': equity
                        })
                        i = j
                        break
            i += 1
        return trades

    def summarize(self, trades):
        df = pd.DataFrame(trades)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        net = df['pnl'].sum()
        total = len(df)
        stats = {
            'Total Trades': total,
            'Win Rate (%)': len(wins) / total * 100 if total else 0,
            'Avg Win': wins['pnl'].mean() if not wins.empty else 0,
            'Avg Loss': losses['pnl'].mean() if not losses.empty else 0,
            'Profit Factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if losses['pnl'].sum() else np.nan,
            'Sharpe Ratio': (df['pnl'] / self.initial_equity).mean() / (df['pnl'] / self.initial_equity).std() * np.sqrt(len(df)) if len(df) > 1 else np.nan,
            'Max Drawdown (%)': (df['equity'] - df['equity'].cummax()).min() / df['equity'].cummax().max() * 100,
            'Net PnL': net
        }
        out = pd.DataFrame([stats])
        print('\nBacktest Results:')
        print(out.to_string(index=False, float_format='%.2f'))

    def run(self):
        df = self.fetch_data()
        data = self.compute_indicators(df)
        trades = self.run_backtest(data)
        if not trades:
            print('No trades executed.')
            sys.exit(0)
        self.summarize(trades)


if __name__ == '__main__':
    MeanRevBacktester().run()
