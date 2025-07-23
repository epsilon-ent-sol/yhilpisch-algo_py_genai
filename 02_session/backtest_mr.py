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
from dateutil.parser import parse
import argparse
from dateutil.parser import parse
import argparse


class MeanRevBacktester:
    """Intraday mean-reversion backtester for DE30_EUR."""

    def __init__(self,
                 config='oanda.cfg', instrument='DE30_EUR',
                 granularity='M10', price='M', equity=100000.0,
                 risk_pct=0.01, leverage=10.0, start=None, end=None):
        self.oanda = tpqoa(config)
        self.instrument = instrument
        self.granularity = granularity
        self.price = price
        self.initial_equity = equity
        self.risk_pct = risk_pct
        self.leverage = leverage
        self.plot = False
        # parse or default end/start (default: end=now UTC, start=3mo back)
        if end is None:
            self.end = datetime.now(timezone.utc)
        else:
            dt = parse(end)
            self.end = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        if start is None:
            self.start = self.end - relativedelta(months=3)
        else:
            dt = parse(start)
            self.start = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

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
        open_trades = 0

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
            if long_entry and open_trades < max_concurrent:
                entry = row['c']
                stop = entry - 1.5 * row['atr']
                target = row['mb']
                # position sizing with leverage
                risk = equity * self.risk_pct
                units = int((risk * self.leverage) / (entry - stop)) if entry > stop else 0
                cutoff = time + timedelta(hours=1)
                open_trades += 1
                for j in range(i + 1, n):
                    r = df.iloc[j]; t = df.index[j]
                    if r['c'] <= stop or r['c'] >= target or 45 <= r['rsi'] <= 55 or t >= cutoff:
                        pnl = (r['c'] - entry) * units
                        equity += pnl
                        trades.append({
                            'entry_time': time, 'exit_time': t,
                            'direction': 'long', 'entry': entry,
                            'exit': r['c'], 'stop': stop, 'target': target,
                            'units': units, 'pnl': pnl, 'equity': equity
                        })
                        i = j
                        open_trades -= 1
                        break
            # Short
            elif short_entry and open_trades < max_concurrent:
                entry = row['c']
                stop = entry + 1.5 * row['atr']
                target = row['mb']
                # position sizing with leverage
                risk = equity * self.risk_pct
                units = int((risk * self.leverage) / (stop - entry)) if stop > entry else 0
                cutoff = time + timedelta(hours=1)
                open_trades += 1
                for j in range(i + 1, n):
                    r = df.iloc[j]; t = df.index[j]
                    if r['c'] >= stop or r['c'] <= target or 45 <= r['rsi'] <= 55 or t >= cutoff:
                        pnl = (entry - r['c']) * units
                        equity += pnl
                        trades.append({
                            'entry_time': time, 'exit_time': t,
                            'direction': 'short', 'entry': entry,
                            'exit': r['c'], 'stop': stop, 'target': target,
                            'units': units, 'pnl': pnl, 'equity': equity
                        })
                        i = j
                        open_trades -= 1
                        break
            i += 1
        return trades

    def summarize(self, df, trades):
        tr_df = pd.DataFrame(trades)
        wins = tr_df[tr_df['pnl'] > 0]
        losses = tr_df[tr_df['pnl'] <= 0]
        net = tr_df['pnl'].sum()
        total = len(tr_df)
        stats = {
            'Total Trades': total,
            'Win Rate (%)': len(wins) / total * 100 if total else 0,
            'Avg Win': wins['pnl'].mean() if not wins.empty else 0,
            'Avg Loss': losses['pnl'].mean() if not losses.empty else 0,
            'Profit Factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if losses['pnl'].sum() else np.nan,
            'Sharpe Ratio': (tr_df['pnl'] / self.initial_equity).mean() / (tr_df['pnl'] / self.initial_equity).std() * np.sqrt(len(tr_df)) if len(tr_df) > 1 else np.nan,
            'Max Drawdown (%)': (tr_df['equity'] - tr_df['equity'].cummax()).min() / tr_df['equity'].cummax().max() * 100,
            'Net PnL': net
        }
        out = pd.DataFrame([stats])
        print('\nBacktest Results:')
        print(out.to_string(index=False, float_format='%.2f'))
        # plot if requested
        if self.plot:
            self.plot_trades(df, trades)

    def plot_trades(self, df, trades):
        """Plot price series with trade markers, stops, and targets."""
        import plotly.graph_objects as go

        from plotly.subplots import make_subplots

        # create two-row subplot: price (with candlesticks) and equity curve
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02,
                            row_heights=[0.7, 0.3])
        # plot OHLC candlesticks in top row
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['o'], high=df['h'], low=df['l'], close=df['c'],
            name='OHLC'
        ), row=1, col=1)
        shown = {
            'entry_long': False, 'entry_short': False,
            'exit': False, 'stop': False, 'target': False
        }
        for tr in trades:
            et, xt = tr['entry_time'], tr['exit_time']
            eprice, xprice = tr['entry'], tr['exit']
            stop, target = tr['stop'], tr['target']
            direction = tr['direction']
            # entry marker
            entry_key = 'entry_long' if direction == 'long' else 'entry_short'
            entry_name = 'Entry (Long)' if direction == 'long' else 'Entry (Short)'
            fig.add_trace(go.Scatter(
                x=[et], y=[eprice], mode='markers',
                marker_symbol='triangle-up' if direction == 'long' else 'triangle-down',
                marker_color='green' if direction == 'long' else 'red',
                marker_size=12, name=entry_name,
                showlegend=not shown[entry_key]
            ))
            shown[entry_key] = True
            fig.add_trace(go.Scatter(
                x=[xt], y=[xprice], mode='markers',
                marker_symbol='x', marker_color='black', marker_size=12,
                name='Exit', showlegend=not shown['exit']
            ))
            shown['exit'] = True
            fig.add_trace(go.Scatter(
                x=[et, xt], y=[stop, stop], mode='lines',
                line=dict(color='red', dash='dash'),
                name='Stop', showlegend=not shown['stop']
            ))
            shown['stop'] = True
            fig.add_trace(go.Scatter(
                x=[et, xt], y=[target, target], mode='lines',
                line=dict(color='green', dash='dash'),
                name='Target', showlegend=not shown['target']
            ))
            shown['target'] = True
        # add equity curve to bottom row, anchored at start and end
        idx0 = df.index[0]
        idxN = df.index[-1]
        eq_times = [idx0] + [tr['exit_time'] for tr in trades] + [idxN]
        last_eq  = trades[-1]['equity']
        eq_vals  = [self.initial_equity] + [tr['equity'] for tr in trades] + [last_eq]
        fig.add_trace(go.Scatter(
            x=eq_times, y=eq_vals, mode='lines',
            line=dict(color='blue', shape='hv'), name='Equity'
        ), row=2, col=1)
        # layout updates
        fig.update_layout(
            title=f'Trades and Equity Curve for {self.instrument}',
            xaxis=dict(rangeslider=dict(visible=False)),
            hovermode='x unified'
        )
        # autorange y-axes on zoom
        fig.update_yaxes(autorange=True, row=1, col=1)
        fig.update_yaxes(autorange=True, row=2, col=1)
        fig.show()

    def run(self):
        df = self.fetch_data()
        data = self.compute_indicators(df)
        trades = self.run_backtest(data)
        if not trades:
            print('No trades executed.')
            sys.exit(0)
        self.summarize(data, trades)
        if self.plot:
            self.plot_trades(data, trades)


def parse_args():
    p = argparse.ArgumentParser(description='Mean-reversion backtest')
    p.add_argument('--config',      default='oanda.cfg')
    p.add_argument('--instrument',  default='DE30_EUR')
    p.add_argument('--granularity', default='M10')
    p.add_argument('--price',       default='M')
    p.add_argument('--equity',      default=100000.0, type=float)
    p.add_argument('--risk-pct',    default=0.01,    type=float,
                   help='fraction of equity to risk per trade')
    p.add_argument('--leverage',    default=10.0,    type=float,
                   help='leverage multiplier for position sizing')
    p.add_argument('--start',       default=None,
                   help='start datetime (ISO8601)')
    p.add_argument('--end',         default=None,
                   help='end datetime (ISO8601)')
    p.add_argument('--plot',        action='store_true',
                   help='show trade plot')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    bt = MeanRevBacktester(
        config=args.config,
        instrument=args.instrument,
        granularity=args.granularity,
        price=args.price,
        equity=args.equity,
        risk_pct=args.risk_pct,
        leverage=args.leverage,
        start=args.start,
        end=args.end
    )
    bt.plot = args.plot
    bt.plot = args.plot
    bt.run()
