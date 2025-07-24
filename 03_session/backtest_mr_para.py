#!/usr/bin/env python3
"""
Backtest intraday mean-reversion strategy on DE30_EUR using tpqoa.
Refactored as a class; uses timezone-aware datetime.now(timezone.utc).
"""

import sys
from datetime import datetime, timedelta, timezone, time

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
                 config='oanda.cfg',
                 instrument='DE30_EUR',
                 granularity='M10',
                 price='M',
                 equity=100000.0,
                 risk_pct=0.01,
                 leverage=10.0,
                 start=None,
                 end=None,
                 bb_window=20,
                 bb_std_dev=2,
                 rsi_period=14,
                 ema_period=20,
                 atr_period=14,
                 avg_range_window=20,
                 rsi_lower=30,
                 rsi_upper=70,
                 entry_atr_multiplier=2,
                 stop_atr_multiplier=1.5,
                 momentum_window=3,
                 momentum_multiplier=2,
                 cutoff_hours=1,
                 max_concurrent_trades=3):
        self.oanda = tpqoa(config)
        self.instrument = instrument
        self.granularity = granularity
        self.price = price
        self.initial_equity = equity
        self.risk_pct = risk_pct
        self.leverage = leverage
        # strategy parameters
        self.bb_window = bb_window
        self.bb_std_dev = bb_std_dev
        self.rsi_period = rsi_period
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.avg_range_window = avg_range_window
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.entry_atr_multiplier = entry_atr_multiplier
        self.stop_atr_multiplier = stop_atr_multiplier
        self.momentum_window = momentum_window
        self.momentum_multiplier = momentum_multiplier
        self.cutoff_hours = cutoff_hours
        self.max_concurrent_trades = max_concurrent_trades
        self.plot = False
        # parse or default end/start (default: end=now UTC, start=3mo back)
        if end is None:
            # default end at UTC midnight of today's date
            today = datetime.now(timezone.utc).date()
            self.end = datetime.combine(today, time(0, 0), tzinfo=timezone.utc)
        else:
            dt = parse(end)
            self.end = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        if start is None:
            # default start three months before end date at UTC midnight
            start_date = self.end.date() - relativedelta(months=3)
            self.start = datetime.combine(start_date, time(0, 0), tzinfo=timezone.utc)
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
        df['mb'] = df['c'].rolling(self.bb_window).mean()
        df['std'] = df['c'].rolling(self.bb_window).std()
        df['ub'] = df['mb'] + self.bb_std_dev * df['std']
        df['lb'] = df['mb'] - self.bb_std_dev * df['std']
        # RSI
        delta = df['c'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(com=self.rsi_period - 1, adjust=False).mean()
        roll_down = down.ewm(com=self.rsi_period - 1, adjust=False).mean()
        rs = roll_up / roll_down
        df['rsi'] = 100 - 100 / (1 + rs)
        # EMA
        df['ema'] = df['c'].ewm(span=self.ema_period, adjust=False).mean()
        # ATR
        tr1 = df['h'] - df['l']
        tr2 = (df['h'] - df['c'].shift()).abs()
        tr3 = (df['l'] - df['c'].shift()).abs()
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(self.atr_period).mean()
        # Range & average range
        df['range'] = df['h'] - df['l']
        df['avg_range'] = df['range'].rolling(self.avg_range_window).mean()
        return df.dropna()

    def run_backtest(self, df):
        equity = self.initial_equity
        trades = []
        i, n = 0, len(df)
        open_trades = 0

        while i < n:
            row = df.iloc[i]
            time = df.index[i]
            # momentum filter: skip if any of last bars range > momentum_multiplier * avg_range
            start = max(i - self.momentum_window, 0)
            recent = df['range'].iloc[start:i]
            avg_rng = df['avg_range'].iloc[start:i]
            if recent.gt(self.momentum_multiplier * avg_rng).any():
                i += 1
                continue
            # entry conditions
            long_entry = (
                row['c'] < row['lb'] and row['rsi'] < self.rsi_lower
                and row['c'] >= row['ema'] - self.entry_atr_multiplier * row['atr']
            )
            short_entry = (
                row['c'] > row['ub'] and row['rsi'] > self.rsi_upper
                and row['c'] <= row['ema'] + self.entry_atr_multiplier * row['atr']
            )
            # Long trade
            if long_entry and open_trades < self.max_concurrent_trades:
                entry = row['c']
                stop = entry - self.stop_atr_multiplier * row['atr']
                target = row['mb']
                risk = equity * self.risk_pct
                units = int((risk * self.leverage) / (entry - stop)) if entry > stop else 0
                cutoff = time + timedelta(hours=self.cutoff_hours)
                open_trades += 1
                for j in range(i + 1, n):
                    r = df.iloc[j]; t = df.index[j]
                    if (r['c'] <= stop or r['c'] >= target
                        or 45 <= r['rsi'] <= 55 or t >= cutoff):
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
            # Short trade
            elif short_entry and open_trades < self.max_concurrent_trades:
                entry = row['c']
                stop = entry + self.stop_atr_multiplier * row['atr']
                target = row['mb']
                risk = equity * self.risk_pct
                units = int((risk * self.leverage) / (stop - entry)) if stop > entry else 0
                cutoff = time + timedelta(hours=self.cutoff_hours)
                open_trades += 1
                for j in range(i + 1, n):
                    r = df.iloc[j]; t = df.index[j]
                    if (r['c'] >= stop or r['c'] <= target
                        or 45 <= r['rsi'] <= 55 or t >= cutoff):
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
    # strategy parameter arguments
    p.add_argument('--bb-window',           type=int,   default=20,
                   help='window for Bollinger Bands')
    p.add_argument('--bb-std-dev',          type=float, default=2,
                   help='std deviation multiplier for Bollinger Bands')
    p.add_argument('--rsi-period',          type=int,   default=14,
                   help='period for RSI calculation')
    p.add_argument('--ema-period',          type=int,   default=20,
                   help='period for EMA calculation')
    p.add_argument('--atr-period',          type=int,   default=14,
                   help='period for ATR calculation')
    p.add_argument('--avg-range-window',    type=int,   default=20,
                   help='window for average range calculation')
    p.add_argument('--rsi-lower',           type=float, default=30,
                   help='lower threshold for RSI-entry')
    p.add_argument('--rsi-upper',           type=float, default=70,
                   help='upper threshold for RSI-entry')
    p.add_argument('--entry-atr-multiplier',type=float, default=2,
                   help='ATR multiplier for entry condition')
    p.add_argument('--stop-atr-multiplier', type=float, default=1.5,
                   help='ATR multiplier for stop distance')
    p.add_argument('--momentum-window',     type=int,   default=3,
                   help='number of bars for momentum filter')
    p.add_argument('--momentum-multiplier', type=float, default=2,
                   help='multiplier for momentum filter')
    p.add_argument('--cutoff-hours',        type=float, default=1,
                   help='hours until cutoff for trade exit')
    p.add_argument('--max-concurrent-trades', type=int, default=3,
                   help='maximum concurrent open trades')
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
        end=args.end,
        bb_window=args.bb_window,
        bb_std_dev=args.bb_std_dev,
        rsi_period=args.rsi_period,
        ema_period=args.ema_period,
        atr_period=args.atr_period,
        avg_range_window=args.avg_range_window,
        rsi_lower=args.rsi_lower,
        rsi_upper=args.rsi_upper,
        entry_atr_multiplier=args.entry_atr_multiplier,
        stop_atr_multiplier=args.stop_atr_multiplier,
        momentum_window=args.momentum_window,
        momentum_multiplier=args.momentum_multiplier,
        cutoff_hours=args.cutoff_hours,
        max_concurrent_trades=args.max_concurrent_trades
    )
    bt.plot = args.plot
    bt.run()
