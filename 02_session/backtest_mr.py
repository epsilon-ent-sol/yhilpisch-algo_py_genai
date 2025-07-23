#!/usr/bin/env python3
"""
Backtest intraday mean-reversion strategy on DE30_EUR using tpqoa.
Strategy parameters adapted from mr_technical_strategy.md:
Bollinger Bands (20,2), RSI(14), EMA(20), ATR(14);
entries on mid price extremes, exits on mean reversion or timeout;
risk 1% equity, stop-loss 1.5*ATR.
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from tpqoa import tpqoa


def compute_indicators(df):
    # Bollinger Bands
    df['mb'] = df['c'].rolling(20).mean()
    df['std'] = df['c'].rolling(20).std()
    df['ub'] = df['mb'] + 2 * df['std']
    df['lb'] = df['mb'] - 2 * df['std']
    # RSI
    delta = df['c'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(com=14 - 1, adjust=False).mean()
    roll_down = down.ewm(com=14 - 1, adjust=False).mean()
    rs = roll_up / roll_down
    df['rsi'] = 100 - 100 / (1 + rs)
    # EMA
    df['ema'] = df['c'].ewm(span=20, adjust=False).mean()
    # ATR
    tr1 = df['h'] - df['l']
    tr2 = (df['h'] - df['c'].shift()).abs()
    tr3 = (df['l'] - df['c'].shift()).abs()
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    # Range for momentum filter
    df['range'] = df['h'] - df['l']
    df['avg_range20'] = df['range'].rolling(20).mean()
    return df.dropna()


def run_backtest(df, initial_equity=100000.0):
    equity = initial_equity
    trades = []
    i = 0
    n = len(df)
    max_concurrent = 3
    open_trades = 0
    while i < n:
        row = df.iloc[i]
        time = df.index[i]
        # Entry conditions
        long_entry = (
            row['c'] < row['lb'] and row['rsi'] < 30
            and row['c'] >= row['ema'] - 2 * row['atr']
            and not df['range'].iloc[max(i-3, 0):i].gt(2 * df['avg_range20'].iloc[max(i-3, 0):i]).any()
        )
        short_entry = (
            row['c'] > row['ub'] and row['rsi'] > 70
            and row['c'] <= row['ema'] + 2 * row['atr']
            and not df['range'].iloc[max(i-3, 0):i].gt(2 * df['avg_range20'].iloc[max(i-3, 0):i]).any()
        )
        if long_entry and open_trades < max_concurrent:
            # open long
            entry_price = row['c']
            stop_price = entry_price - 1.5 * row['atr']
            take_price = row['mb']
            risk_amount = equity * 0.01
            stop_dist = entry_price - stop_price
            units = int(risk_amount / stop_dist) if stop_dist > 0 else 0
            max_exit = time + timedelta(hours=1)
            # scan for exit
            for j in range(i+1, n):
                r = df.iloc[j]
                t = df.index[j]
                exit_price = None
                if r['c'] >= take_price or (45 <= r['rsi'] <= 55) or t >= max_exit or r['c'] <= stop_price:
                    exit_price = r['c']
                if exit_price is not None:
                    pnl = (exit_price - entry_price) * units
                    equity += pnl
                    trades.append({
                        'entry_time': time, 'exit_time': t,
                        'direction': 'long', 'entry': entry_price,
                        'exit': exit_price, 'units': units,
                        'pnl': pnl, 'equity': equity
                    })
                    open_trades = max(open_trades - 1, 0)
                    i = j
                    break
        elif short_entry and open_trades < max_concurrent:
            # open short
            entry_price = row['c']
            stop_price = entry_price + 1.5 * row['atr']
            take_price = row['mb']
            risk_amount = equity * 0.01
            stop_dist = stop_price - entry_price
            units = int(risk_amount / stop_dist) if stop_dist > 0 else 0
            max_exit = time + timedelta(hours=1)
            for j in range(i+1, n):
                r = df.iloc[j]
                t = df.index[j]
                exit_price = None
                if r['c'] <= take_price or (45 <= r['rsi'] <= 55) or t >= max_exit or r['c'] >= stop_price:
                    exit_price = r['c']
                if exit_price is not None:
                    pnl = (entry_price - exit_price) * units
                    equity += pnl
                    trades.append({
                        'entry_time': time, 'exit_time': t,
                        'direction': 'short', 'entry': entry_price,
                        'exit': exit_price, 'units': units,
                        'pnl': pnl, 'equity': equity
                    })
                    open_trades = max(open_trades - 1, 0)
                    i = j
                    break
        i += 1
    return trades


def summarize(trades, initial_equity):
    df = pd.DataFrame(trades)
    total_trades = len(df)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    net_pnl = df['pnl'].sum()
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if losses['pnl'].sum() != 0 else np.nan
    returns = df['pnl'] / initial_equity
    sharpe = returns.mean() / returns.std() * np.sqrt(len(returns)) if len(returns) > 1 else np.nan
    eq_curve = df['equity'].tolist()
    peak = initial_equity
    drawdowns = []
    for e in eq_curve:
        peak = max(peak, e)
        drawdowns.append((e - peak) / peak)
    max_dd = min(drawdowns) if drawdowns else 0

    stats = pd.DataFrame({
        'Total Trades': [total_trades],
        'Win Rate (%)': [win_rate],
        'Avg Win': [avg_win],
        'Avg Loss': [avg_loss],
        'Profit Factor': [profit_factor],
        'Sharpe Ratio': [sharpe],
        'Max Drawdown (%)': [max_dd * 100],
        'Net PnL': [net_pnl]
    })
    print('\nBacktest Results:')
    print(stats.to_string(index=False, float_format='%.2f'))


def main():
    oanda = tpqoa('oanda.cfg')
    end = datetime.utcnow()
    start = end - relativedelta(months=1)
    df = oanda.get_history('DE30_EUR', start, end, 'M10', 'M')
    if df.empty:
        print('No data retrieved.')
        sys.exit(1)
    data = compute_indicators(df)
    initial_equity = 100000.0
    trades = run_backtest(data, initial_equity)
    if not trades:
        print('No trades executed.')
        sys.exit(0)
    summarize(trades, initial_equity)


if __name__ == '__main__':
    main()
