#!/usr/bin/env python3
"""Brute-force optimization for MeanRevBacktester strategy parameters."""

import argparse
import itertools
import pandas as pd
import time

from backtest_mr_para import MeanRevBacktester


def main():
    parser = argparse.ArgumentParser(
        description='Optimize mean-reversion strategy parameters'
    )
    parser.add_argument('--config',      default='oanda.cfg')
    parser.add_argument('--instrument',  default='DE30_EUR')
    parser.add_argument('--granularity', default='M10')
    parser.add_argument('--price',       default='M')
    parser.add_argument('--equity',      type=float, default=100000.0)
    parser.add_argument('--risk-pct',    type=float, default=0.01)
    parser.add_argument('--leverage',    type=float, default=10.0)
    parser.add_argument('--start',       default=None)
    parser.add_argument('--end',         default=None)
    parser.add_argument('--top',         type=int,   default=10,
                        help='number of top combinations to display')
    args = parser.parse_args()

    base_bt = MeanRevBacktester(
        config=args.config,
        instrument=args.instrument,
        granularity=args.granularity,
        price=args.price,
        equity=args.equity,
        risk_pct=args.risk_pct,
        leverage=args.leverage,
        start=args.start,
        end=args.end,
    )
    raw_df = base_bt.fetch_data()
    # echo exact data window so backtests can be repeated on identical history
    print(f"Data window: start={base_bt.start.isoformat()} end={base_bt.end.isoformat()}"
          f" ({len(raw_df)} bars)")
    print("To reproduce exactly, rerun backtest_mr_para.py with these --start/--end values and your chosen strategy flags.")

    grid = {
        'bb_window':   [10, 20, 30],
        'bb_std_dev':  [1.5, 2.0, 2.5],
        'rsi_period':  [10, 14, 20],
        'ema_period':  [10, 20],
        'atr_period':  [10, 14],
    }
    keys, values = zip(*grid.items())
    combos = list(itertools.product(*values))
    total = len(combos)
    results = []
    start_time = time.time()
    for idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        print(f"Testing combo {idx}/{total}: {params}", flush=True)
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
            bb_window=params['bb_window'],
            bb_std_dev=params['bb_std_dev'],
            rsi_period=params['rsi_period'],
            ema_period=params['ema_period'],
            atr_period=params['atr_period'],
        )
        df = bt.compute_indicators(raw_df.copy())
        trades = bt.run_backtest(df)
        pnl = sum(t['pnl'] for t in trades)
        n = len(trades)
        wins = [t for t in trades if t['pnl'] > 0]
        win_rate = len(wins) / n * 100 if n else 0.0
        results.append({**params, 'net_pnl': pnl,
                        'n_trades': n, 'win_rate': win_rate})

    elapsed = time.time() - start_time
    df_res = pd.DataFrame(results)
    df_top = df_res.sort_values('net_pnl', ascending=False).head(args.top)
    print(f"Optimization completed in {elapsed:.1f}s")
    print(f'Top {args.top} parameter combinations:')
    print(df_top.to_string(index=False, float_format='%.2f'))


if __name__ == '__main__':
    main()
