<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Intraday Mean-Reversion Algorithmic Trading Strategy

Below is a concrete, step-by-step algorithmic trading strategy for intraday mean reversion, utilizing several technical indicators and specifying all thresholds and parameters. This approach is designed for liquid equities but can be modified for other instruments.

## Objective

Capitalize on short-term price deviations from a defined mean within the day, betting that prices revert toward an average after becoming briefly overbought or oversold.

## Indicators Used

- **Bollinger Bands (20-period, 2 standard deviations)**
- **Relative Strength Index (RSI, 14-period)**
- **Exponential Moving Average (EMA, 20-period)**
- **Average True Range (ATR, 14-period) – for volatility-based position sizing/stop-loss**


## Exact Strategy Rules

### 1. Data \& Timeframe

- **Timeframe**: 5-minute candles.
- **Instruments**: Only trade stocks with high intraday liquidity (e.g., S\&P 500 stocks).


### 2. Entry Conditions

#### **Long Entry (Buy)**

- The close of the current 5-min candle is **below the lower Bollinger Band** (20-period, 2 SD).
- RSI (14) of current candle is **below 30**.
- The price is **not further than 2×ATR below** the 20-period EMA (prevents entry when far outside the expected range).
- No other open long position in the same instrument.


#### **Short Entry (Sell)**

- The close of the current 5-min candle is **above the upper Bollinger Band** (20-period, 2 SD).
- RSI (14) is **above 70**.
- The price is **not further than 2×ATR above** the 20-period EMA.
- No other open short position in the same instrument.


### 3. Position Sizing

- Risk **1% of account equity per trade**.
- **Position size = (Account Equity × 1%) ÷ (Entry Price – Stop Loss distance)** where Stop Loss distance is dynamically set (see below)[^1].


### 4. Stop-Loss and Exit

- **Stop-Loss**: Set at **1.5×ATR** away from entry price, in the direction of the stop.
- **Exit Take-Profit**: When price:
    - Closes at or crosses **back to the middle Bollinger Band (20-period moving average)** or
    - RSI returns to between 45 and 55, or
    - After a **maximum of 1 hour** after entry (timed exit to enforce intraday discipline).
- Optionally, **partial exit** when price reaches the EMA; move stop to breakeven on the remainder[^2][^1][^3].


### 5. Additional Risk Controls

- **Maximum of 3 concurrent trades** open at any time.
- **No entries during first 5 minutes after market open** or last 15 minutes before close (to avoid opening/closing volatility).
- **No entry if one of the last three candles was a high-momentum bar** (>2× average 5-min range for the last 20 candles).


## Example

| Indicator/Value | Long Entry Signal | Short Entry Signal |
| :-- | :-- | :-- |
| Bollinger Band (20, 2 SD) | Close < Lower Band | Close > Upper Band |
| RSI (14-period) | < 30 | > 70 |
| Distance from 20-EMA | > (but not >2×ATR) | < (but not <−2×ATR) |
| Volatility Filter (ATR) | Used for stops | Used for stops |
| Position Sizing | 1% equity | 1% equity |
| Exit | MA cross or time | MA cross or time |

## Practical Notes

- **Backtest and optimize** the thresholds (e.g., 2 SD, RSI 30/70, ATR multipliers) for your specific universe and timeframe.
- Monitor for slippage and transaction costs; avoiding very illiquid names is essential for intraday execution.
- Avoid trading during major economic news releases to minimize whipsaw risk.


## Strategy Rationale

This strategy combines three perspectives to maximize mean-reversion reliability:

- **Price extremes:** Bollinger Bands detect statistical outliers.
- **Momentum confirmation:** RSI confirms that the move is truly overextended.
- **Adaptive risk:** ATR and EMA ensure entries/exits/position sizes adapt to current volatility.

These details provide a precisely specified, highly testable blueprint for an intraday mean-reversion algorithmic system[^2][^1][^3].

<div style="text-align: center">⁂</div>

[^1]: https://www.luxalgo.com/blog/mean-reversion-strategies-for-algorithmic-trading/

[^2]: https://tradefundrr.com/mean-reversion-trading-techniques/

[^3]: https://macrogmsecurities.com.au/long-only-algorithmic-trading-strategies-for-stocks/

[^4]: https://www.cmcmarkets.com/en-gb/trading-guides/mean-reversion

[^5]: https://de.tradingview.com/scripts/mean-reversion/

[^6]: https://www.samco.in/knowledge-center/articles/mean-reversion-trading-strategies/

[^7]: https://www.tradingview.com/script/EzDT1Dzf-Intraday-Mean-Reversion-Main/

[^8]: https://howtotrade.com/wp-content/uploads/2023/11/Mean-Reversion-Trading.pdf

[^9]: https://quant.stackexchange.com/questions/78719/how-find-optimal-entry-exit-thresholds-for-a-mean-reverting-process

[^10]: https://static1.squarespace.com/static/5fc9e2e184bf712dcfcc8dcf/t/5fca100cec1d586fb3934ac9/1607077902561/Kami+Export+-+Profitable_Algorithmic_Trading_Strategie+(1)+(6).pdf

[^11]: https://www.quantifiedstrategies.com/mean-reversion-trading-strategy/

[^12]: https://www.reddit.com/r/algotrading/comments/1cwsco8/a_mean_reversion_strategy_with_211_sharpe/

[^13]: https://www.mql5.com/en/blogs/post/753277

[^14]: https://www.interactivebrokers.com/campus/ibkr-quant-news/mean-reversion-strategies-introduction-trading-strategies-and-more-part-i/

[^15]: https://www.quantitativo.com/p/trading-the-mean-reversion-curve

[^16]: https://blog.quantinsti.com/mean-reversion-strategies-introduction-building-blocks/

[^17]: https://machinelearning-basics.com/mean-reversion-trading-strategy-using-python/

[^18]: https://www.fmz.com/lang/en/strategy/430247

[^19]: https://www.reddit.com/r/algotrading/comments/13ivyk6/what_are_your_day_trading_mean_reversion/

[^20]: https://in.tradingview.com/scripts/meanreversion/

