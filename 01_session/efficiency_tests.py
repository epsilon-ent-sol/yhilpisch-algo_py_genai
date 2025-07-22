import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import norm

# ----------------------
# 1. Load data and compute returns
# ----------------------
# Load CSV with 'Date' and stock price columns (e.g., 'AAPL')
df = pd.read_csv('../data/eod_data.csv')  # adjust path if needed
symbol = 'NFLX'
df[f'{symbol}_returns'] = np.log(df[symbol] / df[symbol].shift(1))
returns = df[f'{symbol}_returns'].dropna()

# ----------------------
# 2. Ljung-Box Test (lag=10)
# ----------------------
ljung_box_result = acorr_ljungbox(returns, lags=[10], return_df=True)
lb_stat = ljung_box_result['lb_stat'].values[0]
lb_pval = ljung_box_result['lb_pvalue'].values[0]

# ----------------------
# 3. Variance Ratio Test (Lo–MacKinlay, lag=2)
# ----------------------
def variance_ratio_test(x, lag=2):
    x = x - np.mean(x)
    n = len(x)
    mu = np.mean(x)
    b = np.sum((x - mu)**2) / (n - 1)
    t = np.sum((x[lag:] - x[:-lag])**2) / ((n - lag) * lag)
    vr = t / b
    z = (vr - 1) / np.sqrt((2 * (2 * lag - 1) * (lag - 1)) / (3 * lag * n))
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return vr, z, p_value

vr_stat, vr_z, vr_pval = variance_ratio_test(returns.values, lag=2)

# ----------------------
# 4. Runs Test (sign randomness)
# ----------------------
def runs_test(x):
    median = np.median(x)
    runs, n1, n2 = 0, 0, 0
    signs = []

    for i in x:
        if i >= median:
            signs.append(1)
            n1 += 1
        else:
            signs.append(0)
            n2 += 1

    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1

    runs += 1
    expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) /
                       (((n1 + n2)**2) * (n1 + n2 - 1)))
    z = (runs - expected_runs) / std_runs
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return runs, expected_runs, z, p_value

runs_stat, runs_exp, runs_z, runs_pval = runs_test(returns.values)

# ----------------------
# 5. Print Results
# ----------------------
print(f"{'Test':<30}{'Test Statistic':>20}{'p-value':>15}")
print("-" * 65)
print(f"{'Ljung–Box (lag=10)':<30}{lb_stat:>20.4f}{lb_pval:>15.4e}")
print(f"{'Variance Ratio (lag=2)':<30}{vr_stat:>20.4f}{vr_pval:>15.4e}")
print(f"{'Runs Test':<30}{runs_z:>20.4f}{runs_pval:>15.4e}")
