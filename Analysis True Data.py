import sys
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import powerlaw
from scipy.stats import pearsonr
import statsmodels.api as sm
import scipy.stats

np.seterr(divide='ignore', invalid='ignore')

# Fetch true data BHP
index_data = yf.download('BHP', start='1980-01-01', end='2023-01-01')['Close'].values
log_returns = np.log(index_data[1:] / index_data[:-1])

# Create a subplot with 2 rows and 4 columns
fig, axes = plt.subplots(2, 4, figsize=(18, 10))  # Adjust the figure size as needed

# Time-series
axes[0, 0].plot(log_returns)
axes[0, 0].set_xlabel('Time step t')
axes[0, 0].set_ylabel('Log return')
axes[0, 0].set_title('Time-series')

# Autocorrelation
# Calculate the autocorrelation
autocorrelation = np.correlate(log_returns, log_returns, mode='full') / (np.var(log_returns) * len(log_returns))

# Plot the autocorrelation on a logarithmic x-axis
lags = np.arange(-len(log_returns) + 1, len(log_returns))

axes[0, 1].plot(lags, autocorrelation, marker='o', markersize=2, linestyle='none', color='b')
axes[0, 1].set_xscale('log')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('Autocorrelation')
axes[0, 1].set_title('Autocorrelation Plot')
axes[0, 1].grid()
axes[0, 1].set_ylim(-1, 1)

# Test for auto-correlation
corr = sm.stats.acorr_ljungbox(log_returns, lags=[10], return_df=True, boxpierce=True)
print('Auto-correlation: ', corr)

# Heavy-tails
fit = powerlaw.Fit(log_returns)

# Compute the probability density for each return value using the power-law distribution
alpha = fit.power_law.alpha

# Sort the synthetic data
synthetic_data = np.linspace(2, 100, 1000)
# Calculate the complementary cumulative distribution function (CCDF)
prob_dens = (alpha - 1) * (synthetic_data ** (-alpha))

axes[0, 2].plot(synthetic_data, prob_dens, label=f'α = {alpha}', marker='o', markersize=2, linestyle='none', color='b')
axes[0, 2].set_xlabel('Normalized Log returns')
axes[0, 2].set_ylabel('P(r)')
axes[0, 2].set_title('Power Law Distribution')
axes[0, 2].grid(True)
axes[0, 2].legend()
axes[0, 2].set_yscale('log')

# Test for heavy-tails
abs_returns = np.abs(log_returns)
abs_returns = np.array([value for value in abs_returns if value > 0])

# Generate synthetic data based on a power-law distribution
synthetic_data = np.random.pareto(3, size=10000)

# Fit power-law distributions to the datasets
fit_synthetic = powerlaw.Fit(synthetic_data)
fit_observed = powerlaw.Fit(abs_returns)

# Calculate the PDFs at the observed data points manually
pdf_synthetic = (fit_synthetic.alpha - 1) * (abs_returns / fit_synthetic.xmin) ** (-fit_synthetic.alpha)
pdf_observed = (fit_observed.alpha - 1) * (abs_returns / fit_observed.xmin) ** (-fit_observed.alpha)

# Calculate the likelihood ratio manually
likelihood_ratio_manual = -2 * np.sum(np.log(pdf_observed / pdf_synthetic))
p_value = 1 - scipy.stats.chi2.cdf(likelihood_ratio_manual, 1)

# Print or use the p-value as needed
print("P-value heavy-tails: ", p_value)

# Volatility clustering
abs_returns = np.abs(log_returns)
abs_autocorrelation = np.correlate(abs_returns, abs_returns, mode='full') / (np.var(abs_returns) * len(abs_returns))

# Plot the autocorrelation on a logarithmic x-axis
lags = np.arange(-len(abs_returns) + 1, len(abs_returns))
axes[0, 3].plot(lags, abs_autocorrelation, marker='o', markersize=2, linestyle='none', color='b')
axes[0, 3].set_xscale('log')
axes[0, 3].set_yscale('log')
axes[0, 3].set_xlabel('Lag')
axes[0, 3].set_ylabel('Autocorrelation')
axes[0, 3].set_title('Volatility clustering')
axes[0, 3].grid()

# Test for volatility clustering
corr = sm.stats.acorr_ljungbox(abs_returns, lags=[10], return_df=True, boxpierce=True)
print('Volatility clustering: ', corr)

# Leverage effect
lags = 50
lev_correlation = []

for k in range(1, lags + 1):
    rolled_returns = log_returns[:-k]  # Shift the returns by lag k
    lagged_returns = log_returns[k:]
    num = np.mean((rolled_returns * (np.linalg.norm(lagged_returns) ** 2)) - rolled_returns * (
            np.linalg.norm(rolled_returns) ** 2))
    den = np.mean(np.linalg.norm(rolled_returns) ** 2) ** 2
    lev_correlation.append(num / den)

axes[1, 0].plot(range(0, lags), lev_correlation, color='b')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Lead-lag correlation')
axes[1, 0].set_title('Leverage effect')
axes[1, 0].grid()

# Coarse-fine volatility correlation
tau = 5
max_lag = 4 * tau  # Maximum lag typically considered is 2 * tau


# Calculate the lead-lag correlation function between coarse and fine volatility
def calculate_lead_lag_correlation(r, tau, max_lag):
    lead_lag_correlation = []
    fine_volatility = np.array([np.sum(np.abs(r[i:i + tau])) for i in range(len(r) - tau + 1)])
    for k in range(-max_lag, max_lag + 1):
        coarse_volatility_t_k = np.array(
            [np.abs(np.sum(r[i + k:i + k + tau])) for i in range(len(r) - tau - abs(k) + 1)])
        correlation = np.corrcoef(coarse_volatility_t_k, fine_volatility[:len(coarse_volatility_t_k)])[0, 1]
        lead_lag_correlation.append(correlation)
    return lead_lag_correlation


lead_lag_correlation = calculate_lead_lag_correlation(log_returns, tau, max_lag)

neg_asym = []
for i in range(0, max_lag):
    neg_asym.append(lead_lag_correlation[-i + 2 * max_lag] - lead_lag_correlation[i])

# Create arrays for the x-axis
x1 = np.arange(-max_lag, max_lag + 1)
x2 = np.arange(1, len(neg_asym) + 1)

# Plot the data
axes[1, 1].plot(x1, lead_lag_correlation, marker='o', markersize=2, linestyle='none', color='b',
                label='Correlation function')
axes[1, 1].plot(x2, neg_asym, marker='o', markersize=2, linestyle='none', color='orange', label='Asymmetry')
axes[1, 1].plot(range(-max_lag, max_lag + 1), np.linspace(0, 0, len(lead_lag_correlation)), linestyle='dotted',
                color='black')
axes[1, 1].set_xlabel('Lag k')
axes[1, 1].set_ylabel('Lead-lag correlation')
axes[1, 1].set_title('Coarse-fine volatility correlation')
axes[1, 1].grid()

# Gain/loss asymmetry
target = 0.06
end = len(log_returns)
days_n = np.zeros(end, dtype=int)
days_p = np.zeros(end, dtype=int)
np.set_printoptions(threshold=sys.maxsize)

for d in range(end):
    ret = np.cumsum(log_returns[d:])
    cond_n = ret < -target
    cond_p = ret > target

    days_n[d] = np.min(np.where(cond_n)) if np.any(cond_n) else False
    days_p[d] = np.min(np.where(cond_p)) if np.any(cond_p) else False

days_n_norm = np.bincount(days_n, minlength=end) / np.sum(days_n >= 0)
days_p_norm = np.bincount(days_p, minlength=end) / np.sum(days_p >= 0)

max_y_idx_n = np.argmax(days_n_norm[1:]) + 1
max_y_idx_p = np.argmax(days_p_norm[1:]) + 1

axes[1, 2].plot(days_n_norm, color="red", label="Neg")
axes[1, 2].plot(days_p_norm, color="blue", label="Pos")
axes[1, 2].axvline(x=max_y_idx_n, color="red", linestyle="--", linewidth=1)
axes[1, 2].axvline(x=max_y_idx_p, color="blue", linestyle="--", linewidth=1)
axes[1, 2].set_xscale('log')
axes[1, 2].legend()
axes[1, 2].set_xlim(1, 1000)
axes[1, 2].set_ylim(0, 0.08)
axes[1, 2].set_xlabel('Days')
axes[1, 2].set_ylabel('Density')
axes[1, 2].set_title('Gain/loss asymmetry')
axes[1, 2].grid()

fig.delaxes(axes[1][3])
# Adjust the layout
plt.tight_layout()

# Show the combined figure
plt.show()
