import sys
import powerlaw
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats

np.seterr(divide='ignore', invalid='ignore')


class WasGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, hidden_dim):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, in_dim),  # Updated size to match your checkpoint
                                    nn.Linear(in_dim, out_dim), nn.Tanh())  # Updated size to match your checkpoint

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs


class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.

    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, in_dim, out_dim, n_layers, hidden_dim):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Tanh())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs


def calculate_lead_lag_correlation(r, tau, max_lag):
    lead_lag_correlation = []
    fine_volatility = np.array([np.sum(np.abs(r[i:i + tau])) for i in range(len(r) - tau + 1)])
    for k in range(-max_lag, max_lag + 1):
        coarse_volatility_t_k = np.array(
            [np.abs(np.sum(r[i + k:i + k + tau])) for i in range(len(r) - tau - abs(k) + 1)])
        correlation = np.corrcoef(coarse_volatility_t_k, fine_volatility[:len(coarse_volatility_t_k)])[0, 1]
        lead_lag_correlation.append(correlation)
    return lead_lag_correlation


# Create a device
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the LSTM Generator
in_dim = 99
n_layers = 1
hid_dim = 79
generator = LSTMGenerator(in_dim=in_dim, out_dim=1, n_layers=n_layers, hidden_dim=hid_dim).to(device)

# Load the Generator dictionary
generator.load_state_dict(torch.load("BEST LSTM-TCN Gen BHP", map_location=device))
# Set the generator in evaluation mode
generator.eval()

# Generate numpy array of data used for the analysis
# Can be replaced by true data or any Generator
number_of_batches = 5
skip = 0
batch_length = 1000
random_input = torch.randn(number_of_batches, batch_length, in_dim, device=device)
generator_data = generator(random_input).cpu().detach().numpy().reshape(number_of_batches, -1)[:, skip:]

# start plotting
# Create a subplot with 2 rows and 4 columns
fig, axes = plt.subplots(2, 4, figsize=(18, 10))  # Adjust the figure size as needed

# Time-series
average_data = np.mean(generator_data, axis=0)
axes[0, 0].plot(average_data)
axes[0, 0].set_xlabel('Time step t')
axes[0, 0].set_ylabel('Log return')
axes[0, 0].set_title('Time-series')

# Plot all statistics
avg_autocorrelation = np.zeros(shape=(number_of_batches, (batch_length - skip) * 2 - 1))
avg_abs_autocorrelation = np.zeros(shape=(number_of_batches, (batch_length - skip) * 2 - 1))
heavy_tails_alpha = []
lags_heavy_tails = 50
lev_correlation = np.zeros(shape=(number_of_batches, lags_heavy_tails))
tau = 5
max_lag = 4 * tau  # Maximum lag typically considered is 2 * tau
avg_lead_lag_correlation = np.zeros(shape=(number_of_batches, max_lag * 2 + 1))
avg_neg_asym = np.zeros(shape=(number_of_batches, max_lag))
avg_days_n_norm = np.zeros(shape=(number_of_batches, batch_length - skip))
avg_days_p_norm = np.zeros(shape=(number_of_batches, batch_length - skip))

for i in range(0, number_of_batches):
    print('Batch number: ', str(i))
    # Linear unpredictability
    random_input = torch.randn(1, batch_length, in_dim, device=device)
    generator_data = generator(random_input).cpu().detach().numpy().flatten()[skip:]

    # Use standardized data for autocorrelation, volatility clusteirng, and time series
    # Use unstandardized data for power law, leverage effect, coarse fine volatility and gain loss asymmetry
    avg_autocorrelation[i, :] = np.correlate(generator_data, generator_data, mode='full') / (
            np.var(generator_data) * len(generator_data))
    corr = sm.stats.acorr_ljungbox(generator_data, lags=[10], return_df=True, model_df=4)
    print(corr)

    # Heavy tails
    fit = powerlaw.Fit(generator_data)
    heavy_tails_alpha.append(fit.power_law.alpha)

    # Test for heavy-tails
    abs_returns = np.abs(generator_data)
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
    abs_returns = np.abs(generator_data)
    avg_abs_autocorrelation[i, :] = np.correlate(abs_returns, abs_returns, mode='full') / (
            np.var(abs_returns) * len(abs_returns))

    # Test for volatility clustering
    corr = sm.stats.acorr_ljungbox(abs_returns, lags=[10], return_df=True, boxpierce=True)
    print('Volatility clustering: ', corr)

    # Leverage effect
    for k in range(1, lags_heavy_tails + 1):
        rolled_returns = generator_data[:-k]  # Shift the returns by lag k
        lagged_returns = generator_data[k:]
        num = np.mean((rolled_returns * (np.linalg.norm(lagged_returns) ** 2)) - rolled_returns * (
                np.linalg.norm(rolled_returns) ** 2))
        den = np.mean(np.linalg.norm(rolled_returns) ** 2) ** 2
        lev_correlation[i, k - 1] = num / den

    # Course-fine volatility
    avg_lead_lag_correlation[i, :] = calculate_lead_lag_correlation(generator_data, tau, max_lag)
    for k in range(0, max_lag):
        avg_neg_asym[i, k] = avg_lead_lag_correlation[i, -k + 2 * max_lag] - avg_lead_lag_correlation[i, k]

    # Gain/loss asymmetry
    target = 0.1
    end = len(generator_data)
    days_n = np.zeros(end, dtype=int)
    days_p = np.zeros(end, dtype=int)
    np.set_printoptions(threshold=sys.maxsize)

    for d in range(end):
        ret = np.cumsum(generator_data[d:])

        cond_n = ret < -target
        cond_p = ret > target

        unreached_max_n = np.argmin(ret)
        unreached_max_p = np.argmax(ret)

        days_n[d] = np.min(np.where(cond_n)) if np.any(cond_n) else False
        days_p[d] = np.min(np.where(cond_p)) if np.any(cond_p) else False

    avg_days_n_norm[i, :] = np.bincount(days_n, minlength=end) / np.sum(days_n >= 0)
    avg_days_p_norm[i, :] = np.bincount(days_p, minlength=end) / np.sum(days_p >= 0)

# Plot linear unpredictability
batch_length = batch_length - skip
avg_autocorrelation = np.mean(avg_autocorrelation, axis=0)
lags = np.arange(-batch_length + 1, batch_length)
axes[0, 1].plot(lags, avg_autocorrelation, marker='o', markersize=2, linestyle='none', color='b')
axes[0, 1].set_xscale('log')
axes[0, 1].set_ylim(-1, 1)
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('Auto-correlation')
axes[0, 1].set_title('Auto-correlation plot')
axes[0, 1].grid()

# Plot heavy tailed distribution
# Sort the synthetic data
synthetic_data = np.linspace(2, 100, 1000)
# Calculate the complementary cumulative distribution function (CCDF)
alpha = np.mean(heavy_tails_alpha)

prob_dens = (alpha - 1) * (synthetic_data ** (-alpha))

axes[0, 2].plot(synthetic_data, prob_dens, label=f'Power Law (α = {alpha})', marker='o', markersize=2, linestyle='none',
                color='b')
axes[0, 2].set_xlabel('Normalized Log returns')
axes[0, 2].set_ylabel('P(r)')
axes[0, 2].set_title('Power Law Distribution')
axes[0, 2].legend()
axes[0, 2].grid(True)
axes[0, 2].set_yscale('log')

# Plot volatility clustering
lags = np.arange(-(avg_abs_autocorrelation.shape[1] // 2) - 1, avg_abs_autocorrelation.shape[1] // 2)
avg_abs_autocorrelation = np.mean(avg_abs_autocorrelation, axis=0)
axes[0, 3].plot(lags, avg_abs_autocorrelation, marker='o', markersize=2, linestyle='none', color='b')
axes[0, 3].set_xscale('log')
axes[0, 3].set_yscale('log')
axes[0, 3].set_xlabel('Lag')
axes[0, 3].set_ylabel('Auto-correlation')
axes[0, 3].set_title('Volatility clustering')
axes[0, 3].grid()

# Plot leverage effect
lev_correlation = np.mean(lev_correlation, axis=0)
axes[1, 0].plot(range(0, lags_heavy_tails), lev_correlation, color='b')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Lead-lag ccorrelation')
axes[1, 0].set_title('Leverage effect')
axes[1, 0].grid()

# Coarse-fine volatility
# Create arrays for the x-axis
avg_neg_asym = np.mean(avg_neg_asym, axis=0)
avg_lead_lag_correlation = np.mean(avg_lead_lag_correlation, axis=0)

x1 = np.arange(-max_lag, max_lag + 1)
x2 = np.arange(1, len(avg_neg_asym) + 1)

# Plot the data
axes[1, 1].plot(x1, avg_lead_lag_correlation, marker='o', markersize=2, linestyle='none', color='b')
axes[1, 1].plot(x2, avg_neg_asym, marker='o', markersize=2, linestyle='none', color='orange')
axes[1, 1].plot(range(-max_lag, max_lag + 1), np.linspace(0, 0, len(avg_lead_lag_correlation)), linestyle='dotted',
                color='black')
axes[1, 1].plot(range(-max_lag, max_lag + 1), np.linspace(0, 0, len(avg_lead_lag_correlation)), linestyle='dotted',
                color='black')
axes[1, 1].set_xlabel('Lag k')
axes[1, 1].set_ylabel('Lead-lag correlation')
axes[1, 1].set_title('Coarse-fine volatility correlation')
axes[1, 1].grid()

# Gain/loss asymmetry
avg_days_p_norm = np.mean(avg_days_p_norm, axis=0)
avg_days_n_norm = np.mean(avg_days_n_norm, axis=0)

max_y_idx_n = np.argmax(avg_days_n_norm[1:]) + 1
max_y_idx_p = np.argmax(avg_days_p_norm[1:]) + 1

axes[1, 2].plot(avg_days_n_norm, color="red", label="Neg")
axes[1, 2].plot(avg_days_p_norm, color="blue", label="Pos")
axes[1, 2].axvline(x=max_y_idx_n, color="red", linestyle="--", linewidth=1)
axes[1, 2].axvline(x=max_y_idx_p, color="blue", linestyle="--", linewidth=1)
axes[1, 2].set_xscale('log')
axes[1, 2].legend()
axes[1, 2].set_xlim(1, 1000)
axes[1, 2].set_ylim(0, 0.2)
axes[1, 2].set_xlabel('Days')
axes[1, 2].set_ylabel('Density')
axes[1, 2].set_title('Gain/loss asymmetry')
axes[1, 2].grid()

fig.delaxes(axes[1][3])
# Adjust the layout
plt.tight_layout()

# Show the combined figure
plt.show()
