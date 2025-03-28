import math
import os

import numpy as np
import torch
import yfinance as yf
import pandas as pd
from scipy import stats
from Option import Option
import chaospy as ch
from scipy.stats import norm
from scipy.optimize import newton
import timeit
import sobol_seq as ss
import sys
import random
import qmcpy as qp
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from scipy.stats import qmc
from statsmodels.nonparametric.kde import KDEUnivariate


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


def LoadIndexData(stock, begin_date, end_date):
    index_data = yf.download(stock, start=begin_date, end=end_date)['Close'].to_numpy()  # .values.reshape(-1, 1, 1)
    log_returns = np.log(index_data[1:] / index_data[:-1])
    S_0 = index_data[-1]
    mu = log_returns.mean()
    v = log_returns.std()
    return log_returns, len(index_data), mu, v, S_0


def generate_asset_price(S, v, r, dt, N, length, Sim="MC", device=None, generator=None):
    if Sim == "MC":
        # Generate N x l array of random numbers
        rand_nums = np.random.normal(0, 1, size=(N, length - 1))
        # Create a matrix of zeros to hold price paths
        price_paths = np.zeros((N, length))

        # Set initial prices
        price_paths[:, 0] = S

        # Calculate price paths using vectorized operations
        for j in range(1, length):
            price_paths[:, j] = price_paths[:, j - 1] * np.exp(
                (r - 0.5 * (v ** 2)) * dt + v * np.sqrt(dt) * rand_nums[:, j - 1]
            )
        return price_paths
    elif Sim == "QMC":
        h_qrng = qp.Halton(length - 1, randomize='QRNG', seed=random.randint(1, 1000))
        points = stats.norm.ppf(h_qrng.gen_samples(N))
        price_paths = np.zeros((N, length))
        price_paths[:, 0] = S

        return_paths = np.zeros((N, length - 1))
        for i in range(1, length):
            price_paths[:, i] = price_paths[:, i - 1] * np.exp(
                (r - v ** 2 / 2) * dt + v * points[:, i - 1] * np.sqrt(dt))

        return price_paths

    elif Sim == "GAN-MC":
        price_paths = np.zeros((N, length))

        # Set initial prices
        price_paths[:, 0] = S

        # Calculate price paths using vectorized operations
        for j in range(1, length):
            random_samples = np.random.choice(empirical_distribution, size=N)
            price_paths[:, j] = price_paths[:, j - 1] * np.exp(
                (r - 0.5 * (v ** 2)) * dt + v * np.sqrt(dt) * random_samples
            )

        return price_paths

    elif Sim == "GAN-LHS":
        # 使用拉丁超立方体抽样
        sampler = qmc.LatinHypercube(d=length - 1, seed=random.randint(1, 1000))
        samples = sampler.random(N)
        points = np.floor(samples * len(empirical_distribution)).astype(int)
        samples = empirical_distribution[points]

        price_paths = np.zeros((N, length))
        price_paths[:, 0] = S

        # Calculate price paths using vectorized operations
        for j in range(1, length):
            price_paths[:, j] = price_paths[:, j - 1] * np.exp(
                (r - 0.5 * (v ** 2)) * dt + v * np.sqrt(dt) * samples[:, j-1]
            )

        return price_paths
    return "Error"


def convert_risk_free_rate(tr):
    return (1 + tr) ** (1 / 252) - 1  # Or just divide by 252?


def convert_volatility(annualized_volatility):
    daily_volatility = annualized_volatility / math.sqrt(252)
    return daily_volatility


def calculate_implied_volatility(option_sort, market_price, strike, T, r, S, div=0):
    def black_scholes(option_sort, S, K, T, r, sigma, div):
        d1 = (math.log(S / K) + (r - div + (sigma ** 2) / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_sort == 'call':
            option_price = S * math.exp(-div * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        elif option_sort == 'put':
            option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-div * T) * norm.cdf(-d1)

        return option_price

    def calculate_error(sigma_guess):
        return black_scholes(option_sort, S, strike, T, r, sigma_guess, div) - market_price

    # Initial guess for volatility
    initial_volatility_guess = 0.20  # 20%
    implied_volatility = 0.015
    # Try different initial guesses for implied volatility
    for i in range(10000):
        try:
            # Attempt to calculate implied volatility using Newton-Raphson method
            implied_volatility = newton(calculate_error, x0=initial_volatility_guess, tol=0.0001)
            break  # Exit the loop if successful convergence
        except RuntimeError:
            # Catch a runtime error (failed convergence) and try a different initial guess
            initial_volatility_guess += 0.01  # Try a different initial guess (e.g., increase by 5%)
    return implied_volatility


def get_option_data(file_name):
    """"The input data used for the simulations"""
    return pd.read_excel(file_name).to_numpy()


def main():
    # Get the option data
    file_name = "CBA Option prices.xlsx"
    option_contracts = get_option_data(file_name)
    print(option_contracts)

    N = 200
    results = []

    # Load Generator (TEST)
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    in_dim = 98
    n_layers = 1
    hid_dim = 70
    name = "BEST W-GAN-GP Gen CBA"
    generator = WasGenerator(in_dim=in_dim, out_dim=1, n_layers=n_layers, hidden_dim=hid_dim).to(device)
    generator.load_state_dict(torch.load(name, map_location=device))

    # Set the generator in evaluation mode
    generator.eval()

    global empirical_distribution

    skip = 0
    numb_of_datasets = 130
    batch_length = 4000
    number_of_batches = 50
    length_of_sample = (batch_length - skip) * number_of_batches
    empirical_distribution = np.zeros(length_of_sample * numb_of_datasets)
    for i in range(0, numb_of_datasets):
        random_input1 = torch.randn(number_of_batches, batch_length, in_dim, device=device)
        generated_samples = generator(random_input1)
        generated_samples = generated_samples.cpu().detach().numpy().reshape(number_of_batches, -1)[:,skip:].flatten()
        generated_samples = (generated_samples - np.mean(generated_samples)) / np.std(generated_samples)
        empirical_distribution[i*length_of_sample:i*length_of_sample+length_of_sample] = generated_samples

    empirical_distribution = np.sort(empirical_distribution)

    # Loop through each option?
    dt = 1
    for i in range(len(option_contracts)):
        print(i)
        option_type = option_contracts[i, 0]
        option_sort = option_contracts[i, 1]
        strike = option_contracts[i, 2]
        market_price = option_contracts[i, 3]
        S_0 = option_contracts[i, 5]
        B = option_contracts[i, 7]
        T = (option_contracts[i, 8] - 1)
        r = convert_risk_free_rate(option_contracts[i, 9])
        v = option_contracts[i, 10]

        if v == 0:
            v = calculate_implied_volatility(option_sort, market_price, strike, T, r, S_0)
            if v < 0.002:
                continue
        # else:
        #     # Convert from yearly to daily volatility
        #     v = convert_volatility(v)

        T = int(T/dt)
        for sim_type in ["MC", "QMC", "GAN-MC", "GAN-LHS"]:
            start_time = timeit.default_timer()
            price_paths = generate_asset_price(S_0, v, r, dt, N, T, sim_type, device, generator)
            option_instance = Option(option_type, option_sort, price_paths, strike, r, T, market_price, B)
            price_data = option_instance.price()[0]
            elapsed = timeit.default_timer() - start_time
            results.append([sim_type] + price_data + [S_0, v, elapsed])

    output_file_path = "W-GAN CBA LHS-1.txt"

    # Write the transposed data to the text file
    with open(output_file_path, 'w') as file:
        for row in results:
            file.write('\t'.join(str(entry) for entry in row) + '\n')


if __name__ == '__main__':
    main()
