import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


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


def calculate_pdf(data, color='b'):
    plt.hist(data, bins=40, density=True, alpha=0.6, color=color)
    plt.xlabel('Stock Price')
    plt.ylabel('Density')


# Create a device
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the LSTM Generator
in_dim = 200
n_layers = 2
hid_dim = 50
generator = WasGenerator(in_dim=in_dim, out_dim=1, n_layers=n_layers, hidden_dim=hid_dim).to(device)

# Load the Generator dictionary
generator.load_state_dict(torch.load("BEST W-GAN-GP Gen BHP", map_location=device))
# Set the generator in evaluation mode
generator.eval()

# Generate numpy array of data used for the analysis
# Can be replaced by true data or any Generator
delta_t = 0.01
volatility = 0.02
daily_rf = 0.0002 * delta_t
rf = 0.0002
S = 100
T = 1000

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

generator_data = np.sort(empirical_distribution)
batch_length = int(T / delta_t)
number_of_batches = 10000

# Create risk neutral paths using GAN
gan_price_paths = np.zeros((number_of_batches, batch_length))
gan_price_paths[:, 0] = S

for j in range(1, batch_length):
    random_samples = np.random.choice(generator_data, size=number_of_batches)
    gan_price_paths[:, j] = gan_price_paths[:, j - 1] * np.exp(
        (rf - 0.5 * (volatility ** 2)) * delta_t + volatility * np.sqrt(delta_t) * random_samples
    )

gan_price_paths = np.mean(gan_price_paths, axis=0)

# Create risk neutral paths using the standard normal distribution
rand_nums = np.random.normal(0, 1, size=(number_of_batches, batch_length-1))
price_paths = np.zeros((number_of_batches, batch_length))
price_paths[:, 0] = S

for j in range(1, batch_length):
    price_paths[:, j] = price_paths[:, j - 1] * np.exp(
        (rf - 0.5 * (volatility ** 2)) * delta_t + volatility * np.sqrt(delta_t) * rand_nums[:, j - 1]
    )

price_paths = np.mean(price_paths, axis=0)

print(gan_price_paths)
print(price_paths)

# Calculate returns
gan_price_paths = np.log(gan_price_paths[1:] / gan_price_paths[:-1])
price_paths = np.log(price_paths[1:] / price_paths[:-1])

# Plot Distributions
plt.figure()
calculate_pdf(gan_price_paths, 'black')
calculate_pdf(price_paths, 'pink')
plt.legend(['GAN Returns', 'Standard Normal Returns'])
plt.title('Sanity check for delta t ' + str(delta_t))
plt.show()

print('Mean of GAN: ' + str(np.mean(gan_price_paths)))
print('Mean of Normal ' + str(np.mean(price_paths)))
print('STD of GAN ' + str(np.std(gan_price_paths)))
print('STD of Normal ' + str(np.std(price_paths)))
print('Deviation from expected value GAN ' + str(np.abs((np.mean(gan_price_paths) - daily_rf)/ daily_rf)))
print('Deviation from expected value Normal ' + str(np.abs((np.mean(price_paths) - daily_rf)/ daily_rf)))
