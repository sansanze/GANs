import os
import random
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import datetime
import numpy as np
import io
import math
from torch.nn.utils import weight_norm
from scipy.stats import wasserstein_distance


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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, channel_last=True):
        y1 = self.tcn(x.transpose(1, 2) if channel_last else x)
        return self.linear(y1.transpose(1, 2))


class CausalConvDiscriminator(nn.Module):
    """Discriminator using casual dilated convolution, outputs a probability for each time step

    Args:
        input_size (int): dimensionality (channels) of the input
        n_layers (int): number of hidden layers
        n_channels (int): number of channels in the hidden layers (it's always the same)
        kernel_size (int): kernel size in all the layers
        dropout: (float in [0-1]): dropout rate

    Input: (batch_size, seq_len, input_size)
    Output: (batch_size, seq_len, 1)
    """

    def __init__(self, input_size, n_layers, n_channel, kernel_size, dropout):
        super().__init__()
        # Assuming same number of channels layerwise
        num_channels = [n_channel] * n_layers
        self.tcn = TCN(input_size, 1, num_channels, kernel_size, dropout)

    def forward(self, x, channel_last=True):
        return torch.sigmoid(self.tcn(x, channel_last))


def pretrain_discriminator_real(discriminator, optimizer, real_data, criterion, device):
    # Set labels for real data
    real_label_value = 1

    # Pre-training with real data
    optimizer.zero_grad()
    real_labels = torch.full((real_data.size(0), real_data.size(1), 1), real_label_value, device=device)
    output_real = discriminator(real_data)
    loss_real = criterion(output_real, real_labels.float())
    loss_real.backward()

    optimizer.step()
    return loss_real.item()


def calculate_objective_function(generator, val_set, input_length, device):
    """
    Calculate the Wasserstein distance between the generated samples and val_set.

    Args:
    - generator: The generator model.
    - val_set: The validation set as a PyTorch tensor.
    - device: The device (cpu or cuda) to use for calculations.

    Returns:
    - wasserstein_distance: The Wasserstein distance.
    """

    # Set the generator to evaluation mode
    generator.eval()
    val_set = val_set.flatten()

    # Generate samples using the generator
    batch_size, seq_len = 1, len(val_set)

    # Make 100 samples and comapre with validation set
    N = 100
    tot_wasserstein_dis = 0

    for i in range(0, N):
        noise = torch.randn(batch_size, seq_len, input_length, device=device)
        generated_samples = generator(noise).cpu().detach().numpy()

        # Reshape the generated samples and validation set for Wasserstein distance calculation
        generated_samples = generated_samples.flatten()

        # Test normalize output
        tot_wasserstein_dis += wasserstein_distance(generated_samples, val_set)

    return tot_wasserstein_dis / N


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def uniform_dist(low, up, minimum, maximum):
    return lambda current_value: clamp(
        current_value + random.uniform(low, up) * current_value, minimum, maximum)


# Define your parameter distributions
number_of_batches_dist = uniform_dist(-0.2, 0.2, 8, 64)
input_length_dist = uniform_dist(-0.2, 0.2, 50, 200)
batch_length_dist = uniform_dist(-0.2, 0.2, 50, 400)
step_size_dist = uniform_dist(-0.2, 0.2, 10, 100)
pretrain_epochs_dist = uniform_dist(-0.2, 0.2, 0, 10)
lr_gen_dist = uniform_dist(-0.2, 0.2, 0.00001, 0.0002)
lr_dis_dist = uniform_dist(-0.2, 0.2, 0.00001, 0.0002)
layers_gen_dist = uniform_dist(-1, 1, 1, 5)
layers_dis_dist = uniform_dist(-1, 1, 1, 5)
dim_gen_dist = uniform_dist(-0.2, 0.2, 50, 300)
dropout_dist = uniform_dist(-0.2, 0.2, 0.0, 0.4)

# Dictionary for boundaries
parameter_ranges = [[8, 64], [50, 200], [50, 400], [10, 100], [0, 10], [0.00001, 0.0002], [0.00001, 0.0002], [1, 5],
                    [1, 5], [50, 300], [0, 0.4]]

# Number of tuning iterations
tuning_iterations = 12  # j=1,...,

# Number of different GAN parameter values per iteration
param_adjustment = 5  # k = 1,...,m

# Initial parameters
number_of_batches = 24
input_length = 100
epochs = 50
batch_length = 252
step_size = 50
pretrain_epochs = 5
lr_gen = 0.0001
lr_dis = 0.0001
layers_gen = 1
layers_dis = 1
dim_gen = 100
dropout = 0.2


def train_GAN(number_of_batches, input_length, batch_length, step_size, pretrain_epochs, lr_gen, lr_dis, layers_gen,
              layers_dis, dim_gen, dropout):
    # convert to integer
    number_of_batches = int(number_of_batches)
    input_length = int(input_length)
    batch_length = int(batch_length)
    step_size = int(step_size)
    pretrain_epochs = int(pretrain_epochs)
    layers_gen = int(layers_gen)
    layers_dis = int(layers_dis)
    dim_gen = int(dim_gen)

    cuda = True  # Set to True if you want to enable CUDA
    outf = 'checkpoints'
    imf = 'images'
    manualSeed = 10
    logdir = 'log'
    run_tag = ''
    real_label = 1
    fake_label = 0

    # Make this our training function with all parameters to tune as input
    # Parameters tuning
    with open('train_set.txt', 'r') as file:
        train_set = np.array([float(line) for line in file.read().splitlines()]).reshape(-1, 1, 1)

    with open('val_set.txt', 'r') as file:
        val_set = np.array([float(line) for line in file.read().splitlines()]).reshape(-1, 1, 1)

    # Create a writer for tensorboard
    date = datetime.datetime.now().strftime("%d-%m-%y_%H-%M")  # Replace colons (:) with underscores (-)
    run_name = f"{run_tag}_{date}" if run_tag != '' else date
    options_str = f"batch length: {batch_length}, numberOfBatches: {number_of_batches}, inputLength: {input_length}, epochs: {epochs}, lrGen: {lr_gen}, lrDis: {lr_dis}, step size: {step_size}, " \
                  f"pretrain epochs: {pretrain_epochs}, layers GEN: {layers_gen}, layers DIS: {layers_dis}, imf: {imf}, dim GEN: {dim_gen}, " \
                  f"dropout: {dropout}"
    print(options_str)

    try:
        os.makedirs(outf)
    except OSError:
        pass
    try:
        os.makedirs(imf)
    except OSError:
        pass

    print("Seed: ", manualSeed)
    torch.manual_seed(manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not cuda:
        print("You have a cuda device, so you might want to run with --cuda as an option")

    log_returns_train = np.log(train_set[1:] / train_set[:-1])

    log_returns_val = np.log(val_set[1:] / val_set[:-1])

    # Calculate the number of segments that can be created
    num_segments = math.floor((log_returns_train.size - batch_length) / step_size)

    # Create an empty array to store the segmented log returns
    gan_set = np.zeros(shape=(num_segments, batch_length, 1))  # Reorganize dimensions for LSTM-GAN training

    # Loop to create the segmented log returns
    for i in range(num_segments):
        segment = log_returns_train[i * step_size: i * step_size + batch_length].reshape(batch_length,
                                                                                         1)  # Reshape each segment to have one feature
        gan_set[i, :, :] = segment

    # Convert the NumPy array to a PyTorch tensor
    gan_set = torch.from_numpy(gan_set).float()  # Convert to PyTorch tensor

    numpy_buffer = io.BytesIO()
    np.save(numpy_buffer, gan_set)
    numpy_buffer.seek(0)

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    netD = CausalConvDiscriminator(input_size=1, n_layers=layers_dis, n_channel=10, kernel_size=8, dropout=dropout).to(
        device)
    netG = LSTMGenerator(in_dim=input_length, out_dim=1, n_layers=layers_gen, hidden_dim=dim_gen).to(device)

    # Create a BytesIO buffer to save and load the model
    modelG_buffer = io.BytesIO()
    torch.save(netG.state_dict(), modelG_buffer)
    modelG_buffer.seek(0)
    netG.load_state_dict(torch.load(modelG_buffer))

    assert netG

    # Repeat the process for netD if needed
    modelD_buffer = io.BytesIO()
    torch.save(netD.state_dict(), modelD_buffer)
    modelD_buffer.seek(0)
    netD.load_state_dict(torch.load(modelD_buffer))

    assert netD

    print("|Discriminator Architecture|\n", netD)
    print("|Generator Architecture|\n", netG)

    criterion = nn.BCELoss().to(device)

    real_label = 1
    fake_label = 0

    # Setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr_dis)
    optimizerG = optim.Adam(netG.parameters(), lr=lr_gen)

    discriminator_losses = []
    generator_losses = []
    epoch_counter = 0

    # Pretrain the discriminator
    for pretrain_epoch in range(pretrain_epochs):
        for i in range(0, gan_set.shape[0], number_of_batches):
            real_batch = gan_set[i:i + number_of_batches].clone().detach().to(device)
            pretrain_loss = pretrain_discriminator_real(netD, optimizerD, real_batch, criterion, device)
        print(f"Pretraining Epoch [{pretrain_epoch + 1}/{pretrain_epochs}], Loss: {pretrain_loss}")

    for epoch in range(epochs):
        epoch_counter += 1
        epoch_discriminator_losses = []  # Temporary list for epoch-specific losses
        epoch_generator_losses = []  # Temporary list for epoch-specific losses
        for i in range(0, gan_set.shape[0], number_of_batches):  # Iterate with batch_size_discriminator steps
            niter = epoch * (gan_set.shape[0] // number_of_batches) + (i // number_of_batches)

            # Extract a batch of sequences from your training set
            real_batch = gan_set[i:i + number_of_batches].clone().detach().to(device)

            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # Train with real data
            netD.zero_grad()
            real_batch = real_batch.to(device)
            batch_size, seq_len = real_batch.size(0), real_batch.size(1)
            label = torch.full((batch_size, seq_len, 1), real_label, device=device)
            output = netD(real_batch)

            label = label.float()  # Convert label to float
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake data
            noise = torch.randn(batch_size, seq_len, input_length, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(fake_label)  # HAS BEEN CHANGED TO FAKE LABEL
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            epoch_discriminator_losses.append(errD.item())
            epoch_generator_losses.append(errG.item())

        avg_discriminator_loss = sum(epoch_discriminator_losses) / len(epoch_discriminator_losses)
        avg_generator_loss = sum(epoch_generator_losses) / len(epoch_generator_losses)

        discriminator_losses.append(avg_discriminator_loss)
        generator_losses.append(avg_generator_loss)

    return [calculate_objective_function(netG, log_returns_val, input_length, device), netG]


# Run loops
file_path = "BHP parameters LSTM TCN.txt"
number_of_parameters = 11
wasserstein_values = np.zeros(param_adjustment + 2)  # Position 0 is final best, pos last is combination parameters
generators = [None] * (param_adjustment + 2)  # Position 0 is final best, pos last is combination parameters
parameters = np.zeros(shape=(param_adjustment + 2, number_of_parameters))  # Final position is improved set

parameters[0, 0] = number_of_batches
parameters[0, 1] = input_length
parameters[0, 2] = batch_length
parameters[0, 3] = step_size
parameters[0, 4] = pretrain_epochs
parameters[0, 5] = lr_gen
parameters[0, 6] = lr_dis
parameters[0, 7] = layers_gen
parameters[0, 8] = layers_dis
parameters[0, 9] = dim_gen
parameters[0, 10] = dropout

columns_to_convert = [0, 1, 2, 3, 4, 7, 8, 9, 10]

initial_GAN = train_GAN(parameters[0, 0], parameters[0, 1], parameters[0, 2], parameters[0, 3],
                                  parameters[0, 4],
                                  parameters[0, 5], parameters[0, 6], parameters[0, 7], parameters[0, 8],
                                  parameters[0, 9],
                                  parameters[0, 10])

wasserstein_values[0] = initial_GAN[0]
generators[0] = initial_GAN[1]

for j in range(0, tuning_iterations):

    for k in range(1, param_adjustment + 1):
        # Calculate new random parameter set
        parameters[k, 0] = number_of_batches_dist(parameters[0, 0])
        parameters[k, 1] = input_length_dist(parameters[0, 1])
        parameters[k, 2] = batch_length_dist(parameters[0, 2])
        parameters[k, 3] = step_size_dist(parameters[0, 3])
        parameters[k, 4] = pretrain_epochs_dist(parameters[0, 4])
        parameters[k, 5] = lr_gen_dist(parameters[0, 5])
        parameters[k, 6] = lr_dis_dist(parameters[0, 6])
        parameters[k, 7] = layers_gen_dist(parameters[0, 7])
        parameters[k, 8] = layers_dis_dist(parameters[0, 8])
        parameters[k, 9] = dim_gen_dist(parameters[0, 9])
        parameters[k, 10] = dropout_dist(parameters[0, 10])

        parameters[k, columns_to_convert] = np.round(parameters[k, columns_to_convert]).astype('int')

        # Train GAN on set
        trained_GAN = train_GAN(parameters[k, 0], parameters[k, 1], parameters[k, 2], parameters[k, 3],
                                parameters[k, 4],
                                parameters[k, 5], parameters[k, 6], parameters[k, 7], parameters[k, 8],
                                parameters[k, 9],
                                parameters[k, 10])

        wasserstein_values[k] = trained_GAN[0]
        generators[k] = trained_GAN[1]

        print('The wasserstein value of iteration ', str(k), ' is: ', wasserstein_values[k])
        sys.stdout.flush()

    # Create one more sample set using formula (assume that W dis is always finite in Python)
    s_array = np.zeros(number_of_parameters)

    for i in range(0, number_of_parameters):
        for k in range(1, param_adjustment + 1):
            # Calculate s
            s_array[i] = (wasserstein_values[0] - wasserstein_values[k]) * (
                    parameters[k, i] - parameters[0, i]) / param_adjustment
        # Use formula for optimal m+1
        if parameters[0, i] != 0:
            parameters[param_adjustment + 1, i] = parameters[0, i] + s_array[i] / (
                    parameters[0, i] * wasserstein_values[0])
        else:
            parameters[param_adjustment + 1, i] = s_array[i] / wasserstein_values[0]
        # Ensure that value is between pre-set minimum and maximum
        parameters[param_adjustment + 1, i] = clamp(parameters[param_adjustment + 1, i], parameter_ranges[i][0],
                                                    parameter_ranges[i][1])

    parameters[:, columns_to_convert] = np.round(parameters[:, columns_to_convert]).astype(int)

    trained_GAN = train_GAN(parameters[param_adjustment + 1, 0],
                            parameters[param_adjustment + 1, 1],
                            parameters[param_adjustment + 1, 2],
                            parameters[param_adjustment + 1, 3],
                            parameters[param_adjustment + 1, 4],
                            parameters[param_adjustment + 1, 5],
                            parameters[param_adjustment + 1, 6],
                            parameters[param_adjustment + 1, 7],
                            parameters[param_adjustment + 1, 8],
                            parameters[param_adjustment + 1, 9],
                            parameters[param_adjustment + 1, 10])

    wasserstein_values[param_adjustment + 1] = trained_GAN[0]
    generators[param_adjustment + 1] = trained_GAN[1]

    print('The wasserstein value of the final iteration is ', wasserstein_values[param_adjustment + 1])
    sys.stdout.flush()

    threshold = 0.12
    for i in range(0, len(wasserstein_values)):
        if wasserstein_values[i] < threshold:

            name = f'LSTM-TCN Gen BHP {j} {i}'
            save_wasserstein = wasserstein_values[i]
            save_parameters = parameters[i, :]
            # Save GAN
            torch.save(generators[i].state_dict(), name)
            # Save parameters to file
            with open(file_path, "a") as file:
                file.write(f'Wasserstein distance  {name} is {save_wasserstein}\n')
                file.write(f'Achieved {name} with parameters: {save_parameters}\n')

    # Choose set with lowest Wasserstein distance for new parameters
    # row 0 for both parameters and wasserstein_values to best parameters
    min_wasserstein = np.min(wasserstein_values)
    if min_wasserstein.size > 0:
        # Save Wasserstein values
        with open("BEST LSTM-TCN GAN Tuning Wasserstein Distance BHP.txt", "a") as file:
            np.savetxt(file, wasserstein_values.reshape(1, -1), delimiter=',', fmt='%f')

        parameters[0, :] = parameters[np.argmin(wasserstein_values), :]
        generators[0] = generators[np.argmin(wasserstein_values)]
        wasserstein_values[0] = np.min(wasserstein_values)
        print('The best parameters of the loop ', str(j), ' are ', parameters[0, :])
        print('The best wasserstein of the loop ', str(j), ' is ', wasserstein_values[0])
        sys.stdout.flush()

# Assuming you have already calculated wasserstein_values and parameters
best_wasserstein = wasserstein_values[0]
best_parameters = parameters[0, :]
best_generator = generators[0]

# Print the best Wasserstein distance and its parameters
print('Best Wasserstein distance is:', best_wasserstein)
print('Achieved with parameters:', best_parameters)

# Save the best Generator
torch.save(best_generator.state_dict(), 'BHP LSTM-TCN Gen')

# Save the values to a text file
with open(file_path, "a") as file:
    file.write(f'Best Wasserstein distance is: {best_wasserstein}\n')
    file.write(f'Achieved with parameters: {best_parameters}')
