import yfinance as yf
import numpy as np

train_set = yf.download('BHP', start='1990-01-01', end='2023-11-11')['Adj Close'].values
val_set = yf.download('BHP', start='1990-01-01', end='2023-11-11')['Adj Close'].values

# Write train_set to a text file
with open('train_set.txt', 'w') as train_file:
    train_file.write("\n".join(map(str, train_set)))

# Write val_set to a text file
with open('val_set.txt', 'w') as val_file:
    val_file.write("\n".join(map(str, val_set)))

# Read the data from the text file and convert it back to a NumPy array
with open('train_set.txt', 'r') as file:
    train = np.array([float(line) for line in file.read().splitlines()]).reshape(-1, 1, 1)

# Read the data from the text file and convert it back to a NumPy array
with open('val_set.txt', 'r') as file:
    val = np.array([float(line) for line in file.read().splitlines()]).reshape(-1, 1, 1)

print(train)
print(val)