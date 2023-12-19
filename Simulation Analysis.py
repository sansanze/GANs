import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_moneyness(row):
    if row['Sort'] == 'put':
        return row['Starting Price'] - row['Strike Price']
    else:
        return row['Strike Price'] - row['Starting Price']


multiple_sets = True

if multiple_sets:
    # For BHP
    output_file_bhp = 'W-GAN BHP Output 50000.txt'
    df_results_bhp = pd.read_csv(output_file_bhp, delimiter='\t', header=None,
                             names=['Simulation', 'Type', 'Sort', 'MAPE', 'Market price', 'Strike Price',
                                    'Time to Maturity', 'Estimated Price', 'Starting Price', 'Implied volatility', 'Computational Time'])

    # For CBA
    output_file_cba = 'W-GAN CBA Output 50000.txt'
    df_results_cba = pd.read_csv(output_file_cba, delimiter='\t', header=None,
                             names=['Simulation', 'Type', 'Sort', 'MAPE', 'Market price', 'Strike Price',
                                    'Time to Maturity', 'Estimated Price', 'Starting Price', 'Implied volatility', 'Computational Time'])

    # For CSL
    output_file_csl = 'W-GAN CSL Output 50000.txt'
    df_results_csl = pd.read_csv(output_file_csl, delimiter='\t', header=None,
                             names=['Simulation', 'Type', 'Sort', 'MAPE', 'Market price', 'Strike Price',
                                    'Time to Maturity', 'Estimated Price', 'Starting Price', 'Implied volatility', 'Computational Time'])

    # Combine sets
    frames = [df_results_bhp, df_results_cba, df_results_csl]
    df_results = pd.concat(frames)

else:
    output_file = 'W-GAN BHP Output 1000000 dt 0.3.txt'
    df_results = pd.read_csv(output_file, delimiter='\t', header=None,
                                 names=['Simulation', 'Type', 'Sort', 'MAPE', 'Market price', 'Strike Price',
                                        'Time to Maturity', 'Estimated Price', 'Starting Price', 'Implied volatility',
                                        'Computational Time'])

df_results['Moneyness'] = df_results.apply(calculate_moneyness, axis=1)
print(df_results)
df_results['MAPE'] = df_results['MAPE'] * 100

# Create new dataframes based in Simulation
grouped_df = df_results.groupby(['Simulation'])
dfs_dict = {}
for group_name, group_data in grouped_df:
    simulation = group_name
    dfs_dict[(simulation)] = group_data

# Plot Moneyness against MAPE
color_dic = {'MC': 'blue', 'QMC': 'orange', 'GAN-MC': 'green', 'GAN-QMC': 'red'}
for (simulation), df in dfs_dict.items():
    color = color_dic.get(simulation[0])  # Use a colormap to get distinct colors
    df = df.sort_values(by='Moneyness')
    # plt.plot(df['Moneyness'], df['MAPE'], color, label=simulation[0])
    z = np.polyfit(df['Moneyness'], df['MAPE'], 1)  # Fit a second-degree polynomial (line)
    p = np.poly1d(z)
    plt.plot(df['Moneyness'], p(df['Moneyness']), color=color, linestyle='--', label=simulation[0])

plt.xlabel('Moneyness (AUD)')
plt.ylabel('MAPE (%)')
plt.legend()
plt.title('Moneyness')
plt.show()

# Plot Maturity against MAPE
for (simulation), df in dfs_dict.items():
    color = color_dic.get(simulation[0])
    mean_MAPE = df.groupby('Time to Maturity', as_index=False)['MAPE'].mean()
    # plt.plot(mean_MAPE['Time to Maturity'], mean_MAPE['MAPE'], color, label=simulation[0])
    z = np.polyfit(mean_MAPE['Time to Maturity'], mean_MAPE['MAPE'], 1)  # Fit a second-degree polynomial (line)
    p = np.poly1d(z)
    plt.plot(mean_MAPE['Time to Maturity'], p(mean_MAPE['Time to Maturity']), color=color, linestyle='--', label=simulation[0])

plt.xlabel('Time to Maturity (days)')
plt.ylabel('MAPE (%)')
plt.legend()
plt.title('Time to Maturity')
plt.show()

# Implied Volatility
for (simulation), df in dfs_dict.items():
    color = color_dic.get(simulation[0])  # Use a colormap to get distinct colors
    df = df.sort_values(by='Implied volatility')
    df['Implied volatility'] = df['Implied volatility'] * (252 ** 0.5) * 100
    df = df[df['Implied volatility'] < 35]

    #plt.plot(df['Implied volatility'], df['MAPE'], color, label=simulation[0])
    z = np.polyfit(df['Implied volatility'], df['MAPE'], 1)  # Fit a second-degree polynomial (line)
    p = np.poly1d(z)
    plt.plot(df['Implied volatility'], p(df['Implied volatility']), color=color, linestyle='--', label=simulation[0])

plt.xlabel('Implied Volatility (%)')
plt.ylabel('MAPE (%)')
plt.legend()
plt.title('Implied Volatility')
plt.show()

# Create new dataframes based in Simulation
grouped_df = df_results.groupby(['Simulation', 'Type', 'Sort'])

for group_name, group_df in grouped_df:
    print(group_name[0], group_name[1], group_name[2], group_df['MAPE'].mean() - 1.96 * group_df['MAPE'].std() / np.sqrt(group_df['MAPE'].shape[0]), group_df['MAPE'].mean(), group_df['MAPE'].mean() + 1.96 * group_df['MAPE'].std() / np.sqrt(group_df['MAPE'].shape[0]))

# Create new dataframes based in Simulation
grouped_df = df_results.groupby(['Simulation'])

for group_name, group_df in grouped_df:
    print(group_name[0], group_df['MAPE'].mean() - 1.96 * group_df['MAPE'].std() / np.sqrt(group_df.shape[0]), group_df['MAPE'].mean(), group_df['MAPE'].mean() + 1.96 * group_df['MAPE'].std() / np.sqrt(group_df.shape[0]))
    time_error = group_df['Computational Time']
    mean_time_error = time_error.mean()
    print('Efficiency ', group_name[0], mean_time_error - 1.96 * time_error.std() / np.sqrt(group_df.shape[0]), mean_time_error, mean_time_error + 1.96 * time_error.std() / np.sqrt(group_df.shape[0]))
