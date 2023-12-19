import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Get Data
file_name = 'W-GAN BHP Sensitivity analysis maturity.txt'
df_results = pd.read_csv(file_name, delimiter='\t', header=None,
                             names=['Simulation', 'Type', 'Sort', 'MAPE', 'Market price', 'Strike Price',
                                    'Time to Maturity', 'Estimated Price', 'Starting Price', 'Implied volatility', 'Computational Time'])


# Plot distribution
grouped_df = df_results.groupby(['Simulation', 'Sort'])
dfs_dict = {}
for group_name, group_data in grouped_df:
    simulation = group_name
    dfs_dict[(simulation)] = group_data

color_dic = {'MC': 'blue', 'QMC': 'orange', 'GAN-MC': 'green', 'GAN-QMC': 'red'}
figure, axis = plt.subplots(2)
for (simulation), df in dfs_dict.items():
    color = color_dic.get(simulation[0])  # Use a colormap to get distinct colors
    option_prices = df['Estimated Price']
    #option_prices = option_prices[option_prices > 0.1]
    print(simulation[1])
    if simulation[1] == 'put':
        sns.kdeplot(option_prices, color=color, label=simulation[0], ax=axis[0])
    else:
        sns.kdeplot(option_prices, color=color, label=simulation[0], ax=axis[1])

axis[0].legend()
axis[0].set_title('Maturity')
axis[0].set_xlabel('Estimated Put Price')
axis[0].set_ylabel('Density')

axis[1].legend()
axis[1].set_xlabel('Estimated Call Price')
axis[1].set_ylabel('Density')
plt.show()
