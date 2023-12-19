import numpy as np
import matplotlib.pyplot as plt

files = ['BEST LSTM GAN Tuning Wasserstein Distance BHP.txt', 'BEST LSTM GAN Tuning Wasserstein Distance CBA.txt',
         'BEST LSTM GAN Tuning Wasserstein Distance CSL.txt',
         'BEST LSTM-TCN GAN Tuning Wasserstein Distance BHP.txt',
         'BEST LSTM-TCN GAN Tuning Wasserstein Distance CBA.txt',
         'BEST LSTM-TCN GAN Tuning Wasserstein Distance CSL.txt',
         'BEST W-GAN Tuning Wasserstein Distance BHP.txt', 'BEST W-GAN Tuning Wasserstein Distance CBA.txt',
         'BEST W-GAN Tuning Wasserstein Distance CSL.txt']

all_values = []
counter = 0
counter2 = 0
for element in files:
    set1 = []
    special_formula = False
    with open(element, 'r') as file:
        for line in file:
            set1.append(line.rstrip('\n').split(','))

    set1 = np.array(set1).astype(float)
    average_set = np.mean(set1, axis=1)
    x = range(1, len(average_set) + 1)
    all_values.append(average_set)

    # Count special formula
    lowest = 100
    low_col = 0
    for i in range(len(set1)):
        for j in range(len(set1[0])-1):  # Loop through columns
            if set1[i][j] < lowest:
                lowest = set1[i][j]
                low_col = j
    print(low_col)
    if low_col == 6:
        counter += 1

    # Count per row
    for i in range(len(set1)):
        lowest = 100
        low_col = -1
        for j in range(len(set1[0])):  # Loop through columns
            if set1[i][j] < lowest:
                lowest = set1[i][j]
                low_col = j

        if low_col == 5:
            counter2 += 1

all_values = np.mean(all_values, axis=0)
plt.plot(range(1, len(all_values) + 1), all_values)
plt.ylabel('Wasserstein Distance')
plt.xlabel('Tuning iteration')
plt.xlim(1, None)  # Corrected line
plt.title('Average Wasserstein Distance of 9 GANs')
plt.show()

print('Number of time special formula gave the best set: ', str(counter))

print(str(counter2))