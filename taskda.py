import pandas as pd
import matplotlib.pyplot as plt

data_1 = pd.read_csv('data_1.csv')
data_2 = pd.read_csv('data_2.csv')

plt.plot(data_1.iloc[:, 0], data_1.iloc[:, 1], label='Data 1')
plt.plot(data_2.iloc[:, 0], data_2.iloc[:, 1], label='Data 2')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.legend()
plt.show()
