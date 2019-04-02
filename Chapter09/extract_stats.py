import pandas as pd
import matplotlib.pyplot as plt

from convert_to_timeseries import convert_data_to_timeseries

# Input file containing data
input_file = 'data_timeseries.txt'

# Load data
data1 = convert_data_to_timeseries(input_file, 2)
data2 = convert_data_to_timeseries(input_file, 3)
dataframe = pd.DataFrame({'first': data1, 'second': data2})

# Print max and min
print('Maximum:\n', dataframe.max())
print('Minimum:\n', dataframe.min())

# Print mean
print('Mean:\n', dataframe.mean())
print('Mean row-wise:\n', dataframe.mean(1)[:10])

# Plot rolling mean
DFMean = dataframe.rolling(window=24).mean()
plt.plot(DFMean)

# Print correlation coefficients
print('Correlation coefficients:\n', dataframe.corr())

# Plot rolling correlation
plt.figure()

DFCorr= dataframe.rolling(window=60).corr(pairwise=False)
plt.plot(DFCorr)
plt.show()

