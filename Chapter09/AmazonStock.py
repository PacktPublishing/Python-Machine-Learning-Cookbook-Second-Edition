import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from random import seed

seed(0)

Data = pd.read_csv('AMZN.csv',header=0, usecols=['Date', 'Close'],parse_dates=True,index_col='Date')
print(Data.info())
print(Data.head())
print(Data.describe())

plt.figure(figsize=(10,5))
plt.plot(Data)
plt.show()

DataPCh = Data.pct_change()

LogReturns = np.log(1 + DataPCh) 
print(LogReturns.tail(10))

plt.figure(figsize=(10,5))
plt.plot(LogReturns)
plt.savefig('figure.pdf',format='pdf',dpi=1000)
plt.show()