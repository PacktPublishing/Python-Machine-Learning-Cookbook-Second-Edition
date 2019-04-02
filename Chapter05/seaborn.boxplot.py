import pandas as pd
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
boston = datasets.load_boston()

BostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)

Predictors = BostonDF[BostonDF.columns[0:12]]

sns.set(style="ticks")
sns.boxplot(data = Predictors)
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
DataScaled = scaler.fit_transform(Predictors)


sns.set(style="ticks")
sns.boxplot(data = DataScaled)
plt.show()