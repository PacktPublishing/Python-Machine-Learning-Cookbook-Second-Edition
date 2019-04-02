import json
import datetime
import sys
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
#from mpl_toolkits import quotes_historical_yahoo_ochl as quotes_yahoo
#from mpl_finance import candlestick2_ochl as quotes_yahoo
# Input symbol file

symbol_file = 'symbol_map.json'

with open(symbol_file, 'r') as f:
    symbol_dict = json.loads(f.read())

symbols, names = np.array(sorted(symbol_dict.items())).T

quotes = []

for symbol in symbols:
    print('Fetching quote history for %r' % symbol, file=sys.stderr)
    url = ('https://raw.githubusercontent.com/scikit-learn/examples-data/'
           'master/financial-data/{}.csv')
    quotes.append(pd.read_csv(url.format(symbol)))
    
closing_quotes = np.vstack([q['close'] for q in quotes])
opening_quotes = np.vstack([q['open'] for q in quotes])

# The daily fluctuations of the quotes 
delta_quotes = closing_quotes - opening_quotes

# Build a graph model from the correlations
edge_model = covariance.GraphicalLassoCV(cv=3)

# Standardize the data 
X = delta_quotes.copy().T
X /= X.std(axis=0)

# Train the model
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Build clustering model using affinity propagation
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

# Print the results of clustering
for i in range(num_labels + 1):
    print("Cluster", i+1, "-->", ', '.join(names[labels == i]))
    


