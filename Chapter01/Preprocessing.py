import numpy as np
from sklearn import preprocessing

data = np.array([[3, -1.5,  2, -5.4], [0,  4,  -0.3, 2.1], [1,  3.3, -1.9, -4.3]])

print(data)

#Mean removal
print("Mean: ",data.mean(axis=0))
print("Standard Deviation: ",data.std(axis=0))

data_standardized = preprocessing.scale(data)

print("Mean standardized data: ",data_standardized.mean(axis=0))
print("Standard Deviation standardized data: ",data_standardized.std(axis=0))

#Scaling
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)

print("Min: ",data.min(axis=0))
print("Max: ",data.max(axis=0))

print("Min: ",data_scaled.min(axis=0))
print("Max: ",data_scaled.max(axis=0))

print(data_scaled)

#Normalization
data_normalized = preprocessing.normalize(data, norm='l1',axis=0)

print(data_normalized)

data_norm_abs = np.abs(data_normalized)

print(data_norm_abs.sum(axis=0))

#Binarization
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)

print(data_binarized)

#One Hot Encoding
data = np.array([[1, 1, 2], [0, 2, 3], [1, 0, 1], [0, 1, 0]])
print(data)

encoder = preprocessing.OneHotEncoder()
encoder.fit(data)
encoded_vector = encoder.transform([[1, 2, 3]]).toarray()

print(encoded_vector)