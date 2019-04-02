from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense

IrisData = load_iris() 

X = IrisData.data

Y = IrisData.target.reshape(-1, 1) 

Encoder = OneHotEncoder(sparse=False)
YHE = Encoder.fit_transform(Y)

XTrain, XTest, YTrain, YTest = train_test_split(X, YHE, test_size=0.30)

model = Sequential()

model.add(Dense(10, input_shape=(4,), activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(3, activation='softmax'))


model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])



model.fit(XTrain, YTrain, verbose=2, batch_size=5, epochs=200)


results = model.evaluate(XTest, YTest)

print('Final test set loss:' ,results[0])
print('Final test set accuracy:', results[1])
