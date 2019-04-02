import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers

SetSeed = 1

CreditCardData = pd.read_csv("creditcard.csv")

CountClasses = pd.value_counts(CreditCardData['Class'], sort = True)

print(CountClasses)

print(CreditCardData.Amount.describe())

from sklearn.preprocessing import StandardScaler

Data = CreditCardData.drop(['Time'], axis=1)
Data['Amount'] = StandardScaler().fit_transform(Data['Amount'].values.reshape(-1, 1))

Data.Amount.describe()

XTrain, XTest = train_test_split(Data, test_size=0.3, random_state=SetSeed)
XTrain = XTrain[XTrain.Class == 0]
XTrain = XTrain.drop(['Class'], axis=1)

YTest = XTest['Class']
XTest = XTest.drop(['Class'], axis=1)

XTrain = XTrain.values
XTest = XTest.values

InputDim = XTrain.shape[1]

InputModel = Input(shape=(InputDim,))
EncodedLayer = Dense(16, activation='relu')(InputModel)
DecodedLayer = Dense(InputDim, activation='sigmoid')(EncodedLayer)
AutoencoderModel = Model(InputModel, DecodedLayer)
AutoencoderModel.summary()

NumEpoch = 10
BatchSize = 32

AutoencoderModel.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

history = AutoencoderModel.fit(XTrain, XTrain,
                    epochs=NumEpoch,
                    batch_size=BatchSize,
                    shuffle=True,
                    validation_data=(XTest, XTest),
                    verbose=1,
                    ).history
                          
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');

                          
PredData = AutoencoderModel.predict(XTest)
mse = np.mean(np.power(XTest - PredData, 2), axis=1)
ErrorCreditCardData = pd.DataFrame({'Error': mse,
                        'TrueClass': YTest})

ErrorCreditCardData.describe()

from sklearn.metrics import confusion_matrix

threshold = 3.
YPred = [1 if e > threshold else 0 for e in ErrorCreditCardData.Error.values]
ConfMatrix = confusion_matrix(ErrorCreditCardData.TrueClass, YPred)
print(ConfMatrix)


from sklearn.metrics import accuracy_score

print(accuracy_score(ErrorCreditCardData.TrueClass, YPred))

