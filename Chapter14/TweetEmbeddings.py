#https://towardsdatascience.com/handling-overfitting-in-deep-learning-models-c760ee047c6e
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import models
from keras import layers

TweetData = pd.read_csv('Tweets.csv')
TweetData = TweetData.reindex(np.random.permutation(TweetData.index))
TweetData = TweetData[['text', 'airline_sentiment']]
XTrain, XTest, YTrain, YTest = train_test_split(TweetData.text, TweetData.airline_sentiment, test_size=0.1, random_state=11)

TkData = Tokenizer(num_words=1000,
                 filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=True, split=" ")
TkData.fit_on_texts(XTrain)
XTrainSeq = TkData.texts_to_sequences(XTrain)
XTestSeq = TkData.texts_to_sequences(XTest)

XTrainSeqTrunc = pad_sequences(XTrainSeq, maxlen=24)
XTestSeqTrunc = pad_sequences(XTestSeq, maxlen=24)

LabelEnc = LabelEncoder()
YTrainLabelEnc = LabelEnc.fit_transform(YTrain)
YTestLabelEnc = LabelEnc.transform(YTest)
YTrainLabelEncCat = to_categorical(YTrainLabelEnc)
YTestLabelEncCat = to_categorical(YTestLabelEnc)

XTrainEmb, XValEmb, YTrainEmb, YValEmb = train_test_split(XTrainSeqTrunc, YTrainLabelEncCat, test_size=0.2, random_state=11)

EmbModel = models.Sequential()
EmbModel.add(layers.Embedding(1000, 8, input_length=24))
EmbModel.add(layers.Flatten())
EmbModel.add(layers.Dense(3, activation='softmax'))


EmbModel.compile(optimizer='rmsprop'
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])
    
EmbHistory = EmbModel.fit(XTrainEmb
                       , YTrainEmb
                       , epochs=100
                       , batch_size=512
                       , validation_data=(XValEmb, YValEmb)
                       , verbose=1)

print('Train Accuracy: ', EmbHistory.history['acc'][-1])
print('Validation Accuracy: ', EmbHistory.history['val_acc'][-1])


plt.plot(EmbHistory.history['acc'])
plt.plot(EmbHistory.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()


