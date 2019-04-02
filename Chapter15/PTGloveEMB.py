from numpy import array
from numpy import zeros
from numpy import asarray
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding


# define negative and positive adjectives to describe a person
Adjectives = ['Wonderful',
        'Heroic',
        'Glamorous',
        'Valuable',
        'Excellent',
        'Optimistic',
        'Peaceful',
        'Romantic',
        'Loving',
        'Faithful',
        'Aggressive',
        'Arrogant',
        'Bossy',
        'Boring',
        'Careless',
        'Selfish',
        'Deceitful',
        'Dishonest',
        'Greedy',
        'Impatient']

AdjLabels = array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])

TKN = Tokenizer()
TKN.fit_on_texts(Adjectives)
VocabSize = len(TKN.word_index) + 1

EncodedAdjectives = TKN.texts_to_sequences(Adjectives)
PaddedAdjectives = pad_sequences(EncodedAdjectives, maxlen=4, padding='post')

EmbeddingsIndex = dict()
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
	Values = line.split()
	Word = Values[0]
	Coefs = asarray(Values[1:], dtype='float32')
	EmbeddingsIndex[Word] = Coefs
f.close()

EmbeddingMatrix = zeros((VocabSize, 100))
for word, i in TKN.word_index.items():
	EmbeddingVector = EmbeddingsIndex.get(word)
	if EmbeddingVector is not None:
		EmbeddingMatrix[i] = EmbeddingVector

AdjModel = Sequential()
PTModel = Embedding(VocabSize, 100, weights=[EmbeddingMatrix], input_length=4, trainable=False)
AdjModel.add(PTModel)
AdjModel.add(Flatten())
AdjModel.add(Dense(1, activation='sigmoid'))
print(AdjModel.summary())

AdjModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

AdjModel.fit(PaddedAdjectives, AdjLabels, epochs=50, verbose=1)

loss, accuracy = AdjModel.evaluate(PaddedAdjectives, AdjLabels, verbose=1)
print('Model Accuracy: %f' % (accuracy*100))
