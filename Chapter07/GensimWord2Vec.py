import nltk
nltk.download('abc')
import gensim
from nltk.corpus import abc

model= gensim.models.Word2Vec(abc.sents())
print(model)
X= list(model.wv.vocab)
data=model.wv.most_similar('science')
print(data)