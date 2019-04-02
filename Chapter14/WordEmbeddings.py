import gensim
# define training data
sentences = [['my', 'first', 'book', 'with', 'Packt', 'is', 'on','Matlab'],
			['my', 'second', 'book', 'with', 'Packt', 'is', 'on','R'],
			['my', 'third', 'book', 'with', 'Packt', 'is', 'on','Python'],
			['one', 'more', 'book'],
			['is', 'on', 'Python', 'too']]
# train model
Model1 = gensim.models.Word2Vec(sentences, min_count=1, sg=0)
# summarize the loaded model
print(Model1)
# summarize vocabulary
wordsM1 = list(Model1.wv.vocab)
print(wordsM1)
# access vector for one word
print(Model1.wv['book'])

# train model
Model2 = gensim.models.Word2Vec(sentences, min_count=1, sg=1)
# summarize the loaded model
print(Model2)
# summarize vocabulary
wordsM2 = list(Model1.wv.vocab)
print(wordsM2)
# access vector for one word
print(Model2.wv['book'])



