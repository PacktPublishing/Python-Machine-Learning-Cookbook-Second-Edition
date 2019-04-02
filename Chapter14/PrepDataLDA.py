from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

Doc1 = "Some doctors say that pizza is good for your health."
Doc2 = "The pizza is good to eat, my sister likes to eat a good pizza, but not to my brother."
Doc3 = "Doctors suggest that walking can cause a decrease in blood pressure."
Doc4 = "My brother likes to walk, but my sister don't like to walk."
Doc5 = "When my sister is forced to walk for a long time she feels an increase in blood pressure."
Doc6 = "When my brother eats pizza, he has health problems."

DocList = [Doc1, Doc2, Doc3, Doc4, Doc5, Doc6]

Tokenizer = RegexpTokenizer(r'\w+')

EnStop = get_stop_words('en')

PStemmer = PorterStemmer()

Texts = []

for i in DocList:
    
    raw = i.lower()
    Tokens = Tokenizer.tokenize(raw)

    StoppedTokens = [i for i in Tokens if not i in EnStop]
    
    StemmedTokens = [PStemmer.stem(i) for i in StoppedTokens]
    
    Texts.append(StemmedTokens)

Dictionary = corpora.Dictionary(Texts)
    
CorpusMat = [Dictionary.doc2bow(text) for text in Texts]

LDAModel = models.ldamodel.LdaModel(CorpusMat, num_topics=3, id2word = Dictionary, passes=20)
print(LDAModel.print_topics(num_topics=3, num_words=3))