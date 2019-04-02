from sklearn.datasets import fetch_20newsgroups

NewsClass = ['rec.sport.baseball', 'rec.sport.hockey']

DataTrain = fetch_20newsgroups(subset='train',categories=NewsClass, shuffle=True, random_state=42)

print(DataTrain.target_names)

print(len(DataTrain.data))
print(len(DataTrain.target))

from sklearn.feature_extraction.text import CountVectorizer

CountVect = CountVectorizer()
XTrainCounts = CountVect.fit_transform(DataTrain.data)
print(XTrainCounts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
TfTransformer = TfidfTransformer(use_idf=False).fit(XTrainCounts)
XTrainNew = TfTransformer.transform(XTrainCounts)

TfidfTransformer = TfidfTransformer()
XTrainNewidf = TfidfTransformer.fit_transform(XTrainCounts)


from sklearn.naive_bayes import MultinomialNB
NBMultiClassifier = MultinomialNB().fit(XTrainNewidf, DataTrain.target)

NewsClassPred = NBMultiClassifier.predict(XTrainNewidf)

# compute accuracy of the classifier
accuracy = 100.0 * (DataTrain.target == NewsClassPred).sum() / XTrainNewidf.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")

