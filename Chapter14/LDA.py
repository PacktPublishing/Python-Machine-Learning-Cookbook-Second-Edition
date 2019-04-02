from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

X, Y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, n_labels=2, random_state=1)
LDAModel = LatentDirichletAllocation(n_components=5, random_state=1)
LDAModel.fit(X) 
# get topics for some given samples:
print(LDAModel.transform(X[-10:]))
