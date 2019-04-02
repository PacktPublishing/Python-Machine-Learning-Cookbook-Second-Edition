from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

NGData = fetch_20newsgroups(shuffle=True, random_state=7,
                             remove=('headers', 'footers', 'quotes'))

print(list(NGData.target_names))

NGData = NGData.data[:2000]

NGDataVect = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=1000,
                                stop_words='english')

NGDataVectModel = NGDataVect.fit_transform(NGData)

LDAModel = LatentDirichletAllocation(n_components=10, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

LDAModel.fit(NGDataVectModel)


NGDataVectModelFeatureNames = NGDataVect.get_feature_names()

for topic_idx, topic in enumerate(LDAModel.components_):
    message = "Topic %d: " % topic_idx
    message += " ".join([NGDataVectModelFeatureNames[i]
    for i in topic.argsort()[:-20 - 1:-1]])
    print(message)
