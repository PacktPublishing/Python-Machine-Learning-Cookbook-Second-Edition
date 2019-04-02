import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

Input, Target = sklearn.datasets.load_digits()
XTrain, XTest, YTrain, YTest = sklearn.model_selection.train_test_split(Input, Target, random_state=3)
ASKModel = autosklearn.classification.AutoSklearnClassifier()
ASKModel.fit(XTrain, YTrain)
YPred = ASKModel.predict(XTest)
print("Accuracy score", sklearn.metrics.accuracy_score(YTest, YPred))
