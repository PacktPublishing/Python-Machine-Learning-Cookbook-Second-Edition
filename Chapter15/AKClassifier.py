from keras.datasets import mnist
import autokeras as ak

(XTrain, YTrain), (XTest, YTest) = mnist.load_data()
XTrain = XTrain.reshape(XTrain.shape + (1,))
XTest = XTest.reshape(XTest.shape + (1,))

AKClf = ak.ImageClassifier()
AKClf.fit(XTrain, YTrain)
Results = AKClf.predict(XTest)
