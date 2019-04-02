import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

(XTrain, YTrain), (XTest, YTest) = mnist.load_data()

XTrain = XTrain.reshape((len(XTrain), np.prod(XTrain.shape[1:])))
XTest = XTest.reshape((len(XTest), np.prod(XTest.shape[1:])))


TrainFilter = np.where((YTrain == 0 ) | (YTrain == 1))
TestFilter = np.where((YTest == 0) | (YTest == 1))

XTrain, YTrain = XTrain[TrainFilter], YTrain[TrainFilter]
XTest, YTest = XTest[TestFilter], YTest[TestFilter]


# create dataset for lightgbm
LgbTrain = lgb.Dataset(XTrain, YTrain)
LgbEval = lgb.Dataset(XTest, YTest, reference=LgbTrain)

# specify your configurations as a dict
Parameters = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}


# train
gbm = lgb.train(Parameters,
                LgbTrain,
                num_boost_round=10,
                valid_sets=LgbTrain)

# predict
YPred = gbm.predict(XTest, num_iteration=gbm.best_iteration)
YPred = np.round(YPred)
YPred = YPred.astype(int)
# eval
print('Rmse of the model is:', mean_squared_error(YTest, YPred) ** 0.5)

ConfMatrix = confusion_matrix(YTest, YPred)
print(ConfMatrix)

print(accuracy_score(YTest, YPred))

