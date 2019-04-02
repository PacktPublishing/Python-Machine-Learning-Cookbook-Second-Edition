from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *

paths = ["train.csv","test.csv"] 
target_name = "SalePrice"

data = Reader(sep=",").train_test_split(paths, target_name)
data = Drift_thresholder().fit_transform(data)

space = {

        'ne__numerical_strategy' : {"space" : [0, 'mean']},

        'ce__strategy' : {"space" : ["label_encoding", "random_projection", "entity_embedding"]},

        'fs__strategy' : {"space" : ["variance", "rf_feature_importance"]},
        'fs__threshold': {"search" : "choice", "space" : [0.1, 0.2, 0.3]},

        'est__strategy' : {"space" : ["XGBoost"]},
        'est__max_depth' : {"search" : "choice", "space" : [5,6]},
        'est__subsample' : {"search" : "uniform", "space" : [0.6,0.9]}

        }

best = opt.optimise(space, data, max_evals = 5)

Predictor().fit_predict(best, data)

