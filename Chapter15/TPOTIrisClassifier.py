from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

IrisData = load_iris()
XTrain, XTest, YTrain, YTest = train_test_split(IrisData.data.astype(np.float64),
    IrisData.target.astype(np.float64), train_size=0.70, test_size=0.30)

TpotCL = TPOTClassifier(generations=5, population_size=50, verbosity=2)
TpotCL.fit(XTrain, YTrain)
print(TpotCL.score(XTest, YTest))
TpotCL.export('TPOTIrisPipeline.py')