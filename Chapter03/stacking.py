# load boston dataset from sklearn
from heamy.dataset import Dataset
from heamy.estimator import Regressor
from heamy.pipeline import ModelsPipeline

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data = load_boston()
X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# create dataset
Data = Dataset(X_train,y_train,X_test)

# initialize RandomForest & LinearRegression
RfModel = Regressor(dataset=Data, estimator=RandomForestRegressor, parameters={'n_estimators': 50},name='rf')
LRModel = Regressor(dataset=Data, estimator=LinearRegression, parameters={'normalize': True},name='lr')


# Stack two models
# Returns new dataset with out-of-fold predictions
Pipeline = ModelsPipeline(RfModel,LRModel)
StackModel = Pipeline.stack(k=10,seed=2)

# Train LinearRegression on stacked data (second stage)
Stacker = Regressor(dataset=StackModel, estimator=LinearRegression)
Results = Stacker.predict()
# Validate results using 10 fold cross-validation
Results = Stacker.validate(k=10,scorer=mean_absolute_error)
