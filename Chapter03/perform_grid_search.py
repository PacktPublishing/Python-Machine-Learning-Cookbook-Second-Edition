from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import utilities 

# Load input data
input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)

###############################################
# Train test split

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)

# Set the parameters by cross-validation
parameter_grid = {"C": [1, 10, 50, 600],
                  'kernel':['linear','poly','rbf'],
                  "gamma": [0.01, 0.001],
                  'degree': [2, 3]}

metrics = ['precision']

for metric in metrics:

    print("#### Grid Searching optimal hyperparameters for", metric)
          
    classifier = GridSearchCV(svm.SVC(C=1), 
            parameter_grid, cv=5,scoring=metric,return_train_score=True) 

    classifier.fit(X_train, y_train)

    print("Scores across the parameter grid:")
    GridSCVResults = pd.DataFrame(classifier.cv_results_)
    for i in range(0,len(GridSCVResults)):
        print(GridSCVResults.params[i], '-->', round(GridSCVResults.mean_test_score[i],3))
    
    print("Highest scoring parameter set:", classifier.best_params_)
    y_true, y_pred = y_test, classifier.predict(X_test)
    print("Full performance report:\n")
    print(classification_report(y_true, y_pred))
    
# Perform a randomized search on hyper parameters

from sklearn.model_selection import RandomizedSearchCV
    
parameter_rand = {"C": [1, 10, 50, 600],
                  'kernel':['linear','poly','rbf'],
                  "gamma": [0.01, 0.001],
                  'degree': [2, 3]}

metrics = ['precision']

for metric in metrics:

    print("#### Randomized Searching optimal hyperparameters for", metric)
          
    classifier = RandomizedSearchCV(svm.SVC(C=1), 
            param_distributions=parameter_rand,n_iter=30, cv=5,return_train_score=True)

    classifier.fit(X_train, y_train)

    print("Scores across the parameter grid:")
    RandSCVResults = pd.DataFrame(classifier.cv_results_)
    for i in range(0,len(RandSCVResults)):
        print(RandSCVResults.params[i], '-->', round(RandSCVResults.mean_test_score[i],3))
    
    print("Highest scoring parameter set:", classifier.best_params_)
    y_true, y_pred = y_test, classifier.predict(X_test)
    print("Full performance report:\n")
    print(classification_report(y_true, y_pred))
    

