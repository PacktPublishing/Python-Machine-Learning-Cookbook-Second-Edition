from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection

import tensorflow as tf

# Load Iris data
iris = datasets.load_iris()
# Load features and classes
x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data, 
                                                                    iris.target, 
                                                                    test_size=0.3, 
                                                                    random_state=42)

# ------------------------------------------
# TensorFlow Implementation
# ------------------------------------------
# Building a 3-layer DNN with 50 units each.
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
classifier_tf = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, 
                                               hidden_units=[50, 50, 50], 
                                               n_classes=3)
classifier_tf.fit(x_train, y_train, steps=5000)
predictions = list(classifier_tf.predict(x_test, as_iterable=True))
score = metrics.accuracy_score(y_test, predictions)

print('TensorFlow Accuracy: {0:f}'.format(score))