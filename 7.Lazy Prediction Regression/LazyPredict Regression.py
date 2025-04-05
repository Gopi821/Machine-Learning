
pip install lazypredict

# import libraries

import lazypredict

from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np

# Load the dataset

diabetes  = datasets.load_diabetes()
X, y = shuffle(diabetes.data, diabetes.target, random_state=13)
X = X.astype(np.float32)

offset = int(X.shape[0] * 0.9)

# Splitting the data into train set and test set\
    
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# Fit the lazypredict regressor and find the predictions

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)

