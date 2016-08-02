from sklearn import grid_search
from sklearn.metrics import make_scorer, log_loss
from keras.utils.np_utils import to_categorical
import numpy as np
import logging

class Model(object):

    def __init__(self):
        self.model = None

    def fit(self, x_train, y_train, x_test):
        pass

    def predict(self, x_test):
        pass

    def grid_search_fit_(self, clf, param_grid, x_train, y_train, cv=2):
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, verbose=20, scoring="f1_weighted")


        logging.info("y_train.shape = %s" % str(np.array(y_train).shape))
        model.fit(x_train, y_train)
        logging.info("Best parameters found by grid search:")
        print(model.best_params_)
        logging.info("Best CV score:")
        print(model.best_score_)
        return model        
