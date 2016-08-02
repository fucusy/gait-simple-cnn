#encoding=utf8

from sklearn.ensemble import RandomForestClassifier
from model.base_model import Model
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import classification_report
from sklearn import svm
import logging
from sklearn.cross_validation import PredefinedSplit
import numpy as np


class RandomForestClassification(Model):

    def __init__(self):
        Model.__init__(self)
        self.model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=2016, verbose=1, max_depth=1000, max_features=400)

    def fit(self, x_train, y_train, x_validation=None, y_validation=None):
        param_grid = {'n_estimators': [1500, 3000, 4000], 'max_depth': [100, 400] }
        train_validation_feature = list(x_train) + list(x_validation)
        train_validation_y = list(y_train) + list(y_validation)
        test_fold = [-1] * len(x_train) + [1] * len(x_validation)
        cv = PredefinedSplit(test_fold=test_fold)
        self.model = self.grid_search_fit_(self.model, param_grid, train_validation_feature, train_validation_y, cv)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class MultiSVMClassification(Model):

    def __init__(self):
        Model.__init__(self)
        self.model = svm.SVC(decision_function_shape='ovr', probability=True)

    def fit(self, x_train, y_train, x_test=None):
        train_pred = cross_val_predict(self.model, x_train, y_train, cv=2)
        report = classification_report(y_train, train_pred)
        logging.info("the cross validation report:\n %s" % report)
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
