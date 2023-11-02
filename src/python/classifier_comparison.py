import os
import numpy as np
import pandas as pd
from pmlb import classification_dataset_names
from sklearn.model_selection import train_test_split

from sklearn import metrics
from catboost import CatBoostClassifier, Pool

import xgboost as xgb


ROOT_PATH = "/home/charles/Data/"

def _path(dataset_name, isX=True):
    suff = "_X" if isX else "_y"
    return ".".join([os.path.join(ROOT_PATH, dataset_name) + suff, "csv"])

def _train_test_path(dataset_name, isTrain=True, isX=True):
    suff = "train" if isTrain else "test"
    return _path("_".join([dataset_name, suff]), isX=isX)

def _X_path(dataset_name, isTrain=True):
    return _train_test_path(dataset_name, isTrain=isTrain, isX=True)

def _y_path(dataset_name, isTrain=True):
    return _train_test_path(dataset_name, isTrain=isTrain, isX=False)
            
def _dataset(dataset_name):
    return [pd.read_csv(x, header=None)  for x in
                       [ _X_path(dataset_name, isTrain=True), \
                         _y_path(dataset_name, isTrain=True), \
                         _X_path(dataset_name, isTrain=False),\
                         _y_path(dataset_name, isTrain=False)
                       ]
           ]
                       
# dataset_name = "colic"
dataset_name = "coil2000"
X_train, y_train, X_test, y_test = _dataset(dataset_name)

##############
## CATBOOST ##
##############
if __name__ == '__main__':
    iterations = 25000
    learning_rate = 0.35
    catboost_loss_function = "CrossEntropy"
    # catboost_loss_function = "Logloss"
    cls_cb = CatBoostClassifier(iterations=iterations,
                                depth=None,
                                learning_rate=learning_rate,
                                loss_function=catboost_loss_function,
                                verbose=False)
    
    cls_cb.fit(X_train, y_train)
    yhat_train_cb = cls_cb.predict(X_train)
    acc_IS_cb = metrics.accuracy_score(yhat_train_cb, y_train)
    yhat_test_cb = cls_cb.predict(X_test)
    acc_OOS_cb = metrics.accuracy_score(yhat_test_cb, y_test)
    
    print("[{}] CATBOOST: ACC_IS: {:2.2%} ACC_OOS: {:2.2%}".format(dataset_name,
                                                                   acc_IS_cb,
                                                                   acc_OOS_cb))

#############
## XGBOOST ##
#############

    eta = 0.35
    max_depth = 0
    subsample = 1. # default is 1.
    objective = "binary:logistic"
    
    cls_xgb = xgb.XGBClassifier(eta=eta,
                                objective=objective,
                                max_depth=max_depth)
    cls_xgb.fit(X_train, y_train)

    yhat_train_xgb = cls_xgb.predict(X_train)
    acc_IS_xgb = metrics.accuracy_score(yhat_train_xgb, y_train)
    yhat_test_xgb = cls_xgb.predict(X_test)
    acc_OOS_xgb = metrics.accuracy_score(yhat_test_xgb, y_test)
    
    print("[{}] XGBOOST:  ACC_IS: {:2.2%} ACC_OOS: {:2.2%}".format(dataset_name,
                                                                   acc_IS_xgb,
                                                                   acc_OOS_xgb))
    
