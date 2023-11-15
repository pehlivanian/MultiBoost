import os
import numpy as np
import pandas as pd
from pmlb import classification_dataset_names
from sklearn.model_selection import train_test_split

from sklearn import metrics
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb


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


def _create_synthetic_disc_data(dim=2):
    dataset_name = "synthetic"
    ROOT_DATA = "/home/charles/Data/"
    METHOD = ['spherical', 'eggholder'][0]

    if METHOD in ('spherical',):
        coord = np.arange(-np.sqrt(np.pi), np.sqrt(np.pi), .1)
        n = coord.shape[0]
        data = np.zeros([np.power(coord.shape[0],2), 2*(dim-1) + 1])
        for i in range(n):
            for j in range(n):
                arg = coord[i]*coord[i] + coord[j]+coord[i]; label = 0.
                label += (1./1.)*np.cos(arg)
                label -= (2./1.)*np.cos(2*arg)
                label += (1./1.)*np.cos(4*arg)
                label -= (4./1.)*np.cos(8*arg)
                label += (1./1.)*np.cos(16*arg)
                label -= (8./1.)*np.cos(32*arg)
                label += (1./1.)*np.cos(64*arg)
                data[j+i*n] = np.array([coord[i], coord[j], label])
                data[j+i*n,2] = data[j+i*n,2] < 0.5
    elif METHOD in ('eggholder',):
        A = 10
        coord = np.arange(-.10, .10, .001)
        n = coord.shape[0]
        data = np.zeros([np.power(coord.shape[0],2), 2*(dim-1) + 1])
        for i in range(n):
            for j in range(n):
                data[j+i*n] = np.array([coord[i],
                                        coord[j],
                                        -(coord[j] + 47)*np.sin(np.abs(coord[i]/2. + \
                                         (coord[j] + 47))) - coord[i]*np.sin(np.abs(coord[i] - (coord[j] + 47)))])
                data[j+i*n,2]  = data[j+i*n,2] < -5.5

    TRAIN_SIZE = 250
    TEST_SIZE = 750
    X_train, X_test, y_train, y_test = train_test_split(data[:,:2], data[:,2], random_state=55, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

    np.savetxt( '{}/{}_train_X.csv'.format(ROOT_DATA, dataset_name), X_train, delimiter=',')
    np.savetxt( '{}/{}_train_y.csv'.format(ROOT_DATA, dataset_name), y_train, delimiter=',')
    np.savetxt( '{}/{}_test_X.csv'. format(ROOT_DATA, dataset_name), X_test,  delimiter=',')
    np.savetxt( '{}/{}_test_y.csv'. format(ROOT_DATA, dataset_name), y_test,  delimiter=',')
    
def _create_synthetic_cont_data(dim=2):
    dataset_name = "synthetic"
    ROOT_DATA = "/home/charles/Data/"
    MODEL_TYPE = "Regression"
    METHOD = ['spherical', 'beale', 'rastrigin', 'rosenbrock', 'bukin', 'levi', 'eggholder'][0]

    if METHOD in ('spherical',):
        coord = np.arange(-np.sqrt(np.pi), np.sqrt(np.pi), .1)
        n = coord.shape[0]
        data = np.zeros([np.power(coord.shape[0],2), 2*(dim-1) + 1])
        for i in range(n):
            for j in range(n):
                arg = coord[i]*coord[i] + coord[j]+coord[i]
                data[j+i*n] =  np.array([coord[i], coord[j], np.cos((1./1.)*arg)])
                data[j+i*n] += np.array([coord[i], coord[j], (1./1.)*np.cos(2*arg)])
                data[j+i*n] += np.array([coord[i], coord[j], (1./1.)*np.cos(4*arg)])
                data[j+i*n] += np.array([coord[i], coord[j], (1./1.)*np.cos(8*arg)])
                data[j+i*n] += np.array([coord[i], coord[j], (1./1.)*np.cos(16*arg)])
                data[j+i*n] += np.array([coord[i], coord[j], (1./1.)*np.cos(32*arg)])
                data[j+i*n] += np.array([coord[i], coord[j], (1./1.)*np.cos(64*arg)])
    elif METHOD in ('beale',):
        coord = np.arange(-1, 1., .01)
        n = coord.shape[0]
        data = np.zeros([np.power(coord.shape[0],2), 2*(dim-1) + 1])
        for i in range(n):
            for j in range(n):
                data[j+i*n] = np.array([coord[i], coord[j], np.power(1.5   - coord[i] + coord[i]*coord[j], 2) + \
                              np.power(2.25  - coord[i] + coord[i]*np.power(coord[j], 2), 2) + \
                              np.power(2.625 - coord[i] + coord[i]*np.power(coord[j], 3), 2)])
    elif METHOD in ('rastrigin',):
        A = 10
        coord = np.arange(-2.*np.pi, 2.*np.pi, .01)
        n = coord.shape[0]
        data = np.zeros([np.power(coord.shape[0],2), 2*(dim-1) + 1])
        for i in range(n):
            for j in range(n):
                data[j+i*n] = np.array([coord[i], coord[j], A*2 + (np.power(coord[i], 2) - A*np.cos(2*np.pi*coord[i])) + (np.power(coord[j], 2) - A*np.cos(2*np.pi*coord[j]))])
    elif METHOD in ('rosenbrock',):
        A = 10
        coord = np.arange(-2, 2., .01)
        n = coord.shape[0]
        data = np.zeros([np.power(coord.shape[0],2), 2*(dim-1) + 1])
        for i in range(n):
            for j in range(n):
                data[j+i*n] = np.array([coord[i], coord[j], 100.*np.power(coord[j] - np.power(coord[i], 2), 2) + np.power(1-coord[i], 2)])
    elif METHOD in ('bukin',):
        A = 10
        coord = np.arange(-15, 15., .01)
        n = coord.shape[0]
        data = np.zeros([np.power(coord.shape[0],2), 2*(dim-1) + 1])
        for i in range(n):
            for j in range(n):
                data[j+i*n] = np.array([coord[i], coord[j], 100*np.sqrt(np.abs(coord[j] - 0.01*np.power(coord[i],2))) + 0.01*np.abs(coord[i] + 10)])
    elif METHOD in ('levi',):
        A = 10
        coord = np.arange(-5, 5., .01)
        n = coord.shape[0]
        data = np.zeros([np.power(coord.shape[0],2), 2*(dim-1) + 1])
        for i in range(n):
            for j in range(n):
                data[j+i*n] = np.array([coord[i], coord[j], np.power(np.sin(3*np.pi*coord[i]), 2) + np.power(coord[i] - 1,2)*(1 + np.power(np.sin(3*np.pi*coord[j]), 2)) + np.power(coord[j] - 1, 2)*(1 + np.power(np.sin(2*np.pi*coord[j]), 2))])
    elif METHOD in ('eggholder',):
        A = 10
        coord = np.arange(-.10, .10, .001)
        n = coord.shape[0]
        data = np.zeros([np.power(coord.shape[0],2), 2*(dim-1) + 1])
        for i in range(n):
            for j in range(n):
                data[j+i*n] = np.array([coord[i], coord[j], -(coord[j] + 47)*np.sin(np.abs(coord[i]/2. + (coord[j] + 47))) - coord[i]*np.sin(np.abs(coord[i] - (coord[j] + 47)))])
                
    if (False):
        xaxis=coord; yaxis=coord;
        xaxis,yaxis = np.meshgrid(xaxis,yaxis)
        zaxis = -(yaxis + 47)*np.sin(np.abs(xaxis/2. + (yaxis + 47))) - xaxis*np.sin(np.abs(xaxis - (yaxis + 47)))
        
        import matplotlib.pyplot as plot
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator
        
        fig, ax = plot.subplots(subplot_kw={"projection": "3d"})
        
        surf = ax.plot_surface(xaxis, yaxis, zaxis, linewidth=0, antialiased=False)
        ax.set_zlim(-1.01, 1.01)
        fig.colorbar(surf)
        plot.show()

    TRAIN_SIZE = 1000
    TEST_SIZE = 200
    X_train, X_test, y_train, y_test = train_test_split(data[:,:2], data[:,2], random_state=54, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

    np.savetxt( '{}/{}/{}_train_X.csv'.format(ROOT_DATA, MODEL_TYPE, dataset_name), X_train, delimiter=',')
    np.savetxt( '{}/{}/{}_train_y.csv'.format(ROOT_DATA, MODEL_TYPE, dataset_name), y_train, delimiter=',')
    np.savetxt( '{}/{}/{}_test_X.csv'. format(ROOT_DATA, MODEL_TYPE, dataset_name), X_test,  delimiter=',')
    np.savetxt( '{}/{}/{}_test_y.csv'. format(ROOT_DATA, MODEL_TYPE, dataset_name), y_test,  delimiter=',')
    
            

def _R2(y, yhat):
    mn = np.mean(y)
    num = np.sum(np.power(y - yhat.reshape(-1,1), 2))
    den = np.sum(np.power(y - mn, 2))
    return (1. - (num/den)).iloc[0]

def _R2_sklearn(y, yhat):
    return metrics.r2_score(y, yhat)

# Classification datasets
# dataset_name = "colic"
# dataset_name = "coil2000"
#
# Regression datasets
# dataset_name = "Regression/606_fri_c2_1000_10"
# dataset_name = "Regression/564_fried"
# dataset_name = "Regression/1199_BNG_echoMonths"
# dataset_name = "Regression/197_cpu_act"
# dataset_name = "Regression/1201_BNG_breastTumor"
# dataset_name = "Regression/1193_BNG_lowbwt"
# dataset_name = "Regression/1203_BNG_pwLinear"
# dataset_name = "Regression/529_pollen"
# _create_synthetic_cont_data(); dataset_name = "Regression/synthetic"
# X_train, y_train, X_test, y_test = _dataset(dataset_name);
_create_synthetic_disc_data(); dataset_name = "synthetic"
X_train, y_train, X_test, y_test = _dataset(dataset_name);



if __name__ == '__nowhere__':
    #############
    ## XGBOOST ##
    #############
    eta = 0.35
    objective = "reg:linear"

    reg_xgb = xgb.XGBRegressor(objective=objective,
                               eta=eta,
                               n_estimators=10,
                               seed=414)

    reg_xgb.fit(X_train, y_train)
    yhat_train_xgb = reg_xgb.predict(X_train)
    yhat_test_xgb  = reg_xgb.predict(X_test)
    R2_IS_xgb = _R2(y_train, yhat_train_xgb)
    R2_OOS_xgb = _R2(y_test, yhat_test_xgb)
    print("[{}_train]: {} x {}".format(dataset_name, X_train.shape[0], X_train.shape[1]))
    print("[{}_test]:  {} x {}".format(dataset_name, X_test.shape[0], X_test.shape[1]))
    print("[{}] XGBOOST:  R2_IS: {:2.2%} R2_OOS: {:2.2%}".format(dataset_name,
                                                                   R2_IS_xgb,
                                                                   R2_OOS_xgb))

    ##############
    ## CATBOOST ##
    ##############

    loss_function = "RMSE"
    iterations = 1000
    learning_rate = 0.25
    
    reg_cb = CatBoostRegressor(loss_function=loss_function,
                               iterations=iterations,
                               learning_rate=learning_rate,
                               verbose=False)

    reg_cb.fit(X_train, y_train)
    yhat_train_cb = reg_cb.predict(X_train)
    yhat_test_cb = reg_cb.predict(X_test)
    R2_IS_cb = _R2(y_train, yhat_train_cb)
    R2_OOS_cb = _R2(y_test, yhat_test_cb)
    print("[{}_train]: {} x {}".format(dataset_name, X_train.shape[0], X_train.shape[1]))
    print("[{}_test]:  {} x {}".format(dataset_name, X_test.shape[0], X_test.shape[1]))
    print("[{}] CATBOOST:  R2_IS: {:2.2%} R2_OOS: {:2.2%}".format(dataset_name,
                                                                  R2_IS_cb,
                                                                  R2_OOS_cb))
    ##############
    ## LIGHTGBM ##
    ##############
    params = {
        "task": "train",
        "boosting": "gbdt",
        "objective": "regression",
        "num_leaves": 10,
        "learning_rate": 0.05,
        "verbose": -1
        }
    train_lgb = lgb.Dataset(X_train, y_train)
    test_lgb = lgb.Dataset(X_test, y_test)
    reg_lgb = lgb.train(params,
                        train_set=train_lgb,
                        valid_sets=test_lgb)
    yhat_train_lgb = reg_lgb.predict(X_train)
    yhat_test_lgb = reg_lgb.predict(X_test)
    R2_IS_lgb = _R2(y_train, yhat_train_lgb)
    R2_OOS_lgb = _R2(y_test, yhat_test_lgb)
    print("[{}_train]: {} x {}".format(dataset_name, X_train.shape[0], X_train.shape[1]))
    print("[{}_test]:  {} x {}".format(dataset_name, X_test.shape[0], X_test.shape[1]))
    print("[{}] LIGHTGBM:  R2_IS: {:2.2%} R2_OOS: {:2.2%}".format(dataset_name,
                                                                  R2_IS_lgb,
                                                                  R2_OOS_lgb))

    
    
if __name__ == '__main__':
    ##############
    ## CATBOOST ##
    ##############

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
    yhat_test_cb = cls_cb.predict(X_test)
    acc_IS_cb = metrics.accuracy_score(yhat_train_cb, y_train)    
    acc_OOS_cb = metrics.accuracy_score(yhat_test_cb, y_test)
    
    print("[{}] CATBOOST:  ACC_IS: {:>4.2%} ACC_OOS: {:>4.2%}".format(dataset_name,
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
    yhat_test_xgb = cls_xgb.predict(X_test)    
    acc_IS_xgb = metrics.accuracy_score(yhat_train_xgb, y_train)
    acc_OOS_xgb = metrics.accuracy_score(yhat_test_xgb, y_test)
    
    print("[{}] XGBOOST:   ACC_IS: {:>4.2%} ACC_OOS: {:>4.2%}".format(dataset_name,
                                                                   acc_IS_xgb,
                                                                   acc_OOS_xgb))

    ##############
    ## LIGHTGBM ##
    ##############
    params = {
        "task": "train",
        "boosting": "gbdt",
        "objective": "multiclass",
        "num_class": 2,
        "num_leaves": 10,
        "learning_rate": 0.05,
        "verbose": -1
        }
    train_lgb = lgb.Dataset(X_train, y_train)
    test_lgb = lgb.Dataset(X_test, y_test)
    cls_lgb = lgb.train(params,
                        train_set=train_lgb,
                        valid_sets=test_lgb)
    yhat_train_lgb_prob = cls_lgb.predict(X_train)
    yhat_test_lgb_prob = cls_lgb.predict(X_test)
    yhat_train_lgb = np.array(yhat_train_lgb_prob[:,0] < yhat_train_lgb_prob[:,1]).astype('int')
    yhat_test_lgb = np.array(yhat_test_lgb_prob[:,0] < yhat_test_lgb_prob[:,1]).astype('int')
    acc_IS_lgb = metrics.accuracy_score(yhat_train_lgb, y_train)
    acc_OOS_lgb = metrics.accuracy_score(yhat_test_lgb, y_test)
    
    print("[{}] LIGHTGBM:  ACC_IS: {:>4.2%} ACC_OOS: {:>4.2%}".format(dataset_name,
                                                                    acc_IS_lgb,
                                                                    acc_OOS_lgb))
    

# /home/charles/src/C++/sandbox/Inductive-Boost/src/script/incremental_classifier_fit.sh 5 750 150 25 150 750 1 1 1 1 1 0.001 0.001 0.001 0.001 0.001 0.450 0.50 0.50 0.50 0.50 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 coil2000_train 50 12 2.25 1 1 1 1 -1.0 1.0 0    
# ./src/script/incremental_regressor_fit.sh 2 100 10 1 1 0.01 0.01 0.5 0.5 0 0 1 1 0 0 Regression/606_fri_c2_1000_10_train 150 12 1 1 1 1 -1.0 1.0 0

# /home/charles/src/C++/sandbox/Inductive-Boost/src/script/incremental_regressor_fit.sh 3 750 300 100 1 1 1 0.05 0.05 0.05 0.35 0.35 0.35 0 0 0 1 1 1 0 0 0 Regression/529_pollen_train 200 0 3 1.0 1 1 -1.0 1.0 0
# /home/charles/src/C++/sandbox/Inductive-Boost/src/script/incremental_regressor_fit.sh 3 750 300 100 1 1 1 0.01 0.01 0.01 0.55 0.55 0.55 0 0 0 1 1 1 0 0 0 Regression/1199_BNG_echoMonths_train 200 0 3 1.0 1 1 -1.0 1.0 0
# /home/charles/src/C++/sandbox/Inductive-Boost/src/script/incremental_regressor_fit.sh 5 750 500 300 100 20 1 1 1 1 1 0.01 0.01 0.01 0.01 0.01 0.55 0.55 0.55 0.55 0.55 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 Regression/1193_BNG_lowbwt_train 200 0 3 1.0 1 1 -1.0 1.0 0
