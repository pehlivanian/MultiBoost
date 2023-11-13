import os
import openml
import numpy as np
import pandas as pd

ROOT_DATA = "/home/charles/Data/"

################################
# OPENML tabular data datasets #
################################
# SUITE_ID = 337 # classification on numerical features
SUITE_ID = 334 # classification on numerical and categorical features
# SUITE_ID = 336 # regression on numerical features
# SUITE_ID = 335 # regression on numerical and categorical features


benchmark_suite = openml.study.get_suite(SUITE_ID)
for task_id in benchmark_suite.tasks:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    print('PROCESSING: {}'.format(dataset.name), end="... ")        
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    if SUITE_ID in (334,):
        subPath = "open_ml/classification_mixed"
        absPath = os.path.join(ROOT_DATA, subPath)
    elif SUITE_ID in (335,):
        subPath = "open_ml/regression_mixed"
        absPath = os.path.join(ROOT_DATA, subPath)        
    elif SUITE_ID in (336,):
        subPath = "open_ml/regression"
        absPath = os.path.join(ROOT_DATA, subPath)        
    elif SUITE_ID in (337,):
        subPath = "open_ml/classification"
        absPath = os.path.join(ROOT_DATA, subPath)
    if not os.path.exists(absPath):
        os.makedirs(absPath)
    np.savetxt(os.path.join(absPath, '{}_X.csv'.format(dataset.name)), X, delimiter=',')
    np.savetxt(os.path.join(absPath, '{}_y.csv'.format(dataset.name)), y, delimiter=',')
    print('COMPLETE: ==> {}'.format(absPath))

#################
# pmlb datasets #
#################

DATA_TYPES = ("classification", "regression")

import pmlb
from pmlb import classification_dataset_names, regression_dataset_names


for DATA_TYPE in DATA_TYPES:
    class_datasets = set()
    if DATA_TYPE in ("classification",):
        dataset_names = classification_dataset_names
    else:
        dataset_names = regression_dataset_names
    for dataset_name in dataset_names:
        if dataset_name not in ('1191_BNG_pbc','1196_BNG_pharynx', '1595_poker'): # too large?
            print('PROCESSING: {}'.format(dataset_name), end='... ')
            X,y = pmlb.fetch_data(dataset_name, return_X_y=True)
            if X.shape[0] < 100:
                continue
            if DATA_TYPE == 'classification' or len(np.unique(y)) > 100:
                if DATA_TYPE in ("classification",):
                    subPath = "pmlb/classification"
                    absPath = os.path.join(ROOT_DATA, subPath)
                elif DATA_TYPE in ("regression",):
                    subPath = "pmlb.regression"
                    absPath = os.path.join(ROOT_DATA, subPath)

        if not os.path.exists(absPath):
            os.makedirs(absPath)
                
        np.savetxt(os.path.join(absPath, '{}_X.csv'.format(dataset_name)), X, delimiter=',')                
        np.savetxt(os.path.join(absPath, '{}_y.csv'.format(dataset_name)), y, delimiter=',')
        print('COMPLETE: ==> {}'.format(absPath))
                

        
###############
# single case #
###############

import pmlb
import numpy as np
import pandas as pd
from pmlb import classification_dataset_names, regression_dataset_names
from sklearn.model_selection import train_test_split

ROOT_DATA = "/home/charles/Data/"
# MODEL_TYPE = "Classification"
MODEL_TYPE = "Regression"

# dataset_name = "606_fri_c2_1000_10"
# dataset_name = "529_pollen"
# dataset_name = "197_cpu_act"
# dataset_name = "1199_BNG_echoMonths"
# dataset_name = "1030_ERA"
# dataset_name = "564_fried"
# dataset_name = "1193_BNG_lowbwt"
# dataset_name = "1201_BNG_breastTumor"
dataset_name = "1203_BNG_pwLinear"

# dataset_name = "spambase"
# dataset_name = "backache"
# dataset_name = "GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1"
# dataset_name = "GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1"
# dataset_name = "flare"
# dataset_name = "income"
# dataset_name = "australian"
# dataset_name = "german"
# dataset_name = "breast_cancer_wisconsin"
# dataset_name = "hypothyroid"
# dataset_name = "coil2000"
# dataset_name = "adult"
# dataset_name = "income_2000"
# dataset_name = "house_votes_84"
# dataset_name = "colic"
# dataset_name = "buggyCrx"
# dataset_name = "ring"

# X = pd.read_csv("/home/charles/Data/income_train_X.csv")
# y = pd.read_csv("/home/charles/Data/income_train_y.csv")
X,y = pmlb.fetch_data(dataset_name, return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# X = X_train; y = y_train

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=44, train_size=800, test_size=200)


if MODEL_TYPE in ("Regression", ):
    np.savetxt( '{}/{}/{}_train_X.csv'.format(ROOT_DATA, MODEL_TYPE, dataset_name), X_train, delimiter=',')
    np.savetxt( '{}/{}/{}_train_y.csv'.format(ROOT_DATA, MODEL_TYPE, dataset_name), y_train, delimiter=',')
    np.savetxt( '{}/{}/{}_test_X.csv'. format(ROOT_DATA, MODEL_TYPE, dataset_name), X_test,  delimiter=',')
    np.savetxt( '{}/{}/{}_test_y.csv'. format(ROOT_DATA, MODEL_TYPE, dataset_name), y_test,  delimiter=',')
else:
    np.savetxt( '{}/{}_train_X.csv'.format(ROOT_DATA, dataset_name), X_train, delimiter=',')
    np.savetxt( '{}/{}_train_y.csv'.format(ROOT_DATA, dataset_name), y_train, delimiter=',')
    np.savetxt( '{}/{}_test_X.csv'. format(ROOT_DATA, dataset_name), X_test,  delimiter=',')
    np.savetxt( '{}/{}_test_y.csv'. format(ROOT_DATA, dataset_name), y_test,  delimiter=',')    
