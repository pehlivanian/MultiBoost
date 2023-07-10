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
                

        
