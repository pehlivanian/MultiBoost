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
        np.savetxt('{}open_ml/tabular/classification/mixed/{}_X.csv'.format(ROOT_DATA, dataset.name), X, delimiter=',')
    elif SUITE_ID in (335,):
        np.savetxt('{}open_ml/tabular/regression/mixed/{}_X.csv'.format(ROOT_DATA, dataset.name), X, delimiter=',')
    elif SUITE_ID in (336,):
        np.savetxt('{}open_ml/tabular/regression/{}_y.csv'.format(ROOT_DATA, dataset.name), y, delimiter=',')
    elif SUITE_ID in (337,):
        np.savetxt('{}open_ml/tabular/classification/{}_X.csv'.format(ROOT_DATA, dataset.name), X, delimiter=',')
        
    print('FINISHED')


#################
# pmlb datasets #
#################
DATA_TYPE = "classification"
# DATA_TYPE = "regression"

import pmlb
from pmlb import classification_dataset_names, regression_dataset_names
class_datasets = set()
if DATA_TYPE in ("classification",):
    dataset_names = classification_dataset_names
else:
    dataset_names = regression_dataset_names
for dataset_name in dataset_names:
    if dataset_name not in ('1191_BNG_pbc','1196_BNG_pharynx', '1595_poker'): # too large?
        print('LOADING {}'.format(dataset_name))
        X,y = pmlb.fetch_data(dataset_name, return_X_y=True)
        print('{} x {}'.format(X.shape[0], X.shape[1]))
        if X.shape[0] < 100:
            continue
        if len(np.unique(y)) > 100:
            if DATA_TYPE in ("classification",):
                np.savetxt('{}pmlb/classification/{}_X.csv'.format(ROOT_DATA, dataset_name), X, delimiter=',')
            elif DATA_TYPE in ("regression",):
                np.savetxt('{}pmlb/regression/{}_y.csv'.format(ROOT_DATA, dataset_name), y, delimiter=',')
        print(dataset_name)

        
