import abc
import logging
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
import string
from datetime import timedelta
from collections import defaultdict
import datetime

from sqlalchemy import create_engine, Table, Column, MetaData, inspect, text
from sqlalchemy.orm import Session
from contextlib import contextmanager

import matplotlib.pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages

from pymongo import MongoClient
from sqlalchemy import create_engine, Table, Column, MetaData
from sqlalchemy_utils import database_exists, create_database
from contextlib import contextmanager

# Duplicated from utils
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.offsets import CustomBusinessMonthBegin

US_Federal_Calendar = USFederalHolidayCalendar()
bmth_us = CustomBusinessMonthBegin(calendar=US_Federal_Calendar)
bday_us = CustomBusinessDay(calendar=US_Federal_Calendar)

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)

# Suppress scientific notation in console display, etc.
np.set_printoptions(suppress=True, edgeitems=30, linewidth=10000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

Dtype_Mapping = {
    'object' : 'TEXT',
    'int64'  : 'INT',
    'float64' : 'FLOAT',
    'datetime64' : 'DATETIME',
    'bool' : 'TINYINT',
    'category' : 'TEXT',
    'timedelta[ns]' : 'TEXT'
    }

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Credentials(metaclass=Singleton):

    def __init__(self):
        # client = MongoClient()
        
        # self._coll = client.MULTISCALEGB.get_collection('credentials')
        pass

    def DB_creds(self):
        # creds = self._coll.find_one()

        # return creds['TS_DB']['user'], creds['TS_DB']['passwd']
        return "charles", "gongzuo"

class DBExt(object):
    def __init__(self, dbName='MULTISCALEGB_CLASS'):
        self.credentials = Credentials()
        user, passwd = self.credentials.DB_creds()
        dbName = dbName or ('MULTISCALEGB_CLASS' if is_classifier else 'MULTISCALEGB_REG')
        self.engine = create_engine('mysql+mysqldb://{}:{}@localhost/{}'.format(user,passwd,dbName))
        self.conn = Session(self.engine)
        self.insp  = inspect(self.engine)
        
    def list_databases(self):
        return self.insp.get_schema_names()

    def list_tables(self, schema):
        return self.issp.get_table_names(schema=schema)

    def execute(self, q):
        req = self.conn.execute(text(q))
        df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        return df

if __name__ == "__main__":
    dbext = DBExt()
    # dataset_name = "synthetic_case_1_train"
    # dataset_name = "buggyCrx_train"
    
    # dataset_name = "adult_train"
    # dataset_name = "magic_train"
    # dataset_name = "synthetic_case_0_train"
    # dataset_name = "spambase_train"
    # dataset_name = "magic_reverse_train"
    # dataset_name = "adult_reverse_train"
    # dataset_name = "magic_wtd_priority_train"
    # dataset_name = "magic_cum_wtd_priority_train"
    # dataset_name = "magic_cum_wtd_priority_a2_b2_train"
    # dataset_name = "magic_cum_wtd_priority_b2_a2_train"
    # dataset_name = "magic_cum_wtd_priority_a2_over_b2_train"
    dataset_name = "magic_cum_wtd_priority_a2_over_b2_sym_train"
    
    
    sql_oos = text("""select * from outofsample where dataset_name = :name_""")
    req_oos = dbext.conn.execute(sql_oos, {"name_": dataset_name})
    df_oos = pd.DataFrame(columns=req_oos.keys(), data=req_oos.fetchall())

    sql_run = text("""select * from run_specification where dataset_name = :name_""")
    req_run = dbext.conn.execute(sql_run, {"name_": dataset_name})
    df_run = pd.DataFrame(columns=req_run.keys(), data=req_run.fetchall())

    merged = pd.merge(df_oos, df_run, left_on=['run_key'], right_on=['run_key'], how='left')
    merged = merged[merged['active_partition_ratio0'] != 1.0]

    grouped = merged.groupby(by=['active_partition_ratio0'], as_index=False)['err'].mean()

    plot.plot(grouped.active_partition_ratio0[:-1], grouped.err[:-1])
    plot.xlabel('active_partition_ratio')
    plot.ylabel('error')
    plot.title('Error by active partition ratio')
    plot.show()

    grouped = merged.groupby(by=['learning_rate0'], as_index=False)['err'].mean()
    plot.plot(grouped.learning_rate0[:-1], grouped.err[:-1])
    plot.xlabel('learning_rate')
    plot.ylabel('error')
    plot.title('Error by active learning rate')
    plot.show()

    optimal_learning_rate0 = grouped[grouped['err'] == grouped['err'].min()].iloc[0].learning_rate0
    
    

    
