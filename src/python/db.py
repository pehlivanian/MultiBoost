import abc
import logging
import pandas as pd
import numpy as np
import quandl
import string
from datetime import timedelta
import datetime
import mariadb

from sqlalchemy import create_engine, Table, Column, MetaData, inspect
from contextlib import contextmanager

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

class Visitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def visit(self, element):
        pass

class Credentials(metaclass=Singleton):

    def __init__(self):
        client = MongoClient()
        
        self._coll = client.MULTISCALEGB.get_collection('credentials')

    def DB_creds(self):
        creds = self._coll.find_one()

        return creds['TS_DB']['user'], creds['TS_DB']['passwd']

class DBExt(object):
    def __init__(self, dbName=None):
        self.credentials = Credentials()
        user, passwd = self.credentials.DB_creds()
        dbName = dbName or 'MULTISCALEGB_CLASS'
        self.conn = create_engine('mysql+mysqldb://{}:{}@localhost/{}'.format(user,passwd,dbName))        

    def list_databases(self):
        insp = inspect(self.conn)
        return insp.get_schema_names()

    def list_tables(self, dbName):
        return self.conn.table_names()        
        return self.conn_table_names()

    def get_top_n_OOS(self, n=5, use_min_criteria=True):
        if use_min_criteria:
            agg_fn = {'err' : min}
        else:
            agg_fn = {'err' : 'last'}
        req = self.conn.execute('select * from outofsample')
        df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        groups = df.groupby(by=['dataset_name', 'run_key'], as_index=False).aggregate(agg_fn)
        groups = groups.sort_values(by=['dataset_name', 'err'])
        groups = groups.groupby(by=['dataset_name'], as_index=False).head(n)
        req = self.conn.execute('select run_key,recursive from run_specification')
        run_specs = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        joined = pd.merge(groups, run_specs, left_on=['run_key'], right_on=['run_key'], how='left')
        return joined 

    def get_complete_OOS_run(self, run_key):
        req = self.conn.execute('select * from outofsample where run_key={}'.format(run_key))
        df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        return df

    def get_complete_IS_run(self, run_key):
        req = self.conn.execute('select * from insample where run_key={}'.format(run_key))
        df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        return df    

    def get_top_n_run_specifications(self, n=5):
        df = self.get_top_n_OOS(n)
        run_specs = pd.DataFrame()
        for i,r in df.iterrows():
            ddf,_ = self.get_run_specification(r.run_key)
            run_specs = pd.DataFrame.append(run_specs,ddf)
        return run_specs

    def get_run_specification(self, run_key):
        req = self.conn.execute('select * from run_specification where run_key={}'.format(run_key))
        df = pd.DataFrame(columns=req.keys(), data=req.fetchall())

        num_partitions = np.zeros(10)
        num_steps      = np.zeros(10)
        learning_rate  = np.zeros(10)
        max_depth      = np.zeros(10)
        min_leafsize   = np.zeros(10)
        min_gainsplit  = np.zeros(10)
        for i in range(0,10):
            num_partitions[i] = df.iloc[0].get('num_partitions{}'.format(i))
            num_steps[i]      = df.iloc[0].get('num_steps{}'.format(i))
            learning_rate[i]  = df.iloc[0].get('learning_rate{}'.format(i))
            max_depth[i]      = df.iloc[0].get('max_depth{}'.format(i))
            min_leafsize[i]   = df.iloc[0].get('min_leafsize{}'.format(i))
            min_gainsplit[i]  = df.iloc[0].get('min_gainsplit{}'.format(i))
        params = np.array([num_partitions, num_steps, learning_rate, max_depth,
                           min_leafsize, min_gainsplit])
        return df,params

               
