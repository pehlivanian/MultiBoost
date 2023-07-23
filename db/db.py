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
    def __init__(self):
        self.credentials = Credentials()
        user, passwd = self.credentials.DB_creds()
        self.conn = create_engine('mysql+mysqldb://{}:{}@localhost'.format(user,passwd))        

    def list_databases(self):
        insp = inspect(self.conn)
        return insp.get_schema_names()

    def list_tables(self, dbName):
        user, passwd = self.credentials.DB_creds()
        conn = create_engine('mysql+mysqldb://{}:{}@localhost/{}'.format(user, passwd, dbName))
        return conn.table_names()        
        return self.conn_table_names()

