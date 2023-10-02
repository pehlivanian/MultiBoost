import abc
import logging
import pandas as pd
import numpy as np
import string
from datetime import timedelta
from collections import defaultdict
import datetime

from sqlalchemy import create_engine, Table, Column, MetaData, inspect, text
from sqlalchemy.orm import Session
from contextlib import contextmanager

import matplotlib.pyplot as plot

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
    def __init__(self, dbName=None, is_classifier=True):
        self.is_classifier = is_classifier
        self.credentials = Credentials()
        user, passwd = self.credentials.DB_creds()
        dbName = dbName or ('MULTISCALEGB_CLASS' if is_classifier else 'MULTISCALEGB_REG')
        self.engine = create_engine('mysql+mysqldb://{}:{}@localhost/{}'.format(user,passwd,dbName))
        self.conn = Session(self.engine)

    def list_databases(self):
        insp = inspect(self.conn)
        return insp.get_schema_names()

    def list_tables(self, dbName):
        return self.conn.table_names()        
        return self.conn_table_names()

    @staticmethod
    def last_min_occ(df, colName):
        return df[colName][df[colName] == df[colName].min()].index.tolist().pop()

    @staticmethod
    def last_max_occ(df, colName):
        return df[colName][df[colName] == df[colName].max()].index.tolist().pop()
    
    def get_top_n_OOS(self, n=5, criterion='min_IS'):
        if self.is_classifier:
            sort_keys = ['dataset_name', 'err']
            sort_ascending=True
            if use_min_criterion:
                agg_fn = {'err' : min}
            else:
                agg_fn = {'err' : 'last'}
        else:
            sort_keys = ['dataset_name', 'r2']
            sort_ascending=False
            if use_min_criterion:
                agg_fn = {'r2' : min}
            else:
                agg_fn = {'r2' : 'last'}
            
        req = self.conn.execute(text('select * from outofsample'))
        df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        if criterion in ('last', 'min_OOS'):
            groups = df.groupby(by=['dataset_name', 'run_key'], as_index=False).aggregate(agg_fn)
            groups = groups.sort_values(by=sort_keys, ascending=sort_ascending)
            grouped = groups.groupby(by=['dataset_name'], as_index=False).head(n)
        elif criterion in ('min_IS'):
            groups = df.groupby(by=['dataset_name', 'run_key'], as_index=False)
            if self.is_classifier:
                grouped = pd.DataFrame(columns=['dataset_name', 'run_key', 'IS_best_ind', 'err'])
            else:
                grouped = pd.DataFrame(columns=['dataset_name', 'run_key', 'IS_best_ind', 'loss'])
            req = self.conn.execute(text('select * from insample'))
            df_is = pd.DataFrame(columns=req.keys(), data=req.fetchall())
            for (dataset_name, run_key),group in groups:
                df2 = df_is[df_is['run_key'] == run_key]
                df2.reset_index(inplace=True)
                # ind = DBExt.last_min_occ(df2, 'err')
                ind = DBExt.last_max_occ(df2, 'F1')
                grouped.loc[grouped.shape[0]] = [dataset_name, run_key, ind, group.iloc[ind].err]
            grouped = grouped.sort_values(by=sort_keys, ascending=sort_ascending)
            grouped = grouped.groupby(by=['dataset_name'], as_index=False).head(n)
        
        req = self.conn.execute(text('select run_key,rcsive from run_specification'))
        run_specs = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        joined = pd.merge(grouped, run_specs, left_on=['run_key'], right_on=['run_key'], how='left')
        return joined 

    def get_complete_OOS_run(self, run_key):
        req = self.conn.execute(text('select * from outofsample where run_key={}'.format(run_key)))
        df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        return df

    def get_complete_IS_run(self, run_key):
        req = self.conn.execute(text('select * from insample where run_key={}'.format(run_key)))
        df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        return df    

    def get_top_n_run_specifications(self, n=5, criterion='min_IS'):
        dd = defaultdict(dict)
        df = self.get_top_n_OOS(n, criterion=criterion)
        run_specs = pd.DataFrame()
        for i,r in df.iterrows():
            ddf,params = self.get_run_specification(r.run_key)
            run_specs = pd.DataFrame.append(run_specs,ddf)
            dd[r.dataset_name][r.run_key] = params
        return df,run_specs,dd

    def get_run_specification(self, run_key):
        req = self.conn.execute(text('select * from run_specification where run_key={}'.format(run_key)))
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

    def plot_part_v_perf(self, dataset_name="Regression/195_auto_price"):
        if self.is_classifier:
            req = self.conn.execute(text('select dataset_name, run_key, min(err) as err from outofsample group by run_key'))
        else:
            req = self.conn.execute(text('select dataset_name, run_key, max(r2) as r2 \
            from outofsample group by run_key'))
        df_oos = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        req = self.conn.execute(text('select dataset_name, run_key, num_partitions0, \
            num_steps0, learning_rate0 from run_specification where num_partitions1 = 0'))
        df_rs = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        joined = pd.merge(df_oos, df_rs, left_on=['dataset_name', 'run_key'], right_on=['dataset_name', 'run_key'],
                          how='inner')
        joined = joined[joined['dataset_name'] == dataset_name]
        if self.is_classifier:
            joined = joined
        else:
            if dataset_name not in ("Regression/1199_BNG_echoMonths_train",):
                joined = joined[joined['r2'] > .5]
            else:
                joined = joined[joined['r2'] > 0.]
        # XXX
        joined = joined[joined['learning_rate0'] < 0.05]
        xaxis = joined['num_partitions0'].values
        if self.is_classifier:
            yaxis = joined['err'].values
        else:
            yaxis = joined['r2'].values
        return xaxis,yaxis

    def plot_part_v_r2_2dim(self, dataset_name="Regression/195_auto_price", random_only=False):
        req = self.conn.execute(text('select dataset_name, run_key, max(r2) as r2 \
            from outofsample group by run_key'))
        df_oos = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        req = self.conn.execute(text('select dataset_name, run_key, num_partitions0, num_partitions1, num_partitions2, \
            learning_rate0, learning_rate1, learning_rate2 from run_specification where rcsive=1'))
        df_rs = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        joined = pd.merge(df_oos, df_rs, left_on=['dataset_name', 'run_key'], right_on=['dataset_name', 'run_key'],
                          how='inner')
        joined = joined[joined['dataset_name'] == dataset_name]
        joined = joined[(joined['num_partitions0'] > 0) & (joined['num_partitions1'] > 0)]
        joined = joined[(joined['learning_rate0'] == 0.10) & (joined['learning_rate1'] == 0.10)]
        joined = joined[joined['r2'] > .5]
        joined = joined[(joined['num_partitions0'] == 28) & (joined['num_partitions1'] > 0)]
        # joined = joined[(joined['num_partitions0'] == 100) & (joined['num_partitions1'] > 0)]
        # joined = joined[(joined['num_partitions0'] == 30) & (joined['num_partitions1'] > 0)]
        if random_only:
            joined = joined[(joined['num_partitions0'] != 30) | (joined['num_partitions0'] != 90)]
        xaxis = joined['num_partitions1'].values
        yaxis = joined['r2'].values
        return xaxis,yaxis

    def plot_part_v_r2_3dim(self, dataset_name="Regression/195_auto_price", random_only=False):
        req = self.conn.execute(text('select dataset_name, run_key, max(r2) as r2 \
            from outofsample group by run_key'))
        df_oos = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        req = self.conn.execute(text('select dataset_name, run_key, num_partitions0, num_partitions1, num_partitions2, \
            learning_rate0, learning_rate1, learning_rate2 from run_specification where rcsive=1'))
        df_rs = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        joined = pd.merge(df_oos, df_rs, left_on=['dataset_name', 'run_key'], right_on=['dataset_name', 'run_key'],
                          how='inner')
        joined = joined[joined['dataset_name'] == dataset_name]
        # joined = joined[(joined['num_partitions0'] > 0) & (joined['num_partitions1'] > 0) & (joined['num_partitions2'] > 0)]
        if random_only:
            joined = joined[(joined['num_partitions0'] != 30) | (joined['num_partitions0'] != 40)]
        else:
            joined = joined[(joined['num_partitions0'] == 40) & (joined['num_partitions1'] == 4)]
        joined = joined[joined['r2'] > .5]
        xaxis = joined['num_partitions2'].values
        yaxis = joined['r2'].values
        return xaxis,yaxis

    def plot_part_v_r2_random(self, dataset_name="Regression/195_auto_price"):
        req = self.conn.execute(text('select dataset_name, run_key, max(r2) as r2 \
            from outofsample group by run_key'))
        df_oos = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        req = self.conn.execute(text('select dataset_name, run_key, num_partitions0, num_partitions1, num_partitions2, \
            learning_rate0, learning_rate1, learning_rate2 from run_specification where rcsive=1'))
        df_rs = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        joined = pd.merge(df_oos, df_rs, left_on=['dataset_name', 'run_key'], right_on=['dataset_name', 'run_key'],
                          how='inner')
        joined = joined[joined['dataset_name'] == dataset_name]
        joined = joined[(joined['num_partitions0'] > 0) & (joined['num_partitions1'] > 0) & (joined['num_partitions2'] > 0)]
        joined = joined[(joined['num_partitions0'] != 30) & (joined['num_partitions0'] != 90)]
        joined = joined[joined['r2'] > .5]
        xaxis = joined['num_partitions0'].values
        yaxis = joined['r2'].values
        return xaxis,yaxis

    def plot_OOS_fits(self, run_keys):
        xaxis = np.array([]); erraxis = np.array([]); F1axis = np.array([])
        for run_key in run_keys:
            req = self.conn.execute(text('select iteration, err, prcsn, recall, F1 from outofsample where run_key={}'.format(
                run_key)))
            df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
            xaxis = xaxis if xaxis.shape[0] > 0 else df.iteration.values
            erraxis = np.concatenate([erraxis, df.err.values.reshape(1,-1)]) if erraxis.shape[0] > 0 else df.err.values.reshape(1,-1)
            F1axis = np.concatenate([F1axis, df.F1.values.reshape(1,-1)]) if F1axis.shape[0] > 0 else df.F1.values.reshape(1,-1)

        # plot
        fig, ax1 = plot.subplots()

        colors = ['tab:red', 'tab:green', 'tab:orange', 'tab:blue']
        labels = ['exp loss', 'binom dev loss', 'square loss', 'synth loss']
        linestyles = {'dense dashdot': (0, (3, 1, 1, 1))}
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Out of sample error')
        ax1.set_title('Out of sample convergence [income]')
        ax1.tick_params(axis='y')
        for i,_ in enumerate(run_keys):
            ax1.plot(xaxis, erraxis[i,:],
                     linestyle='solid',
                     color=colors[i],
                     marker='.',
                     linewidth=1.0,
                     label=labels[i])
        ax1.legend()
        ax1.grid(True)
        
        # ax2 = ax1.twinx()
        
        # color = 'tab:green'
        # ax2.set_ylabel('F1', color=color)
        # for i,_ in enumerate(run_keys):
        #     ax2.plot(xaxis, F1axis[i,:], color=color)
        # ax2.tick_params(axis='y', labelcolor=color)

        # fig.tight_layout()
        plot.show()

if __name__ == "__main__":
    # dbext = DBExt(is_classifier=False);
    # xaxis,yaxis = dbext.plot_part_v_r2("Regression/195_auto_price")
    # xaxis,yaxis = dbext.plot_part_v_r2("Regression/207_autoPrice")
    # xaxis,yaxis = dbext.plot_part_v_r2("Regression/584_fri_c4_500_25")
    # xaxis,yaxis = dbext.plot_part_v_r2("Regression/606_fri_c2_1000_10_train")
    # xaxis,yaxis = dbext.plot_part_v_r2("Regression/529_pollen_train")    
    # xaxis,yaxis = dbext.plot_part_v_r2("Regression/197_cpu_act_train")
    # xaxis,yaxis = dbext.plot_part_v_r2("Regression/1199_BNG_echoMonths_train")
    # xaxis,yaxis = dbext.plot_part_v_r2("Regression/564_fried_train")            
    # xaxis,yaxis = dbext.plot_part_v_r2("Regression/529_pollen_train")    
    # xaxis,yaxis = dbext.plot_part_v_r2_2dim("Regression/195_auto_price")
    # xaxis,yaxis = dbext.plot_part_v_r2_2dim("Regression/584_fri_c4_500_25")    
    # xaxis,yaxis = dbext.plot_part_v_r2_3dim("Regression/195_auto_price", random_only=False)
    # xaxis,yaxis = dbext.plot_part_v_r2_random("Regression/195_auto_price")
    # xaxis,yaxis = dbext.plot_part_v_r2_random("Regression/207_autoPrice")

    dbext = DBExt(is_classifier=True)
    # xaxis,yaxis = dbext.plot_part_v_perf("GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1_train")
    # xaxis,yaxis = dbext.plot_part_v_perf("GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1_train")
    # xaxis,yaxis = dbext.plot_part_v_perf("flare_train")
    # xaxis,yaxis = dbext.plot_part_v_perf("phoneme_train")
    
    # spambase_train
    # dbext.plot_OOS_fits([13784078293975702669, 9187967397797350836, 15767095704017143287, 12793747121298945138])
    # income_small_train
    # dbext.plot_OOS_fits([7533277102691265751, 8251155077064940421, 7400247794797131665, 12836108145017863223])
    # phoneme
    # dbext.plot_OOS_fits([8705424173097098323, 2421851584358366385, 1820609185731366400, 1998113784327814407])
    # flare
    # dbext.plot_OOS_fits([1917161425195422475, 1274523705138131147, 197195856475585082, 7986938987171996673])
    # GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1_train
    dbext.plot_OOS_fits([12806597599666742865, 18329239330487574321, 8061260327215002817, 15890096115503869579])
    
    # sortind = [x[0] for x in sorted(enumerate(xaxis), key=lambda x: x[1])]
    # xaxis = xaxis[[sortind]]
    # yaxis = yaxis[[sortind]]
    # plot.scatter(xaxis,yaxis)
    # plot.plot(xaxis, yaxis)
    # plot.show()

    
