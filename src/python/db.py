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

class Visitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def visit(self, element):
        pass

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
            req = self.conn.execute(text('select dataset_name, run_key, err from outofsample'))
        else:
            req = self.conn.execute(text('select dataset_name, run_key, max(r2) as r2 \
            from outofsample group by run_key'))
        df_oos = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        req = self.conn.execute(text('select dataset_name, run_key, num_partitions0, \
            num_steps0, learning_rate0 from run_specification where loss_fn = 3 and num_partitions1 = 0'))
        # req = self.conn.execute(text('select dataset_name, run_key, num_partitions0, \
        #     num_steps0, learning_rate0, num_partitions1 from run_specification where \
        #     loss_fn = 3 and num_partitions1 > 0'))
        # req = self.conn.execute(text('select dataset_name, run_key, num_partitions0, \
        #     num_steps0, learning_rate0, num_partitions1, num_partitions2 from \
        #     run_specification where loss_fn = 3 and num_partitions2 > 0'))
        df_rs = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        joined = pd.merge(df_oos, df_rs, left_on=['dataset_name', 'run_key'], right_on=['dataset_name', 'run_key'],
                          how='inner')
        joined = joined[joined['dataset_name'] == dataset_name]

        # process
        joined = joined[joined['num_partitions0'] >= 10]
        # joined = joined[joined['num_partitions1'] >= 10]
        # joined = joined[joined['num_partitions2'] >= 10]
        joined = joined.drop_duplicates(subset=['num_partitions0'])
        # joined = joined.drop_duplicates(subset=['num_partitions0', 'num_partitions1'])
        # joined = joined.drop_duplicates(subset=['num_partitions0', 'num_partitions1', 'num_partitions2'])        
        
        if self.is_classifier:
            joined = joined
        else:
            if dataset_name not in ("Regression/1199_BNG_echoMonths_train",):
                joined = joined[joined['r2'] > .5]
            else:
                joined = joined[joined['r2'] > 0.]

        xaxis = joined['num_partitions0'].values
        # xaxis = joined['num_partitions1'].values
        # xaxis = joined['num_partitions0'].values + joined['num_partitions1'].values + joined['num_partitions2'].values
        if self.is_classifier:
            yaxis = joined['err'].values
        else:
            yaxis = joined['r2'].values
        return xaxis,yaxis

    def plot_OOS_simple(self, dataset, with_priority=True):
        # q = "select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where B.loss_fn = 12 and B.num_partitions0 = 1000 and A.dataset_name = \"{}\" and B.num_partitions1 = 100 and B.basesteps = 50 and B.active_partition_ratio0 = .50 group by A.run_key order by B.loss_power".format(dataset)
        q ="select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where A.dataset_name = \"{}\" group by A.run_key order by B.loss_power".format(dataset)

        fig = plot.figure()
        fig.set_size_inches(9, 6)
        fig.subplots_adjust(left=0.075)
        fig.subplots_adjust(right=0.925)

        axis = fig.add_subplot(211)
        i = 4; level = 2
        req = self.conn.execute(text(q))
        df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        xaxis = df[(df['loss_power'] > 0.5) & (df['loss_power'] <= 5.0)]['loss_power'].values
        yaxis = df[(df['loss_power'] > 0.5) & (df['loss_power'] <= 5.0)]['err'].values
        
        b, m, c = polyfit(xaxis, yaxis, 2)
        opt_x = -m/2/c
        opt_y = b+m*opt_x+c*np.power(opt_x,2)
        fitaxis = b + m*xaxis + c*np.power(xaxis,2)
        DBExt._add_plot(dataset, xaxis, yaxis, fitaxis, opt_x, opt_y, level, axis, metric="error (%)")
        # plot.show()

        axis = fig.add_subplot(212)
        i = 4; level = 2
        req = self.conn.execute(text(q))
        df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        xaxis = df[(df['loss_power'] > 0.5) & (df['loss_power'] <= 5.0)]['loss_power'].values
        yaxis = df[(df['loss_power'] > 0.5) & (df['loss_power'] <= 5.0)]['prcsn'].values
        
        b, m, c = polyfit(xaxis, yaxis, 2)
        opt_x = -m/2/c
        opt_y = b+m*opt_x+c*np.power(opt_x,2)
        fitaxis = b + m*xaxis + c*np.power(xaxis,2)
        DBExt._add_plot(dataset, xaxis, yaxis, fitaxis, opt_x, opt_y, level, axis, metric="precision")
        # plot.show()


        filename = "Summary_by_p_{}.pdf".format(dataset)
        with PdfPages(filename) as pdf:
            pdf.savefig(fig)
        filename = "Summary_by_p_{}.png".format(dataset)
        fig.savefig(filename)

        
    def plot_OOS_by_power(self, dataset, with_priority=True):
        if with_priority:
            q1 = "select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where B.loss_fn = 12 and B.num_partitions0 > 0 and A.dataset_name = \"{}\" and B.num_partitions1 = 0 and B.basesteps = 25 and B.active_partition_ratio0 > .1 group by A.run_key order by B.loss_power".format(dataset)
            q3 = "select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where B.loss_fn = 12 and B.num_partitions0 > 0 and A.dataset_name = \"{}\" and B.num_partitions2 > 0 and B.num_partitions4 = 0 and B.basesteps = 25 and B.active_partition_ratio0 > .1 group by A.run_key order by B.loss_power".format(dataset)
            q5 = "select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where B.loss_fn = 12 and B.num_partitions0 > 0 and A.dataset_name = \"{}\" and B.num_partitions4 > 0 and B.num_partitions6 = 0 and B.basesteps = 25 and B.active_partition_ratio0 > .1 group by A.run_key order by B.loss_power".format(dataset)
            q7 = "select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where B.loss_fn = 12 and B.num_partitions0 > 0 and A.dataset_name = \"{}\" and B.num_partitions6 > 0 and B.num_partitions8 = 0 and B.basesteps = 25 and B.active_partition_ratio0 > .1 group by A.run_key order by B.loss_power".format(dataset)
            q9 = "select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where B.loss_fn = 12 and B.num_partitions0 > 0 and A.dataset_name = \"{}\" and B.num_partitions8 > 0 and B.basesteps = 25 and B.active_partition_ratio0 > .1 group by A.run_key order by B.loss_power".format(dataset)
        else:
            q1 = "select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where B.loss_fn = 12 and B.num_partitions0 > 0 and A.dataset_name = \"{}\" and B.num_partitions1 = 0 and B.basesteps = 25 and B.active_partition_ratio0 < 0.05 group by A.run_key order by B.loss_power".format(dataset)
            q3 = "select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where B.loss_fn = 12 and B.num_partitions0 > 0 and A.dataset_name = \"{}\" and B.num_partitions2 > 0 and B.num_partitions4 = 0 and B.basesteps = 25 and B.active_partition_ratio0 < 0.05 group by A.run_key order by B.loss_power".format(dataset)
            q5 = "select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where B.loss_fn = 12 and B.num_partitions0 > 0 and A.dataset_name = \"{}\" and B.num_partitions4 > 0 and B.num_partitions6 = 0 and B.basesteps = 25 and B.active_partition_ratio0 < 0.05 group by A.run_key order by B.loss_power".format(dataset)
            q7 = "select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where B.loss_fn = 12 and B.num_partitions0 > 0 and A.dataset_name = \"{}\" and B.num_partitions6 > 0 and B.num_partitions8 = 0 and B.basesteps = 25 and B.active_partition_ratio0 < 0.05 group by A.run_key order by B.loss_power".format(dataset)
            q9 = "select A.dataset_name, A.run_key, A.err as err, A.prcsn as prcsn, A.F1 as F1, B.loss_fn, B.loss_power, B.num_partitions0, B.num_partitions1, B.num_partitions2, B.learning_rate0, B.learning_rate1, B.learning_rate2, B.num_steps0, B.num_steps1, B.num_steps2 from outofsample A join run_specification B on A.run_key=B.run_key where B.loss_fn = 12 and B.num_partitions0 > 0 and A.dataset_name = \"{}\" and B.num_partitions8 > 0 and B.basesteps = 25 and B.active_partition_ratio0 < 0.05 group by A.run_key order by B.loss_power".format(dataset)
        qs = [q1,q3,q5,q7]

        if with_priority:
            path = 'recursive_{}_priority_level_{}.pdf'
        else:
            path = 'recursive_{}_level_{}.pdf'
            
        
        fig,ax = plot.subplots(1,1)
        subind0 = [0, 0, 0, 0]
        fig.set_size_inches(9, 6)
        fig.subplots_adjust(left=0.075)
        fig.subplots_adjust(right=0.925)
        
        for i,q in enumerate(qs):
            level = i*2+1
            req = self.conn.execute(text(q))
            df = pd.DataFrame(columns=req.keys(), data=req.fetchall())

            xaxis = df[(df['loss_power'] > 0.) & (df['loss_power'] <= 5.0)]['loss_power'].values
            yaxis = df[(df['loss_power'] > 0.) & (df['loss_power'] <= 5.0)]['err'].values

            b, m, c = polyfit(xaxis, yaxis, 2)
            opt_x = -m/2/c
            opt_y = b+m*opt_x+c*np.power(opt_x,2)
            fitaxis = b + m*xaxis + c*np.power(xaxis,2)
            axis = ax
            DBExt._add_plot(dataset, xaxis, yaxis, fitaxis, opt_x, opt_y, level, axis)             

            if (True):
                filename = path.format(i, dataset)
                with PdfPages(filename) as pdf:
                    pdf.savefig(fig)
                    
                fig,ax = plot.subplots(1,1)
                fig.set_size_inches(9, 6)
                fig.subplots_adjust(left=0.075)
                fig.subplots_adjust(right=0.925)


        fig1,ax1 = plot.subplots(1,1)
        fig.set_size_inches(9, 6)
        fig.subplots_adjust(left=0.075)
        fig.subplots_adjust(right=0.925)
        
        q = q9
        i = 4; level = 9
        req = self.conn.execute(text(q))
        df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
        
        xaxis = df[(df['loss_power'] > 0.) & (df['loss_power'] <= 5.0)]['loss_power'].values
        yaxis = df[(df['loss_power'] > 0.) & (df['loss_power'] <= 5.0)]['err'].values
        
        b, m, c = polyfit(xaxis, yaxis, 2)
        opt_x = -m/2/c
        opt_y = b+m*opt_x+c*np.power(opt_x,2)
        fitaxis = b + m*xaxis + c*np.power(xaxis,2)
        axis = ax1
        DBExt._add_plot(dataset, xaxis, yaxis, fitaxis, opt_x, opt_y, level, axis)

        filename = path.format(i, dataset)
        with PdfPages(filename) as pdf:
            pdf.savefig(fig1)

        

    @staticmethod
    def _add_plot(dataset, xaxis, yaxis, fitaxis, opt_x, opt_y, level, axis, metric="error"):
        axis.grid(True)
        axis.set_xlabel('p')
        # axis.set_ylabel('Out of sample error (%)')
        axis.set_ylabel('Out of sample {}'.format(metric))
        # axis.set_title('Recursive[{}] OOS error (%)'.format(level))
        axis.set_title('OOS {}'.format(metric))

        axis.scatter(xaxis, yaxis, marker='x', label="OOS error")
        axis.plot(xaxis, fitaxis, '-', label="quad fit", color="orange")
        if (opt_x > 0.0) and (opt_x <= 5.0):
            axis.plot([opt_x], [opt_y], marker='o', markersize=8, markerfacecolor="black")
        axis.legend(loc='upper right')
        
    def plot_OOS_fits(self, run_keys, dataset):
        xaxis = np.array([]); erraxis = np.array([]); F1axis = np.array([]); prcsnaxis = np.array([])
        for run_key in run_keys:
            req = self.conn.execute(text('select iteration, err, prcsn, recall, F1 from outofsample where run_key={}'.format(
                run_key)))
            df = pd.DataFrame(columns=req.keys(), data=req.fetchall())
            xaxis = xaxis if xaxis.shape[0] > 0 else df.iteration.values
            erraxis = np.concatenate([erraxis, df.err.values.reshape(1,-1)]) if erraxis.shape[0] > 0 else df.err.values.reshape(1,-1)
            prcsnaxis=np.concatenate([prcsnaxis, df.prcsn.values.reshape(1,-1)]) if prcsnaxis.shape[0] > 0 else df.prcsn.values.reshape(1,-1)
            F1axis = np.concatenate([F1axis, df.F1.values.reshape(1,-1)]) if F1axis.shape[0] > 0 else df.F1.values.reshape(1,-1)

        # plot
        fig, ax1 = plot.subplots()

        colors = ['tab:red', 'tab:green', 'tab:orange', 'tab:blue', 'tab:pink', 'tab:gray']
        labels = ['exp loss               ', 'binom dev loss    ', 'square loss          ', 'synth loss cubic   ', 'synth loss cube rt', 'synth loss cubicm']
        # labels = ['All subsets   ', '50% subsets']
        labels = [x+" [error: {}%]" for x in labels]
        linestyles = {'dense dashdot': (0, (3, 1, 1, 1))}
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Out of sample error (%)')
        ax1.set_title('Out of sample convergence [{}]'.format(dataset[:16]))
        ax1.tick_params(axis='y')
        for i,_ in enumerate(run_keys):
            label = labels[i].format(str(np.round(erraxis[i,-1],2)))
            ax1.plot(xaxis, erraxis[i,:],
                     linestyle='solid',
                     color=colors[i],
                     marker='.',
                     markersize=5.0,
                     linewidth=1.0,
                     label=label)

        if False:
            ax2 = ax1.twinx()
            
            color = 'tab:green'
            ax2.set_ylabel('F1')
            labels = ['All subsets   ', '50% subsets']
            labels = [x+" [F1: {}]" for x in labels]        
            for i,_ in enumerate(run_keys):
                label = labels[i].format(str(np.round(F1axis[i,-1],2)))            
                ax2.plot(xaxis, F1axis[i,:],
                         linestyle=linestyles['dense dashdot'],
                         color=colors[i],
                         marker='.',
                         markersize=5.0,
                         label=label)
                
            ax1.legend(loc='lower right')
            ax2.legend(loc='upper right')
            ax1.grid(True)
            ax2.grid(True)
            
            fig.tight_layout()                     

        ax1.grid(True)
        ax1.legend(loc='upper right')
        plot.show()
        # filename = 'priority_adult_fig.pdf'
        # with PdfPages(filename) as pdf:
        #     pdf.savefig(fig)
        

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
    # xaxis,yaxis = dbext.plot_part_v_perf("income_small_train")
    
    # spambase_train
    # dbext.plot_OOS_fits([13784078293975702669, 9187967397797350836, 15767095704017143287, 12793747121298945138],
    #                     "spambase")
    # income_small_train
    # dbext.plot_OOS_fits([7533277102691265751, 8251155077064940421, 7400247794797131665, 12836108145017863223],
    #                     "income")
    # phoneme
    # dbext.plot_OOS_fits([8705424173097098323, 2421851584358366385, 1820609185731366400, 1998113784327814407])
    # flare
    # dbext.plot_OOS_fits([1917161425195422475, 1274523705138131147, 197195856475585082, 7986938987171996673])
    # GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1_train
    # dbext.plot_OOS_fits([12806597599666742865, 18329239330487574321, 8061260327215002817, 15890096115503869579],

    #                     "GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1") 200 iterations
    # dbext.plot_OOS_fits([16658721250403211986, 9305782506885980955, 558962598135559380, 3006904790200134207, 836796850612459713], "GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1")

    # LATEST
    #                     "income"
    # dbext.plot_OOS_fits([2450778176480913663, 16245809754040591623, 5702964362226338383, 12141359753427671061, 17087407651129546284], "income")
    # phoneme
    # dbext.plot_OOS_fits([2198770098628379437, 7919356570590355038, 5870206050745901991, 14972031634584792342, 5155973280160634802], "phoneme")
    # spambase
    # dbext.plot_OOS_fits([7509760514710782388, 16160241094113825097, 17044766082791278599, 6234424463209312435, 9046709106798984035], "spambase")
    # flare
    # dbext.plot_OOS_fits([13918540862625454121, 12497941719585348570, 15063258873168302088, 3699344882172787925, 10852222179632322186],"flare")
    # german
    # dbext.plot_OOS_fits([8363277825869133099, 2193534154822646370, 7712989217054850562, 1820462959196204091, 16021844930615959795], "german")

    # 6 loss functions
    # income
    # dbext.plot_OOS_fits([9896831864593018725, 11760346465113099430, 11470328976658411635, 1593918338463665465, 13128485379685894848, 1471927161598063725], "income")
    # spambase
    # dbext.plot_OOS_fits([13037801800753975719, 10554336323616544769, 12397366513681523567, 13999283684645320268, 166732925607201951, 5753093941682235975], "spambase")
    # phoneme
    # dbext.plot_OOS_fits([3665393597474860163, 5991368650252470032, 1710142948180159165, 8431531754912409505, 15398622298968073629, 16530157184505497296], "phoneme")
    # flare
    # dbext.plot_OOS_fits([3665123017970236105, 6890981365057713202, 8248659231312751020, 830653885356967330, 2823164921657540188, 7126282586633062304], "flare")
    # adult
    # dbext.plot_OOS_fits([8453587131535466825, 536224941498796699, 277025322160564859, 6160569975937583803, 4077672419385012766, 7660754858450613426], "adult")
    
    # phoneme
    # dbext.plot_OOS_fits([10983252967684601218, 2264980010301009553], "phoneme")
    # spambase
    # dbext.plot_OOS_fits([838851549671652446, 8906307970632292207], "spambase")
    # income
    # dbext.plot_OOS_fits([881529938279876822, 9083113082298868931], "income")
    # hypothyroid
    # dbext.plot_OOS_fits([12533667763751356557, 1969787373940308242], "hypothyroid")
    # flare
    # dbext.plot_OOS_fits([7801503716704362743, 9229190595195867291], "flare")
    # coil2000
    # dbext.plot_OOS_fits([16490765667426907248, 17096735190734304193], "coil2000")
    # adult
    # dbext.plot_OOS_fits([13716798805666073679, 7761421272482278056], "adult")
    # dbext.plot_OOS_fits([17486348340806686583, 11723447567148917324], "adult")
    # dbext.plot_OOS_fits([13561509335051850106, 8257621172237294579], "adult")
    # sortind = [x[0] for x in sorted(enumerate(xaxis), key=lambda x: x[1])]
    # xaxis = xaxis[[sortind]]
    # yaxis = yaxis[[sortind]]

    # xaxis,yaxis = dbext.plot_OOS_by_power("income_2000_train")
    # xaxis,yaxis = dbext.plot_OOS_by_power("phoneme_train")
    # dbext.plot_OOS_by_power("house_votes_84_train")
    # dbext.plot_OOS_by_power("colic_train")
    # dbext.plot_OOS_by_power("buggyCrx_train")

    dbext.plot_OOS_simple("adult_sm_train")

    # b, m, c = polyfit(xaxis, yaxis, 2)
    # print("optimal x: {}".format(-m/2/c))
    # plot.scatter(np.array(xaxis), np.array(yaxis))
    # plot.scatter(xaxis, yaxis)
    # plot.plot(xaxis, yaxis)
    # plot.plot(xaxis, b + m*xaxis + c*np.power(xaxis,2), '-')
    # plot.show()


    
