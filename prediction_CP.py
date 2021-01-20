#!/usr/bin/env python
# coding: utf-8

# In[92]:


import csv
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from datetime import datetime
from breakout import detect_breakout
import os
import matplotlib.dates as mdate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import RandomizedSearchCV


# In[4]:


data = {}

data["CPU"] = pd.DataFrame()

source_dir = './CPU'
for f in os.listdir(source_dir):
    f_df = pd.read_csv("%s/%s" % (source_dir, f))
    hw_type, testname = f.split("-")
    f_df["hw_type"] = hw_type
    f_df["testname"] = testname
    data["CPU"] = pd.concat([data["CPU"], f_df])
print (data["CPU"].hw_type.value_counts())


# In[51]:


data["Memory"] = pd.DataFrame()

source_dir = './Memory'
for f in os.listdir(source_dir):
    f_df = pd.read_csv("%s/%s" % (source_dir, f))
    hw_type, testname = f.split("-")
    f_df["hw_type"] = hw_type
    f_df["testname"] = testname
    data["Memory"] = pd.concat([data["Memory"], f_df])
print (data["Memory"].hw_type.value_counts())


# In[52]:


data["Disk"] = pd.DataFrame()

source_dir = './Disk'
for f in os.listdir(source_dir):
    f_df = pd.read_csv("%s/%s" % (source_dir, f))
    hw_type, testname = f.split("-")
    f_df["hw_type"] = hw_type
    f_df["testname"] = testname
    data["Disk"] = pd.concat([data["Disk"], f_df])
print (data["Disk"].hw_type.value_counts())


# In[53]:


class Data:
    X = pd.DataFrame()
    y = pd.DataFrame()
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def xgb(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=8)
        xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.7, learning_rate = 0.1,
                                  max_depth = 5, subsample = 0.9, n_estimators = 1000,random_state = 3)
        xg_reg.fit(X_train,y_train)
        predict_xgb = xg_reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predict_xgb))
        print("RMSE(XgBoost): %f" %(rmse/3600/24),"days")
        fo = open("pred_results.txt", "a+")
        fo.write("RMSE(XgBoost): %f days" %(rmse/24/3600)+"\n")
        fo.close()
    def lgb(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=8)
        lgb_reg = lgb.LGBMRegressor(objective = 'mse', colsample_bytree = 0.7, learning_rate = 0.1,
                                    max_depth = 5, subsample = 0.9, n_estimators = 1000, random_state = 3, 
                                    num_leaves = 30)
        lgb_reg.fit(X_train,y_train)
        predict_lgb = lgb_reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predict_lgb))
        print("RMSE(LightGBM): %f" %(rmse/3600/24),"days")
        fo = open("pred_results.txt", "a+")
        fo.write("RMSE(LightGBM): %f days" %(rmse/24/3600)+"\n")
        fo.close()
    def cb(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=8)
        cb_reg = cb.CatBoostRegressor(learning_rate = 0.1, max_depth = 5, verbose = 0, iterations = 1000,
                                      early_stopping_rounds = 200, eval_metric = 'RMSE',random_state = 3)
        cb_reg.fit(X_train,y_train)
        predict_cb = cb_reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predict_cb))
        print("RMSE(CatBoost): %f" %(rmse/3600/24),"days")
        fo = open("pred_results.txt", "a+")
        fo.write("RMSE(CatBoost): %f days" %(rmse/24/3600)+"\n")
        fo.close()


# In[54]:


cpu_raw = data["CPU"]
memory_raw = data["Memory"]
disk_raw = data["Disk"]


# In[55]:


cpd = {}

for dataset in ["CPU", "Memory", "Disk"]:

    results_dir = "./%s_results" % dataset
    if dataset not in cpd.keys():
        cpd[dataset] = {} 

    for f in os.listdir("%s" % results_dir):
        # Exclude hidden files (most likely thumbnails)
        if f[:2] != "._":
            print("Loading: %s/%s" % (results_dir, f))
            df = pd.read_csv("%s/%s" % (results_dir, f))
            print(len(df))

            if "aggr" in f:
                # Extract CPD threshold from the filename 
                lthreshold = float(f.split("-")[1])

                df["lthreshold"] = lthreshold
                if "aggr" not in cpd[dataset].keys():
                    cpd[dataset]["aggr"] = pd.DataFrame()
                cpd[dataset]["aggr"] = pd.concat([cpd[dataset]["aggr"], df], sort=False)

            if "indiv" in f:
                if "indiv" not in cpd[dataset].keys():
                    cpd[dataset]["indiv"] = pd.DataFrame()
                cpd[dataset]["indiv"] = pd.concat([cpd[dataset]["indiv"], df], sort=False)

            if "meta" in f:
                cpd[dataset]["meta"] = df

            if "segment-durations" in f:
                cpd[dataset]["segment-durations"] = df


# In[56]:


cpu_cp = cpd['CPU']['indiv']
memory_cp = cpd['Memory']['indiv']
disk_cp = cpd['Disk']['indiv']


# In[59]:


print(cpu_cp.columns)
print(memory_cp.columns)
print(disk_cp.columns)


# In[60]:


cpu_raw.reset_index(drop=True, inplace=True)
memory_raw.reset_index(drop=True, inplace=True)
disk_raw.reset_index(drop=True, inplace=True)


# In[66]:


cpu_result = pd.merge(cpu_cp,cpu_raw,how='inner',on=['hw_type', 'testname', 'total_threads', 'dvfs',
       'socket_num', 'timestamp'])
memory_result = pd.merge(memory_cp,memory_raw,how='inner',on=['hw_type', 'testname', 'dvfs',
       'socket_num', 'timestamp'])
disk_result = pd.merge(disk_cp,disk_raw,how='inner',on=['hw_type', 'testname', 'device',
       'iodepth', 'timestamp'])


# In[68]:


print(cpu_result.columns)
print(memory_result.columns)
print(disk_result.columns)


# In[74]:


# X = cpu_result[['nodeid', 'kernel_release', 'os_release', 'gcc_ver', 
#                 'version', 'version_hash', 'exec_time']]
# y = cpu_result[['timestamp']]
# categorical_cols = X.columns[X.dtypes==object].tolist()
# le = LabelEncoder()
# # X.columns
# X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

# Cpu = Data(X,y)
# Cpu.xgb()
# Cpu.lgb()
# Cpu.cb()


# In[75]:


# X = memory_result[['nodeid','kernel_release', 'os_release', 'gcc_ver', 
#                    'version_hash','mean']]
# y = memory_result[['timestamp']]
# categorical_cols = X.columns[X.dtypes==object].tolist()
# le = LabelEncoder()
# # X.columns
# X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

# Memory = Data(X,y)
# Memory.xgb()
# Memory.lgb()
# Memory.cb()


# In[86]:


# X = disk_result[['disk_model','disk_serial','disk_size','nodeid','mean','device']]
# y = disk_result[['timestamp']]
# categorical_cols = X.columns[X.dtypes==object].tolist()
# le = LabelEncoder()
# X['disk_serial'] = le.fit_transform(X['disk_serial'].astype(str))
# X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

# Disk = Data(X,y)
# Disk.xgb()
# Disk.lgb()
# Disk.cb()


# In[94]:


class Data_RandomSearch:
    X = pd.DataFrame()
    y = pd.DataFrame()
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def xgb(self):
        param_grid = {
            'num_leaves': list(range(30, 100)),
            'n_estimators' : list(range(500,2500,100)),
            'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base = np.exp(1), num = 1000)),
            'subsample': list(np.arange(0.3,0.9,0.1)),
            'colsample_bytree': list(np.linspace(0.6, 1, 10)),
            'max_depth': list(np.arange(3,7,1))
        }
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=8)
        xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')
        
        clf = RandomizedSearchCV(xg_reg, 
                         param_distributions = param_grid, 
                         n_iter = 1000, 
                         scoring = 'neg_root_mean_squared_error', 
                         error_score = 0, 
                         verbose = 0,
                         n_jobs = -1,
                         random_state = 50)
        clf.fit(X_train,y_train)
        predict_xgb = clf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predict_xgb))
        print("RMSE(XgBoost): %f" %(rmse/24/3600),"days")
        fo = open("all_features.txt", "a+")
        fo.write("RMSE(XgBoost): %f days" %(rmse/24/3600)+"\n")
        fo.close()
    def lgb(self):
        param_grid = {
            'class_weight': [None, 'balanced'],
            'boosting_type': ['gbdt', 'goss', 'dart'],
            'num_leaves': list(range(30, 100)),
            'n_estimators' : list(range(500,2500,100)),
            'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base = np.exp(1), num = 1000)),
            'subsample': list(np.arange(0.3,0.9,0.1)),
            'colsample_bytree': list(np.linspace(0.6, 1, 10)),
            'max_depth': list(np.arange(3,7,1))
        }
    
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=8)
        
        lgb_reg = lgb.LGBMRegressor(objective = 'mse')
        clf = RandomizedSearchCV(lgb_reg, 
                         param_distributions = param_grid, 
                         n_iter = 1000, 
                         scoring = 'neg_root_mean_squared_error', 
                         error_score = 0, 
                         verbose = 0,
                         n_jobs = -1,
                         random_state = 50)
        clf.fit(X_train,y_train)
        predict_lgb = clf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predict_lgb))
        print("RMSE(LightGBM): %f" %(rmse/24/3600),"days")
        fo = open("all_features.txt", "a+")
        fo.write("RMSE(LightGBM): %f days" %(rmse/24/3600)+"\n")
        fo.close()
        
    def cb(self):
        param_grid = {
            'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base = np.exp(1), num = 1000)),
            'max_depth': list(np.arange(3,7,1))
        }
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=8)
        cb_reg = cb.CatBoostRegressor(eval_metric = 'RMSE')
        clf = RandomizedSearchCV(cb_reg, 
                         param_distributions = param_grid, 
                         n_iter = 1000, 
                         scoring = 'neg_root_mean_squared_error', 
                         error_score = 0, 
                         verbose = 0,
                         n_jobs = -1,
                         random_state = 50)
        clf.fit(X_train,y_train)
        predict_cb = clf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predict_cb))
        print("RMSE(CatBoost): %f" %(rmse/24/3600),"days")
        fo = open("all_features.txt", "a+")
        fo.write("RMSE(CatBoost): %f days" %(rmse/24/3600)+"\n")
        fo.close()


# In[93]:


X = cpu_result[['hw_type', 'lthreshold', 'testname', 'total_threads', 'dvfs',
       'socket_num', 'percent_change', 'class', 'nodeid', 'kernel_release', 
       'os_release', 'gcc_ver', 'version', 'version_hash', 'exec_time']]
y = cpu_result[['timestamp']]
categorical_cols = X.columns[X.dtypes==object].tolist()
le = LabelEncoder()
# X.columns
X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

Cpu = Data_RandomSearch(X,y)
Cpu.xgb()
Cpu.lgb()
Cpu.cb()


# In[ ]:


X = memory_result[['hw_type', 'lthreshold', 'testname', 'dvfs', 'socket_num',
       'percent_change', 'nodeid', 'kernel_release', 'os_release', 'gcc_ver',
       'version_hash', 'mem_clock_speed', 'units', 'mean', 'min', 'max',
       'stdev']]
y = memory_result[['timestamp']]
categorical_cols = X.columns[X.dtypes==object].tolist()
le = LabelEncoder()
# X.columns
X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

Memory = Data_RandomSearch(X,y)
Memory.xgb()
Memory.lgb()
Memory.cb()


# In[ ]:


X = disk_result[['hw_type', 'lthreshold', 'testname', 'device', 'iodepth',
          'percent_change', 'disk_model', 'disk_serial', 'disk_size', 
          'nodeid', 'units', 'mean', 'min', 'max', 'stdev']]
y = disk_result[['timestamp']]
categorical_cols = X.columns[X.dtypes==object].tolist()
le = LabelEncoder()
X['disk_serial'] = le.fit_transform(X['disk_serial'].astype(str))
X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

Disk = Data_RandomSearch(X,y)
Disk.xgb()
Disk.lgb()
Disk.cb()


# In[ ]:




