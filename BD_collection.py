#!/usr/bin/env python
# coding: utf-8

# ## Breakout Collection for CCGrid20 paper dataset and save them into files##

# In[1]:


import csv
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import datetime as dt
import os


# In[ ]:


# Install breaout package from https://github.com/ZhaoNeil/anomaly-detection
import BreakoutDetection


# In[ ]:


data = {}

data["CPU"] = pd.DataFrame()
data["Memory"] = pd.DataFrame()
data["Disk"] = pd.DataFrame()

source_dir = './cpu'
for f in os.listdir(source_dir):
    f_df = pd.read_csv("%s/%s" % (source_dir, f))
    hw_type, testname = f.split("-")
    f_df["hw_type"] = hw_type
    f_df["testname"] = testname
    data["CPU"] = pd.concat([data["CPU"], f_df])
print (data["CPU"].hw_type.value_counts())


# In[ ]:


source_dir = './memory'
for f in os.listdir(source_dir):
    f_df = pd.read_csv("%s/%s" % (source_dir, f))
    hw_type, testname = f.split("-")
    f_df["hw_type"] = hw_type
    f_df["testname"] = testname
    data["Memory"] = pd.concat([data["Memory"], f_df])
print (data["Memory"].hw_type.value_counts())


# In[ ]:


source_dir = './disk'
for f in os.listdir(source_dir):
    f_df = pd.read_csv("%s/%s" % (source_dir, f))
    hw_type, testname = f.split("-")
    f_df["hw_type"] = hw_type
    f_df["testname"] = testname
    data["Disk"] = pd.concat([data["Disk"], f_df])
print (data["Disk"].hw_type.value_counts())


# In[ ]:


df = data["CPU"]
factor_list = ["testname", "total_threads", "dvfs", "socket_num"]
metric = "exec_time"

minsize_list = [10,13,16,19,22,25]

dest_dir = "./CPU_BD"
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

meta_df = pd.DataFrame(columns=["hw_type", "total_measurements", "configurations", 
                                    "start_timestamp", "end_timestamp"])
indiv_cps = pd.DataFrame(columns = ["hw_type", "min_size"] + 
                  factor_list +  
                  ["timestamp", "percent_change"])
indiv_durations = pd.DataFrame(columns = ["hw_type", "min_size"] + 
                                     factor_list +  
                                     ["segment_duration"])
BreakoutDetection.cpd_aggr(df, factor_list, metric, meta_df, minsize_list, indiv_cps, indiv_durations,dest_dir)


# In[ ]:


df = data["Memory"]
factor_list = ["testname", "dvfs", "socket_num"]
metric = "mean"

minsize_list = [10,13,16,19,22,25]

dest_dir = "./Memory_BD"
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

meta_df = pd.DataFrame(columns=["hw_type", "total_measurements", "configurations", 
                                    "start_timestamp", "end_timestamp"])
indiv_cps = pd.DataFrame(columns = ["hw_type", "min_size"] + 
                  factor_list +  
                  ["timestamp", "percent_change"])
indiv_durations = pd.DataFrame(columns = ["hw_type", "min_size"] + 
                                     factor_list +  
                                     ["segment_duration"])

BreakoutDetection.cpd_aggr(df, factor_list, metric, meta_df, minsize_list, indiv_cps, indiv_durations,dest_dir)


# In[ ]:


df = data["Disk"]
factor_list = ["testname", "device", "iodepth"]
metric = "mean"

minsize_list = [10,13,16,19,22,25]

dest_dir = "./Disk_BD"
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Initialize dataframes where results will be saved by giving names to the columns
meta_df = pd.DataFrame(columns=["hw_type", "total_measurements", "configurations", 
                                    "start_timestamp", "end_timestamp"])
indiv_cps = pd.DataFrame(columns = ["hw_type", "min_size"] + 
                  factor_list +  
                  ["timestamp", "percent_change"])
indiv_durations = pd.DataFrame(columns = ["hw_type", "min_size"] + 
                                     factor_list +  
                                     ["segment_duration"])
BreakoutDetection.cpd_aggr(df, factor_list, metric, meta_df, minsize_list, indiv_cps, indiv_durations,dest_dir)


# In[ ]:




