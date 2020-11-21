#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Install breaout package from https://github.com/ZhaoNeil/anomaly-detection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from breakout import detect_breakout


# In[3]:


def cpd_aggr(df, factor_list, metric, meta_df, minsize_list, indiv_cps, indiv_durations, dest_dir):
    """ 
    Run cpd_indiv() a bunch of times and aggregate/save results; repeat for different CPD thresholds.
    This function will be reused for memory and disk CPD. 
    """
    global processed_configs_count
    global total_measurements 
    processed_configs_count = 0
    total_measurements = 0

    for hw_type, hw_type_grp in df.groupby("hw_type"):

        print ("\nProcessing hw_type:", hw_type)
        total_measurements += len(hw_type_grp)

        processed_config_count = 0
        for idx, grp in hw_type_grp.groupby(factor_list):
            if len(grp) >= 200:
                processed_config_count += 1

        meta_df.loc[len(meta_df)] = [hw_type, len(hw_type_grp), processed_config_count,
                                            hw_type_grp.timestamp.min(), hw_type_grp.timestamp.max()]
         
        for min_size in minsize_list:  
      
            # Run the analysis for selected configuration
            n = cpd_indiv(hw_type, hw_type_grp, min_size, factor_list, metric, indiv_cps, indiv_durations, processed_configs_count)
            processed_configs_count = n

        print ("Processed configs:", processed_configs_count)
        print ("Processed measurements:", total_measurements)

    meta_df.to_csv("%s/meta" % (dest_dir), index=False)
    indiv_durations.to_csv("%s/segment-durations" % (dest_dir), index=False)

    for min_size, grp in indiv_cps.groupby("min_size"):
        grp.to_csv("%s/indiv-%.2f" % (dest_dir, int(float(min_size))), index=False)


# In[5]:


def cpd_indiv(hw_type, df, min_size, factor_list, metric, indiv_cps, indiv_durations, processed_configs_count):
    

    for idx, grp in df.groupby(factor_list):
        
        # Do not consider subsets that include less than 200 measurements
        if len(grp) < 200:
            continue
            
        processed_configs_count += 1

        vals = list(grp[metric].values)
        timestmaps = grp.timestamp.values
        changepoint = detect_breakout(vals, min_size=min_size, method='multi', beta=0.001, degree=1)
        cp_index = [0]+changepoint+[len(vals)]
        mean_vals = []
        percent_change = []
        for i in range(len(cp_index)-1):
            mean_vals.append(np.mean(vals[cp_index[i]:cp_index[i+1]]))
        for i in range(len(changepoint)):
            percent_change.append((mean_vals[i+1]-mean_vals[i])/mean_vals[i])
            indiv_cps.loc[len(indiv_cps)] = [hw_type, "%.2f" % min_size] + [str(x) for x in list(idx)] + [timestmaps[changepoint[i]], percent_change[i]] 


        timestmaps_min = timestmaps.min()
        timestmaps_max = timestmaps.max()
        if len(changepoint) == 0:
            segment_lengths = [timestmaps_max-timestmaps_min]
        elif len(changepoint) == 1:
            segment_lengths = [timestmaps[changepoint[0]]-timestmaps_min, timestmaps_max-timestmaps[changepoint[0]]]
        else:
            segment_lengths = [timestmaps[changepoint[0]]-timestmaps_min, timestmaps_max-timestmaps[changepoint[-1]]]
            for cpt_idx in range(len(changepoint))[1:]:
                  segment_lengths.append(timestmaps[changepoint[cpt_idx]]-timestmaps[changepoint[cpt_idx-1]])
        for sl in segment_lengths:
            indiv_durations.loc[len(indiv_durations)] = [hw_type, "%.2f" % min_size] + [str(x) for x in list(idx)] + [sl / (3600.0 * 24)] 
    return processed_configs_count


# In[ ]:




