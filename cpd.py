#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr


get_ipython().system('git clone https://github.com/guillemr/robust-fpop')
get_ipython().system('R CMD build robust-fpop')
get_ipython().system('R CMD INSTALL robseg_*.tar.gz')


robjects.r("install.packages('changepoint', lib = '/usr/local/lib/R/site-library')")
robseg = importr('robseg')
robjects.r.source("robust-fpop/Simulation.R");


def cpd_aggr(df, factor_list, metric, meta_df, lthreshold_list, indiv_cps, indiv_durations, dest_dir):
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
         
        for lthreshold in lthreshold_list:  
      
            # Run the analysis for selected configuration
            n = cpd_indiv(hw_type, hw_type_grp, lthreshold, factor_list, metric, indiv_cps, indiv_durations, processed_configs_count)
            processed_configs_count = n

        print ("Processed configs:", processed_configs_count)
        print ("Processed measurements:", total_measurements)

    meta_df.to_csv("%s/meta" % (dest_dir), index=False)
    indiv_durations.to_csv("%s/segment-durations" % (dest_dir), index=False)

    for lthreshold, grp in indiv_cps.groupby("lthreshold"):
        
        grp.to_csv("%s/indiv-%.2f" % (dest_dir, float(lthreshold)), index=False)

        aggr_df = pd.DataFrame()
        for hw, subgrp in grp.groupby("hw_type"):
            
            # Working only for CPU data/factor_list
            #configs_str = (subgrp["testname"] + "-" + subgrp["total_threads"] + "-" + subgrp["dvfs"]+ "-" + subgrp["socket_num"] + ";").tolist()
            # More general version: 
            configs_str = ["-".join(x.values)+";" for idx, x in subgrp[factor_list].astype(str).iterrows()]

            timeline = pd.DataFrame({"configs": configs_str,
                                    "direction": subgrp.percent_change.apply(lambda x: "U" if x > 0 else "D").tolist(),
                                    "percent_changes": subgrp.percent_change.apply(lambda x: str(x) + ";").tolist()},
                                    index=pd.to_datetime(subgrp.timestamp, unit='s'))
            t_count = timeline.resample('D').sum()
            t_count["count"] = t_count["configs"].apply(lambda x: len(str(x).split(";"))-1)
            t_count["hw_type"] = hw
            aggr_df = pd.concat([aggr_df, t_count])

        aggr_df.to_csv("%s/aggr-%.2f" % (dest_dir, float(lthreshold)))


def cpd_indiv(hw_type, df, lthreshold, factor_list, metric, indiv_cps, indiv_durations, processed_configs_count):
    

    for idx, grp in df.groupby(factor_list):
        
        # Do not consider subsets that include less than 200 measurements
        if len(grp) < 200:
            continue
            
        processed_configs_count += 1

        vals = grp[metric].values
        timestmaps = grp.timestamp.values
        vals_str = ",".join([str(el) for el in vals])
        # Create a list with these values and pass it into the R environment
        robjects.r("x.data <- c(%s)" % (vals_str))

        # Run CPD with the selected threshold
        res=robjects.r("""
            est.sd <- varDiff(x.data)
            res.ou <- fpop_intern(x.data/est.sd,  test.stat="Outlier", pen.value=2*log(length(x.data)), lthreshold=%f)
            cpt <- res.ou$cpts[-length(res.ou$cpts)]
            """ % lthreshold)

        mean_vals = robjects.r("res.ou$smt.signal * est.sd") 
        for v in robjects.r("cpt"):
            # For each changepoint, estimate the corresponding change in means (relative change, %)
            percent_change = (mean_vals[v] - mean_vals[v-1]) / mean_vals[v-1] * 100.0
            indiv_cps.loc[len(indiv_cps)] = [hw_type, "%.2f" % lthreshold] + [str(x) for x in list(idx)] + [timestmaps[v], percent_change] 

        # Estimate legths of segments, in seconds
        timestmaps_min = timestmaps.min()
        timestmaps_max = timestmaps.max()
        if len(robjects.r("cpt")) == 0:
            segment_lengths = [timestmaps_max-timestmaps_min]
        elif len(robjects.r("cpt")) == 1:
            segment_lengths = [timestmaps[robjects.r("cpt")[0]]-timestmaps_min, timestmaps_max-timestmaps[robjects.r("cpt")[0]]]
        else:
            segment_lengths = [timestmaps[robjects.r("cpt")[0]]-timestmaps_min, timestmaps_max-timestmaps[robjects.r("cpt")[-1]]]
            for cpt_idx in range(len(robjects.r("cpt")))[1:]:
                  segment_lengths.append(timestmaps[robjects.r("cpt")[cpt_idx]]-timestmaps[robjects.r("cpt")[cpt_idx-1]])
        for sl in segment_lengths:
            indiv_durations.loc[len(indiv_durations)] = [hw_type, "%.2f" % lthreshold] + [str(x) for x in list(idx)] + [sl / (3600.0 * 24)] 
    return processed_configs_count





