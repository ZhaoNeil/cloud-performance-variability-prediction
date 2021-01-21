# cloud-performance-variability-prediction
This repository contains analysis code for paper 'Cloud Performance Variability Prediction', submitted for [4th Workshop on Hot Topics in Cloud Computing Performance](https://hotcloudperf.spec.org). 

Authors:
*Yuxuan Zhao(y.zhao.2@umail.leidenuniv.nl)
*Dmitry Duplyakin(dmdu@cs.utah.edu)
*Robert Ricci(ricci@cs.utah.edu)
*Alexandru Uta(a.uta@liacs.leidenuniv.nl)

**CPD_collection** and **CPD_counts** are for the changepoint detection (robseg) on [CCGrid20 paper's dataset](https://zenodo.org/record/3686952/#.X2u5yy8RrVU) and analysis methods refer to [this paper](https://arxiv.org/abs/2003.04824). Analysis code refers to [this repository](https://gitlab.flux.utah.edu/emulab/cloudlab-ccgrid20).

**BD_collection** and **BD_counts** are for the changepoint detection (Breakout Detection)  on [CCGrid20 paper's dataset](https://zenodo.org/record/3686952/#.X2u5yy8RrVU).

**BreakoutDetection** and **cpd** are two methods for changepoint detection.

**changepoint_compare** is comparison for two methods.

**feature_importance** shows the feature importance in prediction.

**prediction_CP** predicts timestamp of changepoint.

With comments and questions related to this repository, please contact: Yuxuan Zhao (y.zhao.2@umail.leidenuniv.nl)