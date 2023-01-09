import pandas as pd
import numpy as np
import os
import cdsw
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from joblib import dump, load

pipe_rf = load("saved_models/pipe_rf.joblib")
#pipe_bclf = load("saved_models/pipe_bclf.joblib")

features = ['return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_4', 'return_lag_5', 'return_lag_6', 'vol_lag_1','vol_lag_2', 'vol_lag_3', 'vol_lag_4', 'vol_lag_5', 'vol_lag_6', 'mom_lag_1', 'mom_lag_2', 'mom_lag_3', 'mom_lag_4', 'mom_lag_5', 'mom_lag_6', 'sma_lag_1', 'sma_lag_2', 'sma_lag_3', 'sma_lag_4', 'sma_lag_5', 'sma_lag_6', 'min_lag_1', 'min_lag_2', 'min_lag_3', 'min_lag_4', 'min_lag_5', 'min_lag_6', 'max_lag_1', 'max_lag_2', 'max_lag_3', 'max_lag_4', 'max_lag_5', 'max_lag_6']

@cdsw.model_metrics
def mkt_movement(args):
    filtArgs = {key:[args[key]] for key in features}
    data = pd.DataFrame.from_dict(filtArgs)
    
    prediction = pipe_rf(data).tolist()
        
    return prediction
    
