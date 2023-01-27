import pandas as pd
import numpy as np
import os
import cdsw
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

from joblib import dump, load

from transform_functions import stock_data_pull, fill_blanks, feature_create

exp_cols = ['symbol','symbol_time','close', 'volume']


@cdsw.model_metrics
def mkt_movement(args):
    
    window = 20  
    features = ['return', 'vol', 'mom', 'sma', 'min', 'max']  
    lags = 6
    
    # read in data direction from kudu (in json format)
    data = pd.DataFrame.from_dict(args['data'])
    print('show the names of columns:',data.columns)
    print('show first few records:',data.head())
    data = data[exp_cols]
    
    # fill missing data
    date_range = data.symbol_time.drop_duplicates()
    date_range = pd.DataFrame(date_range)
    stock_ready = fill_blanks(data,'symbol',date_range,'symbol_time');
    
    # transform dataset
    tckr_list = stock_ready.symbol.unique().tolist()
    stk_rdy_2, cols =  feature_create(stock_ready,tckr_list, window,lags,features,clip=True,tckr='symbol',date='symbol_time',close='close')
    
    # predict
#    pipe_rf = load("pipe_rf.joblib")
    pipe_rf = load("pipe_rf.joblib")

            
        
    prediction = pipe_rf.predict(stk_rdy_2[cols])      
    
    # output prep ## added this section to provide other relavent into
    pred_time_list = [stk_rdy_2.symbol_time.max()]*len(stk_rdy_2)
    output = {'symbol':stk_rdy_2.symbol.tolist(),'pred_time':pred_time_list,'prediction':prediction}
    #out_df = pd.DataFrame.from_dict(output)
    
    return output