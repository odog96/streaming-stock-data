import pandas as pd
import numpy as np
import os
import cdsw
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

from joblib import dump, load

import tensorflow as tf
from tensorflow import keras

from transform_forecaster import fill_blanks

exp_cols = ['symbol','symbol_time','close', 'volume']

@cdsw.model_metrics
def forecast_price(args):
    
    window = 60      
    #data = pd.DataFrame.from_dict(args)
    data = pd.DataFrame.from_dict(args['data'])
    data = data[exp_cols]
    
    # fill missing data
    date_range = data.symbol_time.drop_duplicates()
    date_range = date_range.sort_values().tail(window)
    date_range = pd.DataFrame(date_range)
    stock_ready = fill_blanks(data,'symbol',date_range,'symbol_time');
    
    # Configure input data for model
    symbol_list = []
    for i, tckr in enumerate(stock_ready.symbol.unique()):
        symbol_list.append(tckr)
        temp_input = stock_ready[stock_ready.symbol == tckr].copy().close.values
        if i == 0:
            input_array = temp_input
        else:
            input_array = np.append(input_array,temp_input)
    
    # reshape
    input_array = input_array.reshape(i+1,window,1)
       # predict
    #recon_model = keras.models.load_model("home/cdsw/saved_models/model_01_17_23")
    try:
        recon_model = keras.models.load_model("model_01_17_23")
    except:
        cwd = os.getcwd()
        print('current working directory is: ',cwd)
        path = "/home/cdsw/model_01_17_23"
        dir_list = os.listdir(path)
        print("Files and directories in '", path, "' :")
        recon_model = keras.models.load_model(path)
        
    
    prediction = recon_model.predict(input_array).tolist()
    
    
    # output prep ## added this section to provide other relavent into
    pred_time_list = [stock_ready.symbol_time.max()]*(i+1)
    output = {'symbol':symbol_list,'pred_time':pred_time_list,'prediction':prediction}

    
    return output