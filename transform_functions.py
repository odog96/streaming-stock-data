import json, requests
import pandas as pd
import numpy as np
import io
import random
import time

from random import sample
import yfinance as yf

import tensorflow as tf
from tensorflow import keras
#import tensorflow_datasets as tfds
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Conv1D,BatchNormalization, Dropout, Flatten, Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential

def stock_data_pull(basket,verb=True):
  """
  Download Stock data
  """
  stock_final = pd.DataFrame()
  # iterate over each symbol
  for i in basket:  
      
      # print the symbol which is being downloaded
      if verb:
        print( str(basket.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)  
      try:
          # download the stock price 
          stock = []
          #stock = yf.download(i,period = "2y",interval='5m', progress=False)
          # for intraday yahoo finance only let's you pull past 60 days. 
          # however there is only a 1 month option
          stock = yf.download(i,period = "1mo",interval='5m', progress=False)

          # append the individual stock prices 
          if len(stock) == 0:
              None
          else:
              stock['Name']=i
              stock_final = stock_final.append(stock,sort=False)
      except Exception:
          None
  print(stock_final.columns)
  stock_final.reset_index(inplace=True)
  # return dataframe with only key columns: 'Datetime','Close','Volume','Name'
  stock_final = stock_final[['Datetime','Close','Volume','Name']].copy()
  return stock_final

#def fill_blanks(df,srl_num,range,value_variable,date_variable):
def fill_blanks(df,srl_num,range,date_variable):

  """
  fills missing observations
  some stocks may not have the same number of observations
  """
  stage_df = df.copy()
  for comb in stage_df[srl_num].unique(): 
    temp = stage_df[stage_df[srl_num] == comb].copy()
    stage_df = stage_df[stage_df[srl_num] != comb] # remove existing series detail
    temp2 = range.merge(temp,how='left',on=date_variable)
    # since we are filling missing data, other fields will be missing to
    # value info will be filled by zero, for other fields, we simply forward fill
    # Then back fill for full missing value coverage    
    temp2.fillna(method='ffill',inplace=True)
    temp2.fillna(method='bfill',inplace=True)
    # replace with new, full data subset
    stage_df = stage_df.append(temp2,ignore_index=True)
  return stage_df

def feature_create(df_in,tckr_list, window,lags,features,clip=False):
  """
  Takes in dataframe with stock information for a sinlge ticker. Returns
  df with return, volatility, momentum, sma, min, and max 
  """
  df_int = pd.DataFrame()

  # validate input data frame has equal number of observations per symbol
  sizes = df_in.groupby('Name').size().unique()
  size_count = sizes.shape[0]

  if size_count == 1:
    print (f'all good, all data contains {sizes[0]} observations')
  else:
    print(f'error, your stock contain tckrs with varying number of observations:{sizes}')
    return

  for tckr in tckr_list:
    temp = df_in[df_in["Name"] == tckr]
    temp.sort_values('Datetime', inplace=True)
    # create core features
    temp['return'] = np.log(temp.Close / temp.Close.shift(1))  
    temp['vol'] = temp['return'].rolling(window).std()      # measure of volatility
    temp['mom'] = np.sign(temp['return'].rolling(window).mean())  
    temp['sma'] = temp['Close'].rolling(window).mean()  
    temp['min'] = temp['Close'].rolling(window).min()  
    temp['max'] = temp['Close'].rolling(window).max()
    temp.dropna(inplace=True)
    # create lag variables
    cols = []
    for f in features:
      for lag in range(1, lags + 1):
        col = f'{f}_lag_{lag}'
        temp[col] = temp[f].shift(lag)  
        cols.append(col)
    print('size of temp is',temp.shape[0])
    if clip:
      # just take only last n, required observations
      temp = temp.tail(1)
      print('size of temp is after clipping',temp.shape[0])
    df_int = df_int.append(temp,ignore_index=True)
    #df_int = df_int.append(temp,ignore_index=True)

  df_int.dropna(inplace=True)
  ## Create lags after core feaures created inside of df_int

  df_int.dropna(inplace=True)

  df_int['direction'] = np.where(df_int['return'] > 0, 1, -1) 
  return df_int, cols

def change_shape(temp):
  """ puts the input vector into the shape needed by the 
      DNN
  """
  temp = temp[['Close','Datetime']].copy()  # create single series dataframe
  y = temp.Close.values          # extract values
  y = y.reshape(y.shape[0],1)           # reshape
  return y

def dataset(x_train,y_train,x_valid,y_valid, batch):
  """
  take in arrays, output tensors. Original code sourced from 
  https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/10_time_series_forecasting_in_tensorflow.ipynb
  """
  train_features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

  # Orin

  test_features_dataset = tf.data.Dataset.from_tensor_slices(x_valid)
  test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_valid)

  train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
  test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

  train_dataset = train_dataset.batch(batch).prefetch(tf.data.AUTOTUNE)
  test_dataset = test_dataset.batch(batch).prefetch(tf.data.AUTOTUNE)

  return train_dataset, test_dataset

def window_gen(x,window,horizon,stride=1):
  """
  Generate training set with features and labels (from time series data)
  assumes a stride of 1, can modify for loop to make larget stride
  """
  x_t = []
  y_t = []
  for i in range(window,len(x)-horizon,stride):
    x_t.append(x[i-window:i,0])
    y_t.append(x[i:i+horizon,0])
  return x_t,y_t