import json, requests
import pandas as pd
import numpy as np
import io
import random
import time

from random import sample
import yfinance as yf

import psutil


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

def feature_create(df_in,tckr_list, window,lags,features,clip=False,tckr='Name',date='Datetime',close='Close'):
    """
      Takes in dataframe with stock information for a sinlge ticker. Returns
      df with return, volatility, momentum, sma, min, and max 
    """
    df_int = pd.DataFrame()

    # validate input data frame has equal number of observations per symbol
    sizes = df_in.groupby(tckr).size().unique()
    size_count = sizes.shape[0]

    if size_count == 1:
        print (f'all good, all data contains {sizes[0]} observations')
    else:
        print(f'error, your stock contain tckrs with varying number of observations {sizes}')
        return 
    count = 1
    for symb in tckr_list:
        temp = df_in[df_in[tckr] == symb]
        temp.sort_values(date, inplace=True)
        # create core features
        temp['return'] = np.log(temp[close] / temp[close].shift(1))  
        temp['vol'] = temp['return'].rolling(window).std()      # measure of volatility
        temp['mom'] = np.sign(temp['return'].rolling(window).mean())  
        temp['sma'] = temp[close].rolling(window).mean()  
        temp['min'] = temp[close].rolling(window).min()  
        temp['max'] = temp[close].rolling(window).max()
        temp.dropna(inplace=True)
        # create lag variables
        cols = []
        for f in features:
            for lag in range(1, lags + 1):
                col = f'{f}_lag_{lag}'
                temp[col] = temp[f].shift(lag)  
                cols.append(col)
        if clip:
      # just take only last n, required observations
            temp = temp.tail(1)
            print('size of temp is after clipping',temp.shape[0])
        if temp.shape[0] == 0:
            print('for ',symb,' size of temp is',temp.shape[0])
            # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            # Getting usage of virtual_memory in GB ( 4th field)
            print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
        else:
            df_int = df_int.append(temp,ignore_index=True)
            count += 1
            #print('stock:',symb,count,'th item',' temp size',temp.shape[0], 'df_int shape',len(df_int))
            del(temp)

#    df_int.dropna(inplace=True)

    df_int['direction'] = np.where(df_int['return'] > 0, 1, -1) 
    return df_int, cols
