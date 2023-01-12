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