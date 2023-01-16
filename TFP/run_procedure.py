# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:11:16 2022

@author: lenovo
"""
import os
import json
import time
import math
import matplotlib.pyplot as plt
from data_process import DataProcessor
from data_transform import DataTransformer
import pandas as pd
from model import Model
import numpy as np
from sklearn.metrics import *
import matplotlib.dates as mdates
import keras

def plot_results(predicted_data, true_data, time_index, tp, p, m):
    plt.rcParams['font.sans-serif']=['SimHei']
    fig,ax = plt.subplots()
    res = pd.DataFrame({'true_data':true_data.reshape(true_data.shape[0]),
                        'predicted_data':predicted_data})
    res.index = time_index
    res['true_data'].plot(label='true_data',ax=ax)
    res['predicted_data'].plot(label='predicted_data',ax=ax)
    plt.legend()
    pattern = {'weekday':'工作日', 'weekend':'双休日','congestion':'拥堵','allday':'全场景'}
    plt.title('模型：{}，交通模式：{}，时间周期：{}分钟'.format(m,pattern[tp],str(p)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.grid()
    plt.show()
    
def score(predicted_data, true_data):
    print('MSE: ' + str(mean_squared_error(true_data, predicted_data, squared=False)))
    print('RMSE: ' + str(mean_squared_error(true_data, predicted_data, squared=True)))
    print('MAE: ' + str(mean_absolute_error(true_data, predicted_data)))
    # MAPE = np.mean(np.abs((predicted_data - true_data) / true_data)) * 100
    print('MAPE: ' + str(100*mean_absolute_percentage_error(true_data, predicted_data))+'%')


configs = json.load(open('config.json', 'r'))
if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
period = configs['data']['periods']
seq_len = configs['data']['sequence_length']
tp = configs['traffic_pattern']
m = configs['model']['name']
    
data_process = DataProcessor(configs['data']['filepath']) 
data_process.time_process()
data_process.delete_lane([1,2,3,4,5])
data_process.process(configs['data']['periods'])
data_process.normalize()
data_process.traffic_pattern(tp)
# df.to_csv('D://df708-726_{p}min.csv'.format(p=period),index=False)
    
df = data_process.df
    
      
data = DataTransformer(df,configs['data']['train_test_split'],configs['data']['columns'])

model = Model()
model.build_model(configs)
keras.utils.plot_model(model.model)
x, y = data.get_train_data(seq_len=configs['data']['sequence_length'],
                           normalise=configs['data']['normalise'])

# 	in-memory training5
model.load_model('./saved_models/26082022-132503-e10.h5')
   
steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])

x_test, y_test = data.get_test_data(
    seq_len=configs['data']['sequence_length'],
    normalise=configs['data']['normalise']
)

# predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
# predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
# print(predictions)
# print(y_test)
   
# np.savetxt('y_test.csv',y_test)
predictions = model.predict_point_by_point(x_test)

# plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
plot_results(predictions, y_test, df.index[data.len_train+seq_len:],tp,period,m)
# res = pd.DataFrame({'timestamp':data_process.dfAllLane.index,
#                     'vol':y_test})
score(predictions, y_test)

# 提供前端展示所用数据
# predictions从标准化数据转换为正常流量
vol_pred = data_process.norm_transform(predictions.reshape(len(predictions),1));
vol_pred = vol_pred.reshape(len(vol_pred))
vol_true = data_process.norm_transform(y_test)
vol_true = vol_true.reshape(len(vol_true))
# vol_test = 








