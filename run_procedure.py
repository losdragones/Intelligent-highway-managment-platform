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
import datetime as dt
import pandas as pd
from model import Model
import numpy as np
from sklearn.metrics import *
import matplotlib.dates as mdates
import keras
from attention import attention

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
    
def plot_results_multiple(vol_pred, vol_true, prediction_len, tp, p, m):
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(range(vol_true.shape[0]),vol_true[:,0],label='truth',alpha=0.7)
    for i in np.arange(0,vol_true.shape[0],20):
        plt.scatter(range(i,i+prediction_len),vol_pred[i,:],label='predict',marker='X', edgecolors='k',
                      c='#ff7f0e', s=48)
        if i==0:
            plt.legend()
    pattern = {'weekday':'工作日', 'weekend':'双休日','congestion':'拥堵','allday':'全场景'}
    plt.title('模型：{}，交通模式：{}，时间周期：{}分钟'.format(m,pattern[tp],str(p)))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.grid(True)
    plt.ylabel('交通量/5min')
    # plt.show()
    plt.draw()
    plt.savefig('./plot_results/{}.jpg'.format(dt.datetime.now().strftime('%d%m%Y-%H%M%S')))
    plt.show()
    
def score(predicted_data, true_data):
    print('MSE: ' + str(mean_squared_error(true_data, predicted_data, squared=False)))
    print('RMSE: ' + str(mean_squared_error(true_data, predicted_data, squared=True)))
    print('MAE: ' + str(mean_absolute_error(true_data, predicted_data)))
    # MAPE = np.mean(np.abs((predicted_data - true_data) / true_data)) * 100
    print('MAPE: ' + str(100*mean_absolute_percentage_error(true_data, predicted_data))+'%')

configs = json.load(open('config_mulstep.json', 'r'))
if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
period = configs['data']['periods']
seq_len = configs['data']['sequence_length']
tp = configs['traffic_pattern']
m = configs['model']['name']
timestp = configs['model']['layers'][0]['input_timesteps']
file = configs['data']['filepath']
df, dfSmooth = pd.DataFrame(), pd.DataFrame()
for i in range(len(file)-1,-1,-1):
    
    data_process = DataProcessor(file[i]) 
    data_process.time_process()
    data_process.delete_lane([1,2,3,4,5])
    data_process.process(period)
    # data_process.aggregate()
    # data_process.traffic_pattern(tp)
    data_process.missing_data(period,tp)
    data_process.smooth(7)
    data_process.abnormal_data()#可能会处理掉拥堵数据
    data_process.normalize()
        
    df[['Day sin','Day cos','volNorm{}'.format(i)]] = data_process.df[['Day sin','Day cos','volNorm']]
    dfSmooth[['Day sin','Day cos','volSN{}'.format(i)]] = data_process.df[['Day sin','Day cos','volSmoothNorm']]

# 读取指定时间范围内的流量
now = dt.datetime(2022, 7, configs['start_day'], configs['start_hour'], configs['start_min'], 0)
t1 = now - dt.timedelta(hours=4)
t2 = now + dt.timedelta(minutes=70)
df = df[(df.index>t1) & (df.index<t2)]
dfSmooth = dfSmooth[(dfSmooth.index>t1) & (dfSmooth.index<t2)]

# 预测指定时间的流量

data = DataTransformer(df,0,df.columns.values)
dataSmooth = DataTransformer(dfSmooth,0,dfSmooth.columns.values)

model = Model()
model.build_model_improve_mul(configs)

# 	in-memory training5
model.load_model('./saved_models/f1-i24-27092022-101251.h5')#在save_model中替换训练模型
   
steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])

x_test, y_test = data.get_test_data_seq(seq_len,timestp)
x1_test, y1_test = dataSmooth.get_test_data_seq(seq_len,timestp)

# np.savetxt('y_test.csv',y_test)
predictions = model.predict_sequence([x_test, x1_test])

vol_pred = data_process.norm_transform(predictions);
vol_true = data_process.norm_transform(y_test.reshape(y_test.shape[0],y_test.shape[1]))

vol_input = data_process.norm_transform(x_test[:,:,-1])

plot_results_multiple(vol_pred, vol_true, 12, tp,period,m)

t = df.index
day = [i.strftime('%m-%d') for i in t]
t = [i.strftime('%H:%M') for i in t]

vol1 = [list(vol_true[i-12:i][:,0]) for i in range(24,len(vol_true))]
vol2 = [list(vol_pred[i-12:i][:,0]) + list(vol_pred[i]) for i in range(24,len(vol_pred))]
# vol1 = [[index, row['vol']] for index,row in volume.iterrows()]
# vol2 = [[index, row['vol']+10] for index,row in volume.iterrows()]
time_show = [t[i:i+24] for i in range(36, len(vol_true)+12)]

vol = [vol1, vol2, time_show, day]
