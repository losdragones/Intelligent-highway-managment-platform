# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 09:38:27 2022

@author: lenovo
"""

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

def plot_data(df, cols, p):
    # df.index = range(len(df))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plot_features = df[cols]
    # fig,ax = plt.subplots()
    plot_features.plot(subplots=True)
    # plt.ylabel('交通量/5min')
    plt.show()

def plot_results_multiple(vol_pred, vol_true, prediction_len, tp, p, m):
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
            
    
def score(predicted_data, true_data):
    print('MSE: ' + str(mean_squared_error(true_data, predicted_data, squared=False)))
    print('RMSE: ' + str(mean_squared_error(true_data, predicted_data, squared=True)))
    print('MAE: ' + str(mean_absolute_error(true_data, predicted_data)))
    # MAPE = np.mean(np.abs((predicted_data - true_data) / true_data)) * 100
    print('MAPE: ' + str(100*mean_absolute_percentage_error(true_data, predicted_data))+'%')

def main():
    configs = json.load(open('config_mulstep.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    period = configs['data']['periods']
    seq_len = configs['data']['sequence_length']
    tp = configs['traffic_pattern']
    m = configs['model']['name']
    pattern = configs['data']['pattern']
    timestp = configs['model']['layers'][0]['input_timesteps']
    
    data_process = DataProcessor(configs['data']['filepath']) 
    data_process.time_process()
    data_process.delete_lane([1,2,3,4,5])
    data_process.process(period)
    # data_process.aggregate()
    # data_process.traffic_pattern(tp)
    data_process.missing_data(period,tp)
    data_process.smooth(13)
    data_process.abnormal_data()#可能会处理掉拥堵数据
    data_process.normalize()
    
    df = data_process.df
    # df.to_csv('D://df708-726_{p}min.csv'.format(p=period),index=False)
    
    plot_data(df, ['vol','headTimeInterval','avgSpeed'], period)
    # plot_data(df[(df['timestamp'].dt.day==11)|(df['timestamp'].dt.day==12)], ['vol'], period)
    # from datetime import datetime as dt
    # plot_data(df[df['timestamp'].dt.day==14], ['vol'], period)
    
    data = DataTransformer(df,configs['data']['train_test_split'],configs['data']['columns'])

    model = Model()
    model.build_model(configs)
    keras.utils.plot_model(model.model,
                           show_shapes=True,
                           show_dtype=True,
                           expand_nested=True)
    # 多步长预测
    x, y = data.get_train_data_seq(seq_len,timestp)

# 	in-memory training5
    history = model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
    # out-of memory generative training
    
    # 多点预测的测试数据
    x_test, y_test = data.get_test_data_seq(seq_len,timestp)
                                        
    # 使用平滑后的数据进行预测的验证
    # y_test = df['volNorm'].values[-len(x_test)-1:-1] 

    plt.rcParams['font.sans-serif']=['SimHei']
    plt.plot(history.history['loss'],marker='o')
    plt.ylabel('训练损失')
    plt.xlabel('训练次数')
    plt.show()
    # np.savetxt('y_test.csv',y_test)
    
    predictions = model.predict_sequence(x_test)
    
    # np.savetxt('predictions.csv',predictions)

    vol_pred = data_process.norm_transform(predictions);
    vol_true = data_process.norm_transform(y_test.reshape(y_test.shape[0],y_test.shape[1]))
    
    vol_input = data_process.norm_transform(x_test[:,:,-1])

    plot_results_multiple(vol_pred, vol_true, 12, tp,period,m)    
    # score(predictions, y_test)
    score(vol_pred, vol_true)

if __name__ == '__main__':
    main()
    
