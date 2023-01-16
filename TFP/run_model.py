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

def plot_data(df, cols, p):
    # df.index = range(len(df))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plot_features = df[cols]
    # fig,ax = plt.subplots()
    plot_features.plot(subplots=True)
    # plt.title('交通流密速指标/{}分钟'.format(p),loc='left')
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  
    # plt.scatter(df['vol'].index,df['vol'])
    plt.show()

def plot_results(predicted_data, true_data, time_index, tp, p, m):
    # fig = plt.figure(facecolor='white')
    # ax = fig.add_subplot(111)
    # ax.plot(true_data, label='True Data')
    # plt.plot(predicted_data, label='Prediction')
    plt.rcParams['font.sans-serif']=['SimHei']
    fig,ax = plt.subplots()
    res = pd.DataFrame({'true_data':true_data.reshape(true_data.shape[0]),
                        'predicted_data':predicted_data})
    res.index = time_index
    # res['true_data'].plot(label='true_data')
    # res['predicted_data'].plot(label='predicted_data')
    res['true_data'].plot(label='true_data',ax=ax)
    res['predicted_data'].plot(label='predicted_data',ax=ax)
    # plt.plot(res['true_data'].values, label='true_data')
    # plt.plot(res['predicted_data'].values, label='predicted_data')
    plt.legend()
    
    pattern = {'weekday':'工作日', 'weekend':'双休日','congestion':'拥堵','allday':'全场景'}
    plt.title('模型：{}，交通模式：{}，时间周期：{}分钟'.format(m,pattern[tp],str(p)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.grid()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [0 for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
    
def score(predicted_data, true_data):
    print('MSE: ' + str(mean_squared_error(true_data, predicted_data, squared=False)))
    print('RMSE: ' + str(mean_squared_error(true_data, predicted_data, squared=True)))
    print('MAE: ' + str(mean_absolute_error(true_data, predicted_data)))
    # MAPE = np.mean(np.abs((predicted_data - true_data) / true_data)) * 100
    print('MAPE: ' + str(100*mean_absolute_percentage_error(true_data, predicted_data))+'%')

# configs = json.load(open('config.json', 'r'))
# data_process = DataProcessor(configs['data']['filepath']) 
# data_process.time_process()
# data_process.delete_lane([1,2,3,4,5])
# data_process.process(configs['data']['periods'])
# data_process.normalize()

# if configs['traffic_pattern'] == 'weekday':
#     df = data_process.dfWeekday
# elif configs['traffic_pattern'] == 'weekend':
#     df = data_process.dfWeekend

# df = data_process.df
# df.describe().transpose()

def main():
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
    # data_process.aggregate()
    data_process.normalize()
    data_process.traffic_pattern(tp)
    # df.to_csv('D://df708-726_{p}min.csv'.format(p=period),index=False)
    
    df = data_process.df
    
    plot_data(df, ['vol','headTimeInterval','avgSpeed'], period)
    
    # plt.plot(range(len(df)), df['vol'].values)
      
    data = DataTransformer(df,configs['data']['train_test_split'],configs['data']['columns'])

    model = Model()
    model.build_model(configs)
    keras.utils.plot_model(model.model)
    x, y = data.get_train_data(seq_len=configs['data']['sequence_length'],
                               normalise=configs['data']['normalise'])

# 	in-memory training5
    history = model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
    # out-of memory generative training
       
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    # history = model.train_generator(
    #     data_gen=data.generate_train_batch(
    #         seq_len=configs['data']['sequence_length'],
    #         batch_size=configs['training']['batch_size'],
    #         normalise=configs['data']['normalise']
    #     ),
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     steps_per_epoch=steps_per_epoch,
    #     save_dir=configs['model']['save_dir']
    # )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.plot(history.history['loss'],marker='o')
    plt.ylabel('训练损失')
    plt.xlabel('训练次数')
    plt.show()
    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # print(predictions)
    # print(y_test)
   
    # np.savetxt('y_test.csv',y_test)
    predictions = model.predict_point_by_point(x_test)
    # predictions从标准化数据转换为正常流量
    # predictions = predictions.reshape(len(predictions),1)
    # predictions = data_process.norm_transform(predictions)
    # np.savetxt('predictions.csv',predictions)

    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    plot_results(predictions, y_test, df.index[data.len_train+seq_len:],tp,period,m)
    # res = pd.DataFrame({'timestamp':data_process.dfAllLane.index,
    #                     'vol':y_test})
    score(predictions, y_test)
    

if __name__ == '__main__':
    main()
    