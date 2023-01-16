# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:22:44 2022

@author: lenovo
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn import svm
import sklearn
import pandas as pd
import numpy as np
import random
import joblib
import json
from data_process import DataProcessor
import matplotlib.pyplot as plt

configs = json.load(open('config.json', 'r'))
data_process = DataProcessor(configs['data']['filepath']) 
data_process.time_process()
data_process.process(configs['data']['periods'])
# data_process.aggregate()
data_process.normalize()

#time时间列     single1   信号值     取前多少个X_data预测下一个数据
def time_slice(time,single,X_lag):
    sample = []
    label = []
    for k in range(len(time) - X_lag - 1):
        t = k + X_lag
        sample.append(single[k:t])
        label.append(single[t])
    return sample,label

def score(predicted_data, true_data):
    print('MSE: ' + str(mean_squared_error(predicted_data, true_data, squared=False)))
    print('RMSE: ' + str(mean_squared_error(predicted_data, true_data, squared=True)))
    print('MAE: ' + str(mean_absolute_error(predicted_data, true_data)))
    print('MAPE: ' + str(mean_absolute_percentage_error(predicted_data, true_data)))

# indicator1 = data_process.dfAllLane['vol']
indicator1 = data_process.df['volNorm']

time = range(len(indicator1))
 
X_lag = 24
sample,label = time_slice(time,indicator1,24)

#数据集划分
X_train, X_test, y_train, y_test = train_test_split(sample, label, test_size=0.15, random_state=42)
 
#数据集掷乱
# random_seed = 13
# X_train, y_train = random.shuffle(X_train, y_train, random_state=random_seed)
 
#参数设置SVR准备
parameters = {'kernel':['rbf'], 
              'gamma':np.logspace(-5, 0, num=6, base=2.0),
              'C':np.logspace(-5, 5, num=11, base=2.0)}
 
#网格搜索：选择十折交叉验证
svr = svm.SVR()
grid_search = GridSearchCV(svr, #model
                           parameters, 
                           cv=10, #cross_validation
                           n_jobs=4, #num of paraller jobs
                           scoring='neg_mean_absolute_error')

svr.fit(X_train, y_train)
y_hat = svr.predict(X_test)

#SVR模型训练
grid_search.fit(X_train,y_train)
#输出最终的参数
print(grid_search.best_params_)
 
#模型的精度
print(grid_search.best_score_)
 
#SVR模型保存
joblib.dump(grid_search,'svr.pkl')
 
#SVR模型加载
svr=joblib.load('svr.pkl')
 
#SVR模型测试
y_hat = svr.predict(X_test)
 
#计算预测值与实际值的残差绝对值
 
plt.plot(y_test[:100],label='true_data')
plt.plot(y_hat[:100],label='predicted_data')
plt.xlabel('data')
plt.ylabel('target')
plt.ylim(0, 1)
plt.title('Support Vector Regression')
plt.legend()
plt.show()

score(y_hat, y_test)























# svr.fit(in_put, out_put)

# svr.predict(sample)

# from sklearn.svm import SVR
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# n_samples, n_features = 10, 5
# rng = np.random.RandomState(0)
# y = rng.randn(n_samples)
# X = rng.randn(n_samples, n_features)
# regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
# regr.fit(X, y)