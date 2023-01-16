# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 09:48:54 2022

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import pandas as pd
from sklearn.metrics import *

'''这部分knn的预测更像是一种回归'''

df = pd.read_csv('D://df708-726_5min.csv')#读取文件，这里直接用的绝对路径
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')#将时间戳转换时间格式，%d整数转成十进位，%f倍精确度数字转成浮点数，%o整数转成八进位

def generate(df):
    x = []
    y = []
    for i,row in df.iterrows():
        try:
            y.append(df.loc[i+12,'vol'])#i+12代表行选，'vol'代表列选
            x.append(list(df.loc[i:i+11,'vol'].values))#i+12代表行选，'vol'代表列选，list是清单
            # print(i)
        except:
            break
    return x,y
    
# ---------------------------------------------------------------------------------------

# from sklearn import svm
# from sklearn.model_selection import GridSearchCV
# svr = svm.SVR()
# parameters = {'kernel':['rbf'], 
#               'gamma':np.logspace(-5, 0, num=6, base=2.0),
#               'C':np.logspace(-5, 5, num=11, base=2.0)}
# grid_search = GridSearchCV(svr, #model
#                            parameters, 
#                            cv=10, #cross_validation
#                            n_jobs=4, #num of paraller jobs
#                            scoring='neg_mean_absolute_error')
# svr.fit(x, y)
# y_svr = svr.predict(x_test)

# score(y_svr, y_test)

# plot_results(y_svr, y_test,df.loc[l+12:,'timestamp'].values,'weekday',15,'SVR')

# ---------------------------------------------------------------------------------------

def score(predicted_data, true_data):
    print('MSE: ' + str(mean_squared_error(true_data, predicted_data, squared=False)))#平均方误差
    print('RMSE: ' + str(mean_squared_error(true_data, predicted_data, squared=True)))#根均方误差
    print('MAE: ' + str(mean_absolute_error(true_data, predicted_data)))#平均绝对值误差
    # MAPE = np.mean(np.abs((predicted_data - true_data) / true_data)) * 100
    print('MAPE: ' + str(100*mean_absolute_percentage_error(true_data, predicted_data))+'%')#平均绝对百分比误差

def plot_results(predicted_data, true_data, time_index, tp, p, m):
    # fig = plt.figure(facecolor='white')
    # ax = fig.add_subplot(111)
    # ax.plot(true_data, label='True Data')
    # plt.plot(predicted_data, label='Prediction')
    plt.rcParams['font.sans-serif']=['SimHei']
    fig,ax = plt.subplots()
    res = pd.DataFrame({'true_data':true_data,
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
    plt.ylabel('交通量/5分钟')
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.grid()
    plt.show()

# for i, weights in enumerate(["uniform", "distance"]):
#     knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
#     y_ = knn.fit(X, y).predict(T)

#     plt.subplot(2, 1, i + 1)
#     plt.scatter(X, y, color="darkorange", label="data")
#     plt.plot(T, y_, color="navy", label="prediction")
#     plt.axis("tight")
#     plt.legend()
#     plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

# plt.tight_layout()
# plt.show()

# line = [1,2,3,4,5]
# for i,j in enumerate(["uniform", "distance"]):
#     print(i)
#     print(j)
l = int(len(df)*0.85)   

# X = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]
# from sklearn.neighbors import KNeighborsRegressor
# neigh = KNeighborsRegressor(n_neighbors=2)
# neigh.fit(X, y)

# print(neigh.predict([[1.5]]))

x, y = generate(df[:l])

x_test, y_test = generate(df[l:])

n_neighbors = 12  #设置12是什么意思
knn = neighbors.KNeighborsRegressor(n_neighbors)

y_knn = knn.fit(x,y).predict(x_test)

score(y_knn, y_test)  #运行输出结果，在控制台查看

plot_results(y_knn, y_test,df.loc[l+12:,'timestamp'].values,'weekday',15,'KNN')




