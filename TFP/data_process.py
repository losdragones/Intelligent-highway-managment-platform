# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 10:15:17 2022

@author: lenovo
"""

import pandas as pd
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
from datetime import datetime as dt
import datetime
from sklearn.preprocessing import MinMaxScaler

# df = pd.read_excel('F:\\TFP\\data\\demo.xlsx')
# df = pd.read_csv('F:\\TFP\\data\\tab_data_radar_traffic_flow0_3.csv')
# df['dev_id'].unique()#'K41+400.LD'
# df = pd.read_csv('F:\\TFP\\data\\volData.csv')
# df4 = pd.read_csv('F:\\TFP\\data\\7.8流量数据.csv')

# df0 = pd.read_csv('F:\\TFP\\data\\tab_data_radar_traffic_flow.csv')   
# df0.head().transpose()
# df.sort_values(by=['lane','timestamp'],inplace=True);
# df.index = range(len(df))   
# df.head().transpose()
# df.to_csv('F:\\TFP\\data\\volData.xlsx',index=False)
# df.columns = data.columns[:-1]

class DataProcessor(object):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        # self.df = pd.read_excel(filepath)
        self.dfAllLane = None
        self.dfWeekday = None
        self.dfWeekend = None
        
    '''描述性统计'''
    def describe(self):
        # 分析各个雷达、各个断面、各个车道的数据量
        df = self.df
        describ = df.pivot_table(index='lane',
                             columns = ['radar_no','coil_no','coil_pos'],
                             aggfunc={'vol':len})
        print(describ)

# '''只包含一个radar_no，因为另一个radar_no数据过少，删除无效数据，删除无效车道'''
# # df = df[df['coil_pos']==90]
# # df['radar_no'].unique()#2949172, 3276856
# df = df[df['radar_no']==3276856]
    '''对时间数据进行处理'''
    def time_process(self):
        df = self.df           
        # 对时间乱码的行进行删除
        try:
            df = df[df.apply(lambda row: '-' in row['timestamp'], axis=1)]
        except:
            1
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        df['day'] = df['timestamp'].dt.day
        # df['intTime'] = df['timestamp'].map(pd.Timestamp.timestamp)
        self.df = df
        
    def delete_lane(self, deleteLane):
        df = self.df
        df = df[df.apply(lambda row: row['lane'] not in deleteLane, axis=1)]
        self.df = df
        
    def periodTime(self, t, period):
        minute = t.minute
        if minute % period == 0:
            return t
        else:
            delta_time = period - (minute % period)
            t = t + datetime.timedelta(minutes=delta_time)
            return t

    '''对数据进行排序 调整时间周期 多个车道数据合并成总流量'''
    def process(self, period):
        day = 24*60*60
        '''period: 不同时间间隔的周期，80ms/1min时period=750,80ms/1min时period=3750''' 
        # 对数据按lane/direction/timestamp进行排序
        df = self.df
        df['timestamp'] = df['timestamp'].apply(lambda t: t - datetime.timedelta(seconds=t.second))
        df['period_time'] = df['timestamp'].apply(lambda t: self.periodTime(t, period))
        # df['period_int'] = df['period_int'].apply(lambda row: str(row))
        # '''前提必须timestamp是整数分钟！！！'''
        df.rename(columns={'totalVol':'vol'}, inplace=True)
        
        for colName in ['avgSpeed','laneTimeCccupancy','laneSpaceCccupancy']:
            df[colName] = df[colName].apply(lambda row: nan if row==0 else row)
        df['headTimeInterval'] = df['headTimeInterval'].apply(lambda row: nan if row==65535 else row)
        df['headSpaceInterval'] = df['headSpaceInterval'].apply(lambda row: nan if row==255 else row)
        
        #除去应急车道计算速度等
        dfFastLane = df[df['lane']!=10]
        
        dfFastLane = dfFastLane.pivot_table(index = 'period_time',
                           aggfunc = {'avgSpeed':np.mean,
                                      'laneTimeCccupancy':np.mean,
                                      'laneSpaceCccupancy':np.mean,
                                      'headTimeInterval':np.mean,
                                      'headSpaceInterval':np.mean})
                                      
        df = df.pivot_table(index = 'period_time',
                           aggfunc = {'vol':sum})
        
        # df['intTime'] = df.index
        dfFastLane['timestamp'] = dfFastLane.index
        df['timestamp'] = df.index
        df = pd.merge(df, dfFastLane, on='timestamp')
        df.index = df['timestamp'].values
        # df.index = df['timestamp'].values
        df.sort_values(by='timestamp', inplace=True)
        df['intTime'] = df['timestamp'].map(pd.Timestamp.timestamp)
        # 这个时间戳→intTime是正确对应的，但是intTime到时间戳会产生8小时误差
        # df['totalVol'] = df['vol'].cumsum()
        # df = df[df['intTime'].apply(lambda row: row % (60*period) == 0)]
        
        # df['vol'] = df['totalVol'].diff()
        # df['vol'].iloc[0] = df['totalVol'].iloc[0]
        df['Day sin'] = np.sin(2 * np.pi * (df['intTime'] % day / day)) 
        df['Day cos'] = np.cos(2 * np.pi * (df['intTime'] % day / day))
        df['weekday'] = df['timestamp'].dt.weekday
        
        # 将时间采样的周期增大
        # timeList = df[df['intTime'].apply(lambda row: row % (60*period) == 0)]['intTime']
                
        # df.sort_values(by=['radar_no','coil_no','lane','timestamp'], inplace=True)
        # df.index = range(len(df))
    
        # df = df[::period]
    
        # 对累计流量进行差分得到每分钟流量
        # data = pd.DataFrame(columns=df.columns)
        # for lane in df['lane'].unique():
        #     tep = df[df['lane']==lane]   
        #     tep['vol'] = tep['totalVol'].diff()
        #     # 由差分法计算的首个数据为Nan
        #     data = pd.concat([data,tep])
        # df = data
        #     # 雷达在运行过程中出现过一次重启 因此对负值进行替换，替换为空
        #     # 对其它雷达累计流量异常情况，替换为空
        # df['vol'] = df['vol'].apply(lambda row: nan if (row<0) | (row>30*period) else row)
        # df.sort_values(by=['lane','timestamp'],inplace=True)
        # df.index = range(len(df))
        # df = df[df['timestamp'].dt.day.apply(lambda d: d in range(19,28))]
        self.df = df
        # return df      
    
    '''检索设备离线情况'''
    def offline_check(self, second_interval):
        df = self.df 
        l = pd.DataFrame(columns=['开始离线时间','重新上线时间'])
        df = df[df['lane']==df['lane'].unique()[0]]
        for i in range(1,len(df)):
            if df['intTime'][i] - df['intTime'][i-1] > second_interval:
                l = l.append({'开始离线时间':df['timestamp'][i-1],
                              '重新上线时间':df['timestamp'][i]}, ignore_index=True)
                # print('开始离线：' + df['timestamp'][i-1])
                # print('重新上线：' + df['timestamp'][i])
        return l
# DF = pd.concat([df0,df1,df2,df3])
    # def aggregate(self):
    #     day = 24*60*60
    #     df = self.df
    #     # df['intMinute'] = df['intTime'] - df['intTime'] % 60

    #     df = df.pivot_table(index = 'intTime',
    #                        aggfunc = {'vol':sum,
    #                                   'avgSpeed':np.mean})
    #     # df.columns = ['avgSpeed','vol']
    #     df['intTime'] = df.index
    #     df['intTime'] = df['intTime'].apply(lambda row: int(row))
    #     df['Day sin'] = np.sin(2 * np.pi * (df['intTime'] % day / day)) 
    #     df['Day cos'] = np.cos(2 * np.pi * (df['intTime'] % day / day))
    #     df.index = df['intTime'].apply(lambda row: dt.fromtimestamp(row))
    #     self.dfAllLane = df
    
    '''数据标准化 分布在0-1间'''
    def normalize(self):
        df = pd.DataFrame(self.df['vol'])
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)
        self.df['volNorm'] = df
        # inverse_data = scaler.inverse_transform(self.df)
    
    def norm_transform(self, data):
        df = pd.DataFrame(self.df['vol'])
        scaler = MinMaxScaler()
        scaler.fit_transform(df);
        return scaler.inverse_transform(data)
        
    '''选择不同模式的交通数据进行预测'''
    def traffic_pattern(self, tp):
        df = self.df
        if tp == 'weekday':
            df = df[df['weekday'].apply(lambda row: row not in [5,6])]
            self.df = df
        elif tp == 'weekend':
            df = df[df['weekday'].apply(lambda row: row in [5,6])]
            self.df = df
        elif tp == 'allday':
            self.df = df
        

    '''数据透视表分车道对各个指标进行统计'''
    def table(df):
        res = df.pivot_table(index=['radar_no','lane'],
                         aggfunc={'vol':[np.max,np.mean,np.std],
                             'avgSpeed':[np.max,np.mean,np.min,np.std]})
        return res

# res = table(df)

# df = df[df.apply(lambda row: row['lane'] in np.arange(6,11),axis=1)]















    
    
    
    

