# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:22:00 2022

@author: lenovo
"""
import pandas as pd
import numpy as np
from scipy import interpolate
from datetime import datetime as dt

df = pd.read_csv('D://df708-726_5min.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])

day = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="{}min".format(5))

df = df.set_index('timestamp',drop=False).reindex(day)

df.timestamp = df.index

df = df[df['timestamp'].dt.day.apply(lambda d: d not in [8,9,13,15,16,17,18,26])]

for index, row in df[pd.isna(df['vol'])].iterrows():
    minute = index.minute
    hour = index.hour
    df.loc[index,'vol'] = np.median(df[(df['timestamp'].dt.minute==minute) & (df['timestamp'].dt.hour==hour)]['vol'])
    
    
l = [1,2,3,np]    
    

def missing_data(df,p):
    
    df.timestamp = df.timestamp.astype('datetime64')

    day = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="{}min".format(p))

    df = df.set_index('timestamp',drop=False).reindex(day)

    df = df[df['timestamp'].dt.day.apply(lambda d: d not in [8,9,13,15,18,26])]

    for index, row in df[df['vol']==np.nan].iterrows():
        minute = index.minute
        hour = index.hour
        df.loc[index,'vol'] = np.mean(df[(df['timestamp'].dt.minute==minute) and (df['timestamp'].dt.hour==hour)]['vol'])
        
    return df
df = missing_data(df, 5)

df['vol'].plot()




df.index = df.timestamp

df = df.reindex(day)

df.timestamp = df.index
df = df[df['timestamp'].dt.day.apply(lambda d: d not in [8,9,13,15,18,26])]

d = df[df['timestamp'].dt.day==23]
d.index = range(len(d))

day = pd.date_range(start=dt.datetime(2022,7,23), end=dt.datetime(2022,7,24), freq="1min")[:-1]
d = d.set_index('timestamp',drop=False).reindex(day)

d['vol'] = list(smooth(d['vol'].values,13))
# d['vol'].interpolate(method='spline',inplace=True,order=3)
d['vol'].interpolate(method='cubic',inplace=True)
plt.plot(d['vol'])

x = np.arange(0,len(y))

tep = pd.DataFrame({'x':x,'y':y})

tep['y'].interpolate(inplace=True)

tep.dropna(inplace=True)

f = interpolate.interp1d(tep.x,tep.y,kind='quadratic')

predict = f(x)

plt.plot(predict)
plt.plot(tep.y)

def smooth(df,n):
    k = int(np.floor(n/2))
    x=[]
    for i in range(0,k):
        x.append(df[i])
    for i in range(k, len(df)-k):
        t = np.mean(df[i-k:i+k+1])
        x.append(t)
    for i in range(len(df)-k,len(df)):
        x.append(df[i])
    return pd.Series(x)
    

    