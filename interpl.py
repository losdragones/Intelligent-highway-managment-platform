# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 22:55:18 2022

@author: lenovo
"""

import pandas as pd
import numpy as np
import json
import datetime as dt
from matplotlib import pyplot as plt 

df = pd.read_csv('D:\\trajectory_data.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

configs = json.load(open('config_mulstep.json', 'r'))
d = configs["start_day"]
h = configs["start_hour"]
minute = configs["start_min"]

t = dt.datetime(2022, 7, d, h, minute, 0)

df = df[(df['timestamp']>t) & (df['timestamp']<t+dt.timedelta(minutes=10))]

speed = []
for t in df['timestamp'].unique():
    tep = df[df['timestamp']==t]
    v = np.mean(tep['y_speed'])
    speed.append(v)

traj = pd.DataFrame({'x':[],'y':[],'timestamp':[]})
                     
for car in df['target_id'].unique():
    
    d = df[df['target_id']==car]
    x, y, t, y_speed = d.x.values, d.y.values, d.intTime.values, d.y_speed.values
    l = len(t) - 1
    tvals = np.linspace(t[0], t[0]+l, 10*l+1)
    xinter = np.interp(tvals, t, x)
    yinter = np.interp(tvals, t, y)
    v = np.interp(tvals, t, y_speed)
    traj = pd.concat([traj,pd.DataFrame({'x':xinter,'y':yinter,'timestamp':tvals,'y_speed':v,'car':car})])
    
    # traj['timestamp'] = traj['timestamp'].apply(lambda row: str(row))
    # traj['car'] = traj['car'].apply(lambda row: str(row))
    # df['intTime'] = df['intTime'].apply(lambda row: str(row))
    
    # t = [str(i) for i in t]
    # tvals = [str(i) for i in tvals]
# traj.to_csv('D:\\trajectory_demo.csv')
# volume = pd.read_csv('D:\\df708-726_5min.csv')
df = traj

traj = []
for t in df['timestamp'].unique():
    tep = df[df['timestamp']==t]
    tep = [[row['x'], row['y']] for index,row in tep.iterrows()]
    traj.append(tep)
    
    
    