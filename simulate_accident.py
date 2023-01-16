# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:16:24 2022

@author: lenovo
"""

import sys,os
import random
import sumolib
import traci  # noqa
import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import seaborn as sns
import xml.etree.ElementTree as ET
from run_procedure import *
import numpy as np

accident_time = 600 #事故发生的时间：第10min
accident_pos = 9000 #事故发生的位置：300m
duration = 1800 #事故持续时间：30min
n_lane = 2 #影响车道数

tree = ET.parse('./simulate/S321.rou.xml')   ##这里的路径为放置路由文件的路径
root = tree.getroot()

# root[5].attrib['vehsPerHour']
# 添加预测流量
l = [0,1,3,4,5,6,7,8,9,10,11,12]
for i in l:
    #小车流量
    root[2*i+5].attrib['vehsPerHour']=str(12*0.9*vol2[0][i])
    root[2*i+5].attrib['departPos']=str(8000)
    #大车流量
    root[2*i+6].attrib['vehsPerHour']=str(12*0.1*vol2[0][i])
    root[2*i+6].attrib['departPos']=str(8000)
    # root[2*i+5].attrib['begin']=str(300*i)
    # root[2*i+6].attrib['end']=str(300*(i+1))
    # root[i+18].attrib['begin']=str(300*i)
    # root[i+18].attrib['end']=str(300*(i+1))
tree.write('./simulate/S321.rou.xml')

os.system('sumo ./simulate/S32.sumocfg')

os.system('python ./simulate/xml2csv.py ./simulate/detect.xml')

d = pd.read_csv('./simulate/detect.csv', sep=';')

d.columns = [i[9:] for i in d.columns]

def smooth(df, k):
    df = df.values
    x=[]
    for i in range(0,k):
        x.append(df[i])
    for i in range(k, len(df)-k):
        t = np.mean(df[i-k:i+k+1])
        x.append(t)
    for i in range(len(df)-k,len(df)):
        x.append(df[i])
    return x

def maxTime(df, v):
    
    d1 = df[df['id']=='detect4']
    
    dv1 = d1[d1['meanSpeed']>v]
    
    dv1.index = range(len(dv1))
    l = dv1['begin'] - dv1['begin'].shift(1)
    l.dropna(inplace=True)
    maxl = max(l)
    return maxl

d1 = d[d['id']=='detect4']
d1['maxJamLengthInMeters'] = smooth(d1['maxJamLengthInMeters'],20)

congestion_time = maxTime(d, 28)

max_length = str(int(np.ceil(d['maxJamLengthInMeters'].max()))) + 'm'

sns.set_theme(style="ticks")
sns.lineplot(x='begin', y='maxJamLengthInMeters', hue='id', style='id', data=d1)#所有车道的可视化
sns.lineplot(x='begin', y='meanSpeed', hue='id', style='id', data=d1)#所有车道的可视化
sns.lineplot(x='begin', y='meanTimeLoss', hue='id', style='id', data=d)

queue_length = list(d1['maxJamLengthInMeters'][::300].values[3:9]+100)

congestion = [max_length, congestion_time, queue_length]

