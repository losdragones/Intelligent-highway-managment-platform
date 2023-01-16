#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2020/8/26 14:48
# @Author : way
# @Site :
# @Describe:

from flask import Flask, render_template
from data import *
import pandas as pd
import numpy as np
# from run_procedure import *
# df = pd.read_csv('D:\\trajectory_data.csv')
from interpl import *
from simulate_accident import *
# import sys,os
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:  
#     sys.exit("please declare environment variable 'SUMO_HOME'")

    
app = Flask(__name__)


@app.route('/')
def index():
    data = SourceData(traj,speed,vol,congestion)
    # data['echart1']['title']
    # print(data.echart2['series'])
    return render_template('index.html', form=data, title=data.title)


@app.route('/corp')
def corp():
    data = CorpData()
    return render_template('index.html', form=data, title=data.title)


@app.route('/job')
def job():
    data = JobData()
    return render_template('index.html', form=data, title=data.title)


if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=False)
