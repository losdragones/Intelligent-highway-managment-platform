# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:07:41 2022

@author: lenovo
"""

from sklego.preprocessing import RepeatingBasisFunction
rbf = RepeatingBasisFunction(n_periods=12,
                             remainder='passthrough',
                             input_range=(1,1440))
from datetime import datetime as dt

range_of_dates = pd.date_range(start='2017-01-01', end='2020-12-30')

x = df['timestamp'].dt.minute + df['timestamp'].dt.hour*60

x = pd.DataFrame(x)

x_1 = rbf.fit_transform(x)

x_1 = pd.DataFrame(x_1, index=x.index)

x_1.plot(subplots=True)


df[['Day sin','Day cos']].plot(subplots=True)

import numpy as np

a = np.linspace(0.1, 1, 24)

b = np.vstack((a for i in range(10))).T
