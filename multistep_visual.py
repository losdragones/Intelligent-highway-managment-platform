# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:16:33 2022

@author: lenovo
"""

l = x_test[:,0,-1]
plt.plot(range(len(l)),l,label='truth',alpha=0.7)
for i in np.arange(0,len(l),20):
    plt.scatter(range(i+24,i+36),predictions[i,:],label='predict',marker='X', edgecolors='k',
                  c='#ff7f0e', s=32)
    if i==0:
        plt.legend()

    
l = x_test[:,0,-1]
plt.plot(range(len(l)),l,label='truth',alpha=0.7)
for i in np.arange(0,len(l),20):
    plt.scatter(range(i+24,i+36),vol_pred[i,:],label='predict',marker='X', edgecolors='k',
                  c='#ff7f0e', s=32)
    if i==0:
        plt.legend()