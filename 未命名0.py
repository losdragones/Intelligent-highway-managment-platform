# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 22:51:25 2023

@author: lenovo
"""
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif']=['SimSun']
plt.title("哈哈")

my_font = fm.FontProperties(fname='C:\\Users\\lenovo\\Desktop\\视频数据处理\\simheittf.ttf')


from matplotlib.font_manager import FontManager
import subprocess
 
mpl_fonts = set(f.name for f in FontManager().ttflist)
print('all font list get from matplotlib.font_manager:')
for f in sorted(mpl_fonts):
    print('\t' + f)
