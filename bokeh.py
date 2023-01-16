# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:51:11 2022

@author: lenovo
"""

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]
# create a new plot with a title and axis labels
# p = figure(title="Simple line example", x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness to the plot
# p.line(x, y, legend_label="Temp.", line_width=2)
# show(p)

source = ColumnDataSource(data=dict(foo=[], bar=[]))

# has new, identical-length updates for all columns in source
new_data = {
    'foo' : [10, 20],
    'bar' : [100, 200],
}

source.stream(new_data)

p = figure(title="Simple line example", x_axis_label='x', y_axis_label='y')
p.line(source, legend_label="Temp.", line_width=2)
