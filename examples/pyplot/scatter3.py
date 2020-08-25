"""Create a scatter plot to overlay
three different distributions"""
from vedo import *
from numpy.random import randn


### first scatter plot in blue
x = randn(2000) * 3
y = randn(2000) * 2

# scatter cloud, place it at z=0:
pts1 = Points([x,y], c="blue", alpha=0.5).z(0.0)


### second scatter plot in red
x = randn(1200) + 4
y = randn(1200) + 2
pts2 = Points([x,y], c="red", alpha=0.5).z(0.1)


### third scatter plot with marker in black
x = randn(20) + 4
y = randn(20) - 4
mark = Marker('*', s=0.2, filled=True)
pts3 = Glyph([x,y], mark, c='k').z(0.2)


label = Text("preliminary\nresults!", font='Inversionz', s=.8, pos=(-8,4,.2))
label.c('green').rotateZ(20)

show(pts1, pts2, pts3, label, __doc__,
     title='A simple scatter plot', axes=1, viewup='2d')
