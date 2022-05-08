"""Use array to vary radius and color
of a line represented as a tube"""
from vedo import *
import numpy as np

settings.defaultFont = 'Quikhand'

ln = [[sin(x), cos(x), x / 2] for x in np.arange(0,9, 0.1)]
N = len(ln)

############################### a simple tube( along ln
t1 = Tube(ln, c="blue", r=0.08)

############################### vary radius
rads = [0.3*(cos(6.0*ir/N))**2+0.1 for ir in range(N)]
t2 = Tube(ln, r=rads, c="tomato", res=24)

############################### vary color
cols = [i for i in range(N)]
cols = makeBands(cols, 5)  # make color bins
t3 = Tube(ln, r=rads, c=cols, res=24)

############################### visualize
plt = Plotter(N=3, axes=dict(textScale=4))
plt.at(0).show(t1, __doc__)
plt.at(1).show(t2)
plt.at(2).show(t3, viewup="z")
plt.interactive().close()
