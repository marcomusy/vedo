"""Form a surface by joining two lines"""
from vedo import *
import numpy as np

l1 = [[sin(x), cos(x), x/3] for x in np.arange(0,9, 0.1)]
l2 = [[sin(x)+0.2, cos(x) + x/15, x/3] for x in np.arange(0,9, 0.1)]

t1 = Tube(l1, r=0.02).color("green5")
t2 = Tube(l2, r=0.02).color("blue3")

r12 = Ribbon(l1, l2, res=(200,5)).alpha(0.5)
r1  = Ribbon(l1, width=0.1).color('orange',0.5)

plt = Plotter(N=2, axes=1)
plt.at(0).show(__doc__, r12, t1, t2)
plt.at(1).show("..or along a single line", r1, t1, viewup="z")
plt.interactive().close()
