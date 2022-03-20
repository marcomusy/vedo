"""Use of plot() function analogous to matplotlib"""
import numpy as np, vtk
from vedo import *
from vedo.pyplot import plot

x = np.linspace(0, 5, 10)

fig1 = plot(x, x*x,    'sg-',  title='Plot1: y=x*x')
fig2 = plot(x, cos(x), 'pr--', title='Plot2: y=cos(x)')
fig3 = plot(x, sqrt(x),'Db-',  title='Plot3: y=sqrt(x)')
fig4 = plot(x, sin(x), '*t--', title='Plot4: y=sin(x)')

# window shape can be expressed as "n/m" or "n|m"
show(fig1, fig2, fig3, fig4,
     shape="3|1", sharecam=False, size=(1300,900)).interactive().close()

