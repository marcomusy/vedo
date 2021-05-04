"""Use of plot() function analogous to matplotlib"""
import numpy as np, vtk
from vedo import *
from vedo.pyplot import plot

x = np.linspace(0, 5, 10)

plt1 = plot(x, x*x,    'sg-',  title='Plot1: y=x*x')
plt2 = plot(x, cos(x), 'pr--', title='Plot2: y=cos(x)')
plt3 = plot(x, sqrt(x),'Db-',  title='Plot3: y=sqrt(x)')
plt4 = plot(x, sin(x), '*t--', title='Plot4: y=sin(x)')

printc('plt1 is vtkAssembly?', isinstance(plt1, vtk.vtkAssembly))

# window shape can be expressed as "n/m" or "n|m"
show(plt1, plt2, plt3, plt4,
     shape="3|1", sharecam=False, size=(1300,900), interactive=True).close()

