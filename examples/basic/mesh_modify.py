# Modify mesh vertex positions.
# (vtkActor transformation matrix is reset when mesh is modified)
#
from vtkplotter import Plotter
import numpy as np


vp = Plotter(axes=4, interactive=0)

dsc = vp.disc()

coords = dsc.coordinates()

for i in range(50):
    noise = np.random.randn(len(coords),3)*0.02
    noise[:,0] = 0 # only perturb z
    noise[:,1] = 0  
    dsc.points(coords+noise) # modify mesh
    vp.show(elevation=-1)
    
vp.show(interactive=1) 
