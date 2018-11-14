# Load a whole directory of 2D shapes
# Make a cloud of points, add noise and smooth it with 
# Moving Least Squares (smoothMLS3D).
# This make a simultaneus fit in 4D (space+time).
# smoothMLS3D returns a vtkAssembly where points are code colored
# in bins of fitted time. 
# The nr of time slices is specified by N.
# The natural pread of the data is estimated by dx and dt.
# A min nr of neighbours in the local fitting can be imposed,
#  if condition is not met the point is discarded.
# Artificial gaussian noise is added for the purpose of testing.
# 
from vtkplotter import Plotter
from vtkplotter.analysis import smoothMLS3D


vp = Plotter()

acts = vp.load('data/timecourse/reference_28*', legend=0)

for i,a in enumerate(acts): 
    a.pos([0,0,i*.1]).lineWidth(3).color(i).alpha(0.5)

pts4d, lost = smoothMLS3D(acts, dx=0.1, dt=0.1, neighbours=5,
                    	  N=10, addnoise=0.2)

vp.show(acts+[pts4d, lost])


