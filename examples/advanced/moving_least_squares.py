from __future__ import division, print_function
from plotter import vtkPlotter


vp = vtkPlotter(shape=(1,3), axes=0, verbose=0)

s0 = vp.load('data/beethoven.ply', alpha=1)#.subdivide()
scopy = s0.clone(alpha=1) # make a copy

s = vp.smoothMLS(s0, f=.8, recursive=0, decimate=1)

vp.write(s,'out.vtk')
vp.show(scopy, at=0) # the original
vp.show(s,  at=1)    # show the procedure
vp.show(s0, at=2, interactive=1) # 'smoothed'
