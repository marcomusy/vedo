#################################################################
from __future__ import division, print_function
import numpy as np
import plotter


vp = plotter.vtkPlotter(size=(800,800), verbose=0, projection=1)
p1 = [0,0,0]
p2 = [0,0,1]

vers   = [.5,1,-2]
rpoint = [0,0,0] ## point not working in vtk?? currently ignored

#######################################################
actor = vp.helix(p1, p2, r=.1, thickness=.01, c='r')
#actor = vp.arrow(p1, p2, c='r')
#actor = vp.line(p1, p2, lw=2, c='r')
#actor = vp.cylinder([p1, p2], r=.03, c='r')
#actor = vp.cone(p1, r=.1, c='r')
#######################################################

#for i, c in enumerate(actor.coordinates()) :
#    if c[2]>.9999: print(i, c)
#vp.show(actor, interactive=1)

vp.arrow(rpoint, np.array(rpoint)+vers, c='g', alpha=.1)#dont change

for i in vp.arange(0,360, 10):
    vp.rotate(actor, 10, vers, rpoint)  ## rotate by 10 deg around vers
    # startPoint, endPoint = actor.orientation() ### get base and tip
    # vp.points([startPoint, endPoint], c='b')
    vp.show(interactive=0)

vp.show(interactive=1)




