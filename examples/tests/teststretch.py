#################################################################
from __future__ import division, print_function
import plotter, numpy as np


### send the spring from p12 to q12, keeping its section size

vp = plotter.vtkPlotter(verbose=0, projection=1)

p1 = [-.6,0.1,-0.5]   
p2 = [3  ,1  , 1  ]
q1 = [0.9,0.2,0.7 ]
q2 = [2  ,0.1,1   ]

vp.points([p1,p2])
vp.line(p1, p2, lw=3, c='m')
vp.points([q1,q2])

#######################################################
# actor = vp.helix(p1, p2, r=.2, thickness=.03, c='r')
# actor = vp.arrow(p1, p2, c='r/.2')
# actor = vp.line(p1, p2, c='r')
# actor = vp.cylinder([p1, p2], r=.03, c='r')
actor = vp.cone(p1, r=.1, c='r/.2')
#######################################################

vp.show()

for i in range(314*2): 
    q2 = np.array(q2) +  [np.sin(i/100)/500, 0, 0]
    vp.point(q2)
    actor.stretch(q1, q2) ##### <---
    vp.camera.Azimuth(.1)
    vp.camera.Elevation(.1)
    vp.render(resetcam=1)
    
vp.show(interactive=1)

