# Shows an ideal gas diffusing in space
#
from __future__ import division, print_function
from random import uniform as u
import plotter


N = 10      # nr of particles along axis
s = 0.01    # random step size

scene = plotter.vtkPlotter(verbose=0)
scene.plane(pos=[.45,.45,-.05], texture='wood7')

for i in range(N):              # generate a grid of points
    for j in range(N): 
        for k in range(N):
            p = [i/N, j/N, k/N]
            scene.point(p, c=p) # color point by its position
print ('Scene is ready, press q to continue')
scene.show()

for t in range(500):           # time loop
    if not t%100: print(t, 'elapsed time:', int(scene.clock),'s')
        
    for i in range(1, N*N*N):   # for each particle
        actor = scene.actors[i]
        r = [u(-s,s), u(-s,s), u(-s,s)] # random step
        p = actor.pos()         # get point position
        q = p + r 
        if q[2]<0: q[2] *= -1   # if bounce on the floor
        actor.pos(q)            # set its new position
    scene.render(resetcam=0)

scene.show(resetcam=0)
