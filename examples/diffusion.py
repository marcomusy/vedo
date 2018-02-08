from __future__ import division, print_function
from random import uniform as u
from plotter import printc, ProgressBar, vtkPlotter

N = 10      # nr of particles along axis
s = 0.01    # random step size

scene = vtkPlotter(verbose=0)
scene.plane(pos=[.45,.45,-.05], texture='wood7')

for i in range(N):              # generate a grid of points
    for j in range(N): 
        for k in range(N):
            p = [i/N, j/N, k/N]
            scene.point(p, c=p) # color point by its own position
printc('Scene is ready, press q to continue', c='green')
scene.show()

pb = ProgressBar(0,200, c='red')
for t in pb.range():            # loop of 400 steps
    pb.print()   
    
    for i in range(1, N*N*N):   # for each particle
        actor = scene.actors[i]
        r = [u(-s,s), u(-s,s), u(-s,s)] # random step
        p = actor.pos()         # get point position
        q = p + r               # add the noise
        if q[2]<0: q[2] *= -1   # if bounce on the floor
        actor.pos(q)            # set its new position
    scene.render(resetcam=0)

scene.show(resetcam=0)
