import plotter
from random import uniform as u


N = 10      # total nr of particles
s = 0.005   # random step size

scene = plotter.vtkPlotter(verbose=0)
scene.plane(center=[.45,.45,-.05], texture='wood7')

Nf = float(N)
for i in range(N):              # generate a grid of points
    for j in range(N): 
        for k in range(N):
            p = [i/Nf, j/Nf, k/Nf]
            scene.point(p, c=p) # color by its position
print('Scene is ready, press q to continue')
scene.show()

for t in range(1000):           # time loop
    for i in range(1, N*N*N):   # for each particle
        actor = scene.actors[i]
        r = [u(-s,s), u(-s,s), u(-s,s)] # random step
        p = actor.pos()         # get point position
        q = p + r 
        if q[2]<0: q[2] *= -1   # bounce on the floor
        actor.pos(q)            # set its new position
    scene.render(resetcam=0)
scene.interact()
