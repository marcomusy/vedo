# Draw the shadow and trailing lines of 2 planes. This is not really
# a simulation.. just a way to illustrate how to move objects around!
from vtkplotter import *

world = Box([0,0,0], 30, 15, 8).wire()

p1 = load(datadir+"cessna.vtk").c("green").addTrail(lw=2, n=50)
p2 = p1.clone().c("tomato").addTrail(lw=2, n=50) # make a copy

# Setup the scene, creating the Plotter object is automatic
show(world, p1, p2, axes=1, bg="white", viewup="z", resetcam=0, interactive=0)

for x in arange(0, 3.5, 0.01):
    p1.pos(9*x-15,  2-x, sin( 3-x)).rotateX(0+x) # make up some fancy movement
    p2.pos(8*x-15, -2+x, sin(-3+x)).rotateX(2-x)

    shad = Shadow(p1, p2, direction='z').z(-4) # fix z position of the shadow
    show(world, p1, p2, shad)

interactive()
