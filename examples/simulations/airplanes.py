# Draw the shadow and trailing lines of 2 planes. This is not really
# a simulation.. just a way to illustrate how to move objects around!
from vedo import *
import numpy as np

settings.allowInteraction = True # if ESC button is hit during the loop

world = Box([0,0,0], 30, 15, 8).wireframe()

plane1 = Mesh(dataurl+"cessna.vtk").c("green").addTrail().addShadow(plane='z', point=-4)
plane2 = plane1.clone().c("tomato").addShadow(plane='z', point=-4)

# Setup the scene
plotter = show(world, plane1, plane2, axes=1, viewup="z", interactive=0)

for t in np.arange(0, 3.2, 0.01):
    plane1.pos(9*t-15, 2-t, sin(3-t)).rotateX(0+t) # make up some movement
    plane2.pos(8*t-15, t-2, sin(t-3)).rotateX(2-t) # for the 2 planes
    plotter.show(world, plane1, plane2)
    if plotter.escaped:
        break  # if ESC button is hit during the loop

plotter.interactive().close()
