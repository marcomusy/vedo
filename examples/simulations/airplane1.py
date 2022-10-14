"""Draw the shadow and trailing line of a flying plane. Not really
a simulation.. just a way to illustrate how to move objects around!"""
from vedo import *

settings.allow_interaction = True # if ESC button is hit during the loop

world = Box([0,0,0], 30, 15, 8).wireframe()

plane = Mesh(dataurl+"cessna.vtk").c("green").add_shadow('z', -4)
plane.pos(-15, 2, 0.14).add_trail(n=200)

# Setup the scene
plt = Plotter(axes=1, interactive=False)

for t in np.arange(0, 3.2, 0.02):

    plane.pos(9*t-15, 2-t, sin(3-t)).rotate_x(0+t) # make up some movement

    plt.show(world, plane, __doc__, viewup="z")
    if plt.escaped:
        break  # if ESC button is hit during the loop

plt.interactive().close()
