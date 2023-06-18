"""Draw the shadows and trailing lines of 2 planes. Not really
a simulation.. just a way to illustrate how to move objects around!"""
from vedo import *

world = Box([0,0,0], 30, 16, 8).wireframe()

plane1 = Mesh(dataurl+"cessna.vtk").c("green")
plane1.pos(-15, 2, 0.15).add_trail(n=100)
plane1.add_shadow('z', -4).add_shadow('y', 8)

plane2 = plane1.clone().c("tomato")
plane2.pos(-15,-2,-0.20).add_trail(n=100)
plane2.add_shadow('z', -4).add_shadow('y', 8)

# Setup the scene
plt = Plotter(interactive=False)
plt.show(world, plane1, plane2, __doc__, viewup="z")

for t in np.arange(0, 3.2, 0.04):
    plane1.pos(9*t-15, 2-t, sin(3-t)).rotate_x(0+t) # make up some movement
    plane2.pos(8*t-15, t-2, sin(t-3)).rotate_x(2-t) # for the 2 planes
    plane1.update_trail().update_shadows()
    plane2.update_trail().update_shadows()
    plt.render()

plt.interactive().close()
