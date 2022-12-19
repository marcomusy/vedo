"""Draw the shadow and trailing line of a flying plane. Not really
a simulation.. just a way to illustrate how to move objects around!"""
from vedo import *

world = Box(size=(30,15,8)).wireframe()
airplane = Mesh(dataurl+"cessna.vtk").c("green")

plt = Plotter(axes=1, interactive=False)

for t in np.arange(0, 3.2, 0.02):

    # make up some movement
    airplane.pos(9*t-15, 2-t, sin(3-t)).rotate_x(t) 
    if t==0:
        airplane.add_trail(n=200).add_shadow('z', -4)
    plt.show(world, airplane, __doc__, viewup="z", resetcam=False)

    if plt.escaped:
        break  # if ESC button is hit during the loop

plt.interactive().close()

