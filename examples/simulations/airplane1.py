"""Draw the shadow and trailing line of a moving object."""
from vedo import *

world = Box(size=(30,15,8)).wireframe()
airplane = Mesh(dataurl+"cessna.vtk").c("green")
airplane.pos(-15, 2.0, 0.15)
airplane.add_trail(n=100).add_shadow('z', -4)

plt = Plotter(interactive=False)
plt.show(world, airplane, __doc__, viewup="z")

for t in np.arange(0, 3.2, 0.04):
    pos = (9*t-15, 2-t, sin(3-t))  # make up some movement
    airplane.pos(pos).rotate_x(t)
    airplane.update_trail()
    airplane.update_shadows()
    plt.render()

plt.interactive().close()

