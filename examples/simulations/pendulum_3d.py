"""Double pendulum in 3D"""
# Original idea and solution using sympy from:
# https://www.youtube.com/watch?v=MtG9cueB548
import time
from vedo import *

# Load the solution:
x1, y1, z1,  x2, y2, z2 = np.load(download(dataurl+'3Dpen.npy'))
p1, p2 = np.c_[x1,y1,z1], np.c_[x2,y2,z2]

ball1 = Sphere(p1[0], r=0.1).color("green5")
ball2 = Sphere(p2[0], r=0.1).color("blue5")

ball1.add_shadow('z', -3)
ball2.add_shadow('z', -3)

ball1.add_trail(n=10)
ball2.add_trail(n=10)
ball1.trail.add_shadow('z', -3) # make trails project a shadow too
ball2.trail.add_shadow('z', -3)

rod1 = Line([0,0,0], ball1, lw=4).add_shadow('z', -3)
rod2 = Line(ball1, ball2, lw=4).add_shadow('z', -3)

axes = Axes(xrange=(-3,3), yrange=(-3,3), zrange=(-3,3))

# show the solution
plt = Plotter(interactive=False)
plt.show(ball1, ball2, rod1, rod2, axes, __doc__, viewup='z')

i = 0
for b1, b2 in zip(p1,p2):
    ball1.pos(b1)
    ball2.pos(b2)
    ball1.update_trail().update_shadows()
    ball2.update_trail().update_shadows()
    rod1.vertices = [[0,0,0], b1]
    rod2.vertices = [b1, b2]
    rod1.update_shadows()
    rod2.update_shadows()
    plt.render()
    time.sleep(0.03)
    i += 1
    if i > 100:
        break

plt.interactive().close()
