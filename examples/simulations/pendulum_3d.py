"""Double pendulum in 3D
Press ESC to quit"""
# Original idea and solution using sympy from Mr. P Solver at:
# https://www.youtube.com/watch?v=MtG9cueB548
from vedo import *

settings.allowInteraction = True

# Load the solution:
x1, y1, z1,  x2, y2, z2 = np.load(download(dataurl+'3Dpen.npy'))
p1, p2 = np.c_[x1,y1,z1], np.c_[x2,y2,z2]

ball1 = Sphere(c="green5", r=0.1, pos=p1[0]).add_trail(n=10).add_shadow('z', -3)
ball2 = Sphere(c="blue5",  r=0.1, pos=p2[0]).add_trail(n=10).add_shadow('z', -3)
ball1.trail.add_shadow('z', -3) # make trails project a shadow too
ball2.trail.add_shadow('z', -3)

rod1 = Line([0,0,0], ball1, lw=4)
rod2 = Line(ball1, ball2, lw=4)

axes = Axes(xrange=(-3,3), yrange=(-3,3), zrange=(-3,3))

plt = Plotter(interactive=False)
plt.show(axes, __doc__, viewup='z')

pb = ProgressBar(0, len(p1))
for b1, b2 in zip(p1,p2):
    ball1.pos(b1)
    ball2.pos(b2)
    rod1.stretch([0,0,0], b1)
    rod2.stretch(b1, b2)
    # show at max frame rate of 15 Hz
    plt.show(ball1, ball2, rod1, rod2, resetcam=False, rate=15)
    pb.print()
    if plt.escaped:  # if ESC is pressed
        break

plt.interactive()
