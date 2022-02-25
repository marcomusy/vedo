"""Add a trailing line to a moving object"""
from vedo import Plotter, sin, Sphere, Point


s = Sphere().c("green").bc("tomato")
s.cutWithPlane([-0.8, 0, 0])  # cut left part of sphere

p = Point([1,1,1], r=12, c="black")

# add a trail to point p with max length 0.5 and 50 segments
p.addTrail(lw=3, maxlength=0.5, n=50)

plt = Plotter(axes=6, interactive=False)

# add meshes to Plotter list
plt += [s, p, __doc__]

for i in range(200):
    p.pos(-2+i/100.0, sin(i/5.0)/15, 0)
    plt.show(azimuth=-0.2)
    if plt.escaped:
        break # if ESC is hit during the loop

# stay interactive and after pressing q close
plt.interactive().close()
