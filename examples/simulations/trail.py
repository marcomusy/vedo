"""Add a trailing line to a moving object"""
from vedo import Plotter, sin, Sphere, Point


s = Sphere().c("green").bc("tomato")
s.cut_with_plane([-0.8, 0, 0])  # cut left part of sphere

p = Point([-2,0,0]).ps(12).color("black")

# add a trail to point p with 50 segments
p.add_trail(lw=3, n=50)

plt = Plotter(axes=6, interactive=False)

# add meshes to Plotter list
plt += [s, p, __doc__]

for i in range(150):
    p.pos(-2+i/100.0, sin(i/5.0)/15, 0).update_trail()
    plt.show(azimuth=-0.2)

# stay interactive and after pressing q close
plt.interactive().close()
