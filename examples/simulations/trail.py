"""
Example usage of addTrail().
Add a trailing line to a moving object.
"""
print(__doc__)
from vedo import Plotter, sin, Sphere, Point


plt = Plotter(axes=6, interactive=False)

s = Sphere().c("green").bc("tomato")
s.cutWithPlane([-0.9, 0, 0])  # cut left part of sphere

p = Point([1,1,1], r=12, c="black")

# add a trail to point p with max length 0.5 and 50 segments
p.addTrail(lw=3, maxlength=0.5, n=50)

# add meshes to Plotter list
plt += [s, p]

for i in range(200):
    p.pos(-2+i/100.0, sin(i/5.0)/15, 0)
    plt.camera.Azimuth(-0.2)
    plt.show()
    if plt.escaped: break # if ESC is hit during the loop

plt.close()
