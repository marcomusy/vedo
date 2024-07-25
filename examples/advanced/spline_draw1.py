from vedo import dataurl, Image, Mesh
from vedo.applications import SplinePlotter  # ready to use class!

pic = Image(dataurl + "images/embryo.jpg")

# Works with surfaces too
# pic = Mesh(dataurl + "bunny.obj").scale(80).shift(dz=-1)
# pic.color("blue9").alpha(0.75).backface_culling()

plt = SplinePlotter(pic)
plt.show(mode="image", zoom='tightest')

if plt.line:
    print("Npts =", len(plt.points()), "NSpline =", plt.line.npoints)
