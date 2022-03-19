"""Extract the mesh region that
has the largest connected surface"""
from vedo import dataurl, Volume, printc, Plotter

mesh1 = Volume(dataurl+"embryo.tif").isosurface(80).c("yellow")
printc("area1 =", mesh1.area(), c="yellow")

mesh2 = mesh1.extractLargestRegion().color("lb")
printc("area2 =", mesh2.area(), c="lb")

plt = Plotter(shape=(2,1), axes=7)
plt.at(0).show(mesh1, __doc__)
plt.at(1).show(mesh2)
plt.interactive().close()
