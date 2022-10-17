"""
Perform other simple mathematical operation between 3d Volumes.
Possible operations are: +, -, /, 1/x, sin, cos, exp, log, abs, **2, sqrt,
  min, max, atan, atan2, median, mag, dot, gradient, divergence, laplacian.
Alphas defines the opacity transfer function in the scalar range.
"""
from vedo import *
printc(__doc__)

plt = Plotter(N=6)

v0 = Volume(dataurl+'embryo.slc').c(0)
v0.add_scalarbar3d()
plt.at(0).show("original", v0)

v1 = v0.clone().operation("gradient").operation("mag")
v1.add_scalarbar3d()
# print(v1.pointdata.keys())
plt.at(1).show("gradient", v1)

v2 = v0.clone().operation("divergence").c(2)
v2.add_scalarbar3d()
plt.at(2).show("divergence", v2)

v3 = v0.clone().operation("laplacian").c(3)
v3.add_scalarbar3d()
plt.at(3).show("laplacian", v3)

v4 = v0.clone().operation("median").c(4)
v4.add_scalarbar3d()
plt.at(4).show("median", v4)

v5 = v0.clone().operation("dot", v0).c(7)
v5.add_scalarbar3d()
plt.at(5).show("dot(v0,v0)", v5, zoom=1.3)

plt.interactive().close()


############################################################# example application
#Start with creating a masked Volume then compute its gradient and probe 2 points
msh = Ellipsoid()

vol = msh.signed_distance(dims=(20, 20, 20))
vol.threshold(above=0.0, replace=0.0)  # replacing all values outside to 0
vol.c("blue").alpha([0.9, 0.0]).alpha_unit(0.1).add_scalarbar3d()

vgrad = vol.operation("gradient")
printc(vgrad.pointdata.keys(), c='g')

grd = vgrad.pointdata['ImageScalarsGradient']
pts = vol.points()  # coords as numpy array
arrs = Arrows(pts, pts + grd*0.1).lighting('off')

pts_probes = [[0.2,0.5,0.5], [0.2,0.3,0.4]]
vects = probe_points(vgrad, pts_probes).pointdata['ImageScalarsGradient']
arrs_pts_probe = Arrows(pts_probes, pts_probes+vects)

plt = Plotter(axes=1, N=2)
plt.at(0).show("A masked Volume", vol)
plt.at(1).show("..compute its gradient and probe 2 points", arrs, arrs_pts_probe)
plt.interactive().close()
