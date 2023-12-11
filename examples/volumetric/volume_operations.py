"""
Perform other simple mathematical operation between 3D Volumes.
Possible operations are: +, -, /, 1/x, sin, cos, exp, log, abs, **2, sqrt,
  min, max, atan, atan2, median, mag, dot, gradient, divergence, laplacian.
Alphas defines the opacity transfer function in the scalar range.
"""
from vedo import *
printc(__doc__)

plt = Plotter(N=4)

v0 = Volume(dataurl+'embryo.slc').cmap(0).add_scalarbar3d()
v0.scalarbar = v0.scalarbar.clone2d("center-right", 0.1) # substitute to a 2d scalarbar
plt.at(0).show("original", v0)

v1 = v0.clone().operation("gradient").operation("mag").add_scalarbar3d()
v1.scalarbar = v1.scalarbar.clone2d("center-right", 0.1)
plt.at(1).show("gradient", v1)

v2 = v0.clone().operation("median").cmap(4).add_scalarbar3d()
v2.scalarbar = v2.scalarbar.clone2d("center-right", 0.1)
plt.at(2).show("median", v2)

v3 = v0.clone().operation("dot", v0).cmap(7).add_scalarbar3d()
v3.scalarbar = v3.scalarbar.clone2d("center-right", 0.1)
plt.at(3).show("dot(v0,v0)", v3, zoom=1.3)

plt.interactive().close()


####################################################################################
#Start with creating a masked Volume then compute its gradient and probe 2 points
msh = Ellipsoid()

vol = msh.signed_distance(dims=[20, 20, 20])
vol.threshold(above=0.0, replace=0.0)  # replacing all values outside to 0
vol.cmap("blue").alpha([0.9, 0.0]).alpha_unit(0.1).add_scalarbar3d()
vol.scalarbar = vol.scalarbar.clone2d("center-right", 0.15)

vgrad = vol.operation("gradient")
printc(vgrad.pointdata, c='g')

grd = vgrad.pointdata['ImageScalarsGradient']
pts = vol.vertices  # coords as numpy array
arrs = Arrows(pts, pts + grd*0.1, c="jet")

pts_probes = [[0.2,0.5,0.5], [0.2,0.3,0.4]]
vpts_probes = Points(pts_probes).probe(vgrad)
vects = vpts_probes.pointdata['ImageScalarsGradient']

arrs_pts_probe = Arrows(pts_probes, pts_probes + vects, c='black')

plt = Plotter(axes=1, N=2)
plt.at(0).show("A masked Volume", vol)
plt.at(1).show("..compute its gradient and probe 2 points", arrs, arrs_pts_probe)
plt.interactive().close()
