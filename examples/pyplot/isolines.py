"""Draw the isolines and isobands of
a scalar field H (height) on a surface"""
from vedo import *

mesh0 = ParametricShape('RandomHills')

# ParametricShapes already have a scalar associated to points
printc('Mesh point arrays:', mesh0.pointdata.keys())
# so assign it a colormap:
mesh0.cmap('terrain')

isol = mesh0.isolines(n=10).color('w')
isob = mesh0.isobands(n=5).add_scalarbar("H=Elevation")

# make a copy and interpolate the Scalars from points to cells
mesh1 = mesh0.clone().map_points_to_cells()
printc('Mesh cell arrays :', mesh1.celldata.keys())

gvecs = mesh1.gradient(on='cells')
cc = mesh1.cell_centers
ars = Arrows(cc, cc + gvecs*0.01, c='bone_r').lighting('off')
ars.add_scalarbar3d(title='|:nablaH|:dot0.01 [arb.units]')

# colormap the gradient magnitude directly on the mesh
mesh2 = mesh1.clone().cmap('jet', mag(gvecs), on='cells')
mesh2.add_scalarbar3d(title='|:nablaH| [arb.units]')

plt = Plotter(N=4, size=(1200,900), axes=11)
plt.at(0).show(mesh0, isol, __doc__)
plt.at(1).show(isob)
plt.at(2).show(mesh1, isol, ars, "Arrows=:nablaH")
plt.at(3).show(mesh2, "Color=|:nablaH|")
plt.interactive().close()
