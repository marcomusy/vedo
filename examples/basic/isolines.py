"""Draw the isolines and isobands of
a scalar field H (height) on a surface"""
from vedo import *

mesh0 = ParametricShape('RandomHills')
# ParametricShapes already have a scalar associated to points
printc('Mesh arrays are:    ', mesh0.getArrayNames())
# so assign it a colormap:
mesh0.cmap('terrain')

isol = mesh0.isolines(n=10).color('w')
show(mesh0, isol, __doc__, at=0, N=4, size=(1200,900), axes=11)

isob = mesh0.isobands(n=5).addScalarBar(title="H=Elevation")
show(isob, at=1)

# make a copy and interpolate the Scalars from points to cells
mesh1 = mesh0.clone(deep=False).mapPointsToCells()
printc('Mesh arrays are now:', mesh1.getArrayNames())

gvecs = mesh1.gradient('Scalars', on='cells')
cc = mesh1.cellCenters()
ars = Arrows(cc, cc + gvecs*0.01, c='bone_r').lighting('off')
ars.addScalarBar3D(title='|\nablaH|~\dot~0.01 [arb.units]')
show(mesh1, isol, ars, "Arrows=\nablaH", at=2)

# colormap the gradient magnitude directly on the mesh
mesh2 = mesh1.clone(deep=False).lw(0.1).cmap('jet', mag(gvecs), on='cells')
mesh2.addScalarBar3D(title='|\nablaH| [arb.units]')
cam = dict(pos=(2.57, -1.92, 3.25), # get these nrs by pressing C
           focalPoint=(0.195, 1.06, 8.65e-3),
           viewup=(-0.329, 0.563, 0.758),
           distance=5.00,
           clippingRange=(1.87, 8.79))
show(mesh2, "Color=|\nablaH|", at=3, camera=cam, interactive=True)
