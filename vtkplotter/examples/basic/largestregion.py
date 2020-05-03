"""Extract the mesh region that
has the largest connected surface
"""
from vtkplotter import *

mesh1 = load(datadir+"embryo.tif", threshold=80, c="y")
printc("area1 =", mesh1.area(), c="y")

mesh2 = mesh1.extractLargestRegion().color("lb")
printc("area2 =", mesh2.area(), c="b")

show(mesh1, __doc__, at=0, shape=(2, 1))
show(mesh2, at=1, zoom=1.2, interactive=1)
