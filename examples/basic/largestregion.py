# Extract the mesh region that has the largest connected surface
#
from vtkplotter.analysis import extractLargestRegion
from vtkplotter.utils import area
from vtkplotter import Plotter, printc

vp = Plotter(shape=(2,1))

act1 = vp.load('data/embryo.slc', c='y')
printc('area1 =', area(act1), c='y')

act2 = extractLargestRegion(act1).color('b')
printc('area2 =', area(act2), c='b')

vp.show(act1, at=0)
vp.show(act2, at=1, zoom=1.2, interactive=1)
