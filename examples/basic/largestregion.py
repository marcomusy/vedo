# Extract the mesh region that has the largest surface
#
from vtkplotter.analysis import extractLargestRegion
from vtkplotter.utils import area
from vtkplotter import Plotter, printc

vp = Plotter(shape=(2,1))

act1 = vp.load('examples/data/embryo.slc')

act2 = extractLargestRegion(act1)

printc('areas =', area(act1), area(act2))

vp.show(act1, at=0)
vp.show(act2, at=1, interactive=1)
