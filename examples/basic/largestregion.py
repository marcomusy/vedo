'''
Extract the mesh region that has the largest connected surface
'''
from vtkplotter import *


act1 = load('data/embryo.slc', c='y')
printc('area1 =', act1.area(), c='y')

act2 = extractLargestRegion(act1).color('b')
printc('area2 =', act2.area(), c='b')

show([act1, Text(__doc__)], at=0, shape=(2,1))
show(act2, at=1, zoom=1.2, interactive=1)
