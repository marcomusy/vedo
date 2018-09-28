# Invoke a tool to cut off parts of a mesh
# Press X to save the mesh or to add new cut planes
#
from vtkplotter import Plotter


vp = Plotter(axes=4)

act = vp.load('data/embryo.tif', c='blue')

vp.addCutterTool(act)

vp.show()
