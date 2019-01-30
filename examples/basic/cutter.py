'''
Load a vtkImageData (tif stack) and convert on the the fly to an isosurface.
Invoke a tool to cut off parts of a mesh
Press X to save the mesh or to add new cut planes
'''
print(__doc__)

from vtkplotter import Plotter


vp = Plotter(axes=4)

act = vp.load('data/embryo.tif', c='blue')

vp.addCutterTool(act)
