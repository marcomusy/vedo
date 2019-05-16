"""
Load a Volume (atif stack).
Invoke a tool to cut off parts of it.
"""
print(__doc__)

from vtkplotter import Plotter, datadir


vp = Plotter(axes=4)

act = vp.load(datadir+"embryo.tif")

vp.addCutterTool(act)
vp.show()
