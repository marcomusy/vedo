"""
Load a Volume (tif stack).
Invoke a tool to cut off parts of it.
"""
print(__doc__)

from vtkplotter import Plotter, datadir

vp = Plotter()

vol = vp.load(datadir+"embryo.tif")

vp.addCutterTool(vol)
vp.show()
