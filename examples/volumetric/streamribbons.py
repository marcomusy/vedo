"""Load an existing vtkStructuredGrid and draw the
lines of the velocity field joining them in ribbons"""
import vtk
from vedo import *

######################## vtk
# Read the data and specify which scalars and vectors to read.
pl3d = vtk.vtkMultiBlockPLOT3DReader()
pl3d.SetXYZFileName(download(datadir+"combxyz.bin"))
pl3d.SetQFileName(download(datadir+"combq.bin"))
pl3d.SetScalarFunctionNumber(100)
pl3d.SetVectorFunctionNumber(202)
pl3d.Update()
# this vtkStructuredData already contains a vector field:
domain = pl3d.GetOutput().GetBlock(0)

######################## vedo
box = Mesh(domain, c=None, alpha=0.1)

probe = Line([9,0,28], [11,0,33], res=11).color('k')

stream = streamLines(domain, probe, direction='backwards', ribbons=2)

show(box, probe, stream, __doc__, axes=7, bg='bb')
