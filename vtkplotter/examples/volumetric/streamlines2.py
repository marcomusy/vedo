"""Load an existing vtkStructuredGrid and draw
the streamlines of the velocity field"""
import vtk
from vtkplotter import *

######################## vtk
# Read the data and specify which scalars and vectors to read.
pl3d = vtk.vtkMultiBlockPLOT3DReader()
pl3d.SetXYZFileName(datadir+"combxyz.bin")
pl3d.SetQFileName(datadir+"combq.bin")
pl3d.SetScalarFunctionNumber(100)
pl3d.SetVectorFunctionNumber(202)
pl3d.Update()
# this vtkStructuredData already has a vector field:
domain = pl3d.GetOutput().GetBlock(0)

######################## vtkplotter
probe= Grid(pos=[9,0,30], normal=[1,0,0], sx=5, sy=5, resx=6, resy=6)

stream = streamLines(domain, probe, direction='backwards')

box = Mesh(domain).alpha(0.1)

show(stream, probe, box, __doc__, axes=8, bg='bb')
