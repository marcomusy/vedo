"""Load an existing vtkStructuredGrid and draw
the streamlines of the velocity field"""
from vedo import *

######################## vtk
import vtk
# Read the data and specify which scalars and vectors to read.
pl3d = vtk.vtkMultiBlockPLOT3DReader()
fpath = download(dataurl+"combxyz.bin")
pl3d.SetXYZFileName(fpath)
fpath = download(dataurl+"combq.bin")
pl3d.SetQFileName(fpath)
pl3d.SetScalarFunctionNumber(100)
pl3d.SetVectorFunctionNumber(202)
pl3d.Update()
# this vtkStructuredData already has a vector field:
domain = pl3d.GetOutput().GetBlock(0)

######################## vedo
probe= Grid(pos=[9,0,30], normal=[1,0,0], s=[5,5], res=[6,6])

stream = streamLines(domain, probe, direction='backwards')

box = Mesh(domain).alpha(0.1)

show(stream, probe, box, __doc__, axes=7, bg='bb').close()
