"""
A mixed example with class vtkSignedDistance:

generate a scalar field by the signed distance from a polydata, 
save it to stack.tif file,
then extract an isosurface from the 3d image.
"""
from vtkplotter import Plotter, Points, Text, datadir

vp = Plotter(verbose=0)

act = vp.load(datadir+"290.vtk").normalize().subdivide().computeNormals()

# Generate signed distance function and contour it
import vtk

dist = vtk.vtkSignedDistance()
dist.SetInputData(act.polydata())
dist.SetRadius(0.2)  # how far out to propagate distance calculation
dist.SetBounds(-2, 2, -2, 2, -2, 2)
dist.SetDimensions(80, 80, 80)
dist.Update()

# vp.write(dist.GetOutput(), 'stack.tif')

fe = vtk.vtkExtractSurface()
fe.SetInputConnection(dist.GetOutputPort())
fe.SetRadius(0.2)  # this should match the signed distance radius
fe.Update()

pts = Points(act.coordinates())

vp.show(fe.GetOutput(), pts, Text(__doc__))
