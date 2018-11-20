# A mixed example with class vtkSignedDistance:
# generate a scalar field by the signed distance from
# a polydata, save it to stack.tif file,
# then extract an isosurface from the 3d image.
#
import vtk
from vtkplotter import Plotter

vp = Plotter()

act = vp.load("data/290.vtk").normalize().subdivide()

# Generate signed distance function and contour it
dist = vtk.vtkSignedDistance()
dist.SetInputData(act.polydata())
dist.SetRadius(0.2) #how far out to propagate distance calculation
dist.SetBounds(-2,2, -2,2, -2,2)
dist.SetDimensions(80, 80, 80)
dist.Update()

#vp.write(dist.GetOutput(), 'stack.tif')

fe = vtk.vtkExtractSurface()
fe.SetInputConnection(dist.GetOutputPort())
fe.SetRadius(0.2) # this should match the signed distance radius
fe.Update()

pts = vp.points(act.coordinates())

vp.show([fe.GetOutput(), pts])

