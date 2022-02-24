import vtk
from vedo import Grid, Tensors, show

domain = Grid(res=[5,5], c='gray')

# Generate random attributes on a plane
ag = vtk.vtkRandomAttributeGenerator()
ag.SetInputData(domain.polydata())
ag.GenerateAllDataOn()
ag.Update()

ts = Tensors(ag.GetOutput(), scale=0.1)
ts.print()

show(domain, ts).close()

