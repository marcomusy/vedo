import vtk
from vtkplotter import *

# Create a volume with tensors
pl = vtk.vtkPointLoad()
pl.SetLoadValue(50)
pl.SetSampleDimensions(6,6,6)
pl.ComputeEffectiveStressOn()
pl.SetPoissonsRatio(0.2)
pl.SetModelBounds(-10,10,-10,10,-10,10)
pl.Update()
vol = Volume(pl.GetOutput(), mode=1)

# Extract a slice of the volume data at index 3
zsl = vol.zSlice(3)

# Generate tensor ellipsoids
#tens = Tensors(vol, source='ellipse', scale=10)
tens = Tensors(zsl, source='ellipse', scale=20)

#show([vol, [tens, zsl]], N=2, axes=1, viewup='z')
show(vol, tens, zsl, axes=1, bg='w', viewup='z')
