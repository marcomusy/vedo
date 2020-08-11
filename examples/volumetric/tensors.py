"""Visualize stress tensors as ellipsoids"""
import vtk
from vedo import *

# Create a volume with tensors
pl = vtk.vtkPointLoad()
pl.SetLoadValue(50)
pl.SetSampleDimensions(6,6,6)
pl.ComputeEffectiveStressOn()
pl.SetPoissonsRatio(0.2)
pl.SetModelBounds(-10,10,-10,10,-10,10)

vol = Volume(pl, mode=1)

# Extract a slice of the volume data at index 3
zsl = vol.zSlice(3)

# Generate tensor ellipsoids
tens1 = Tensors(vol, source='ellipse', scale=10)
tens2 = Tensors(zsl, source='ellipse', scale=20)

show([(vol, __doc__), tens1], N=2, axes=9, bg='w', viewup='z')
show(vol, tens2, zsl, axes=9, viewup='z', new=True)
