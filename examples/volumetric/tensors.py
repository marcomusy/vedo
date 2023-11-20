"""Visualize stress tensors as ellipsoids"""
import vtk
from vedo import *

# Create a test volume with tensors
pl = vtk.vtkPointLoad()
pl.SetLoadValue(50)
pl.SetSampleDimensions(6,6,6)
pl.ComputeEffectiveStressOn()
pl.SetPoissonsRatio(0.2)
pl.SetModelBounds(-10,10,-10,10,-10,10)
pl.Update()

vol = Volume(pl.GetOutput()).mode(1)
print(vol)

# Extract a slice of the volume data at index 3
zsl = vol.zslice(3)

# Generate tensor ellipsoids
tens1 = Tensors(vol, source='ellipse', scale=10).cmap("Reds")
print(tens1)

tens2 = Tensors(zsl, source='ellipse', scale=20).cmap("Greens")
print(tens2)

show([(vol, __doc__), tens1], N=2, axes=9, viewup='z').close()

show(vol, tens2, zsl, axes=9, viewup='z').close()

