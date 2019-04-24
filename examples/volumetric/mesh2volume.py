"""
Convert a mesh it into volume (left in grey) representation as vtkImageData
where the foreground voxels are 1 and the background voxels are 0.

Right: the vtkImageData is isosurfaced.
"""
from vtkplotter import *

doc = Text(__doc__, c="k")

s = load(datadir+"shapes/bunny.obj").normalize().wire()

img = actor2ImageData(s, spacing=(0.02, 0.02, 0.02))

v = Volume(img, alphas=[0, 0.5]) 

iso = isosurface(img, smoothing=0.9).color("b")

show(v, s.scale(1.05), doc, at=0, N=2, bg="w")

show(iso, at=1, interactive=1)
