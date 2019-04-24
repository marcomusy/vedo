"""
Use a scalar to paint colored bands on a mesh,
this can be combined with opacities values for each vertex of the mesh.
Keyword depthpeeling improves the rendering of translucent objects.
"""
from vtkplotter import show, Hyperboloid, Torus, Text
from numpy import linspace

doc = Text(__doc__, c="k", bg="lg")


hyp = Hyperboloid()
scalars = hyp.coordinates()[:, 2]  # let z-coord be the scalar
hyp.pointColors(scalars, bands=5, cmap="rainbow")  # make color bands

tor = Torus(thickness=0.3)
scalars = tor.coordinates()[:, 2]  # let z-coord be the scalar
transp = linspace(1, 0.5, len(scalars))  # set transparencies from 1 -> .5
tor.pointColors(scalars, alpha=transp, bands=3, cmap="winter")

show(hyp, tor, doc, viewup="z", depthpeeling=1, axes=2)
