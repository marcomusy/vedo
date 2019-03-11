"""
Modify mesh vertex positions.
(vtkActor transformation matrix is reset when mesh is modified)
"""
from vtkplotter import Plotter, Disc, Text
import numpy as np


vp = Plotter(axes=4, interactive=0)

dsc = Disc()

coords = dsc.coordinates()

for i in range(50):
    noise = np.random.randn(len(coords), 3) * 0.02
    noise[:, 0] = 0  # only perturb z
    noise[:, 1] = 0
    dsc.setPoints(coords + noise)  # modify mesh
    vp.show(dsc, elevation=-1)

vp.add(Text(__doc__))

vp.show(interactive=1)
