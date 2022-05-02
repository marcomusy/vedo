import numpy as np
import vedo

import magpylib  # pip install magpylib
coil1 = magpylib.Collection()
for z in np.linspace(-8, 8, 16):
    winding = magpylib.current.Loop(
        current=100,
        diameter=10,
        position=(0,0,z),
    )
    coil1.add(winding)

#####################
vol = vedo.Volume(
	dims=(41, 41, 41),
    spacing=(2, 2, 2),
    origin=(-40, -40, -40),
)

# compute B-field and add as data to
pts = vol.points()
vol.pointdata['B'] = coil1.getB(pts)

# compute field lines
seeds = vedo.Disc(r1=1, r2=5.2, res=(3,12))

streamlines = vedo.StreamLines(
	vol,
	seeds,
    maxPropagation=180,
    initialStepSize=0.1,
    direction="both",
)
streamlines.cmap("bwr", "B").lw(5).addScalarBar("mT")

vedo.show(streamlines, axes=1)
