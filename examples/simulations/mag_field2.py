import magpylib  # pip install magpylib
import numpy as np
import vedo

coil1 = magpylib.Collection()
for z in np.linspace(-8, 8, 16):
    winding = magpylib.current.Loop(
        current=100,
        diameter=10,
        position=(0,0,z),
    )
    coil1.add(winding)

#####################
volume = vedo.Volume(
    dims=(41, 41, 41),
    spacing=(2, 2, 2),
    origin=(-40, -40, -40),
)

# compute B-field and add as pointdata to volume
volume.pointdata['B'] = coil1.getB(volume.vertices)

# compute field lines
seeds = vedo.Disc(r1=1, r2=5.2, res=(3,12))

streamlines = vedo.StreamLines(
    volume,
    seeds,
    max_propagation=180,
    initial_step_size=0.1,
    direction="both",
)
streamlines.cmap("RdBu_r", "B").lw(5).add_scalarbar("mT")

vedo.show(streamlines, axes=1)
