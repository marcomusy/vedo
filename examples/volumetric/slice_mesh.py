"""Probe a Volume with a Mesh"""
from vedo import *

# Load a Volume
vol = Volume(dataurl + 'embryo.slc')
vol.cmap('bone').mode(1)

# Create a Mesh (can be any mesh)
msh = Paraboloid(res=200).scale(200).pos([100,100,200])

# Probe the Volume with the Mesh
# and colorize it with the probed values
msh.probe(vol)
msh.cmap('Spectral').add_scalarbar().print()

show(vol, msh, __doc__, axes=1).close()
