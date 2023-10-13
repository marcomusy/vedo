"""Slice/probe a Volume with a Mesh"""
from vedo import *

vol = Volume(dataurl + 'embryo.slc')
vol.cmap('bone').mode(1)
msh = Paraboloid(res=200).scale(200).pos(100,100,200)

scalars = vol.probe_points(msh).pointdata[0]
msh.cmap('Spectral', scalars).add_scalarbar()

show(vol, msh, __doc__, axes=True).close()
