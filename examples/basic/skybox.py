"""Embed a mesh into a skybox environment
Mesh lighting is by Physically Based Rendering (PBR)"""
from vedo import *

msh = Mesh(dataurl+"man.vtk").rotateX(-90)

# Use physically based rendering (PBR):
msh.c("white").lighting(metallicity=1, roughness=0.05)

# Specify a skybox environment from a HDR file
# (more skybox example HDR files at https://polyhaven.com/hdris)
cubemap_path = download(dataurl+"kloppenheim_06_4k.hdr")

show(msh, __doc__, bg=cubemap_path).close()

