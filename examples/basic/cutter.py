"""Drag the sphere to cut the mesh interactively
Use mouse buttons to zoom and pan"""
from vedo import *

s = Mesh(dataurl+'cow.vtk')

plt = show(s, __doc__, bg='black', bg2='bb', interactive=False)
plt.addCutterTool(s, mode='sphere') #modes= sphere, plane, box
plt.close()