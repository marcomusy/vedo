"""Streamlines originating from a probing sphere
in a volume domain defined by the pink hyperboloid.
The vector field is given by the coords of the hyperboloid,
this field is interpolated to the whole bounding box.
"""
from vedo import *

mesh = Hyperboloid(pos=(0,0,0)).alpha(0.2)

vects = mesh.clone().points() # let's assume this
mesh.pointdata["hyp_coords"] = vects

probe = Sphere(pos=[0,0.6,0.3], r=0.3, res=8).clean()
probe.wireframe().alpha(0.2).color('g')

stream = streamLines(mesh, probe,
                     maxPropagation=0.3,
                     extrapolateToBoundingBox={'dims':(10,10,10)})

show(stream, probe, mesh, mesh.box(), __doc__, axes=3, viewup='z').close()
