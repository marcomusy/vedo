# Example usage with pygmsh package:
# https://github.com/nschloe/pygmsh
#
import pygmsh

geom = pygmsh.opencascade.Geometry(
  characteristic_length_min=0.1,
  characteristic_length_max=0.1,
  )

rectangle = geom.add_rectangle([-1.0, -1.0, 0.0], 2.0, 2.0)
disk1 = geom.add_disk([-1.2, 0.0, 0.0], 0.5)
disk2 = geom.add_disk([+1.2, 0.0, 0.0], 0.5)
union = geom.boolean_union([rectangle, disk1, disk2])

disk3 = geom.add_disk([0.0, -0.9, 0.0], 0.5)
disk4 = geom.add_disk([0.0, +0.9, 0.0], 0.5)
flat = geom.boolean_difference([union], [disk3, disk4])

geom.extrude(flat, [0, 0, 0.3])

mesh = pygmsh.generate_mesh(geom)


from vedo import *

m1 = Mesh( mesh.points ).pointSize(3)
m2 = Mesh([mesh.points, mesh.cells['triangle']])
m3 = Mesh([mesh.points, mesh.cells['tetra']])
m3.cutWithPlane(normal=(-1,0,-1))
m4 = Mesh([mesh.points, list(mesh.cells['triangle']) + list(mesh.cells['tetra'])])

show([m1, m2, (m3, m1.box()), m4.lw(0.1)], N=4)
