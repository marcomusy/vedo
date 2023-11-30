# Example of usage of the madcad library
# See https://pymadcad.readthedocs.io/en/latest/index.html
import vedo
import numpy as np
import madcad

##########################################################################
mesh = vedo.Mesh(vedo.dataurl+"bunny.obj")
mesh.compute_normals()

madcad_mesh = vedo.utils.vedo2madcad(mesh)
madcad.thicken(madcad_mesh, thickness=0.1)
madcad.show([madcad_mesh])


##########################################################################
vedo_mesh = vedo.utils.madcad2vedo(madcad_mesh)

arrs = vedo.Arrows(vedo_mesh.vertices, vedo_mesh.vertices + 0.01 * vedo_mesh.pointdata["Normals"])
vedo.show(mesh, arrs, axes=1).close()
