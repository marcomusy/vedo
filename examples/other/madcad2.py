"""Convert a vedo mesh to a madcad mesh and vice versa"""
# See https://pymadcad.readthedocs.io/en/latest/index.html
from pathlib import Path
import sys

import vedo
from vedo.external import vedo2madcad, madcad2vedo

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _optional import require_module

madcad = require_module("madcad", "pymadcad")

# Configure inputs and run the visualization workflow.
mesh = vedo.Mesh(vedo.dataurl+"bunny.obj")
mesh.compute_normals()

############################################################
madcad_mesh = vedo2madcad(mesh)

madcad.thicken(madcad_mesh, thickness=0.1)
if vedo.settings.dry_run_mode == 0:
    madcad.show([madcad_mesh])


#############################################################
vedo_mesh = madcad2vedo(madcad_mesh)

verts = vedo_mesh.vertices
norms = vedo_mesh.pointdata["Normals"]
arrs = vedo.Arrows(verts, verts + 0.005 * norms)
vedo.show(mesh, arrs, __doc__, axes=1).close()
