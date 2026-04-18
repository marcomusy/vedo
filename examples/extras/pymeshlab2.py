"""pymeshlab interoperability example:
Surface reconstruction by ball pivoting"""

import sys

import vedo

try:
    import pymeshlab  # tested on pymeshlab-2022.2.post2
except ModuleNotFoundError:
    print("Skipping example: optional dependency 'pymeshlab' is not installed.")
    print("Install with: pip install pymeshlab")
    sys.exit(0)

# Configure inputs and run the visualization workflow.
pts = vedo.Mesh(vedo.dataurl + "cow.vtk").points  # numpy array of vertices

m = pymeshlab.Mesh(vertex_matrix=pts)

ms = pymeshlab.MeshSet()
ms.add_mesh(m)

# API compatibility across pymeshlab versions.
if hasattr(pymeshlab, "Percentage"):
    p = pymeshlab.Percentage(2)
else:
    p = pymeshlab.PercentageValue(2)
ms.generate_surface_reconstruction_ball_pivoting(ballradius=p)

mlab_mesh = ms.current_mesh()
reco_mesh = vedo.Mesh(mlab_mesh).compute_normals().flat().backcolor("t")

vedo.show(
    __doc__,
    vedo.Points(pts),
    reco_mesh,
    axes=True,
    bg2="blue9",
    title="vedo + pymeshlab",
)
