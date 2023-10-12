"""pymeshlab interoperability example:
Surface reconstruction by ball pivoting"""
import vedo
import pymeshlab  # tested on pymeshlab-2022.2.post2

pts = vedo.Mesh(vedo.dataurl+'cow.vtk').vertices # numpy array of vertices

m = pymeshlab.Mesh(vertex_matrix=pts)

ms = pymeshlab.MeshSet()
ms.add_mesh(m)

p = pymeshlab.Percentage(2)
ms.generate_surface_reconstruction_ball_pivoting(ballradius=p)

mlab_mesh = ms.current_mesh()
reco_mesh = vedo.Mesh(mlab_mesh).compute_normals().flat().backcolor('t')

vedo.show(
    __doc__, vedo.Points(pts), reco_mesh,
    axes=True, bg2='blue9', title="vedo + pymeshlab",
)
