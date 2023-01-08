import vedo
import pymeshlab  # tested on pymeshlab-2022.2.post2

filepath = vedo.download(vedo.dataurl+'bunny.obj')

ms = pymeshlab.MeshSet()
ms.load_new_mesh(filepath)

pt = [0.0234, 0.0484, 0.0400]
ms.compute_scalar_by_geodesic_distance_from_given_point_per_vertex(startpoint=pt)
# vedo.show(ms, axes=True) # this already works!

mlab_mesh = ms.current_mesh()

vedo_mesh = vedo.Mesh(mlab_mesh).cmap('Paired').add_scalarbar("distance")

print("We can also convert it back to pymeshlab.MeshSet:",
      type(vedo.utils.vedo2meshlab(vedo_mesh))
)

vedo.show(
    __doc__, vedo_mesh, vedo.Point(pt),
    axes=True, bg='green9', bg2='blue9', title="vedo + pymeshlab",
)

################################################################################
# Full list of filters, https://pymeshlab.readthedocs.io/en/latest/filter_list.html
# pymeshlab.print_filter_list()
# pymeshlab.print_filter_parameter_list('generate_surface_reconstruction_screened_poisson')
