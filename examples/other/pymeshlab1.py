import pymeshlab # pip install pymeshlab==2021.10
import vedo

filepath = vedo.download(vedo.dataurl+'bunny.obj')

ms = pymeshlab.MeshSet()
ms.load_new_mesh(filepath)
#vedo.show(ms, axes=True) # this already works!

pt = [0.0234, 0.0484, 0.0400]
ms.colorize_by_geodesic_distance_from_a_given_point(startpoint=pt)

mlab_mesh = ms.current_mesh()

vedo_mesh = vedo.Mesh(mlab_mesh).cmap('Paired').addScalarBar("distance")

print("Can convert back to pymeshlab.MeshSet:", type(vedo_mesh.to_meshlab()))

vedo.show(__doc__, vedo_mesh, vedo.Point(pt),
          axes=True, bg='green9', bg2='blue9', title="vedo + pymeshlab",
)

################################################################################
# Full list of filters, https://pymeshlab.readthedocs.io/en/latest/filter_list.html
#
# MeshLab offers plenty of useful filters, among which:
#
# ambient_occlusion
# compute_curvature_principal_directions
# colorize_by_geodesic_distance_from_a_given_point
# compute_normals_for_point_sets
# compute_planar_section
# compute_geometric_measures
# compute_topological_measures
# close_holes
# curvature_flipping_optimization
# cut_mesh_along_crease_edges
# define_new_per_vertex_attribute
# dust_accumulation
# estimate_radius_from_density
# global_registration
# hausdorff_distance
# hc_laplacian_smooth
# invert_faces_orientation
# laplacian_smooth
# laplacian_smooth_surface_preserving
# marching_cubes_apss
# marching_cubes_rimls
# merge_close_vertices
# mls_projection_apss
# mls_projection_rimls
# planar_flipping_optimization
# point_cloud_simplification
# points_cloud_movement
# poisson_disk_sampling
# re_compute_vertex_normals
# re_orient_all_faces_coherentely
# remeshing_isotropic_explicit_remeshing
# remove_duplicate_faces
# remove_duplicate_vertices
# repair_non_manifold_edges_by_removing_faces
# repair_non_manifold_edges_by_splitting_vertices
# repair_non_manifold_vertices_by_splitting
# snap_mismatched_borders
# subdivision_surfaces_catmull_clark
# subdivision_surfaces_ls3_loop
# subdivision_surfaces_midpoint
# surface_reconstruction_ball_pivoting
# surface_reconstruction_screened_poisson
# surface_reconstruction_vcg
# taubin_smooth
# transform_scale_normalize
# tri_to_quad_by_4_8_subdivision
# tri_to_quad_by_smart_triangle_pairing
# turn_into_a_pure_triangular_mesh
# twostep_smooth
# volumetric_obscurance
# volumetric_sampling
# voronoi_filtering
# voronoi_scaffolding
#
###################################################################################
