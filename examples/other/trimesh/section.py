import trimesh
import numpy as np
from vedo import show, Plane, printc, download, settings

settings.immediateRendering = False

# load the mesh from filename, file objects are also supported
f = download('https://github.com/mikedh/trimesh/raw/main/models/featuretype.STL')
mesh = trimesh.load_mesh(f)

# get a single cross section of the mesh
txt = 'cross section of the mesh'
mslice = mesh.section(plane_origin=mesh.centroid, plane_normal=[0,0,1])

pl = Plane(mesh.centroid, normal=[0,0,1], s=[6,4], c='green', alpha=0.3)

slice_2D, to_3D = mslice.to_planar()

# show objects on N=2 non-synced renderers:
show([(mesh, pl), (slice_2D, txt)], N=2, sharecam=False, axes=7).close()

# if we wanted to take a bunch of parallel slices, like for a 3D printer
# we can do that easily with the section_multiplane method
# we're going to slice the mesh into evenly spaced chunks along z
# this takes the (2,3) bounding box and slices it into [minz, maxz]
z_extents = mesh.bounds[:,2]
# slice every .125 model units (eg, inches)
z_levels  = np.arange(*z_extents, step=0.125)

# find a bunch of parallel cross sections
sections = mesh.section_multiplane(plane_origin=mesh.bounds[0],
                                   plane_normal=[0,0,1],
                                   heights=z_levels)
N = len(sections)
printc("nr. of sections:", N, c='green')

# summing the array of Path2D objects will put all of the curves
# into one Path2D object, which we can plot easily
combined = np.sum(sections)
sections.append([combined, 'combined'])

# show objects in N synced renderers:
show(sections, N=N, axes=1, new=True).interactive().close()

