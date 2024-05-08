import numpy as np
import vedo
from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn

################################################################################
# Example of mesh relaxation
path = vedo.download(vedo.dataurl + "mouse_brain.stl")
mesh = mm.loadMesh(path)

relax_params = mm.MeshRelaxParams()
relax_params.iterations = 5
mm.relax(mesh, relax_params)

props = mm.SubdivideSettings()
props.maxDeviationAfterFlip = 0.5
mm.subdivideMesh(mesh, props)

plus_z = mm.Vector3f()
plus_z.z = 1.0
rotation_xf = mm.AffineXf3f.linear(mm.Matrix3f.rotation(plus_z, 3.1415 * 0.5))
mesh.transform(rotation_xf)

vedo.Mesh(mesh).show().close()

################################################################################
# Simple triangulation
u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 100j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

# Prepare for MeshLib PointCloud
verts = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1).reshape(-1, 3)
# verts = vedo.Mesh(vedo.dataurl+"bunny.obj").subdivide(2).vertices

# Create MeshLib PointCloud from np ndarray
pc = mn.pointCloudFromPoints(verts)

# Remove duplicate points
pc.validPoints = mm.pointUniformSampling(pc, 0.1)
pc.invalidateCaches()

# Triangulate it
triangulated_pc = mm.triangulatePointCloud(pc)

# Fix possible issues
triangulated_pc = mm.offsetMesh(triangulated_pc, 0.0)

vedo.show(vedo.Points(pc), vedo.Mesh(triangulated_pc, alpha=0.4)).close()

################################################################################
# Example of Boolean operation
# create first sphere with radius of 1 unit
sphere1 = mm.makeUVSphere(1.0, 64, 64)

# create second sphere by cloning the first sphere and moving it in X direction
sphere2 = mm.copyMesh(sphere1)
xf = mm.AffineXf3f.translation(mm.Vector3f(0.7, 0.0, 0.0))
sphere2.transform(xf)

# perform boolean operation
result = mm.boolean(sphere1, sphere2, mm.BooleanOperation.Intersection)
if not result.valid():
    print(result.errorString)

vedo.show(vedo.Mesh(result.mesh, alpha=0.4)).close()

################################################################################
# Example of mesh offset
mesh = mm.loadMesh(path)

# Setup parameters
params = mm.OffsetParameters()
# offset grid precision (algorithm is voxel based)
params.voxelSize = mesh.computeBoundingBox().diagonal() * 5e-3
if mm.findRightBoundary(mesh.topology).empty():
    # use if you have holes in mesh
    params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule

# Make offset mesh
offset = mesh.computeBoundingBox().diagonal() * 0.025
result_mesh = mm.offsetMesh(mesh, offset, params)

vedo.show(vedo.Mesh(result_mesh).lw(1)).close()

################################################################################
# Example of fill holes
path = vedo.download(vedo.dataurl + "bunny.obj")
mesh = mm.loadMesh(path)

# Find single edge for each hole in mesh
hole_edges = mesh.topology.findHoleRepresentiveEdges()

for e in hole_edges:
    #  Setup filling parameters
    params = mm.FillHoleParams()
    params.metric = mm.getUniversalMetric(mesh)
    #  Fill hole represented by `e`
    mm.fillHole(mesh, e, params)

vedo.show(vedo.Mesh(mesh)).close()

################################################################################
# Example of stitch holes
mesh_a = vedo.Mesh(vedo.dataurl + "bunny.obj").cut_with_plane(
    origin=(+0.01, 0, 0), normal=(+1, 0, 0)
)
mesh_b = vedo.Mesh(vedo.dataurl + "bunny.obj").cut_with_plane(
    origin=(-0.01, 0, 0), normal=(-1, 0, 0)
)
mesh_a.write("meshAwithHole.stl")
mesh_b.write("meshBwithHole.stl")

mesh_a = mm.loadMesh("meshAwithHole.stl")
mesh_b = mm.loadMesh("meshBwithHole.stl")

# Unite meshes
mesh = mm.mergeMeshes([mesh_a, mesh_b])

# Find holes
edges = mesh.topology.findHoleRepresentiveEdges()

# Connect two holes
params = mm.StitchHolesParams()
params.metric = mm.getUniversalMetric(mesh)
mm.buildCylinderBetweenTwoHoles(mesh, edges[0], edges[1], params)

vedo.show(vedo.Mesh(mesh).lw(1)).close()
