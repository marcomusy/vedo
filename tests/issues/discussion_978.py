from skimage import measure
from vedo import *

settings.use_parallel_projection = True 

# Create Ellipsoid
ellipsoid = Ellipsoid(axis1=(1,0,0), axis2=(0,2,0), axis3=(0,0,3))
ellipsoid.rotate_z(45).rotate_x(20).pos(30,40,50)

s = 0.1
vol = ellipsoid.binarize(spacing=(s,s,s)).print()

np_vol = vol.tonumpy()
labels = measure.label(np_vol, connectivity=2)
properties = measure.regionprops(labels)

centroid = properties[0].centroid
inertia_tensor = properties[0].inertia_tensor * 0.1
eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)
e0 = eigenvectors[:,0]*eigenvalues[0]
e1 = eigenvectors[:,1]*eigenvalues[1]
e2 = eigenvectors[:,2]*eigenvalues[2]

# Construct Ellipsoid With Inertia Tensor
transform = LinearTransform(inertia_tensor)
transform.rotate(90, e1)
transform.shift(centroid)

tens = Sphere(res=24).apply_transform(transform)
tens.color('w').alpha(0.25).lw(1)

line0 = Line([0,0,0], [1,0,0]).apply_transform(transform).color('r').lw(4)
line1 = Line([0,0,0], [0,1,0]).apply_transform(transform).color('g').lw(4)
line2 = Line([0,0,0], [0,0,1]).apply_transform(transform).color('b').lw(4)

evector0 = Arrow(end_pt=e0, c='r').rotate(90, e1).shift(centroid)
evector1 = Arrow(end_pt=e1, c='g').rotate(90, e1).shift(centroid)
evector2 = Arrow(end_pt=e2, c='b').rotate(90, e1).shift(centroid)

# print("eigenvectors\n", eigenvectors)
# print("eigenvalues ", eigenvalues)
# print(transform)
# print("inertia_tensor\n", inertia_tensor)
show(
    ellipsoid.scale(10).pos(centroid).wireframe(), # should roughly match the tensor
    tens,
    line0, line1, line2, evector0, evector1, evector2,
    axes=1, bg2='lightblue', viewup='z',
)