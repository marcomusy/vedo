"""Glyphs:
at each vertex of a mesh, another mesh
is shown with various orientation options"""
from vedo import *

# Create a sphere with resolution 12, set its color and show as wireframe
s = Sphere(res=12).c("white", 0.1).wireframe()

randvs = np.random.rand(s.npoints, 3)  # random orientation vectors

#######################################
# Create an ellipsoid glyph and scale it down
gly1 = Ellipsoid().scale(0.04)

# create a Glyph object that will show an ellipsoid at each vertex
gsphere1 = Glyph(
    s,
    gly1,
    orientation_array=randvs,
    scale_by_vector_size=True,
    color_by_vector_size=True,
    c="jet",
)


#######################################
# Create a mesh glyph and scale it down
gly2 = Mesh(dataurl + "shuttle.obj").rotate_y(180).scale(0.02)

# Create a Glyph object that will show a shuttle at each vertex
gsphere2 = Glyph(
    s,
    gly2,
    orientation_array="normals",
    c="lightblue",
)

# Show two groups of objects on N=2 renderers:
show([(s, gsphere1, __doc__), (s, gsphere2)], N=2, bg="bb", zoom=1.4).close()
