"""Glyphs:
at each vertex of a mesh, another mesh
is shown with various orientation options"""
from vedo import *

s = Sphere(res=12).c("white", 0.1).wireframe()

randvs = np.random.rand(s.npoints, 3)  # random orientation vectors

#######################################
gly1 = Ellipsoid().scale(0.04)

gsphere1 = Glyph(
    s,
    gly1,
    orientation_array=randvs,
    scale_by_vector_size=True,
    color_by_vector_size=True,
    c="jet",
)


#######################################
gly2 = Mesh(dataurl + "shuttle.obj").rotate_y(180).scale(0.02)

gsphere2 = Glyph(
    s,
    gly2,
    orientation_array="normals",
    c="lightblue",
)

# show two groups of objects on N=2 renderers:
show([(s, gsphere1, __doc__), (s, gsphere2)], N=2, bg="bb", zoom=1.4).close()
