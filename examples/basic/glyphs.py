"""Glyphs:
at each vertex of a mesh, another mesh
is shown with various orientation options"""
from vedo import *
import numpy as np

s = Sphere(res=12).c('white',0.1).wireframe()

randvs = np.random.rand(s.NPoints(), 3)  # random orientation vectors

#######################################
gly1 = Ellipsoid().scale(0.04)

gsphere1 = Glyph(s, gly1,
                 orientationArray=randvs,
                 scaleByVectorSize=True,
                 colorByVectorSize=True,
                 c='jet',
                )


#######################################
gly2 = Mesh(dataurl+"shuttle.obj").rotateY(180).scale(0.02)

gsphere2 = Glyph(s, gly2,
                 orientationArray="normals",
                 tol=0.1,  # minimum seaparation of 10% of bounding box
                 c='lightblue',
                )

# show two groups of objects on N=2 renderers:
show([(s, gsphere1, __doc__), (s, gsphere2)], N=2, bg='bb', zoom=1.4).close()
