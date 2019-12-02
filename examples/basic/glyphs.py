"""Glyphs:
at each vertex of a mesh, another mesh
is shown with various orientation options
"""
from vtkplotter import *
from numpy.random import rand

t = Text(__doc__) # pick the above header as description

s = Sphere(res=12).c('white').alpha(0.1).wireframe()

randvs = rand(s.NPoints(), 3)  # random orientation vectors for each vertex

#######################################
gly1 = Cylinder().rotateY(90).scale(0.03)

gsphere1 = Glyph(s, gly1,
                 c='lightgreen',
                 orientationArray=randvs,
                 scaleByVectorSize=True,
                 )

#######################################
gly2 = load(datadir+"shuttle.obj").rotateY(180).scale(0.02)

gsphere2 = Glyph(s, gly2,
                 c='lavender',
                 orientationArray="normals",
                 tol=0.1,  # impose a minimum seaparation of 10%
                 )

# show two groups of objects on N=2 renderers:
show([(s, gsphere1, t), (s, gsphere2)], N=2, zoom=1.4)
