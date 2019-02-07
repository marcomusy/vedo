"""
Glyphs.
At each vertex of a mesh, another mesh is shown with
various orientation options and coloring.
"""
from vtkplotter import *
from numpy.random import rand

t = text(__doc__)

s = sphere(res=8).wire(True).alpha(0.2)

randvs = rand(s.N(), 3) # random orientation vectors for each vertex

#######################################
gly1 = cylinder().rotateY(90).scale(0.1)

gsphere1 = glyph(s, gly1, c=None, # c=None picks the vector size
                 orientationArray=randvs, 
                 scaleByVectorSize=True)

show([s, gsphere1, t], at=0, N=2, verbose=0)


#######################################
gly2 = load('data/shuttle.obj').rotateY(180).scale(.02)

gsphere2 = glyph(s, gly2, orientationArray='normals')

show([s, gsphere2], at=1, interactive=1)
