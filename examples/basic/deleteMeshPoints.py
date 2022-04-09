"""Remove points and cells from a mesh
which are closest to a specified point."""
from vedo import *

settings.useDepthPeeling = True

msh = Mesh(dataurl+'apple.ply')
msh.c('lightgreen').bc('tomato').lw(0.1)

pt = [1, 0.5, 1]
R = 1.2
ids = msh.closestPoint(pt, radius=R, returnPointId=True)

printc('NPoints before:', msh.NPoints(), c='g')
msh.deleteCellsByPointIndex(ids)
msh.clean()  # remove orphaned vertices (not associated to any cell)
printc('NPoints after :', msh.NPoints(), c='g')

sph = Sphere(pt, r=R, alpha=0.1)

show(Point(pt), sph, msh, __doc__, axes=1).close()
