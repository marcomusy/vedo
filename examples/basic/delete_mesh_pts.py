"""Remove points and cells from a mesh
which are closest to a specified point."""
from vedo import *

settings.use_depth_peeling = True

msh = Mesh(dataurl+'apple.ply')
msh.c('lightgreen').bc('tomato').lw(0.1)

pt = [1, 0.5, 1]
R = 1.2
ids = msh.closest_point(pt, radius=R, return_point_id=True)

printc('NPoints before:', msh.npoints, c='g')
msh.delete_cells_by_point_index(ids)
msh.clean()  # remove orphaned vertices (not associated to any cell)
printc('NPoints after :', msh.npoints, c='g')

sph = Sphere(pt, r=R, alpha=0.1)

show(Point(pt), sph, msh, __doc__, axes=1).close()
