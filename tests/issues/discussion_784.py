
from vedo import *

s = Sphere(quads=True, res=10) # some test points in space
pts = s.points()

vpts = Points(pts)
vpts.compute_normals_with_pca(invert=True)
vpts.print()

normals = vpts.pointdata["Normals"]
arrows  = Arrows(pts, pts + normals/10).c('red5')

show(vpts, arrows, axes=True).close()
