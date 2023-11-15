"""Warp a region of a mesh using Thin Plate Splines.
Red points stay fixed while a single point in space
moves as the arrow indicates."""
from vedo import *

settings.use_depth_peeling = True
mesh = Mesh(dataurl+"man.vtk").color('w')

# a heavily decimated copy with about 200 points
meshdec = mesh.clone().triangulate().decimate(n=200)

sources = [[0.9, 0.0, 0.2]]  # this point moves
targets = [[1.2, 0.0, 0.4]]  # ...to this.
for pt in meshdec.vertices:
    if pt[0] < 0.3:          # these pts don't move
        sources.append(pt)   # (e.i. source = target)
        targets.append(pt)
arrow = Arrows(sources, targets)
apts = Points(sources).c("red")

warp = mesh.clone().warp(sources, targets)
warp.c("blue", 0.3).wireframe()

sphere = Sphere(r=0.3).pos(1,0,-.50)
sphere.apply_transform(warp.transform)
# print(warp.transform)

show(mesh, arrow, warp, apts, sphere, axes=1).close()
