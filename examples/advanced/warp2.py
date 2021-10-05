"""Warp a region of a mesh using Thin Plate Splines.
Red points stay fixed while a single point in space
moves as the arrow indicates."""
from vedo import *


mesh = Mesh(dataurl+"man.vtk").color('w').lineWidth(0.1)

# a heavily decimated copy
meshdec = mesh.clone().triangulate().decimate(N=200)

sources = [[0.9, 0.0, 0.2]]  # this point moves
targets = [[1.2, 0.0, 0.4]]  # to this.
arrow = Arrow(sources[0], targets[0])

for pt in meshdec.points():
    if pt[0] < 0.3:          # these pts don't move
        sources.append(pt)   # source = target
        targets.append(pt)   #

warp = mesh.clone().warp(sources, targets)
warp.c("blue",0.3).lineWidth(0)

apts = Points(sources).c("red")

show(mesh, arrow, warp, apts, __doc__, viewup="z", axes=1)
