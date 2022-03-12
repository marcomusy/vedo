#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://www.redblobgames.com/articles/visibility/
# https://en.wikipedia.org/wiki/Art_gallery_problem
"Hover mouse to illuminate the visible area of the room"
from vedo import *
import numpy as np

rw = Rectangle((0,0), (1,1)).texture(dataurl+"textures/paper1.jpg")
tx = Text3D("эd", font="Comae").pickable(False)
tx.scale(0.475).pos(0.1, 0.2, 0.0002).c('green4')
d = rw.diagonalSize()

objs = [rw, tx]
mobjs = merge(objs).c('grey4').flat()
allpts = mobjs.points()
walls = mobjs.extrude(0.03, cap=False).z(-0.01).pickable(False).flat()

def func(evt):
	p = evt.picked3d
	if p is None:
		return

	pts1 = []
	for q1 in allpts:
		v = versor(q1-p)
		e = cross(v, [0,0,1]) * 0.01  # a small epsilon shift
		for k in (v, v+e, v-e):
			ipts = walls.intersectWithLine(p, p+k*d, tol=1e-04)
			n = len(ipts)
			if n==1 or n>1 and mag(q1-ipts[0]) < 1e-04:
				pts1.append(ipts[0])

	pts2 = []
	for pt in pts1:
		angle = np.arctan2(pt[1]-p[1], pt[0]-p[0])
		pts2.append([pt, angle])
	pts = utils.sortByColumn(pts2,1)[:,0].tolist() # sort by angle

	line = Line(pts, closed=True).z(0.01).lw(4).c('grey3')
	surf = line.clone().triangulate().lw(0).c('yellow4').alpha(0.3)
	surf.pickable(False)
	area = Assembly(surf, line)
	area.name = "Area"
	plt.remove("Area").add(area)

plt = Plotter(bg2="light blue")
plt.addCallback("mouse hover", func)
plt.show(objs, walls, __doc__, zoom=1.1, elevation=-20).close()


