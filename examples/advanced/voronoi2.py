"""Voronoi tessellation of a pointcloud on a grid"""
from vedo import dataurl, Points, Grid, voronoi, show

pts0 = Points(dataurl+'rios.xyz').color('k')
pts1 = pts0.clone().smoothLloyd2D()

grid = Grid([14500,61700], s=[22000,24000], res=[30,30]).ps(1)
allpts = pts1.points().tolist() + grid.points().tolist()

msh = voronoi(allpts, method='scipy')

msh.lw(0.1).wireframe(False).cmap('terrain_r', 'VoronoiID', on='cells')
centers = Points(msh.cellCenters(), c='k')

show(msh, pts0, __doc__, axes=dict(digits=3))
