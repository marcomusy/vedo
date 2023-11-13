"""Voronoi tessellation of a pointcloud on a grid"""
from vedo import dataurl, Points, Grid, show

pts0 = Points(dataurl+'rios.xyz').color('k')
pts1 = pts0.clone().smooth_lloyd_2d()

grid = Grid([14500,61700], s=[22000,24000], res=[30,30]).ps(1)
allpts = pts1.vertices.tolist() + grid.vertices.tolist()

msh = Points(allpts).generate_voronoi(method='scipy')

msh.lw(1).wireframe(False).cmap('terrain_r', 'VoronoiID', on='cells')
centers = Points(msh.cell_centers).color("k")

show(msh, pts0, __doc__, axes=dict(digits=3), zoom=1.3)

