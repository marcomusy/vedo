"""Warp scalars inside of a Volumetric dataset"""
from vedo import Volume, Cube, Arrows, show, dataurl

vol = Volume(dataurl+"embryo.tif")

source = Cube().scale(3000)
target = Cube().scale([4000,5000,6000]).rotate_x(20).wireframe()

arrs = Arrows(source, target, c='k')

vol.warp(source, target, fit=True)

show(vol, arrs, source, target, __doc__, axes=1, viewup='z')
