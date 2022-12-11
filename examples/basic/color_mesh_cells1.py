"""Colorize faces of a Mesh by passing
a 1-to-1 list of colors and transparencies"""
from vedo import *

settings.use_depth_peeling = True

tor = Torus(res=9).linewidth(1)

rgba = np.random.rand(tor.ncells, 4)*255 # RGBA values
tor.cellcolors = rgba

printc(
    'Mesh cell arrays:', tor.celldata.keys(),
    'shape:', tor.celldata['CellsRGBA'].shape,
)
show(tor, __doc__).close()
