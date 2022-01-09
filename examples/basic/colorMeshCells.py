"""Colorize faces of a Mesh by passing
a 1-to-1 list of colors and transparencies"""
from vedo import *
import numpy as np

settings.useDepthPeeling = True

tor = Torus(res=9).lineWidth(1)

rgba = np.random.rand(tor.NCells(), 4)*255 # RGBA values

tor.cellIndividualColors(rgba)
printc('Mesh cell arrays:', tor.celldata.keys(),
       'shape:', tor.celldata['CellIndividualColors'].shape)

show(tor, __doc__).close()
