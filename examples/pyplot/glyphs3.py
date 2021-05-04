"""Orient and scale 'glyphs'
(use a Mesh like a symbol)"""
# Credits: original example and data from https://plotly.com/python/cone-plot
# Adapted for vedo by M. Musy, 2020.
from vedo import Cone, Glyph, show
import numpy as np
import pandas as pd

# Read cvs data
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/vortex.csv")
pts = np.c_[df['x'], df['y'],df['z']]
vecs= np.c_[df['u'], df['v'],df['w']]

# Create a mesh to be used like a symbol (a "glyph") to be attached to each point
cone = Cone().scale(0.3).rotateY(90) # make it smaller and orient tip to positive x
glyph = Glyph(pts, cone, vecs, scaleByVectorSize=True, colorByVectorSize=True)

glyph.lighting('ambient').cmap('Blues').addScalarBar(title='wind speed')

show(glyph, __doc__, axes=True).close()


