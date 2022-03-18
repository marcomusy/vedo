"""Format a histogram so that it can be superimposed to another"""
import numpy as np
from vedo.pyplot import histogram
from vedo import Text3D, show

np.random.seed(3)
data1 = np.random.randn(250)
data2 = (np.random.rand(250)-0.5)*12

histo = histogram(
    data1,
    bins=30,
    errors=True,
    aspect=4/3,
    title='My distributions',
    xtitle='two stochastic variables',
    c='red5',
    marker='o',
)

# Extract the 16th bin and color it purple
histo.unpack(15).c('purple4')
histo += Text3D('Highlight a\nspecial bin', pos=(0.5,20), c='purple4', italic=1)

# A second histogram:
#  make it in same format as histo so it can be superimposed
histo += histogram(data2, alpha=0.8, format=histo)

show(histo, __doc__, mode="image", zoom='tight').close()
