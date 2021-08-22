import numpy as np
from vedo.pyplot import histogram
from vedo import *

np.random.seed(3)
data1 = np.random.randn(250)
data2 = (np.random.rand(250)-0.5)*12

hst1 = histogram(data1,
                 bins=30,
                 errors=True,
                 aspect=4/3,
                 title='My distributions',
                 xtitle='some stochastic x_\mu^0',
                 c='red',
                 marker='o',
)

# pick the 16th bin and color it violet
hst1.unpack(15).c('violet')
hst1 += Text3D('Highlight a\nspecial bin', pos=(0.5,20), c='v')

# A second histogram:
# make it in same format as hst1 so it can be superimposed
hst2 = histogram(data2, format=hst1, alpha=0.5)

# Show both:
show(hst1, hst2, mode="image").close()