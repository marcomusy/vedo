from vedo import *
from vedo.pyplot import violin
import numpy as np

n = 1000

acts = [
    Text3D('gaussian', pos=(0,4.5), s=0.3, c='k', justify='center'),
    violin(np.random.randn(n)),

    Text3D('exponential', pos=(5,-1), s=0.3, c='k', justify='center'),
    violin(np.random.exponential(1, n), x=5, width=3, splined=False, centerline=False, c='t', lc='k'),

    Text3D('chisquare', pos=(10,11), s=0.3, c='k', justify='center'),
    violin(np.random.chisquare(9, n)/4, x=10, vlim=(0,10), c='lg', lc='dg'),
]

show(acts, axes=dict(xtitle=False, ytitle='distribution')).close()
