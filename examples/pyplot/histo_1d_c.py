"""Uniform distribution weighted by sin^2 12x + :onehalf"""
import numpy as np
from vedo import Line, settings
from vedo.pyplot import histogram

settings.default_font = "DejavuSansMono"

data = np.random.rand(10000)
weights = np.ones_like(data) * np.sin(12*data)**2 + 1/2

fig = histogram(
    data,
    weights=weights,
    bins=50,
    aspect=16/9,          # desired aspect ratio of the figure
    xtitle=__doc__,       # x-axis title
    padding=[0,0,0,0.1],  # allow 10% padding space only on the top
    gap=0,                # no gap between bins
    ac='k7',              # axes color
    c='yellow9',
    label='my histogram',
)

x = np.linspace(0,1, 200)
y = 200*np.sin(12*x)**2 + 100
fig += Line(np.c_[x, y], c='red5', lw=3).z(0.001)

fig.add_label('my function', marker='-', mc='red5')
fig.add_legend(pos=[0.7,1.33], alpha=0.2)

fig.show(size=(1000,700), bg='black', zoom='tight').close()

