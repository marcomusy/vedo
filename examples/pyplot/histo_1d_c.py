"""Uniform distribution weighted by sin^2 12x + \onehalf"""
from vedo import Line, settings, np
from vedo.pyplot import histogram

settings.defaultFont = 11

data = np.random.rand(10000)
weights = np.ones_like(data) * np.sin(12*data)**2 + 1/2

fig1 = histogram(
    data,
    weights=weights,
    bins=50,
    aspect=16/9,          # desired aspect ratio of the figure
    xtitle=__doc__,       # x-axis title
    padding=[0,0,0,0.1],  # allow 10% padding space only on the top
    gap=0,                # no gap between bins
    ac='k7',              # axes color
    c='yellow9',
)

x = np.linspace(0,1, 200)
y = 200*np.sin(12*x)**2 + 100
fig1 += Line(x, y, c='red5', lw=3)

fig1.show(size=(1000,700), bg='black', zoom='tight').close()

