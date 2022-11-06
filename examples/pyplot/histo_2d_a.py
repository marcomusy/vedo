"""Histogram of 2 variables"""
from vedo import Marker, Points, np
from vedo.pyplot import histogram

n = 10_000
x = np.random.randn(n) + 20
y = x + np.random.randn(n) + 10
xm, ym = np.mean(x), np.mean(y)

histo = histogram(
    x, y,         # this is interpreted as 2D
    bins=25,
    zlim=(0,150), # saturate color above 150 entries
    cmap='Blues_r',
    ztitle="Number of entries in bin",
)
# Add a marker to the plot
histo += Marker('*', s=0.2, c='r').pos(xm, ym, 0.2)

# Add also the original points
histo += Points([x,y], r=2).z(0.1)

histo.show(zoom='tight', bg='light yellow').close()
