import numpy as np
from scipy import special
from scipy.special import jn_zeros
from vedo import show
from vedo.pyplot import plot

Nr = 2
Nθ = 3

def f(x, y):
    d2 = x ** 2 + y ** 2
    if d2 > 1:
        return np.nan
    else:
        r = np.sqrt(d2)
        θ = np.arctan2(y, x)
        kr = jn_zeros(Nθ, 4)[Nr]
        return special.jn(Nθ, kr * r) * np.cos(Nθ * θ)


p = plot(
    f, xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1],
    show_nan=False, bins=(100, 100),
)

# Unpack the plot objects to customize them
objs = p.unpack() 
# objs[1].off()         # turn off the horizontal levels
# objs[0].lw(1)         # set line width
objs[0].lighting('off') # style of mesh lighting "glossy", "plastic"..
zvals = objs[0].vertices[:, 2] # get the z values
objs[0].cmap("RdBu", zvals, vmin=-0.0, vmax=0.4) # apply the color map
sc = objs[0].add_scalarbar3d(title="Bessel Function").scalarbar

print("range:", zvals.min(), zvals.max()) 
show(p, sc, viewup="z").close()
