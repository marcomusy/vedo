import numpy as np
import colorsys
from scipy import special
from scipy.special import jn_zeros
from vedo import ScalarBar3D, show, settings
from vedo.colors import color_map, build_lut
from vedo.pyplot import plot

Nr = 2
Nθ = 1

settings.default_font = "Theemim"
settings.interpolate_scalars_before_mapping = False

def custom_lut(name, vmin=0, vmax=1, scale_l=0.05, N=256):
    # Create a custom look-up-table
    lut = []
    x = np.linspace(vmin, vmax, N)
    for i in range(N):
        r, g, b = color_map(i, name, 0, N-1)
        h, l, s = colorsys.rgb_to_hls(r,g,b)
        ###### do  something here ######
        l = min(1, l * (1 + 5 * scale_l))
        rgb = colorsys.hls_to_rgb(h, l, s)
        lut.append([x[i], rgb])
    return build_lut(lut)

def f(x, y):
    d2 = x**2 + y**2
    if d2 > 1:
        return np.nan
    else:
        r = np.sqrt(d2)
        θ = np.arctan2(y, x)
        kr = jn_zeros(Nθ, 4)[Nr]
        return special.jn(Nθ, kr * r) * np.exp(1j * Nθ * θ)

p1 = plot(
    lambda x,y: np.abs(f(x,y)),
    xlim=[-1, 1], ylim=[-1, 1],
    bins=(200, 200),
    show_nan=False,
    axes=dict(
        xtitle="x", ytitle="y", ztitle="|f(x,y)|",
        xlabel_rotation=90, ylabel_rotation=90, zlabel_rotation=90,
        xtitle_rotation=90, ytitle_rotation=90, zaxis_rotation=45,
        ztitle_offset=0.03,
    ),
)

# Unpack the plot objects to customize them
msh = p1.unpack(0).triangulate().lighting('glossy')
msh.cut_with_sphere((0,0,0), 0.99, invert=0)

pts = msh.vertices # get the points
zvals = pts[:,2]   # get the z values
θvals = [np.angle(f(*p[:2])) for p in pts]  # get the phases

lut = custom_lut("hsv", vmin=-np.pi/10, vmax=np.pi)

msh.cmap(lut, θvals) # apply the color map
scbar = ScalarBar3D(
    msh, title=f"Bessel Function Nr={Nr} Nθ={Nθ}",
    label_rotation=90, c="black",
)

# Set a specific camera position and orientation (press shift-C to see it)
cam = dict(
    position=(3.88583, 0.155949, 3.88584),
    focal_point=(0, 0, 0),
    viewup=(-0.7, 0, 0.7),
    distance=5.4,
)
show(p1, scbar, camera=cam).close()
