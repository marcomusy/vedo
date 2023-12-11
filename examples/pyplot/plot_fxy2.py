"""Draw a z = BesselJ(x,y) surface with a custom color map
and a custom scalar bar with labels in radians"""
import numpy as np
from scipy import special
from scipy.special import jn_zeros
from vedo import ScalarBar3D, Line, show, settings
from vedo.colors import color_map, build_lut
from vedo.pyplot import plot

Nr = 1
Nθ = 3

settings.default_font = "Theemim"
settings.interpolate_scalars_before_mapping = False
axes_opts = dict(
    xtitle="x", ytitle="y", ztitle="|f(x,y)|",
    xlabel_rotation=90, ylabel_rotation=90, zlabel_rotation=90,
    xtitle_rotation=90, ytitle_rotation=90, zaxis_rotation=45,
    ztitle_offset=0.03,
)

def custom_lut_surface(name, vmin=0, vmax=1, N=256):
    # Create a custom look-up-table for the surface
    table = []
    x = np.linspace(vmin, vmax, N)
    for i in range(N):
        rgb = color_map(i, name, 0, N-1)
        table.append([x[i], rgb])
    return build_lut(table)

def custom_table_scalarbar(name):
    # Create a custom table of colors and labels for the scalarbar
    table = []
    x = np.linspace(-np.pi,np.pi, 401)
    labs = ["-:pi"  , "-3:pi/4", "-:pi/2", "-:pi/4", "0",
            "+:pi/4",  "+:pi/2", "+3:pi/4","+:pi"]  
    for i in range(401):
        rgb = color_map(i, name, 0, 400)
        if i%50 == 0:
            table.append([x[i], rgb, 1, labs[i//50]])
        else:
            table.append([x[i], rgb])
    return table, build_lut(table)

######################################################################
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
    bins=(100, 100),
    show_nan=False,
    axes=axes_opts,
)

# Unpack the 0-element (the surface of the plot) to customize it
msh = p1[0].lighting('glossy')

pts = msh.vertices   # get the points
zvals = pts[:,2]     # get the z values
θvals = [np.angle(f(*p[:2])) for p in pts]  # get the phases

lut = custom_lut_surface("hsv", vmin=-np.pi, vmax=np.pi)
msh.cmap(lut, θvals) # apply the color map

table, lut = custom_table_scalarbar("hsv")
line = Line((1,-1), (1,1))  # a dummy line to attach the scalarbar to
line.cmap("hsv", [0, 1])
scbar = ScalarBar3D(
    line,
    title=f"N_r ={Nr}, N_θ ={Nθ}, phase :theta in radians",
    label_rotation=90,
    categories=table,
    c='black',
)
# convert the scalarbar to a 2D object and place it to the bottom
scbar = scbar.clone2d([-0.6,-0.7], size=0.2, rotation=-90, ontop=True)

# Set a specific camera position and orientation (shift-C to see it)
cam = dict(
    position=(3.88583, 0.155949, 3.88584),
    focal_point=(0, 0, 0), viewup=(-0.7, 0, 0.7), distance=5.4,
)
show(p1, scbar, __doc__, camera=cam).close()
