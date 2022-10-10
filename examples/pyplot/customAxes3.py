"""Customizing Axes.
Cartesian planes can be displaced
from their lower-range default position"""
from vedo import Sphere, Axes, precision, show

sph = Sphere().scale([4, 3, 2]).shift(5, 6, 7).c("green2", 0.1).wireframe()

axs = Axes(
    sph,  # build axes for object sph
    xtitle="x axis",
    ytitle="y axis",
    ztitle="z axis",
    htitle="An ellipsoid at " + precision(sph.center_of_mass(), 2),
    htitle_font=1,
    htitle_color="red3",
    zxgrid=True,
    xyframe_line=2,
    yzframe_line=2,
    zxframe_line=2,
    xyframe_color="red3",
    yzframe_color="green3",
    zxframe_color="blue3",
    xyshift=0.2,  # move xy plane 20% along z
    yzshift=0.2,  # move yz plane 20% along x
    zxshift=0.2,  # move zx plane 20% along y
)

show(sph, axs, __doc__).close()
