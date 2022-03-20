"""Superpose plots of different styles"""
from vedo.pyplot import plot
from vedo import settings
import numpy as np

settings.defaultFont= 'Kanopus'

x = np.linspace(0, 10, num=21)
y = 3 * np.sin(x)
errs = np.ones_like(x) / 2

################# first plot
fig = plot(
    x, y,
    "*r-",           # markers: *,o,p,h,D,d,v,^,s,x,a
	title=__doc__,
    xtitle="t variable (\mus)",
    ytitle="y(x) = \pmK_i \dot\sqrtsin^2 t",
    aspect=16/9,     # aspect ratio x/y of plot
    # xlim=(-1, 14), # specify x range
)

################# plot on top of fig
fig += plot(
    x+3, y,
    "sb--",
    xerrors=errs,    # set error bars on x
    yerrors=errs,    # set error bars on y
    spline=True,     # continous line through points
    lw=3,
)

################## plot again on top of fig
fig += plot(x, y/5, "g")

##################
fig.show(mode="image", size=(800,450), zoom='tight').close()
