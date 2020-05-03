from vtkplotter.pyplot import plot
import numpy as np

x = np.linspace(0, 10, num=21)
y = 3 * np.sin(x)
errs = np.ones_like(x) / 2

################# first plot
plt = plot(
    x, y,
    "*r-",           # markers: *,o,p,h,D,d,v,^,s,x,a
    xtitle="x variable (mm)",
    ytitle="y(x)",
    aspect=16 / 9,   # aspect ratio x/y of plot
    # xlim=(-1, 14), # specify x range
    # ylim=(-4, 5),  # specify y range
)

################# plot on top of plt
plt.plot(
    x+3, y,
    "sb--",
    xerrors=errs,    # set error bars on x
    yerrors=errs,    # set error bars on y
    spline=True,     # continous line through points
    lw=1.5,
)

################## plot again on top of plt
plt.plot(x, y/5, "g")


##################
plt.show()
