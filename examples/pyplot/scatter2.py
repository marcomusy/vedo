# Scatter plot of a gaussian distribution
# with varying color and point sizes
from vedo import Text2D, Plotter
from vedo.pyplot import plot
import numpy as np

n = 1000
x = np.random.randn(n)
y = np.random.randn(n)

# define what size must have each marker:
marker_sizes = np.sin(2*x)/8

# define a (r,g,b) list of colors for each marker:
marker_cols = np.c_[np.cos(2*x), np.zeros(n), np.zeros(n)]


txt0 = Text2D("A scatter plot of a\n2D gaussian distribution")
fig0 = plot(
    x, y,
    lw=0,                      # no joining lines
    ma=0.3,                    # ma = marker alpha
    marker="*",                # marker style
    xtitle="variable A",
    ytitle="variable B",
    grid=False,
)

txt1 = Text2D("marker size proportional to sin(2x) ")
fig1 = plot(
    x, y,
    ma=0.3,
    lw=0,
    marker="*",
    ms=marker_sizes,           # VARIABLE marker sizes
    mc='red',                  # same fixed color for markers
    grid=False,
)

txt2 = Text2D("marker size proportional to sin(2x)\nred level   proportional to cos(2x)")
fig2 = plot(
    x, y, ma=0.3, lw=0,
    marker=">",
    ms=marker_sizes,           # VARIABLE marker sizes
    mc=marker_cols,            # VARIABLE marker colors
    grid=False,
)

plt = Plotter(N=3, size=(1800,500))
plt.at(0).show(fig0, txt0)
plt.at(1).show(fig1, txt1)
plt.at(2).show(fig2, txt2, zoom=1.2)
plt.interactive().close()
