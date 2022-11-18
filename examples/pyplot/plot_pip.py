"""Picture in picture plotting"""
from vedo import np, settings, show
from vedo.pyplot import plot

settings.default_font = 'Theemim'

def f(x):
    return 3*np.exp(-x)*np.cos(2*x)**2
xa = np.arange(0, 0.5, 0.01)
xb = np.arange(0, 4.0, 0.01)

# Build first figure:
fig1 = plot(
    xa, f(xa),
    title=__doc__,
    xtitle='time in seconds',
    ytitle='Intensity [a.u.]',
)

# Build second figure w/ options for axes:
fig2 = plot(
    xb, f(xb),
    title='3 e^-x cos 2x**2  (wider range)',
    xtitle=' ', ytitle=' ',  # leave empty
    c='red5',
    axes=dict(
        xyplane_color='#dae3f0',
        grid_linewidth=0, # make it solid
        xyalpha=1,        # make it opaque
        text_scale=2,     # make text bigger
    )
)
# Scale fig to make it smaller
fig2.scale(0.04).shift(0.05, 0.75)

fig1.insert(fig2)  ############# insert

show(fig1, zoom='tight').close()

