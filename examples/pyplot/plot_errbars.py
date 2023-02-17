"""Superpose plots in different styles"""
from vedo.pyplot import plot
from vedo import np, settings

settings.default_font = 'Theemim'
settings.remember_last_figure_format = True

x = np.linspace(0, 10, num=21)
y = 3 * np.sin(x)

################# first plot
fig = plot(
    x, y,
    "*r-",           # markers: *,o,p,h,D,d,v,^,s,x,a
	title=__doc__,
    xtitle="t variable (:mus)",
    ytitle="y(x) = :pmK_i :dot:sqrtsin^2 t",
    aspect=16/9,     # aspect ratio x/y of plot
    xlim=(-1, 14),   # specify x range
    axes=dict(text_scale=1.2),
    label="3 :dot sin(x)",
)

################# plot on top of fig
fig += plot(
    x + np.pi, y,
    "sb--",
    like=fig,        # format like fig
    splined=True,    # continuous spline through points
    lw=3,            # line width
    label="3 :dot sin(x - :pi)",
)

################## plot again on top of fig
fig += plot(x, y/5, "g", label="3/5 :dot sin(x)")

################## plot again on top of fig
fig += plot(x, y/5-1, "purple5 -", label="3/5 :dot sin(x) - 1")

################## Show! ##################
fig.add_legend(pos=[0.95,1])
fig.show(size=(1400,900), zoom='tight').close()
