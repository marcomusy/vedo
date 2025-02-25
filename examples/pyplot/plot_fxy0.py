import numpy as np
import vedo

def update_plot(widget=None, evt=""):
    k = widget.value if evt else kinit
    #################################################
    y = 2 * k * x / (1 + (k * x)**2)  # hill function
    # y = (k*x)**2 * np.sign(k*x)  # another function
    # y = 2 * (k * x)**3 / (1 + (k * x)**2)
    #################################################

    # use jet colormap to color the line
    # color = "blue4" if k>0 else "red5"
    color = vedo.color_map(k, name="winter_r", vmin=x[0], vmax=x[-1])
    title = (f"y-mean : {np.mean(y):.3f}, std: {np.std(y):.3f},\n"
             f"y-range: [{np.min(y):.3f}, {np.max(y):.3f}], "
             f"integral: {np.trapezoid(y, x):.3f}")
    p = vedo.pyplot.plot(
        x, y,
        xlim=xrange,
        ylim=yrange,
        c=color,    # line color
        lw=3,       # line width
        aspect=1.0, # aspect ratio of the plot
        axes={"number_of_divisions":10, "htitle":title, "htitle_size":0.015},
    )
    plt.remove("PlotXY").add(p)
    if yrange[0] is None or yrange[1] is None:
        plt.reset_camera(tight=0.05)

#####################################################################
kinit  = 0.1            # initial value of the parameter
xrange = (-1, 1)        # set a fixed x axis range
yrange = (-1, 1)        # set a fixed y axis range
# yrange = (None, None) # uncomment to autoscale y axis

vedo.settings.default_font = "Roboto"
x = np.linspace(*xrange, 200, endpoint=True)
plt = vedo.Plotter(size=(900, 1100))
plt.add_slider(
    update_plot,
    x[0], x[-1],
    value=kinit,
    title="parameter value",
    pos=5,
)
update_plot()  # update initial plot
plt.show("1D Function Viewer\nMove the slider to modify the parameter",
         mode="2d", zoom="tight").close()

