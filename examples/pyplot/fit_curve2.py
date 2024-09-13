"""Create slider that changes the value of the k parameter in the function."""
import numpy as np
from vedo.pyplot import plot
from vedo import settings, Line, Text2D, Plotter


settings.default_font = "Brachium"
settings.remember_last_figure_format = True


def func(x, h, a, x0, k):
    return h + a * (x-x0) * np.sin((x-x0)**2 / k)


def callback(w, e):
    y = func(xdata, *true_params[:3], slider.value)
    res = np.sum((ydata - y)**2 / 100)
    txt2d.text(f"Residuals: {res:.3f}")
    # remove previous fit line and insert the new one
    line = Line(np.c_[xdata, y], c="green4", lw=3)
    p.remove(p[2]).insert(line)


true_params = [20, 2, 8, 3]
xdata = np.linspace(3, 10, 100)
ydata_true = func(xdata, *true_params)
ydata = np.random.normal(ydata_true, 3, 100)

p = plot(
    xdata,
    ydata,
    "o", 
    mc="blue2", 
    title="f = h + a*(x-x_0 )*sin((x-x_0 )**2 /k)",
    label="Data",
)
p += plot(xdata, ydata_true, "-g", lw=2, label="Fit")
p.add_legend(pos="top-right")

txt2d = Text2D(pos="bottom-left", bg='yellow5', s=1.2)

plt = Plotter(size=(900, 650))
slider = plt.add_slider(callback, 1, 5, value=3, title="k-value")
plt.show(p, txt2d, __doc__, zoom=1.3, mode="2d")
