"""Fitting a curve to a dataset"""
import numpy as np
from scipy.optimize import curve_fit
from vedo.pyplot import plot
from vedo import settings


def func(x, h, a, x0, k):
    return h + a * (x-x0) * np.sin((x-x0)**2 / k)


# generate simulated data
xdata = np.linspace(3, 10, 80)
true_params = [20, 2, 8, 3]
ydata_true = func(xdata, *true_params)
ydata = np.random.normal(ydata_true, 3, 80)

fit_params, pcov = curve_fit(func, xdata, ydata, p0=[19, 3, 8, 2.5])
ydata_fit = func(xdata, *fit_params)
print("true params = ", true_params)
print("fit  params = ", fit_params)

settings.default_font = "ComicMono"
settings.remember_last_figure_format = True  # when adding with p += ...

p  = plot(xdata, ydata, "o", mc="blue2", title=__doc__, label="Data")
p += plot(xdata, ydata_true, "-g", lw=2, label="Ground Truth")
p += plot(xdata, ydata_fit,  "-r", lw=4, label="Fit")
p.add_legend(pos="bottom-right")

p.show(size=(900, 650), zoom="tight")
