import numpy as np
from scipy.optimize import curve_fit
from vedo.pyplot import plot
from vedo import settings


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-((x - x0)**2) / (2 * sigma**2))


def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    ymin, ymax = min(y), max(y)
    popt, pcov = curve_fit(gauss, x, y, p0=[ymin, ymax-ymin, mean, sigma])
    return popt


# generate simulated data
xdata = np.linspace(3, 10, 100)
ydata_true = gauss(xdata, 20, 5, 6, 1)
ydata = np.random.normal(ydata_true, 1, 100)

H, A, x0, sigma = gauss_fit(xdata, ydata)
# print("Offset   :", H)
# print("Center   :", x0)
# print("Sigma    :", sigma)
# print("Amplitude:", A)

settings.default_font = "ComicMono"
settings.remember_last_figure_format = True  # when adding with p += ...

p  = plot(xdata, ydata, "o", title="Gaussian Fit to points", label="Data")
p += plot(xdata, ydata_true, "-g", label="Ground Truth")
p += plot(xdata, gauss(xdata, *gauss_fit(xdata, ydata)), "-r", label="Fit")
p.add_legend()

p.show(size=(900, 650), zoom="tight")
