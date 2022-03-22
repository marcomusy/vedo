import numpy as np
from scipy import special
from scipy.optimize import curve_fit
from vedo import settings, Marker
from vedo.pyplot import plot

settings.defaultFont = 'Calco'

xdata = [230, 234, 240, 243, 246, 249, 252]
ydata = [0,   0,    11,  62,  15,  21, 100]
tdata = [100, 31,   34, 80,   21,  21, 100]

yerrs = np.sqrt(ydata) /np.array(tdata) +0.05
ydata = np.array(ydata) /np.array(tdata)

def func(x, a, x0):
    return (1+special.erf(a*(x-x0)))/2

p0 = [1/25, 240] # initial guess
popt, _ = curve_fit(func, xdata, ydata, p0)

x = np.linspace(225, 255, 50)
y = func(x, *popt)
x0, y0 = popt[1], func(popt[1], *popt)

fig = plot(
    xdata, ydata,
    'o',
    yerrors=yerrs,
    ylim=(-0.1,1.3),
    title="ERF(x) fit to data",
    ytitle='Embryos with visible HL',
    xtitle='Hind Limb age (h)',
    mc='blue2',
    ms=0.3,
    lw=2,
)
fig += plot(x, y, lw=5)
fig += Marker('*', s=0.5, c='r4').pos(x0,y0, 0.2)
fig.show(zoom=1.2)


