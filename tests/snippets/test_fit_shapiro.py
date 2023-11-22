# https://www.youtube.com/watch?v=yJCSupnOv8w
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import shapiro
from vedo.pyplot import histogram, plot
from vedo import settings

settings.default_font = "ComicMono"
settings.use_parallel_projection = True
settings.remember_last_figure_format = True

data = [
    196,
    193,
    186,
    154,
    151,
    147,
    141,
    138,
    125,
    110,
    109,
    80,
    67,
    32,
    12,
    -103,
    -108,
    -143,
]

# Perform the Shapiro-Wilk test to check for normality
statistic, p_value = shapiro(data)

fig = histogram(
    data,
    title=(
        "Shapiro-Wilk test\n"
        "on cheating chess players\n"
        f"(p-value = {p_value*100:.3f}%)"
    ),
    xtitle="ELO score variation",
    gap=0.02,
    label="Data",
    xlim=(-300, 300),
)

# Fit the data with a double gaussian
def func(x, a0, sigma0, a1, mean1, sigma1):
    g0 = a0 * np.exp(-(x        )**2 /2 /sigma0**2) # background
    g1 = a1 * np.exp(-(x - mean1)**2 /2 /sigma1**2) # signal
    return g0 + g1

xdata = fig.centers
ydata = fig.frequencies
fit_params, pcov = curve_fit(func, xdata, ydata, p0=[2,100,2,150,50])
ydata_fit = func(xdata, *fit_params)
ydata_fit_background = func(xdata, fit_params[0], fit_params[1], 0, 0, 1)
fig += plot(xdata, ydata_fit, "-r 0", lw=4, label="Fit")
fig += plot(xdata, ydata_fit_background, "-b", lw=2, label="Bkg")
fig.add_legend()

print("# of cheaters:", np.sum(ydata_fit - ydata_fit_background))
fig.show(zoom="tight")
