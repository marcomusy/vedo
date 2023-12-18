import numpy as np
from scipy.optimize import curve_fit

############################################# Define fit function
def fit_function(x, A, beta, B, mu, sigma):
    return A * np.exp(-x / beta) + B * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

# Generate exponential and gaussian data
data1 = np.random.exponential(scale=2.0, size=4000)
data2 = np.random.normal(loc=3.0, scale=0.3, size=1000)

# Fit the function to the histogram data
bins = np.linspace(0, 6, 61)
data_entries_1, bins_1 = np.histogram(data1, bins=bins)
data_entries_2, bins_2 = np.histogram(data2, bins=bins)
data_entries = data_entries_1 + data_entries_2  # sum the two sets
binscenters = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
popt, pcov = curve_fit(
    fit_function, xdata=binscenters, ydata=data_entries, p0=[200, 2.0, 200, 3.0, 0.3]
)
# Generate enough x values to make the curves look smooth.
xspace = np.linspace(0, 6, 100)


############################################# vedo
# Plot the histogram and the fitted function.
from vedo import settings
from vedo.pyplot import histogram, plot

settings.default_font = "ComicMono"
settings.remember_last_figure_format = True

h = histogram(
    data1.tolist() + data2.tolist(),
    xlim=(0, 6),
    bins=60,
    title="Exponential decay with gaussian peak",
    xtitle="x axis",
    ytitle="Number of entries",
    label="Histogram entries",
    c='green3',
)
h += plot(
    xspace,
    fit_function(xspace, *popt),
    lc="darkorange",
    lw=3,
    label="Fit function",
)
h.add_legend()
h.show(zoom="tight")

############################################# matplotlib
# Plot the histogram and the fitted function.
import matplotlib.pyplot as plt

plt.bar(
    binscenters,
    data_entries,
    width=bins[1] - bins[0],
    color="g",
    label=r"Histogram entries",
)
plt.plot(
    xspace,
    fit_function(xspace, *popt),
    color="darkorange",
    linewidth=2.5,
    label=r"Fit function",
)
plt.xlim(0, 6)
plt.xlabel(r"x axis")
plt.ylabel(r"Number of entries")
plt.title(r"Exponential decay with gaussian peak")
plt.legend(loc="best")
plt.show()
