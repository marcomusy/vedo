"""Fit y=ax+b and compute error bands"""
from vedo import Text2D, DashedLine, show
from vedo.pyplot import plot, fit
import numpy as np
# np.random.seed(0)

# Generate some noisy data points along a line
a, b = (np.random.rand(2)-0.5)*10 # choose a and b

x = np.linspace(0, 15, 25)
y = a*x + b
noise = np.random.randn(len(x)) * 5 # create gaussian noise

# Plot the points and add the "true" line without noise
fig = plot(x, y+noise, '*k', title=__doc__)
fig += DashedLine(x, y, c='red5')

# Fit points and evaluate, with a boostrap and Monte-Carlo technique,
# the correct error coeffs and error bands. Return a Line object:
pfit = fit(
    [x, y+noise],
    deg=1,          # degree of the polynomial
    niter=500,      # nr. of MC iterations to compute error bands
    nstd=2,         # nr. of std deviations to display
)

fig += [pfit, pfit.errorBand, *pfit.errorLines] # add these objects to fig

msg = f"Generated a, b  : {np.array([a,b])}"\
      f"\nFitted    a, b  : {pfit.coefficients}"\
      f"\nerrors on a, b  : {pfit.coefficientErrors}"\
      f"\nave point spread: \sigma \approx {pfit.dataSigma:.3f} in y units"
msg = Text2D(msg, font='VictorMono', pos='bottom-left', c='red3')

show(fig, msg, mode="image").close()
