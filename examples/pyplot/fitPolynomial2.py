"""A polynomial fit of degree="""
from vedo import precision, Text3D, DashedLine
from vedo.pyplot import plot, fit, histogram, show
import numpy as np
# np.random.seed(0)

n   = 25 # nr of data points to generate
deg = 3  # degree of the fitting polynomial

# Generate some noisy data points
x = np.linspace(0, 12, n)
y = (x-6)**3 /50 + 6              # the "truth" is a cubic fuction!
xerrs = np.linspace(0.4, 1.0, n)  # make last points less precise
yerrs = np.linspace(1.0, 0.4, n)  # make first points less precise
# xerrs = yerrs = None #assume errors are all the same (but unknown)
noise = np.random.randn(n)

# Plot the noisy points with their error bars
plt = plot(x, y+noise, '.k',
           title=__doc__+str(deg),
           xerrors=xerrs,
           yerrors=yerrs,
           aspect=8/9,
           xlim=(-3,15),
           ylim=(-3,15),
          )
plt += DashedLine(x, y)

# Fit points and evaluate, with a boostrap and Monte-Carlo technique,
# the correct errors and error bands. Return a Line object:
pfit = fit([x, y+noise],
           deg=deg,        # degree of the polynomial
           niter=500,      # nr. of MC iterations to compute error bands
           nstd=2,         # nr. of std deviations to display
           xerrors=xerrs,  # optional array of errors on x
           yerrors=yerrs,  # optional array of errors on y
           vrange=(-3,15), # specify the domain of fit
          )

plt += [pfit, pfit.errorBand, *pfit.errorLines] # add these objects to Plot

txt = "fit coefficients:\n " + precision(pfit.coefficients, 2) \
    + "\n\pm" + precision(pfit.coefficientErrors, 2) \
    + "\n\Chi^2_\nu  = " + precision(pfit.reducedChi2, 3)
plt += Text3D(txt, s=0.42, font='VictorMono').pos(2,-2).c('k')

# Create an histo to show the correlation of fit parameters
h = histogram(pfit.MonteCarloCoefficients[:,0],
              pfit.MonteCarloCoefficients[:,1],
              title="parameters correlation",
              xtitle='coeff_0', ytitle='coeff_1',
              cmap='bone_r', scalarbar=True)
h.scale(150).shift(-1,11) # make it a lot bigger and move it

show(plt, h, zoom=1.3, mode="image").close()
