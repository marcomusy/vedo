"""A polynomial fit of degree="""
from vedo import np, precision, Text3D, DashedLine
from vedo.pyplot import plot, fit, histogram, show
# np.random.seed(0)

n   = 25 # nr of data points to generate
deg = 3  # degree of the fitting polynomial

# Generate some noisy data points
x = np.linspace(0, 12, n)
y = (x-6)**3 /50 + 6              # the "truth" is a cubic function!
xerrs = np.linspace(0.4, 1.0, n)  # make last points less precise
yerrs = np.linspace(1.0, 0.4, n)  # make first points less precise
noise = np.random.randn(n)

# Plot the noisy points with their error bars
fig1 = plot(
    x, y+noise,
    ".k",
    title=__doc__+str(deg),
    xerrors=xerrs,
    yerrors=yerrs,
    aspect=4/5,
    xlim=(-3,15),
    ylim=(-3,15),
    padding=0,
)
fig1 += DashedLine(x, y, c='r')

# Fit points and evaluate, with a boostrap and Monte-Carlo technique,
# the correct errors and error bands. Return a Line object:
pfit = fit(
    [x, y+noise],
    deg=deg,        # degree of the polynomial
    niter=500,      # nr. of MC iterations to compute error bands
    nstd=2,         # nr. of std deviations to display
    xerrors=xerrs,  # optional array of errors on x
    yerrors=yerrs,  # optional array of errors on y
    vrange=(-3,15), # specify the domain of fit
)

fig1 += [pfit, pfit.errorBand, pfit.errorLines] # add these objects to Figure

# Add some text too
txt = "fit coefficients:\n " + precision(pfit.coefficients, 2) \
    + "\n\pm" + precision(pfit.coefficientErrors, 2) \
    + "\n\Chi^2_\nu  = " + precision(pfit.reducedChi2, 3)
fig1 += Text3D(txt, s=0.42, font='VictorMono').pos(4,-2).c('k')

# Create a 2D histo to show the correlation of fit parameters
fig2 = histogram(
    pfit.MonteCarloCoefficients[:,0],
    pfit.MonteCarloCoefficients[:,1],
    title="parameters correlation",
    xtitle='coeff_0', ytitle='coeff_1',
    cmap='ocean_r',
    scalarbar=True,
)

show(fig1, fig2, N=2, sharecam=False, zoom='tight').close()
