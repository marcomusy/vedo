"""Superimpose histograms and curves"""
import numpy as np
from vedo.pyplot import histogram, plot, settings

settings.defaultFont = "Bongas"

mu, sigma, n, bins = 100.0, 15.0, 600, 50
samples = np.random.normal(loc=mu, scale=sigma, size=n)
x = np.linspace(min(samples), max(samples), num=50)
y = 1/(sigma*np.sqrt(2*np.pi)) * np.exp( -(x-mu)**2 /(2*sigma**2))
dy = 1/np.sqrt(n)

plt  = histogram(samples, title=__doc__,
                 bins=bins, density=True, c='cyan3', aspect=8/7)

plt += plot(x, y, "-", lc='orange5')
plt += plot(x, y*(1+dy), "--", lc='orange5', lw=2)
plt += plot(x, y*(1-dy), "--", lc='orange5', lw=2)

plt.show(size=(800,700), zoom=1.2, mode="image").close()
