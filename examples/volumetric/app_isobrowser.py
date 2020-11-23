from vedo import datadir, Volume
from vedo.applications import IsosurfaceBrowser

vol = Volume(datadir+'head.vti')

plt = IsosurfaceBrowser(vol, c='gold') # Plotter instance

plt.show(axes=7, bg2='lb')
