from vedo import dataurl, Volume
from vedo.applications import IsosurfaceBrowser

vol = Volume(dataurl+'head.vti')

plt = IsosurfaceBrowser(vol, c='gold') # Plotter instance

plt.show(axes=7, bg2='lb').close()
