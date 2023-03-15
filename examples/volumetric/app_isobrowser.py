from vedo import dataurl, Volume
from vedo.applications import IsosurfaceBrowser

vol = Volume(dataurl+'head.vti')

# IsosurfaceBrowser(Plotter) instance:
plt = IsosurfaceBrowser(vol, use_gpu=True, c='gold')

plt.show(axes=7, bg2='lb').close()
