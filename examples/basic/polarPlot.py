from vtkplotter import polarPlot, show
import numpy as np

title     = "A (splined) polar plot"
angles    = [  0,  20,  60, 160, 200, 250, 300, 340]
distances = [0.1, 0.2, 0.3, 0.5, 0.6, 0.4, 0.2, 0.1]

dn1 = polarPlot([angles, distances], deg=True, spline=False,
                c='green', alpha=0.5, title=title, vmax=0.6)

dn2 = polarPlot([angles, np.array(distances)/2], deg=True,
                c='tomato', alpha=1, showDisc=False, vmax=0.6)
dn2.z(0.01) # set z above 0, so that plot is visible

show(dn1, dn2, axes=None, bg='white')
