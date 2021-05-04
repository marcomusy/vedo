from vedo import Volume, dataurl
from vedo.applications import RayCastPlotter

embryo = Volume(dataurl+"embryo.slc").mode(1).c('jet') # change properties

plt = RayCastPlotter(embryo) # Plotter instance
plt.show(viewup="z", bg='black', bg2='blackboard', axes=7).close()
