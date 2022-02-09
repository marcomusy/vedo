from vedo import Volume, dataurl
from vedo.applications import RayCastPlotter

embryo = Volume(dataurl+"embryo.slc").mode(1).c('jet')  # change properties

plt = RayCastPlotter(embryo, bg='black', bg2='blackboard', axes=7)  # Plotter instance

plt.show(viewup="z").close()
