from vedo import load, dataurl
from vedo.applications import RayCaster
embryo = load(dataurl+"embryo.slc") # vtkVolume
embryo.mode(1).c('jet') # change properties
plt = RayCaster(embryo) # Plotter instance
plt.show(viewup="z", bg='black', bg2='blackboard', axes=7)
