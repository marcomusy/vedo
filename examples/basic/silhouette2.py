"""Make the silhouette of an object
move along with camera position.
"""
from vtkplotter import *
import vtk

plt = Plotter(bg='lightblue', interactive=False)

s = load(datadir+'shark.ply')
s.alpha(0.1).c('gray').lw(0.1).lc('red')
plt.show(s, Text(__doc__)) 

sil = vtk.vtkPolyDataSilhouette()
sil.SetInputData(s.polydata())
sil.SetCamera(plt.camera)
silMapper = vtk.vtkPolyDataMapper()
silMapper.SetInputConnection(sil.GetOutputPort())

actor = Actor()
actor.lw(4).c('black').SetMapper(silMapper)

plt.add(actor) # add() also renders the scene
interactive()