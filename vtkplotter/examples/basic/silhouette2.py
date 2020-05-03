"""Make the silhouette of an object
move along with camera position.
"""
from vtkplotter import *
import vtk

plt = Plotter(bg='lightblue', interactive=False)

s = load(datadir+'shark.ply')
s.alpha(0.1).c('gray').lw(0.1).lc('red')
plt.show(s, __doc__)

sil = vtk.vtkPolyDataSilhouette()
sil.SetInputData(s.polydata())
sil.SetCamera(plt.camera)
silMapper = vtk.vtkPolyDataMapper()
silMapper.SetInputConnection(sil.GetOutputPort())

mesh = Mesh()
mesh.lw(4).c('black').SetMapper(silMapper)

plt.add(mesh) # add() also renders the scene
interactive()
