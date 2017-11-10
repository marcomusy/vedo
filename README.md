# vtkPlotter
A python helper class to easily draw VTK objects

Based on VTK: https://www.vtk.org
To install:
*sudo apt-get install python-vtk*

## Example pyhton usage:
```python
import plotter
vp = plotter.vtkPlotter()
vp.help() # shows a help page
```

Load a vtk file as a vtkActor and visualize it as wireframe
with a ruler on top, no axes. Quit python when closing the window 
```python
actor = vp.loadActor('data/xavis/vtk/290.vtk')
actor.GetProperty().SetRepresentationToWireframe()
vp.show(actor, ruler=1, axes=0, q=1)
#same as vp.show(actors=[actor], ruler=1, axes=0, q=1)
#vp.show() picks what is automatically stored in vp.actors
EX1
```







