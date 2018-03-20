#!/usr/bin/env python
#
# Align 2 shapes and for each vertex of the first draw
# and arrow to the closest point of the second:
#
import plotter


vp = plotter.vtkPlotter()
a1, a2 = vp.load('data/2[79]0.vtk') # load 2 files and assign to a1, a2

# the usual vtk way to assign a property is always available
# (it's easier to set c='g' in the command above!)
a1.GetProperty().SetColor(0,1,0) 

# align a1 to a2, store the new actor in a1b
a1b = vp.align(a1, a2) 

ps1b = a1b.coordinates() # coordinates of actor a1b
# for each point in a1b draw an arrow towards the closest point on a2
for p in ps1b: 
    vp.arrow(p, plotter.closestPoint(a2, p))

vp.show(legend=['Source','Target','Aligned','Links'])
