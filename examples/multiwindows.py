#!/usr/bin/env python
#
# Example of drawing objects on different windows and/or subwindows
# within the same window
#
# We split the main window in a 36 subwindows and draw something 
# in windows nr 12 and nr 33.
# Then open an independent window and draw on it two shapes
#
import plotter


# this is one instance of the class vtkPlotter with 6 raws and 6 columns
vp1 = plotter.vtkPlotter(shape=(6,6), title='many windows example')

# set a different background color for a specific subwindow
vp1.renderers[35].SetBackground(.8,.9,.9)

# don't draw axes
vp1.axes = False 

# don't hold script execution after the show() is called
vp1.interactive=False

# load the actors and give them a name
a = vp1.load('data/250.vtk')
b = vp1.load('data/270.vtk', legend='some legend')
c = vp1.load('data/290.vtk')

vp1.show(at=12, actors=[a,b])
vp1.show(at=33, actors=[b,c])

# declare a second independent instance of the class vtkPlotter
vp2 = plotter.vtkPlotter(bg=(0.9,0.9,1))

vp2.load('data/250.vtk', legend='an other window')
vp2.load('data/270.vtk')

# show and interact with mouse and keyboard on the 3D window
vp2.show(interactive=1) 

