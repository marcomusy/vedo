#!/usr/bin/env python
#
# Example of drawing objects on different windows and/or subwindows
# within the same window
#
# We split the main window in a 25 subwindows and draw something 
# in specific windows numbers
# Then open an independent window and draw on a shape
#
import plotter


# this is one instance of the class vtkPlotter with 5 raws and 5 columns
vp1 = plotter.vtkPlotter(shape=(5,5), title='many windows', axes=0)

# set a different background color for a specific subwindow (the last one)
vp1.renderers[24].SetBackground(.8,.9,.9)

# don't hold script execution after the show() is called
vp1.interactive=False

# load the actors and give them a name
a = vp1.load('data/shapes/airboat.vtk', legend='some legend')
b = vp1.load('data/shapes/alfa147.vtk', c='red')
c = vp1.load('data/shapes/atc.ply')

vp1.show(at=10, actors=[a,b])
vp1.show(at=11, actors=a)
vp1.show(at=12, actors=b)
vp1.show(at=15, actors=[b,c])
vp1.show(at=24, actors=c)

# declare a second independent instance of the class vtkPlotter
vp2 = plotter.vtkPlotter(bg=(0.9,0.9,1))

vp2.load('data/shapes/porsche.ply', legend='an other window')

# show and interact with mouse and keyboard on the 3D window
vp2.show(interactive=1) 

