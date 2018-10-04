# Example of drawing objects on different windows 
# and/or subwindows within the same window.
# We split the main window in a 25 subwindows and draw something 
# in specific windows numbers.
# Then open an independent window and draw a shape on it.
#
from vtkplotter import Plotter


# this is one instance of the class Plotter with 5 raws and 5 columns
vp1 = Plotter(shape=(5,5), title='many windows', axes=0)

# set a different background color for a specific subwindow (the last one)
vp1.renderers[24].SetBackground(.8,.9,.9)

# load the actors and give them a name
a = vp1.load('data/shapes/airboat.vtk', legend='some legend')
b = vp1.load('data/shapes/cessna.vtk', c='red')
c = vp1.load('data/shapes/atc.ply')

# with shape=(a,b) script execution after the show() is not held
vp1.show(a, at=10)
vp1.show(a, at=11)
vp1.show(b, at=12)
vp1.show(c, at=15)
vp1.show(c, at=24)

# declare a second independent instance of the class Plotter
vp2 = Plotter(pos=(500,250), bg=(0.9,0.9,1)) # blue-ish background

vp2.load('data/shapes/porsche.ply', legend='an other window')

# show and interact with mouse and keyboard on the 3D window
vp2.show()
