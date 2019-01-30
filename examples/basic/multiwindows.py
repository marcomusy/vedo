'''
Example of drawing objects on different windows 
and/or subwindows within the same window.
We split the main window in a 25 subwindows and draw something 
in specific windows numbers.
Then open an independent window and draw a shape on it.
'''
print(__doc__)

from vtkplotter import Plotter, text


# this is one instance of the class Plotter with 5 raws and 5 columns
vp1 = Plotter(shape=(5,5), title='many windows', axes=0)
# having set shape=(n,m), script execution after the show() is not held

# set a different background color for a specific subwindow (the last one)
vp1.renderers[24].SetBackground(.8,.9,.9) # use vtk method SetBackground()

# load the actors and give them a name
a = vp1.load('data/shapes/airboat.vtk', legend='some legend')
b = vp1.load('data/shapes/cessna.vtk', c='red')
c = vp1.load('data/shapes/atc.ply')

# show a text in each renderer
for i in range(22):
    txt = text('renderer\nnr.'+str(i), c=i, s=0.5, justify='centered')
    vp1.show(txt, at=i)

vp1.show(a, at=22)
vp1.show(b, at=23)
vp1.show(c, at=24)


############################################################
# declare a second independent instance of the class Plotter
vp2 = Plotter(pos=(500,250), bg=(0.9,0.9,1)) # blue-ish background

vp2.load('data/shapes/porsche.ply', legend='an other window')

vp2.show() # show and interact with mouse and keyboard 
