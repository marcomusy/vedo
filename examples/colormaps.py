# ################################################################
# Example usage of colorMap(value, name)
# This method returns a color from a float in the range [0,1]
# by looking it up in matplotlib database of colormaps
# ################################################################
from __future__ import division, print_function
from plotter import vtkPlotter, colorMap


mapkeys = ['copper', 'gray', 'binary', 'cool', 'rainbow', 'winter', 
           'jet', 'paired', 'hot', 'afmhot', 'bone']

vp = vtkPlotter(shape=(3,4), axes=3, verbose=0, interactive=0)

#load actor and subdivide mesh to increase the nr of vertex points
act = vp.load('data/shapes/mug.ply', c='gray/0.1', wire=1).subdivide()
pts = act.coordinates() 
print('range in y is:', act.ybounds())

for i,key in enumerate(mapkeys): # for each available color map name

    # make a list of colors based on the y position of point p
    cols = [ colorMap(p[1]/.087, name=key) for p in pts ]
    
    apts = vp.points(pts, cols)
    
    vp.show([act, apts], at=i, legend=key)

vp.show(interactive=1)
