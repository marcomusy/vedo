# ################################################################
# Example usage of pointColors
# This method returns a color from a float in the range [0,1]
# by looking it up in matplotlib database of colormaps
# ################################################################
from __future__ import division, print_function
from vtkplotter import Plotter

# these are the available color maps
mapkeys = ['afmhot', 'binary', 'bone', 'cool', 'coolwarm', 'copper', 
           'gist_earth', 'gray', 'hot', 'jet', 'rainbow', 'winter']

vp = Plotter(N=len(mapkeys), axes=0, verbose=0, interactive=0)

#load actor and subdivide mesh to increase the nr of vertex points
# make it invisible:
pts = vp.load('data/shapes/mug.ply', alpha=0).subdivide().coordinates()

for i,key in enumerate(mapkeys): # for each available color map name
    apts = vp.points(pts).pointColors(pts[:,1], cmap=key)    
    vp.show(apts, at=i, legend=key)

vp.show(interactive=1)
