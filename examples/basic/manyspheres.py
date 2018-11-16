# Example that shows how to draw very large number of 
# spheres (same for points, lines) with different colors
# or different radius. Resolution (res) can be specified.
# (vtk versions<8.0 might be slow)
#
from vtkplotter import Plotter
from random import gauss

N = 100000

vp = Plotter(N=2, axes=3, interactive=0)

print('calculating..')
cols = range(N) #color numbers
pts  = [(gauss(0,1), gauss(0,2), gauss(0,1)) for i in cols]
rads = [abs(pts[i][1])/10 for i in cols] # radius=0 for y=0

# all have same radius but different colors:
s0 = vp.spheres(pts, c=cols, r=0.1, res=3) # res=resolution

# all have same color (texture) but different radius along y:
s1 = vp.spheres(pts, r=rads, res=10, texture='gold1') 

vp.show(s0, at=0)
print('..rendering spheres:', N*2)
vp.show(s1, at=1, legend='N='+str(N), interactive=1)

