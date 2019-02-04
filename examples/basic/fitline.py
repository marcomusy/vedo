'''
Usage example of fitLine() and fitPlane()

Draw a line in 3D that fits a cloud of 20 points,
Show the first set of 20 points and fit a plane to them.
'''
from __future__ import division, print_function
import numpy as np
from vtkplotter import Plotter, fitLine, fitPlane, points, text

# declare the class instance
vp = Plotter(verbose=0, title='linear fitting')

# draw 500 fit lines superimposed and very transparent
for i in range(500): 
    
    x = np.linspace(-2, 5, 20) # generate each time 20 points
    y = np.linspace( 1, 9, 20)
    z = np.linspace(-5, 3, 20)
    data = np.array(list(zip(x,y,z)))
    data+= np.random.normal(size=data.shape)*0.8 # add gauss noise
    
    vp.add( fitLine(data, lw=4).alpha(0.03) ) # fit a line

# 'data' still contains the last iteration points
vp.add(points(data, r=10, c='red'))

# the first fitted slope direction is stored 
# in actor.info['slope] and actor.info['normal]
print('Line Fit slope = ', vp.actors[0].info['slope']) 

plane = vp.add( fitPlane(data) ) # fit a plane
print('Plan Fit normal=', plane.info['normal']) 

vp.add(text(__doc__))

vp.show()
