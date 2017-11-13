# -*- coding: utf-8 -*-
#
"""
Created on Mon Nov 13 12:48:43 2017
@author: mmusy
"""

import plotter
vp = plotter.vtkPlotter()
#vp.help() # shows a help web page


#Load a vtk file as a vtkActor and visualize it in wireframe style with a ruler on top. 
#The tridimensional shape corresponds to the outer shape of the embryonic mouse limb 
#at about 12 days of gestation.
#Press Esc to close the window and exit python session, or q to continue:
vp = plotter.vtkPlotter()
actor = vp.loadActor('data/290.vtk')
actor.GetProperty().SetRepresentationToWireframe()
vp.show()  # picks what is automatically stored in vp.actors
#vp.show(actor)           # ignores the content of vp.actors
#vp.show(actors=[actor])  # same as above


#Load 3 actors assigning each a different color, use their file paths as legend entries. 
#No need to use any variables, as actors are stored internally in vp.actors:
vp.loadActor('data/250.vtk', c=(1,0.4,0))
vp.loadActor('data/270.vtk', c=(1,0.6,0))
vp.loadActor('data/290.vtk', c=(1,0.8,0))
print 'Loaded vtkActors: ', len(vp.actors)
vp.show(legend=vp.files)


#Draw a spline that goes through a set of points, don't show the points (nodes=False):
vp = plotter.vtkPlotter()
from random import uniform as u
pts = [(u(0,10), u(0,10), u(0,10)) for i in range(10)]
vp.make_spline(pts, s=.1, nodes=False)
vp.show()


#Draw a PCA ellipsoid that contains 67% of a cluod of points:
vp = plotter.vtkPlotter()
pts = [(u(0,200), u(0,200), u(0,200)) for i in range(50)]
vp.make_points(pts)
vp.make_ellipsoid(pts, pvalue=0.67, axes=True)
vp.show()


#Show 3 planes as a grid, add a dummy sine plot on top left, add 3 axes at the origin:
import numpy as np
xycoords = [(np.exp(i/10.), np.sin(i/5.)) for i in range(40)]
vp = plotter.vtkPlotter()
gr  = vp.make_xyplot( xycoords )
plx = vp.make_grid(center=(0,0.5,0.5), normal=(1,0,0), c=(1,0,0))
ply = vp.make_grid(center=(0.5,0,0.5), normal=(0,1,0), c=(0,1,0))
plz = vp.make_grid(center=(0.5,0.5,0), normal=(0,0,1), c=(0,0,1))
ax  = vp.make_axes()
vp.show(axes=0)


#Show the vtk boundaries of a vtk surface and its normals
#(ratio reduces the total nr of arrows by the indicated factor):
vp = plotter.vtkPlotter()
va = vp.loadActor('data/290.vtk', c=(1,0.1,0.1))
nv = vp.make_normals(va, ratio=5)
sbound = vp.make_boundaries(va)
vp.show(actors=[va,nv, sbound], axes=1)


#Split window in a 49 subwindows and draw something in windows nr 12 and nr 38. 
#Then open an independent window and draw on two shapes:
vp1 = plotter.vtkPlotter(shape=(7,7), size=(900,900))
v290 = vp1.load('data/290.vtk')
v270 = vp1.load('data/270.vtk')
vp1.interactive = False
vp1.show(at=12, polys=[v290,v270])
vp1.show(at=38, polys=[v290,v270]) 
vp2 = plotter.vtkPlotter(bg=(0.9,0.9,1))
v250 = vp2.loadActor('data/250.vtk')
v270 = vp2.loadActor('data/270.vtk')
vp2.show()


#Load a surface and show its curvature based on 4 different schemes. 
#All four shapes share a common vtkCamera:
#0-gaussian, 1-mean, 2-max, 3-min
vp = plotter.vtkPlotter(shape=(1,4), size=(400,1600))
v = vp.load('data/290.vtk')
vp.interactive = False
for i in [0,1,2,3]: 
    c = vp.make_curvatures(v, ctype=i, r=1, alpha=0.8)
    vp.show(at=i, actors=[c])
vp.interact() 


#Draw a bunch of simple objects on separate parts of the rendering window:
vp = plotter.vtkPlotter(shape=(2,3), size=(800,1200))
vp.axes        = True
vp.commoncam   = False
vp.interactive = False
vp.show(at=0, actors=vp.make_arrow( [0,0,0], [1,1,1] ))
vp.show(at=1, actors=vp.make_line(  [0,0,0], [1,2,3] ))
vp.show(at=2, actors=vp.make_points( [ [0,0,0], [1,1,1], [3,1,2] ] ))
vp.show(at=3, actors=vp.make_text('hello', cam=False))
vp.show(at=4, actors=vp.make_sphere([.5,.5,.5], r=0.3))
vp.show(at=5, actors=vp.make_cube(  [.5,.5,.5], r=0.3))
vp.interact()


#Draw a line in 3D that fits a cloud of points, 
#also show the first set of 20 points and fit a plane to them:
vp = plotter.vtkPlotter()
vp.verbose = False
for i in range(500): # draw 500 fit lines superposed
    x = np.mgrid[-2:5 :20j][:, np.newaxis] # generate 20 points
    y = np.mgrid[ 1:9 :20j][:, np.newaxis]
    z = np.mgrid[-5:3 :20j][:, np.newaxis]
    data  = np.concatenate((x, y, z), axis=1)
    data += np.random.normal(size=data.shape)*0.8 # add gauss noise
    if i==0: 
        vp.make_points(data)
        vp.make_fitplane(data)
    vp.make_fitline(data, lw=10, alpha=0.01) # fit
print 'Fit slope=',vp.result['slope'] # access the last fitted slope direction
vp.show()


#Display a tetrahedral mesh (Fenics/Dolfin format). The internal verteces are displayed too:
vp = plotter.vtkPlotter()
actor = vp.loadActor('data/290.xml')
actor.GetProperty().SetRepresentationToWireframe()
vp.show()        
