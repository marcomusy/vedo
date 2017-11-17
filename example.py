#!/usr/bin/env python
#
"""
Created on Mon Nov 13 12:48:43 2017
@author: mmusy
"""
import plotter


# Declare an instance of the class
vp = plotter.vtkPlotter()
#vp.help() # shows a help message


#Load a vtk file as a vtkActor and visualize it.
#The tridimensional shape corresponds to the outer shape of the embryonic mouse limb
#at about 11 days of gestation.
#Choose a tomato color for the internal surface, and no transparency.
#Press Esc to close the window and exit python session, or q to continue:
#vp.flat=1
vp.load('data/250.vtk', c='b', bc='tomato', alpha=1) # c=(R,G,B), #hex, letter or name
vp.show()             # picks what is automatically stored in vp.actors


#Load a vtk file as a vtkActor and visualize it in wireframe style.
a = vp.load('data/290.vtk', wire=1) # same as a.GetProperty().SetRepresentationToWireframe()
vp.showaxes = False
vp.show()             # picks what is automatically stored in vp.actors
#vp.show(a)           # ignores the content of vp.actors
#vp.show(actors=[a])  # same as above


#Load 3 actors assigning each a different color, use their file paths as legend entries.
#No need to use any variables, as actors are stored internally in vp.actors:
vp = plotter.vtkPlotter()
vp.load('data/250.vtk', c=(1,0.4,0))
vp.load('data/270.vtk', c=(1,0.6,0))
vp.load('data/290.vtk', c=(1,0.8,0))
print ('Loaded vtkActors: ', len(vp.actors), vp.names)
vp.show(legend=vp.names)


#Draw a spline that goes through a set of points, don't show the points (nodes=False):
from random import uniform as u
pts = [(u(0,10), u(0,10), u(0,10)) for i in range(20)]
vp = plotter.vtkPlotter()
vp.spline(pts, s=.1, nodes=False)
vp.show(legend='a random spline')


#Draw a PCA ellipsoid that contains 67% of a cloud of points:
vp = plotter.vtkPlotter()
pts = [(u(0,200), u(0,200), u(0,200)) for i in range(50)]
vp.points(pts)
vp.ellipsoid(pts, pvalue=0.67, pcaaxes=True)
vp.show(legend=['points', 'PCA ellipsoid'])


#Show 3 planes as a grid, add a dummy sine plot on top left:
import numpy as np
xycoords = [(np.exp(i/10.), np.sin(i/5.)) for i in range(40)]
vp = plotter.vtkPlotter()
vp.xyplot( xycoords )
vp.grid(center=(0,0.5,0.5), normal=(1,0,0), c=(1,0,0))
vp.grid(center=(0.5,0,0.5), normal=(0,1,0), c=(0,1,0))
vp.grid(center=(0.5,0.5,0), normal=(0,0,1), c=(0,0,1))
vp.show(axes=0)


#Show the vtk boundaries of a vtk surface and its normals
#(ratio reduces the total nr of arrows by the indicated factor):
vp = plotter.vtkPlotter()
va = vp.load('data/290.vtk', c='maroon')
vp.normals(va, ratio=5)
vp.boundaries(va)
vp.show(legend='shape w/ boundaries')


#Split window in a 36 subwindows and draw something in windows nr 12 and nr 33.
#Then open an independent window and draw on two shapes:
vp1 = plotter.vtkPlotter(shape=(6,6), size=(900,900))
vp1.renderers[35].SetBackground(.8,.9,.9)
v270 = vp1.load('data/270.vtk')     #load as vtkActor (default)
v290 = vp1.loadPoly('data/290.vtk') #load as polydata
vp1.interactive = False
vp1.showaxes = False
vp1.show(at=12, actors=[v270,v290]) # polydata are automatically
vp1.show(at=33, actors=[v270,v290]) # transformed into vtkActor
vp2 = plotter.vtkPlotter(bg=(0.9,0.9,1))
vp2.load('data/250.vtk')
vp2.load('data/270.vtk')
vp2.show(legend='an other window')


#Load a surface and show its curvature based on 4 different schemes.
#All four shapes share a common vtkCamera:
#0-gaussian, 1-mean, 2-max, 3-min
vp = plotter.vtkPlotter(shape=(1,4), size=(400,1600))
v = vp.load('data/290.vtk')
vp.interactive = False
vp.showaxes = False
for i in [0,1,2,3]:
    c = vp.curvature(v, ctype=i, r=1, alpha=0.8)
    vp.show(at=i, actors=[c], legend='method #'+str(i+1))
vp.interact()


#Draw a bunch of simple objects on separate parts of the rendering window:
vp = plotter.vtkPlotter(shape=(2,3), size=(800,1200))
vp.axes        = True
vp.commoncam   = False
vp.interactive = False
vp.show(at=0, actors=vp.arrow( [0,0,0], [1,1,1] ) )
vp.show(at=1, actors=vp.line(  [0,0,0], [1,2,3] ) )
vp.show(at=2, actors=vp.points( [ [0,0,0], [1,1,1], [3,1,2] ] ) )
vp.show(at=3, actors=vp.text('hello', cam=False, bc=(0,1,0) ) )
vp.show(at=4, actors=vp.sphere([.5,.5,.5], r=0.3), axes=0 )
vp.show(at=5, actors=vp.cube(  [.5,.5,.5], r=0.3), axes=0 )
vp.interact()


#Draw a line in 3D that fits a cloud of points,
#also show the first set of 20 points and fit a plane to them:
vp = plotter.vtkPlotter()
vp.verbose = False
for i in range(500): # draw 500 fit lines superimposed
    x = np.mgrid[-2:5 :20j][:, np.newaxis] # generate 20 points
    y = np.mgrid[ 1:9 :20j][:, np.newaxis]
    z = np.mgrid[-5:3 :20j][:, np.newaxis]
    data  = np.concatenate((x, y, z), axis=1)
    data += np.random.normal(size=data.shape)*0.8 # add gauss noise
    if i==0:
        vp.points(data, c='red')
        vp.fitplane(data)
    vp.fitline(data, lw=10, alpha=0.01) # fit
print ('Fit slope=', vp.result['slope']) # access the last fitted slope direction
vp.show(legend=['points','fitting plane','fitting line'])


#Display a tetrahedral mesh (Fenics/Dolfin format). #not yet tested on vtk6
#The internal vertices are displayed too:
#vp = plotter.vtkPlotter()
#vp.load('data/290.xml.gz', wire=1)
#vp.show(legend=['tet. mesh','boundary surf.'])


#As a short cut, the filename can be given in the show command directly:
plotter.vtkPlotter().show('data/limb.pcd') # Point cloud (PCL file format)
