#!/usr/bin/env python
#
from __future__ import division, print_function
from random import uniform as u
import numpy as np
import plotter


# 1 ########################################################################################
# Declare an instance of the class
vp = plotter.vtkPlotter(title='Example 1 and 2')
# vp.help() # shows a help message

# Load a vtk file as a vtkActor and visualize it.
# The tridimensional shape corresponds to the outer shape of the embryonic mouse limb
# at about 11 days of gestation.
# Choose a tomato color for the internal surface, and no transparency.
vp.load('data/250.vtk', c='b', bc='tomato', alpha=1) # c=(R,G,B), #hex, letter or name
vp.show()             # picks what is automatically stored in vp.actors list
# Press Esc to close the window and exit python session, or q to continue


# 2 ########################################################################################
# Load a vtk file as a vtkActor and visualize it in wireframe style.
a = vp.load('data/290.vtk', wire=1) # same as a.GetProperty().SetRepresentationToWireframe()
vp.axes = False
vp.show(legend=None) # picks what is automatically stored in vp.actors
#vp.show(a)           # would ignore the content of vp.actors and draws a
#vp.show(actors=[a])  # same as above


# 3 ########################################################################################
# Load 3 actors assigning each a different color, uses their file names as legend entries.
# No need to use any variables, as actors are stored internally in vp.actors:
vp = plotter.vtkPlotter(title='Example 3')
vp.load('data/250.vtk', c=(1,0.4,0))
vp.load('data/270.vtk', c=(1,0.6,0))
vp.load('data/290.vtk', c=(1,0.8,0))
print ('Loaded vtkActors: ', len(vp.actors))
vp.show()


# 4 ########################################################################################
# Draw a spline that goes through a set of points, show the points (nodes=True):
vp = plotter.vtkPlotter(title='Example 4')
pts = [(u(0,10), u(0,10), u(0,10)) for i in range(20)]
vp.spline(pts, s=1.5, nodes=True)
vp.show(legend='a random spline')


# 5 ########################################################################################
# Draw the PCA ellipsoid that contains 67% of a cloud of points:
vp = plotter.vtkPlotter(title='Example 5')
pts = [(u(0,200), u(0,200), u(0,200)) for i in range(50)]
vp.points(pts)
vp.pca(pts, pvalue=0.67, pcaAxes=True)
vp.show(legend=['points', 'PCA ellipsoid'])


# 6 ########################################################################################
# Show 3 planes as a grid, add a dummy sine plot on top left:
xycoords = [(np.exp(i/10.), np.sin(i/5.)) for i in range(40)]
vp = plotter.vtkPlotter(title='Example 6')
vp.xyplot( xycoords )
vp.grid(pos=(0,0.5,0.5), normal=(1,0,0), c=(1,0,0))
vp.grid(pos=(0.5,0,0.5), normal=(0,1,0), c=(0,1,0))
vp.grid(pos=(0.5,0.5,0), normal=(0,0,1), c=(0,0,1))
vp.show(axes=0)


# 7 ########################################################################################
# Show the vtk boundaries of a vtk surface and its normals
# (ratio reduces the total nr of arrows by the indicated factor):
vp = plotter.vtkPlotter(title='Example 7')
va = vp.load('data/290.vtk', c='maroon', legend=0)
vp.normals(va, ratio=5, legend=False)
vp.boundaries(va)
vp.show(legend='shape w/ boundaries')


# 8 ########################################################################################
# Increases the number of points in a vtk mesh using 'subDivideSurface'
# and shows them before and after.
vp = plotter.vtkPlotter(N=2)
vp.interactive = False
a = vp.load('data/290.vtk')
aCoord = plotter.getCoordinates(a)
aPoints = vp.points(aCoord, r=3, legend='# points =' + str(len(aCoord)))

addingPoints = vp.subDivideSurface(a)  # Increasing the number of points of the mesh
addingPointsCoord = plotter.getCoordinates(addingPoints)
newPoints = vp.points(addingPointsCoord, r=3, legend='# points =' + str(len(addingPointsCoord)))

vp.show(at=0, actors=aPoints)
vp.show(at=1, actors=newPoints)
vp.show(interactive=1)


# 9 ########################################################################################
# Split window in a 36 subwindows and draw something in windows nr 12 and nr 33.
# Then open an independent window and draw on two shapes:
vp1 = plotter.vtkPlotter(shape=(6,6), title='Example 9')
vp1.renderers[35].SetBackground(.8,.9,.9)
vp1.axes = False
a = vp1.load('data/250.vtk')
b = vp1.load('data/270.vtk', legend='some legend')
c = vp1.load('data/290.vtk')
vp1.show(at=12, actors=[a,b], interactive=False)
vp1.show(at=33, actors=[b,c])
vp2 = plotter.vtkPlotter(bg=(0.9,0.9,1))
vp2.load('data/250.vtk', legend='an other window')
vp2.load('data/270.vtk')
vp2.show()


# 10 ########################################################################################
# Load a surface and show its curvature based on 4 different schemes.
# All four shapes share a common vtkCamera:
# 0-gaussian, 1-mean, 2-max, 3-min
vp = plotter.vtkPlotter(shape=(1,4), title='Example 10')
v = vp.load('data/290.vtk')
vp.interactive = False
vp.axes = False
for i in [0,1,2,3]:
    c = vp.curvature(v, method=i, r=1, alpha=0.8)
    vp.show(at=i, actors=[c], legend='method #'+str(i+1))
vp.show(interactive=1)


# 11 #######################################################################################
# Draw a simple objects on separate parts of the rendering window:
# split window to best accomodate 6 renderers
vp = plotter.vtkPlotter(N=9, title='Example 11')
vp.commoncam   = False
vp.interactive = False
vp.show(at=0, actors=vp.arrow(),  legend='arrow()' )
vp.show(at=1, actors=vp.line(),   legend='line()' )
vp.show(at=2, actors=vp.points(), legend='points()' )
vp.show(at=3, actors=vp.text('hello', cam=False, bc=(1,0,0) ) )
vp.show(at=4, actors=vp.sphere() )
vp.show(at=5, actors=vp.cube(),   legend='cube()')
vp.show(at=6, actors=vp.ring(),   legend='ring()')
vp.show(at=7, actors=vp.helix(),  legend='helix()')
vp.show(at=8, actors=vp.cylinder(), legend='cylinder()')
vp.show(interactive=1)


# 12 #######################################################################################
# Draw a bunch of objects in many formats. Split window in 3 rows and 3 columns
vp = plotter.vtkPlotter(shape=(3,3), title='Example 12')
vp.commoncam   = False
vp.interactive = False
vp.show(at=0, c=0, actors='data/beethoven.ply', ruler=1, axes=0)
vp.show(at=1, c=1, actors='data/cow.g', wire=1)
vp.show(at=2, c=2, actors='data/limb.pcd')
vp.show(at=3, c=3, actors='data/shapes/spider.ply')
vp.show(at=4, c=4, actors='data/shuttle.obj')
vp.show(at=5, c=5, actors='data/shapes/magnolia.vtk')
vp.show(at=6, c=6, actors='data/shapes/man.vtk', alpha=1, axes=1)
a = vp.getActors('man')        # retrieve actors by matching legend string
a[0].rotateX(90)               #  and rotate it by 90 degrees around x
a[0].rotateY(1.57, rad=True)   #  and then by 90 degrees around y
vp.show(at=7, c=7, actors='data/teapot.xyz')
vp.show(at=8, c=8, actors='data/unstrgrid.vtu')
vp.show(interactive=1)


# 13 #######################################################################################
# Draw the same object with different surface textures
vp = plotter.vtkPlotter(shape=(3,3), verbose=0, axes=0, interactive=0, title='Example 13')
mat = ['aqua','gold2','metal1','ivy','paper','sky','white2','wood3','wood7']
for i,mname in enumerate(mat): # mname can be any jpeg file
    sp = vp.load('data/beethoven.ply', alpha=1, texture=mname)
    vp.show(at=i, actors=sp, legend=mname)
vp.show(interactive=1)


# 14 ########################################################################################
# Draw a line in 3D that fits a cloud of points,
# also show the first set of 20 points and fit a plane to them:
vp = plotter.vtkPlotter(verbose=False, title='Example 14')
for i in range(500): # draw 500 fit lines superimposed
    x = np.linspace(-2, 5, 20) # generate 20 points
    y = np.linspace( 1, 9, 20)
    z = np.linspace(-5, 3, 20)
    data = np.array(list(zip(x,y,z)))
    data+= np.random.normal(size=data.shape)*0.8 # add gauss noise
    if i==0:
        vp.points(data, c='red')
        vp.fitPlane(data)
    vp.fitLine(data, lw=10, alpha=0.01) # fit
print ('Fit slope=', vp.result['slope']) # the last fitted slope direction
vp.show(legend=['points','fitting plane','fitting line'])


# 15 ########################################################################################
# Cut a set of shapes with a plane that goes through the
# point at x=500 and has normal (0, 0.3, -1).
# Wildcards are ok to load multiple files or directories:
vp = plotter.vtkPlotter(title='Example 15')
vp.load('data/*.vtk', c='orange', bc='aqua', alpha=1)
for a in vp.actors:
    vp.cutActor(a, origin=(500,0,0), normal=(0,0.3,-1))
vp.show()


# 16 ########################################################################################
# As a short-cut, the filename can be given in the show command directly:
plotter.vtkPlotter().show('data/limb.pcd') # Point cloud (PCL file format)


# 17 ########################################################################################
# Display a tetrahedral mesh (Fenics/Dolfin format).
# The internal vertices are displayed too:
vp = plotter.vtkPlotter(title='Example 17')
vp.load('data/290.xml.gz')
vp.show(legend='tetrahedral mesh')


# 18 ########################################################################################
# Align 2 shapes and for each vertex of the first draw
# and arrow to the closest point of the second:
vp = plotter.vtkPlotter(title='Example 18')
a1, a2 = vp.load('data/2[79]0.vtk')
a1.GetProperty().SetColor(0,1,0)
a1b = vp.align(a1, a2, rigid=1)
ps1 = plotter.getCoordinates(a1b) # coordinates of actor
for p in ps1: vp.arrow(p, plotter.closestPoint(a2, p))
vp.show(legend=['Source','Target','Aligned','Links'])


# 19 ########################################################################################
# Find closest point in set pts1 to pts2 within a specified radius
pts1 = [(u(0,5), u(0,5), u(0,5)) for i in range(40)]
pts2 = [(u(0,5), u(0,5), u(0,5)) for i in range(20)]
vp = plotter.vtkPlotter(title='Example 19')
vp.points(pts1, r=4,  alpha=1, legend='point set 1')
vp.points(pts1, r=25, alpha=0.1) # make a halo
a = vp.points(pts2, r=4, c='r', alpha=1, legend='point set 2')
for p in pts1:
    cp = plotter.closestPoint(a, p, radius=2)
    vp.line(p, cp)
vp.show()


# 20 ########################################################################################
# Draw a cloud of points each one with a different color
# which depends on its position
vp = plotter.vtkPlotter(title='Example 20')
rgb = [(u(0,255), u(0,255), u(0,255)) for i in range(2000)]
vp.points(rgb, c=rgb, alpha=0.7, legend='RGB points')
vp.show()


# 21 ########################################################################################
a = vp.load('data/cow.g')
a.cutterWidget() # invoke widget


# 22 ########################################################################################
# Make a video (needs cv2 package)
vp = plotter.vtkPlotter(title='Example 22')
vp.load('data/290.vtk', c='b', bc='tomato', alpha=1)
vp.openVideo(duration=3) # will force it to last 3 seconds in total
for i in range(100):
    vp.render(resetcam=True)
    vp.camera.SetPosition(700.-i*20., -10, 4344.-i*80.)
    vp.addFrameVideo()
vp.releaseVideo()
vp.show()

############################################################################################
