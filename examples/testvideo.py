#!/usr/bin/env python
# 
# Make a video (needs to import cv2)
#
import plotter

# declare the class instance
vp = plotter.vtkPlotter(title='Example 22')

vp.load('data/290.vtk', c='b', bc='tomato', alpha=1)

# open a video file
# duration=3 will force it to last 3 seconds in total
vp.openVideo(duration=3) 

# use render() instead of show() inside loop - it's faster!
for i in range(100):
    vp.render(resetcam=True)
    vp.camera.SetPosition(700.-i*20., -10, 4344.-i*80.)
    vp.addFrameVideo() 

vp.releaseVideo() # merges all the recorded frames

vp.show()
