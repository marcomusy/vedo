#!/usr/bin/env python
# 
# Make a video (needs to import cv2). Still experimental!
#
import plotter

# declare the class instance
vp = plotter.vtkPlotter(title='Example 22')

vp.load('data/shapes/spider.ply', c='m', alpha=1)

# open a video file
# duration=3 will force it to last 3 seconds in total
vp.openVideo(duration=3) 

# use render() instead of show() inside loop - it's faster!
for i in range(100):
    vp.render(resetcam=True)
    vp.camera.Azimuth(4) # rotate by 4 deg at each iteration
    vp.addFrameVideo() 

vp.releaseVideo() # merges all the recorded frames

vp.show()
