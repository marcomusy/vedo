"""
Make a video (needs ffmpeg or opencv)
 Set offscreen=True to only produce the video
 without any graphic window showing
"""
print(__doc__)
from vedo import *

# settings.screeshotScale = 2           # to get higher resolution

# declare the class instance
vp = Plotter(bg='beige', axes=10, offscreen=True)

vp.load(dataurl+"spider.ply").texture("leather").rotateX(-90)

# open a video file and force it to last 3 seconds in total
video = Video("spider.mp4", duration=6, backend='ffmpeg') # backend='opencv'

# Any rendering loop goes here, e.g.:
# for i in range(80):
#     vp.show(elevation=1, azimuth=2)  # render the scene
#     video.addFrame()                 # add individual frame

# OR use the automatic video shooting function:
video.action(zoom=1.1)
#Options are:  elevation_range=(0,80),
#              azimuth_range=(0,359),
#              zoom=None,
#              cam1=None, cam2=None   # initial and final camera positions

video.close()                         # merge all the recorded frames
