"""Make a video file
 with or without graphic window
(needs ffmpeg or opencv)"""
from vedo import dataurl, Plotter, Mesh, Video

# settings.screeshotScale = 2           # to get higher resolution

# declare the class instance
plt = Plotter(bg='beige', bg2='lb', axes=10, offscreen=False, interactive=1)

plt += Mesh(dataurl+"spider.ply").rotateX(-90).texture(dataurl+'textures/leather.jpg')
plt += __doc__

# Open a video file and force it to last 3 seconds in total
video = Video("spider.mp4", duration=4, backend='ffmpeg') # backend='opencv'

##############################################################
# Any rendering loop goes here, e.g.:
# for i in range(80):
#     plt.show(elevation=1, azimuth=2)  # render the scene
#     video.addFrame()                  # add individual frame

##############################################################
# OR use the automatic video shooting function:
#Options are:  elevation=(0,80), # range of elevation values
#              azimuth=(0,359),
#              zoom=None,
#              cameras=None      # a list of camera positions, e.g.:
cam1= dict(pos=(5.805, 17.34, -0.8418),
           focalPoint=(3.133, 1.506, -3.132),
           viewup=(-0.3099, 0.1871, -0.9322),
           distance=16.22,
           clippingRange=(12.35, 21.13))
cam2= dict(pos=(-1.167, 3.356, -18.66),
           focalPoint=(3.133, 1.506, -3.132),
           distance=16.22,
           clippingRange=(8.820, 25.58))
cam3 = dict(pos=(-4.119, 0.9889, -0.8867),
           focalPoint=(2.948, 1.048, -3.592),
           viewup=(-0.01864, 0.9995, -0.02682),
           distance=7.567,
           clippingRange=(0.07978, 17.04))

video.action(cameras=[cam1, cam2, cam3, cam1])

video.close()                         # merge all the recorded frames

plt.interactive().close()
