"""
Make a video (needs ffmpeg)
 Set offscreen=True to only produce the video
 without any graphical window showing
"""
print(__doc__)
from vtkplotter import *

# declare the class instance
vp = Plotter(bg='beige', axes=10, interactive=0, offscreen=False)

vp.load(datadir+"spider.ply").texture("leather2").rotateX(-90)

# open a video file and force it to last 3 seconds in total
video = Video(name="spider.mp4", duration=3)

for i in range(80):
    vp.show()  # render the scene first
    vp.camera.Elevation(1)
    vp.camera.Azimuth(2)  # rotate by 5 deg at each iteration
    video.addFrame()

video.close()  # merge all the recorded frames

vp.show(interactive=1)
