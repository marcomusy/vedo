# Make a video (needs ffmpeg)
# Set offscreen=True to only produce the video without 
# any graphical window showing
#
from vtkplotter import Plotter

# declare the class instance
vp = Plotter(axes=0, offscreen=False)

vp.load('data/shapes/spider.ply', texture='leather2', alpha=1)

# open a video file and force it to last 3 seconds in total
video = vp.openVideo(name='spider.mp4', duration=3) 

for i in range(100):
    vp.render()          # render the scene first
    vp.camera.Azimuth(2) # rotate by 5 deg at each iteration
    video.addFrame() 

video.close()            # merge all the recorded frames

vp.show()
