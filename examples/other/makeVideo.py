"""Make a video file with or without graphic window"""
from vedo import dataurl, Plotter, Mesh, Video

# settings.screeshot_scale = 2  # to get higher resolution

msh = Mesh(dataurl+"spider.ply").rotate_x(-90)
msh.texture(dataurl+"textures/leather.jpg")

plt = Plotter(bg="beige", bg2="lb", axes=10, offscreen=False)
plt += [msh, __doc__]

##############################################################
# Open a video file and force it to last 3 seconds in total
video = Video("spider.mp4", duration=3) # or gif

##############################################################
# Any rendering loop goes here, e.g.:
# for i in range(80):
#     plt.show(elevation=1, azimuth=2)  # render the scene
#     video.add_frame()                 # add individual frame

##############################################################
# OR use the automatic video shooting function:
# Options are: elevation=(0,80), # range of elevation values
#              azimuth=(0,359),
#              zoom=None,
#              cameras=None

##############################################################
# OR set a sequence of camera positions, e.g.:
cam1 = dict(
    position=(5.805, 17.34, -0.8418),
    focal_point=(3.133, 1.506, -3.132),
    viewup=(-0.3099, 0.1871, -0.9322),
    distance=16.22,
    clipping_range=(12.35, 21.13),
)
cam2 = dict(
    position=(-1.167, 3.356, -18.66),
    focal_point=(3.133, 1.506, -3.132),
    distance=16.22,
    clipping_range=(8.820, 25.58),
)
cam3 = dict(
    position=(-4.119, 0.9889, -0.8867),
    focal_point=(2.948, 1.048, -3.592),
    viewup=(-0.01864, 0.9995, -0.02682),
    distance=7.567,
    clipping_range=(0.07978, 17.04),
)

video.action(cameras=[cam1, cam2, cam3, cam1])

video.close()  # merge all the recorded frames and write to disk

plt.interactive().close()
