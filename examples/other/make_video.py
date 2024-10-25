"""Make a video file with or without graphic window"""
from vedo import dataurl, Plotter, Mesh, Axes, Video


msh = Mesh(dataurl+"data/teapot.vtk").normalize()#.rotate_x(-90)
msh.shift(-msh.center_of_mass())

plt = Plotter(bg="beige", bg2="lb", offscreen=False)
plt += [msh, Axes(msh), __doc__]

##############################################################
# Open a video file and force it to last 3 seconds in total
video = Video("vedo_video.mp4", duration=3) # or gif

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
    focal_point=(0.133, 0.506, -0.132),
    viewup=(-0.3099, 0.1871, -0.9322),
    clipping_range=(12.35, 21.13),
)
cam2 = dict(
    position=(-1.167, 3.356, -18.66),
    focal_point=(0.133, 0.506, -0.132),
    clipping_range=(8.820, 25.58),
)
cam3 = dict(
    position=(-4.119, 0.9889, -0.8867),
    focal_point=(0.948, 0.048, -0.592),
    viewup=(-0.01864, 0.9995, -0.02682),
    clipping_range=(0.07978, 17.04),
)

video.action(cameras=[cam1, cam2, cam3, cam1])

video.close()  # merge all the recorded frames and write to disk

plt.interactive().close()
