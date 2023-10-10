"""The 3D-ruler axis style,
a flag pole and a goniometer"""
from vedo import *

settings.use_parallel_projection = True  # avoid parallax effects

mesh = Cone().c("steelblue").rotate_y(90).pos(1, 2, 3)

# add a flagpole-style comment
a, v = precision(mesh.area(), 4), precision(mesh.volume(), 4)
fp = mesh.flagpole(
    "S = πr^2 +πr√(h^2 +r^2 )\n  = " + a 
    + "~μm^2 \nV = πr^2 ·h/3\n  = " + v 
    + "~μm^3",
    s=0.1,
)
fp.color("r3").scale(0.7).follow_camera()

# measure the angle formed by 3 points
gon = Goniometer(
    [-0.5, 1, 2], [2.5, 2, 2], [-0.5, 3, 3], 
    prefix=":alpha_c =~", lw=2, s=0.8
)

# show distance of 2 points
rul = Ruler3D(
    (-0.5, 2, 1.9),
    (2.5, 2, 2.9),
    prefix="L_x =",
    units="μm",
    axis_rotation=90,
    tick_angle=70,
)

# make 3d rulers along the bounding box (similar to set axes=7)
ax3 = RulerAxes(mesh, units="μm")

show(mesh, fp, gon, rul, ax3, __doc__, bg2="lb", viewup="z").close()
