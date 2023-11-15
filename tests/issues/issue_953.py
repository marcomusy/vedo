import numpy as np
from colorcet import bmy
from vedo import Points, Grid, show


def add_reconst(name, i=2):
    p = Points(np_pts)

    bb = list(p.bounds())
    if bb[0] == bb[1]:
        bb[1] += 1
        bb[0] -= 1
    if bb[2] == bb[3]:
        bb[3] += 1
        bb[2] -= 1
    if bb[4] == bb[5]:
        bb[5] += 1
        bb[4] -= 1

    m = p.reconstruct_surface(bounds=bb)
    m.cmap(bmy, m.vertices[:, i]).add_scalarbar()
    names.append(name)
    pts.append(p)
    mesh.append(m)


names = []
pts = []
mesh = []

grid = Grid(res=(20, 20))

# grid with constant z=0
np_pts = grid.clone().vertices
add_reconst("z=0")

# grid with constant z=1
np_pts = grid.clone().z(1).vertices
add_reconst("z=1")

# grid with varying z
np_pts = grid.clone().vertices
np_pts[:, 2] = np.sin(np_pts[:, 0])
add_reconst("sin z")

# constant y
np_pts = grid.clone().rotate_x(90).vertices
add_reconst("y=0", 1)

# constant x
np_pts = grid.clone().rotate_y(90).vertices
add_reconst("x=0", 0)

# rotated plan
np_pts = grid.clone().rotate_x(90).rotate_y(40).rotate_z(60).vertices
add_reconst("tilted")

show([t for t in zip(names, pts, mesh)], N=len(mesh), sharecam=False, axes=1)
