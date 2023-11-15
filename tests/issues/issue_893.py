import numpy as np
import vedo

N = np.arange(24).reshape([2, 3, 4])
cubes = []
texts = []
positions = []
xs, ys, zs = N.shape
for x in range(xs):
    for y in range(ys):
        for z in range(zs):
            pos = (x, y, z)
            val = N[x, y, z]
            cubes.append(vedo.Cube(pos=pos, side=0.6, alpha=0.1))
            positions.append(pos)

pts = vedo.Points(positions)
labs= pts.labels2d(font='Quikhand', scale=2, justify="top-center", c="red4")
vedo.show(cubes, labs, axes=4).close()

################################################################### (BUG)
texts = []
xs, ys, zs = [2, 1, 2]
for x in range(xs):
    for y in range(ys):
        for z in range(zs):
            pos = (x, y, z)
            txt = vedo.Text3D(f"{pos}", pos, s=0.05, justify='centered', c='r5')
            txt.rotate_x(0.00001)
            txt.shift(0.00001, 0.00001, 0.00001) # same as rotate_x
            texts.append(txt.follow_camera())

vedo.show(texts, axes=1)
