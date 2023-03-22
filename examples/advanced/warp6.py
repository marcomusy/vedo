"""Press c while hovering to warp a text onto a surface"""
from vedo import *


def on_keypress(event):

    if event.actor is not None and event.keypress == "c":
        p = event.picked3d
        ix = mesh.closest_point(p, return_point_id=True)
        pt = points[ix]
        vec = normals[ix]
        pt = pt + vec / 8

        txt.orientation(vec).pos(pt)

        tpts = txt.clone().subsample(0.05).points()
        kpts = [mesh.closest_point(tp) for tp in tpts]
        warped = txt.clone().warp(tpts, kpts, sigma=0.01, mode="2d")
        warped.c("purple5")

        lines = Lines(tpts, kpts, alpha=0.2)
        plt.remove("Text3D", "Lines").add(txt, warped, lines)


txt = Text3D("Text3D\n01-ABCD", s=0.15, justify="centered", c="red5")

mesh = ParametricShape("RandomHills").c("gray5").alpha(0.25)
points = mesh.points()
normals = mesh.normals()

plt = Plotter()
plt.add_callback("key press", on_keypress)
plt.show(mesh, txt, __doc__, axes=9, viewup="z").close()
