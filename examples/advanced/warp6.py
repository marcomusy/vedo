"""Press c while hovering to warp a Mesh onto another Mesh"""
from vedo import *


def on_keypress(event):
    if event.actor and event.keypress == "c":
        picked = event.picked3d
        idx = mesh.closest_point(picked, return_point_id=True)
        pt = points[idx]
        n = normals[idx]
        pt = pt + n / 5

        txt.orientation(n).pos(pt)

        tpts = txt.clone().subsample(0.05).vertices
        kpts = [mesh.closest_point(tp) for tp in tpts]
        warped = txt.clone().warp(tpts, kpts, sigma=0.01, mode="2d")
        warped.c("purple5")

        lines = Lines(tpts, kpts, alpha=0.2)
        plt.remove("Text3D", "Lines").add(txt, warped, lines).render()


txt = Text3D("Text3D\n01-ABCD", s=0.1, justify="centered", c="red5")

mesh = ParametricShape("RandomHills").scale([1,1,0.5])
mesh.c("gray5").alpha(0.25)
points = mesh.vertices
normals = mesh.vertex_normals

plt = Plotter()
plt.add_callback("key press", on_keypress)
plt.show(mesh, txt, __doc__, axes=9, viewup="z").close()
