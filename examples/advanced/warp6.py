"""Press c while hovering to warp a Mesh onto another Mesh"""
from vedo import *


def on_keypress(event):
    if event.object and event.keypress == "c":
        picked = event.picked3d
        idx = mesh.closest_point(picked, return_point_id=True)
        n = normals[idx]
        p = verts[idx] + n / 5

        txt = Text3D("Text3D\nABCDEF", s=0.1, justify="centered").c("red5")
        txt.reorient([0,0,1], n).pos(p)

        tpts = txt.clone().subsample(0.05).vertices
        kpts = [mesh.closest_point(tp) for tp in tpts]
        warped = txt.clone().warp(tpts, kpts, sigma=0.01, mode="2d")
        warped.c("purple5")

        lines = Lines(tpts, kpts).alpha(0.2)
        plt.remove("Text3D", "Lines").add(txt, warped, lines).render()

mesh = ParametricShape("RandomHills").scale([1,1,0.5])
mesh.c("gray5").alpha(0.25)
verts = mesh.vertices
normals = mesh.vertex_normals

plt = Plotter()
plt.add_callback("key press", on_keypress)
plt.show(mesh, __doc__, axes=9, viewup="z").close()
