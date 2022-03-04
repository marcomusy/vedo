"""Hover mouse to interactively fit a sphere to a region of the mesh"""
from vedo import *


def func(event):  # callback function
    global objs
    p = event.picked3d
    if p is None:
        return
    pts = Points(msh.closestPoint(p, N=50), r=6)
    sph = fitSphere(pts).alpha(0.1)
    txt.text(f'Radius : {sph.radius}\nResidue: {sph.residue}')
    plt.remove(objs).add([sph, pts])
    objs = [sph, pts]

msh = Mesh(dataurl+'290.vtk').subdivide()
msh.addCurvatureScalars(method=2)
msh.cmap('PRGn', vmin=-0.02).addScalarBar()

objs = [None, None]  # placeholders for the callback
txt = Text2D(__doc__, bg='yellow', font='Calco')

plt = Plotter(axes=1)
plt.addCallback('mouse hover', func)
plt.show(msh, txt, viewup='z').close()
