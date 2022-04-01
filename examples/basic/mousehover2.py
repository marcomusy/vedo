"""Hover mouse to interactively fit a sphere to a region of the mesh"""
from vedo import *

def func(event):  # callback function
    p = event.picked3d
    if p is None:
        return
    pts = Points(msh.closestPoint(p, N=50), r=6)
    sph = fitSphere(pts).alpha(0.1)
    pts.name = "mypoints"   # we give it a name to make it easy to
    sph.name = "mysphere"   # remove the old and add the new ones
    txt.text(f'Radius : {sph.radius}\nResidue: {sph.residue}')
    plt.remove("mypoints", "mysphere").add(pts, sph)

txt = Text2D(__doc__, bg='yellow', font='Calco')

msh = Mesh(dataurl+'290.vtk').subdivide()
msh.addCurvatureScalars(method=2)
msh.cmap('PRGn', vmin=-0.02).addScalarBar()

plt = Plotter(axes=1)
plt.addCallback('mouse hover', func)
plt.show(msh, txt, viewup='z')
plt.close()
