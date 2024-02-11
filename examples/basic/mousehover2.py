"""Hover mouse to interactively fit a sphere to a region of the mesh"""
from vedo import *

def func(event):  # callback function
    p = event.picked3d
    if p is None:
        return
    pts = Points(msh.closest_point(p, n=50), r=6)
    sph = fit_sphere(pts).alpha(0.1).pickable(False)
    pts.name = "mypoints"   # we give it a name to make it easy to
    sph.name = "mysphere"   # remove the old and add the new ones
    txt.text(f'Radius : {sph.radius}\nResidue: {sph.residue}')
    plt.remove("mypoints", "mysphere").add(pts, sph).render()

txt = Text2D(__doc__, bg='yellow', font='Calco')

msh = Mesh(dataurl+'290.vtk').subdivide()
msh.compute_curvature(method=2)
msh.cmap('PRGn', vmin=-0.02).add_scalarbar()

plt = Plotter(axes=1)
plt.add_callback('mouse hover', func)
plt.show(msh, txt, viewup='z')
plt.close()
