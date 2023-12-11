"""Remesh a surface mesh using the ACVD algorithm."""
# Needs PyACVD: pip install pyacvd
# See: https://github.com/akaszynski/pyacvd
from vedo import Sphere, Mesh, show
from vedo.pyplot import histogram
from pyvista import wrap
from pyacvd import Clustering

msh1 = Sphere(res=50).cut_with_plane()
msh1.compute_quality().cmap('RdYlGn', on='cells', vmin=0, vmax=70).linewidth(1)

clus = Clustering(wrap(msh1.dataset))
clus.cluster(1000, maxiter=100, iso_try=10, debug=False)
pvremsh1 = clus.create_mesh()

msh2 = Mesh(pvremsh1).shift([2,0,0])
msh2.compute_quality().cmap('RdYlGn', on='cells', vmin=0, vmax=70).linewidth(1)

his1 = histogram(msh1.celldata["Quality"], xlim=(0,70), aspect=2, c='RdYlGn', title='Original Quality')
his2 = histogram(msh2.celldata["Quality"], xlim=(0,70), aspect=2, c='RdYlGn', title='Remeshed Quality')
his1 = his1.clone2d('bottom-left',  0.75)
his2 = his2.clone2d('bottom-right', 0.75)

show(msh1, msh2, his1, his2, __doc__, bg='k5', bg2='wheat')
#remsh1.write('sphere.vtk')
