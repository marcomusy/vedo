import pymeshlab
import vedo


ms = pymeshlab.MeshSet()
ms.load_new_mesh(vedo.download(vedo.datadir+'spider.ply'))
ms.load_new_mesh(vedo.download(vedo.datadir+'panther.stl'))
# vedo.show(ms, axes=True) # this already works!

ms.print_filter_parameter_list('hausdorff_distance')
print(ms.apply_filter('hausdorff_distance', sampledmesh=0, targetmesh=1))

mlab_mesh = ms.current_mesh()

vedo_mesh = vedo.Mesh(mlab_mesh).color('b5').lw(0.1)

print("Can convert back to pymeshlab.MeshSet:\n\t", vedo_mesh.to_meshlab())

vedo.show(ms,
          axes=True, title="pymeshlab + vedo")
