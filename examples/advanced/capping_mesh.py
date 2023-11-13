"""Manual capping of a mesh"""
from vedo import *

def capping(amsh, bias=0, invert=False, res=50):

    bn =  amsh.boundaries().join(reset=True)

    pln = fit_plane(bn)
    cp = [pln.closest_point(p) for p in bn.vertices]
    pts = Points(cp)

    if invert is None:
        cutm = amsh.clone().cut_with_plane(origin=pln.center, normal=pln.normal)
        invert = cutm.npoints > amsh.npoints

    pts2 = pts.clone().reorient(pln.normal, [0,0,1]).project_on_plane('z')
    msh2 = pts2.generate_mesh(invert=invert, mesh_resolution=res)

    source = pts2.vertices.tolist()
    target = bn.vertices.tolist()
    printc(f"..warping {len(source)} points")
    msh3 = msh2.clone().warp(source, target, mode='3d')

    if not invert:
        bias *= -1
        msh3.reverse()

    if bias:
        newpts = []
        for p in msh3.vertices:
            q = bn.closest_point(p)
            d = mag(p-q)
            newpt = p + d * pln.normal * bias
            newpts.append(newpt)
        msh3.points(newpts)
    return msh3


msh = Mesh(dataurl+"260_flank.vtp").c('orange5').bc('purple7').lw(1)

# mcap = msh.cap()  # automatic
mcap = capping(msh, invert=True)

merged_msh = merge(msh, mcap).clean().smooth()
merged_msh.subsample(0.0001).wireframe(False)  # merge duplicate points
printc("merged_msh is closed:", merged_msh.is_closed())

show([[msh, __doc__],
      [merged_msh, merged_msh.boundaries()]],
      N=2, axes=1, elevation=-40,
).close()

