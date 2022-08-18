"""Manual capping of a mesh"""
from vedo import *

def capping(msh, bias=0, invert=False, res=50):

    bn =  msh.boundaries().join(reset=True)

    pln = fitPlane(bn)
    cp = [pln.closestPoint(p) for p in bn.points()]
    pts = Points(cp)
    pts.top = pln.normal

    if invert is None:
        cutm = msh.clone().cutWithPlane(origin=pln.center, normal=pln.normal)
        invert = cutm.N() > msh.N()

    pts2 = pts.clone().orientation([0,0,1]).projectOnPlane('z')
    msh2 = pts2.tomesh(resMesh=(res,res), invert=invert).smooth()

    source = pts2.points().tolist()
    target = bn.points().tolist()
    printc(f"..warping {len(source)} points")
    msh3 = msh2.clone().warp(source, target, mode='3d')

    if not invert:
        bias *= -1
        msh3.reverse()

    if bias:
        newpts = []
        for p in msh3.points():
            q = bn.closestPoint(p)
            d = mag(p-q)
            newpt = p + d * pln.normal * bias
            newpts.append(newpt)
        msh3.points(newpts)
    return msh3


msh = Mesh(dataurl+"260_flank.vtp").c('orange5').bc('purple7').lw(0.1)

# mcap = msh.cap()  # automatic
mcap = capping(msh, invert=True)

merged_msh = merge(msh, mcap)
merged_msh.subsample(0.0001).wireframe(False)  # merge duplicate points
printc("merged_msh is closed:", merged_msh.isClosed())

show([[msh, __doc__],
      [merged_msh, merged_msh.boundaries()]],
      N=2, axes=1, elevation=-40,
).close()

