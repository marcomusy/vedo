"""Orbitals of the Hydrogen atom"""
import sympy as sm
from sympy.physics.hydrogen import Psi_nlm
from vedo import *

n,l,m = 4,2,1

res = 16
vol = Volume(dims=(res,res,res)) # an empty Volume
arr = vol.points()    # Volume -> numpy array

arr = (arr-arr.mean(axis=0)) * 40/res  # shift to origin and scale
x,y,z = arr.T
r, theta, phi = utils.cart2spher(x, y, z)
rtp = np.c_[r, theta, phi]

vals = []
pb = ProgressBar(0, len(rtp))
for r,t,p in rtp:
    prob = sm.Abs(Psi_nlm(n,l,m, r,t,p).evalf())
    vals.append(prob)
    pb.print()

vol.pointdata["PsiSquare"] = np.array(vals, dtype=float)*100
vol.pointdata.select("PsiSquare")
vol.addScalarBar3D("100 \dot |\Psi|^2").print()

show(vol, f"{__doc__}\n for (n,l,m) = {n,l,m}", axes=1)

