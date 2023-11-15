"""A flag-post style marker"""
from vedo import ParametricShape, precision, color_map, show

s = ParametricShape("RandomHills").cmap("coolwarm")

pts = s.clone().decimate(n=10).vertices

fss = []
for p in pts:
    col = color_map(p[2], name="coolwarm", vmin=0, vmax=0.7)
    ht = precision(p[2], 3)
    fs = s.flagpost(f"Heigth:\nz={ht}m", p, c=col)
    fss.append(fs)

show(s, *fss, __doc__, bg="bb", axes=1, viewup="z")
