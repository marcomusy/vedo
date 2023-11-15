
from vedo import *

def make_cap(t1, t2):

    newpoints = t1.vertices.tolist() + t2.vertices.tolist()
    newfaces = []
    for i in range(n-1):
        newfaces.append([i,   i+1, i+n])
        newfaces.append([i+n, i+1, i+n+1])
    newfaces.append([2*n-1,   0, n])
    newfaces.append([2*n-1, n-1, 0])
    capm = Mesh([newpoints, newfaces])
    return capm


pts = [[sin(x), cos(x), x/3] for x in np.arange(0.1, 3, 0.3)]
vline  = Line(pts, lw=3)

t1 = Tube(pts, r=0.2, cap=False)
t2 = Tube(pts, r=0.3, cap=False)

tc1a, tc1b = t1.boundaries().split()
tc2a, tc2b = t2.boundaries().split()
n = tc1b.npoints

tc1b.join(reset=True).clean() # needed because indices are flipped
tc2b.join(reset=True).clean()

capa = make_cap(tc1a, tc2a)
capb = make_cap(tc1b, tc2b)

# show(vline, t1, t2, tc1a, tc1b, tc2a, tc2b, capa, capb, axes=1).close()

thick_tube = merge(t1, t2, capa, capb).lw(1)#.clean()
show("thick_tube", vline, thick_tube, axes=1).close()