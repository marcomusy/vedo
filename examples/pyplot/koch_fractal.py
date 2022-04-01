"""Koch snowflake fractal"""
from vedo import sqrt, Line, show

levels = 7

def koch(level):
    # Compute Koch fractal contour points
    k = sqrt(3)/2
    if level:
        points = koch(level-1) + [(0, 0)]  # recursion!
        kpts = []
        for i in range(len(points)-1):
            p1, p2 = points[i], points[i+1]
            dx, dy = (p2[0]-p1[0])/3, (p2[1]-p1[1])/3
            pa = (p1[0] + dx  , p1[1] + dy  )
            pb = (p1[0] + dx*2, p1[1] + dy*2)
            z = complex(pb[0]-pa[0], pb[1]-pa[1]) * (0.5-k*1j)
            p3 = (pa[0]+z.real, pa[1]+z.imag)
            kpts += [p1, pa, p3, pb]
        return kpts
    else:
        return [(0, 0), (1, 0), (0.5, k)]

kochs = []
for i in range(levels):
    # Create a Line from the points and mesh the inside with minimum resolution
    kmsh = Line(koch(i)).tomesh(resMesh=1).lw(0).color(-i).z(-i/1000)
    kochs.append(kmsh)

show(kochs, __doc__+ f"\nlevels: {levels}\npoints: {kmsh.N()}", bg2='lb').close()
