"""Generate the silhouette of a mesh
as seen along a specified direction

Axes font: """
# Source: Zhi-Qiang Zhou (https://github.com/zhouzq-thu)
from vedo import *

settings.defaultFont = 'Kanopus'
settings.useDepthPeeling = True

plt = Plotter(title="Example of projectOnPlane()")

s = Hyperboloid().rotateX(20)
pts = s.points()
n = len(pts)

plt += [s, __doc__+settings.defaultFont]

# orthogonal projection ###############################
plane1 = Plane(pos=(2,0,2), normal=(1,0,1), s=[5,5]).alpha(0.1)
so = s.clone().projectOnPlane(plane1).c('y')
plt += [plane1, so, so.silhouette('2d')]
pts1 = so.silhouette('2d').points()

# perspective projection ##############################
plane2 = Plane(pos=(3,3,3), normal=(1,1,1), s=[5,5]).alpha(0.1)
point = [6, 6, 6]
sp = s.clone().projectOnPlane(plane2, point=point).c('m')
plt += [plane2, sp, sp.silhouette('2d')]

# oblique projection ##################################
plane3 = Plane(pos=(4,8,-4), normal=(-1,0,1), s=[5,5]).alpha(0.1)
sob = s.clone().projectOnPlane(plane3, direction=(1,2,-1)).c('g')
plt += [plane3, sob, sob.silhouette('2d')]
pts2 = sob.silhouette('2d').points()

# draw the lines
for i in range(0, n, int(n/20)):
     plt += Line(pts1[i], pts[i], c='k', alpha=0.2)
     plt += Line(point,   pts[i], c='k', alpha=0.2)
     plt += Line(pts2[i], pts[i], c='k', alpha=0.2)

plt.show(axes=dict(xtitle='X-axis in \mum',
                   ytitle='Y-axis in \mum',
                   ztitle='Z-axis in \mum',
                   yzGrid=False,
                   textScale=1.5,
                  ),
).close()
