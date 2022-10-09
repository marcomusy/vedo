from vedo import *

ug = Mesh('https://github.com/marcomusy/vedo/files/4602353/domain_unstruct.vtk.gz')

# make up some custom vector field
pts   = ug.points()
x,y,z = pts.T
windx = np.ones_like(x)*4
windy = np.exp(-(x+18)**2/100) * np.sign(y)/(abs(y)+8)*20
wind  = np.c_[windx, windy, np.zeros_like(windy)]

ug.pointdata["wind"] = wind  # add the vectors to the mesh

ars = Arrows(pts-wind/10, pts+wind/10, c='hot')

ypr = np.linspace(-15,15, num=25)
xpr = np.zeros_like(ypr)-40
probes = np.c_[xpr, ypr]

str_lns = StreamLines(ug, probes, max_propagation=80, lw=3)

show(ars, str_lns, zoom=8, bg2='lb').close()
