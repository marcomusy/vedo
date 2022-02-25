"""Give a cartoony appearance to a 3D mesh"""
from vedo import dataurl, settings, Plotter, Mesh, Text2D

settings.useDepthPeeling = True
settings.multiSamples = 8 # antialiasing

plt = Plotter() # this creates a default camera, needed by silhouette()

txt = Text2D(__doc__, pos="bottom-center", font="Bongas", s=2, bg='dg')

man = Mesh(dataurl+'man.vtk')
man.lighting('off').c('pink').alpha(0.5)

ted = Mesh(dataurl+'teddy.vtk').scale(0.4).rotateZ(-45).pos(-1,-1,-1)
ted.lighting('off').c('sienna').alpha(0.1)

plt.show(txt,
         ted, man,
         ted.silhouette(),
         man.silhouette(featureAngle=40).lineWidth(3).color('dr'),
         bg='wheat', bg2='lb', elevation=-80, zoom=1.2,
)
plt.close()
