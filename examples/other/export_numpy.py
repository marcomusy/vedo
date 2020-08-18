from vedo import *

m1 = load(datadir+'bunny.obj').c('g').normalize().rotateX(+90)
m2 = load(datadir+'teddy.vtk').c('v').normalize().rotateZ(-90).pos(3,0,0)

show(m1, m2, axes=1)

exportWindow('scene.npz')
printc("Window exported to numpy file: scene.npz", c='g')


################################################
vp = importWindow('scene.npz')
# vp = load('scene.npz')

vp += Text2D("Imported scene", c='k', bg='b')

vp.show()

printc("\nTry also:\n> vedo scene.npz", c='g')