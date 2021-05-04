from vedo import *

m1 = Mesh(dataurl+'bunny.obj').c('g').normalize().rotateX(+90)
m2 = Mesh(dataurl+'teddy.vtk').c('v').normalize().rotateZ(-90).pos(3,0,0)

p = show(m1, m2, axes=1)

exportWindow('scene.npz')
printc("Window exported to numpy file: scene.npz", c='g')
p.close()

################################################
plt = importWindow('scene.npz')
# plt = load('scene.npz')

plt += Text2D("Imported scene", c='k', bg='b')

plt.show().close()

printc("\nTry also:\n> vedo scene.npz", c='g')