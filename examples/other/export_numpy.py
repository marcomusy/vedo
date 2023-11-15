from vedo import *

m1 = Mesh(dataurl+'bunny.obj').c('g').normalize().rotate_x(+90)
m2 = Mesh(dataurl+'teddy.vtk').c('v').normalize().rotate_z(-90).pos(3,0,0)

plt = show(m1, m2, axes=1)

plt.export('scene.npz')
printc("Window exported to numpy file: scene.npz", c='g')

plt.close()

################################################
plt = import_window('scene.npz')

plt += Text2D("Imported scene", c='k', bg='b')

plt.show().close()

printc("\nTry also:\n> vedo scene.npz", c='g')

