"""Setting illumination properties:
ambient, diffuse, specular, specularPower, specularColor.
"""
from vedo import Plotter, Mesh, dataurl


plt = Plotter(axes=1)

ambient, diffuse, specular = 0.1, 0., 0.
specularPower, specularColor= 20, 'white'

apple = Mesh(dataurl+'apple.ply').normalize().c('gold')

for i in range(8):
    s = apple.clone().pos((i%4)*2.2, int(i<4)*3, 0)

    #s.phong()
    s.flat()

    # modify the default with specific values
    s.lighting('default', ambient, diffuse, specular, specularPower, specularColor)
    #ambient += 0.125
    diffuse += 0.125
    specular += 0.125

    plt += s

plt += __doc__
plt.show().close()
