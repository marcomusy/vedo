"""Setting illumination properties:
ambient, diffuse, specular power and color."""
from vedo import Plotter, Mesh, dataurl

ambient = 0.1
diffuse = 0
specular = 0
specular_power = 20
specular_color = "white"

apple = Mesh(dataurl + "apple.ply")
apple.flat().c("gold")

plt = Plotter(axes=1, bg='black', bg2='white')

for i in range(8):
    x = (i % 4) * 2.2
    y = int(i < 4) * 3
    apple_copy = apple.clone().pos(x, y)

    # modify the default with specific values
    apple_copy.lighting(
        "default", ambient, diffuse, 
        specular, specular_power, specular_color
    )
    plt += apple_copy

    ambient += 0.125
    diffuse += 0.125
    specular+= 0.125

plt += __doc__
plt.show().close()
