
from vedo import *

one = Mesh(dataurl+"bunny.obj", c="green")
two = Mesh(dataurl+"bunny.obj", c="red").shift(0.3,0,0)

one.add_shadow("z", -0.1)
two.add_shadow("z", -0.1)

show(one, two, axes=1).close()