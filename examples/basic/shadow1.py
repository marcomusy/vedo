"""Cast a shadow of 2 meshes onto the wall"""
from vedo import dataurl, Mesh, Sphere, show

spider = Mesh(dataurl+"spider.ply")
# spider.rotate_z(-90).normalize()
spider.texture(dataurl+'textures/leather.jpg')
spider.add_shadow('x', -3)

sphere = Sphere(r=0.4).pos(0.5,0,1).add_shadow('x', -3)

show(spider, sphere, __doc__, axes=1, viewup="z").close()
