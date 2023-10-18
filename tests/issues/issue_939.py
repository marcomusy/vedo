from vedo import *
# cloning generates the same object type (not Mesh)
print(type(Plane().clone()))
print(type(Line([0,0],[1,1]).clone()))
