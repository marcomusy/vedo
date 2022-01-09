from vedo import *

man = Mesh(dataurl+'man.vtk').c('k9').lighting('glossy')
floor = Box(length=9, width=9, height=0.1).z(-1.6).c('white')
cube = Cube().pos(2,-2,-0.4)

p1 = Point([1,0,1], c='red5')
p2 = Point([0,1,2], c='green5')
p3 = Point([-1,-0.5,1], c='blue5')

# Add light sources at the given positions
l1 = Light(p1)
l2 = Light(p2)
l3 = Light(p3)

plt = Plotter(bg='blackboard')
plt.addShadows()
plt.show(man, floor, cube, l1, l2, l3, p1, p2, p3)



