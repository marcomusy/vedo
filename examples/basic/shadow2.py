from vedo import *

man = Mesh(dataurl+'man.vtk').c('k9').lighting('glossy')
floor = Box(length=9, width=9, height=0.1).z(-1.6).c('white')
cube = Cube().pos(2,-2,-1)

p1 = Arrow([4,0,4], [0,0,0],  c='red5').scale(0.2)
p2 = Arrow([0,4,4], [0,0,0],  c='green5').scale(0.2)
p3 = Arrow([-4,-4,4], [0,0,0],  c='blue5').scale(0.2)

# Add light sources at the given positions
# (grab the position and color of the arrow object)
l1 = Light(p1)
l2 = Light(p2)
l3 = Light(p3)

plt = Plotter(bg='blackboard').add_shadows()
plt.show(l1, l2, l3, p1, p2, p3, man, floor, cube)

