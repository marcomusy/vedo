"""Compute main axes of a transformation matrix"""
from vedo import *

settings.default_font = "Calco"

M = np.random.rand(3,3) - 0.5

A = LinearTransform(M)
print(A)
print(M)

p = [1, 2, 3]
pt = Point(p)
print("---------- All these should be equal:")
print("M @ [1,2,3]    =", M @ p)
print("A([1,2,3])     =", A(p))
print("A(pt).vertices =", A(pt).vertices[0])

maxes = A.compute_main_axes()

arr1 = Arrow([0,0,0], maxes[0]).c('r', 0.5)
arr2 = Arrow([0,0,0], maxes[1]).c('g', 0.5)
arr3 = Arrow([0,0,0], maxes[2]).c('b', 0.5)

sphere1 = Sphere().wireframe().lighting('off').alpha(0.2)
sphere1.cmap('hot', sphere1.vertices[:,2])

sphere2 = sphere1.clone().apply_transform(A)

show([[sphere1, __doc__], 
      [sphere2, arr1, arr2, arr3, str(M)]],
     N=2, axes=1, bg='bb')
