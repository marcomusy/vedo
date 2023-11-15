import numpy as np
from vedo.utils import make3d

print('----------------------------------8')
print(make3d([]))
assert str(make3d([])) == '[]'

print('----------------------------------9')
print(make3d([0,1]))
assert str(make3d([0,1])) == '[0 1 0]'

print('----------------------------------11')
print(make3d([[0,1],[9,8]]))
assert str(make3d([[0,1],[9,8]])) == '[[0 1 0]\n [9 8 0]]'

print('----------------------------------7')
print(make3d([[0,1], [6,7], [6,7], [6,7]]))
assert str(make3d([[0,1], [6,7], [6,7], [6,7]])) == '[[0 1 0]\n [6 7 0]\n [6 7 0]\n [6 7 0]]'   

print('----------------------------------10')
print(make3d([0,1,2]))
assert str(make3d([0,1,2])) == '[0 1 2]'

print('----------------------------------4')
print(make3d([[0,1,2]]))
assert str(make3d([[0,1,2]])) == '[[0 1 2]]'

print('----------------------------------5')
print(make3d([[0,1,2], [6,7,8]]))
assert str(make3d([[0,1,2], [6,7,8]])) == '[[0 1 2]\n [6 7 8]]'

print('----------------------------------3')
print(make3d([ [0,1,2], [6,7,8], [6,7,9] ]))
assert str(make3d([ [0,1,2], [6,7,8], [6,7,9] ])) == '[[0 1 2]\n [6 7 8]\n [6 7 9]]'

print('----------------------------------6')
print(make3d([[0,1,2], [6,7,8], [6,7,8], [6,7,4]]))
assert str(make3d([[0,1,2], [6,7,8], [6,7,8], [6,7,4]])) == '[[0 1 2]\n [6 7 8]\n [6 7 8]\n [6 7 4]]'

# print(make3d([[0,1,2,3], [6,7,8,9]])# will CORRECTLY raise error)
# print(make3d([ [0,1,2,3], [6,7,8,9], [6,7,8,8] ]))# will CORRECTLY raise error
# print(make3d([0,1,2,3])) # will CORRECTLY raise error
