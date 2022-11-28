# Tests:
import numpy as np
from vedo.utils import *
from vedo.utils import make3d

print(make3d([]))
assert str(make3d([])) == '[]'

print(make3d([0,1]))
assert str(make3d([0,1])) == '[0 1 0]'

print(make3d([0,1,2]))
assert str(make3d([0,1,2])) == '[0 1 2]'

# print(make3d([0,1,2,3])) # will CORRECTLY raise error

print(make3d([[0,1,2,3], [6,7,8,9]]))
# assert str() == ''

print(make3d([ [0,1,2,3], [6,7,8,9], [6,7,8,8] ]))
# assert str() == ''

print(make3d([ [0,1,2], [6,7,8], [6,7,9] ]))
# assert str() == ''

print(make3d([ [0,1,2], [6,7,8], [6,7,9] ], transpose=True))
# assert str() == ''

print(make3d([[0,1,2]]))
# assert str() == ''

print(make3d([[0,1,2], [6,7,8]]))
# assert str() == ''

print(make3d([[0,1,2], [6,7,8], [6,7,8], [6,7,4]]))
# assert str() == ''

print(make3d([[0,1], [6,7], [6,7], [6,7]]))
# assert str() == ''
