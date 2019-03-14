'''
for i in range(10):
    Cone(...) # no variable assigned!
show(collection())
'''
from vtkplotter import Cone, collection, Text, show

for i in range(10):
    Cone(pos=[3*i, 0, 0], axis=[i, i-5, 0]) # no variable assigned

Text(__doc__,  font='courier')

# collection() retrieves the list of all created actors
show(collection())
