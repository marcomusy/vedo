'''
Intersect a vtkImageData (voxel dataset) with planes
'''
from vtkplotter import show, loadImageData, probePlane, vector, Text

img = loadImageData('data/embryo.slc')

planes=[]
for i in range(6):
    print('probing slice plane #',i)
    pos = img.GetCenter() + vector(0,0,(i-3)*25.)
    a = probePlane(img, origin=pos, normal=(0,0,1)).alpha(0.2)
    planes.append(a)
    #print(max(a.scalars(0))) # access scalars this way, 0 means first

show(planes + [Text(__doc__)], axes=4, verbose=0, bg='w')