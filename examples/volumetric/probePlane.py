# Intersect a vtkImageData (voxel dataset) with planes
#
from vtkplotter import Plotter, vtkio, analysis, vector

vp = Plotter(axes=4)

img = vtkio.loadImageData('data/embryo.slc')

planes=[]
for i in range(6):
    print('probing slice plane #',i)
    pos = img.GetCenter() + vector(0,0,(i-3)*25.)
    a = analysis.probePlane(img, origin=pos, normal=(0,0,1)).alpha(0.2)
    planes.append(a)
    #print(max(a.scalars(0))) # access scalars this way, 0 means first
    
vp.show(planes)