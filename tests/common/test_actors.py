from vedo import Cone, Sphere, merge, Volume
import numpy as np
import vtk

print('---------------------------------')
print('vtkVersion', vtk.vtkVersion().GetVTKVersion())
print('---------------------------------')


#####################################
cone = Cone(res=48)
sphere = Sphere(res=24)

carr = cone.cellCenters()[:, 2]
parr = cone.points()[:, 0]
cone.addCellArray(carr, 'carr')
cone.addPointArray(parr, 'parr')

carr = sphere.cellCenters()[:, 2]
parr = sphere.points()[:, 0]
sphere.addCellArray(carr, 'carr')
sphere.addPointArray(parr, 'parr')

sphere.addPointArray(np.sin(sphere.points()), 'pvectors')
sphere.addElevationScalars()

cone.computeNormals()
sphere.computeNormals()


###################################### test clone()
c2 = cone.clone()
print('clone()', cone.N(), c2.N())
assert cone.N() == c2.N()
print('clone()', cone.NCells(), c2.NCells())
assert cone.NCells() == c2.NCells()


###################################### test merge()
m = merge(sphere, cone)
print('merge()', m.N(), cone.N() + sphere.N())
assert m.N() == cone.N() + sphere.N()
print('merge()', m.NCells(), cone.NCells() + sphere.NCells())
assert m.NCells() == cone.NCells() + sphere.NCells()


###################################### inputdata
print('inputdata', [cone.inputdata()], "vtk.vtkPolyData")
assert isinstance(cone.inputdata(), vtk.vtkPolyData)


###################################### mapper
print('mapper',[cone.mapper()], "vtk.vtkPolyDataMapper")
assert isinstance(cone.mapper(), vtk.vtkPolyDataMapper)


###################################### pickable
cone.pickable(False)
cone.pickable(True)
print('pickable', cone.pickable(), True)
assert cone.pickable()


###################################### pos
cone.SetPosition(1,2,3)
print('pos', [1,2,3], cone.pos())
assert np.allclose([1,2,3], cone.pos())
cone.pos(5,6)
print('pos',[5,6,0], cone.pos())
assert np.allclose([5,6,0], cone.pos())


###################################### addPos
cone.pos(5,6,7).addPos(3,0,0)
print('addPos',[8,6,7], cone.pos())
assert np.allclose([8,6,7], cone.pos())


###################################### x y z
cone.pos(10,11,12)
cone.x(1.1)
print('x y z',[1.1,11,12], cone.pos())
assert np.allclose([1.1,11,12], cone.pos())
cone.y(1.2)
print('x y z',[1.1,1.2,12], cone.pos())
assert np.allclose([1.1,1.2,12], cone.pos())
cone.z(1.3)
print('x y z',[1.1,1.2,1.3], cone.pos())
assert np.allclose([1.1,1.2,1.3], cone.pos())


###################################### rotate
cr = cone.pos(0,0,0).clone().rotate(90, axis=(0, 1, 0))
print('rotate', np.max(cr.points()[:,2]) ,'<', 1.01)
assert np.max(cr.points()[:,2]) < 1.01


###################################### orientation
cr = cone.pos(0,0,0).clone().orientation(newaxis=(1, 1, 0))
print('orientation',np.max(cr.points()[:,2]) ,'<', 1.01)
assert np.max(cr.points()[:,2]) < 1.01

####################################### scale
cr.scale(5)
print('scale',np.max(cr.points()[:,2]) ,'>', 4.99)
assert np.max(cr.points()[:,2]) > 4.99


###################################### box
bx = cone.box()
print('box',bx.N(), 24)
assert bx.N() == 24
print('box',bx.clean().N() , 8)
assert bx.clean().N() == 8

###################################### getTransform
ct = cone.clone().rotateX(10).rotateY(10).rotateY(10)
print('getTransform', [ct.getTransform()], [vtk.vtkTransform])
assert isinstance(ct.getTransform(), vtk.vtkTransform)
ct.applyTransform(ct.getTransform())
print('getTransform',ct.getTransform().GetNumberOfConcatenatedTransforms())
assert ct.getTransform().GetNumberOfConcatenatedTransforms()


###################################### getArrayNames
arrnames = cone.getArrayNames()
print('getArrayNames',arrnames['PointData'][0] , 'parr')
assert arrnames['PointData'][0] == 'parr'
print('getArrayNames',arrnames['CellData'][0] , 'carr')
assert arrnames['CellData'][0] == 'carr'


###################################### getPointArray
arr = sphere.getPointArray('parr')
print('getPointArray',len(arr))
assert len(arr)
print('getPointArray',np.max(arr) ,'>', .99)
assert np.max(arr) > .99

arr = sphere.getCellArray('carr')
print('getCellArray',[arr])
assert len(arr)
print('getCellArray',np.max(arr) ,'>', .99)
assert np.max(arr) > .99


######################################__add__
print('__add__', [cone+sphere], [vtk.vtkAssembly])
assert isinstance(cone+sphere, vtk.vtkAssembly)


###################################### points()
s2 = sphere.clone()
pts = sphere.points()
pts2 = pts + [1,2,3]
pts3 = s2.points(pts2).points()
print('points()',sum(pts3-pts2))
assert np.allclose(pts2, pts3)


###################################### faces
print('faces()', np.array(sphere.faces()).shape , (2112, 3))
assert np.array(sphere.faces()).shape == (2112, 3)


###################################### texture
st = sphere.clone().texture('wood2')
print('texture test')
assert isinstance(st.GetTexture(), vtk.vtkTexture)


###################################### deletePoints
sd = sphere.clone().deletePoints(range(100))
print('deletePoints',sd.N() , sphere.N())
assert sd.N() == sphere.N()
print('deletePoints',sd.NCells() ,'<', sphere.NCells())
assert sd.NCells() < sphere.NCells()


###################################### reverse
# this fails on some archs (see issue #185)
# lets comment it out temporarily
sr = sphere.clone().reverse().cutWithPlane()
print('DISABLED: reverse test', sr.N(), 576)
rev = vtk.vtkReverseSense()
rev.SetInputData(sr.polydata())
rev.Update()
print('DISABLED: reverse vtk nr.pts, nr.cells')
print(rev.GetOutput().GetNumberOfPoints(),sr.polydata().GetNumberOfPoints(),
      rev.GetOutput().GetNumberOfCells(), sr.polydata().GetNumberOfCells())
# assert sr.N() == 576


###################################### quantize
sq = sphere.clone().quantize(0.1)
print('quantize',sq.N() , 834)
assert sq.N() == 834


###################################### bounds
ss = sphere.clone().scale([1,2,3])
print('bounds',ss.xbounds())
assert np.allclose(ss.xbounds(), [-1,1], atol=0.01)
print('bounds',ss.ybounds())
assert np.allclose(ss.ybounds(), [-2,2], atol=0.01)
print('bounds',ss.zbounds())
assert np.allclose(ss.zbounds(), [-3,3], atol=0.01)


###################################### averageSize
print('averageSize',sphere.averageSize())
assert 0.9 < sphere.averageSize() < 1.0
print('diagonalSize',sphere.diagonalSize())
assert 3.3 < sphere.diagonalSize() < 3.5
print('maxBoundSize',sphere.maxBoundSize())
assert 1.9 < sphere.maxBoundSize() < 2.1
print('centerOfMass',sphere.centerOfMass())
assert np.allclose(sphere.centerOfMass(), [0,0,0])
print('volume',sphere.volume())
assert 4.1 < sphere.volume() < 4.2
print('area',sphere.area())
assert 12.5 < sphere.area() < 12.6


###################################### closestPoint
pt = [12,34,52]
print('closestPoint',sphere.closestPoint(pt), [0.19883616, 0.48003298, 0.85441941])
assert np.allclose(sphere.closestPoint(pt),
                   [0.19883616, 0.48003298, 0.85441941])


###################################### findCellsWithin
ics = sphere.findCellsWithin(xbounds=(-0.5, 0.5))
print('findCellsWithin',len(ics) , 1404)
assert len(ics) == 1404


######################################transformMesh
T = cone.clone().pos(35,67,87).getTransform()
s3 = sphere.clone().applyTransform(T)
print('transformMesh',s3.centerOfMass(), (35,67,87))
assert np.allclose(s3.centerOfMass(), (35,67,87))


######################################normalize
s3 = sphere.clone().pos(10,20,30).scale([7,8,9]).normalize()
print('normalize',s3.centerOfMass(), (10,20,30))
assert np.allclose(s3.centerOfMass(), (10,20,30))
print('normalize',s3.averageSize())
assert 0.9 < s3.averageSize() < 1.1


###################################### stretch
c2 = cone.clone().stretch([0,0,0], [3,4,5])
print('stretch',c2.maxBoundSize(), '>', 5)
assert c2.maxBoundSize() > 5


###################################### crop
c2 = cone.clone().crop(left=0.5)
print('crop',np.min(c2.points()[:,0]), '>', -0.001)
assert np.min(c2.points()[:,0]) > -0.001


###################################### subdivide
s2 = sphere.clone().subdivide(4)
print('subdivide',s2.N() , 270338)
assert s2.N() == 270338


###################################### decimate
s2 = sphere.clone().decimate(0.2)
print('decimate',s2.N() , 213)
assert s2.N() == 213

###################################### normalAt
print('normalAt',sphere.normalAt(12), [9.97668684e-01, 1.01513637e-04, 6.82437494e-02])
assert np.allclose(sphere.normalAt(12), [9.97668684e-01, 1.01513637e-04, 6.82437494e-02])

###################################### isInside
print('isInside',)
assert sphere.isInside([0.1,0.2,0.3])

###################################### intersectWithLine
pts = sphere.intersectWithLine([-2,-2,-2], [2,3,4])
print('intersectWithLine',pts[0])
assert np.allclose(pts[0], [-0.8179885149002075, -0.522485613822937, -0.2269827425479889])
print('intersectWithLine',pts[1])
assert np.allclose(pts[1], [-0.06572723388671875, 0.41784095764160156, 0.9014091491699219])


############################################################################
############################################################################ Assembly
asse = cone+sphere

######################################
print('unpack',len(asse.unpack()) , 2)
assert len(asse.unpack()) ==2
print('unpack', asse.unpack(0).name)
assert asse.unpack(0) == cone
print('unpack',asse.unpack(1).name)
assert asse.unpack(1) == sphere
print('unpack',asse.diagonalSize(), 4.15)
assert 4.1 < asse.diagonalSize() < 4.2


############################################################################ Volume
X, Y, Z = np.mgrid[:30, :30, :30]
scalar_field = ((X-15)**2 + (Y-15)**2 + (Z-15)**2)/225
print('Test Volume, scalar min, max =', np.min(scalar_field), np.max(scalar_field))

vol = Volume(scalar_field)
volarr = vol.getPointArray()

print('Volume',volarr.shape[0] , 27000)
assert volarr.shape[0] == 27000
print('Volume',np.max(volarr) , 3)
assert np.max(volarr) == 3
print('Volume',np.min(volarr) , 0)
assert np.min(volarr) == 0

###################################### isosurface
iso = vol.isosurface(threshold=1.0)
print('isosurface', iso.area())
assert 2540 < iso.area() <  3000

