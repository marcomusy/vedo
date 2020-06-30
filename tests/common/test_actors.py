from vedo import Cone, Sphere, merge, Volume, show
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
assert cone.N() == c2.N()
assert cone.NCells() == c2.NCells()


###################################### test merge()
m = merge(sphere, cone)
assert m.N() == cone.N() + sphere.N()
assert m.NCells() == cone.NCells() + sphere.NCells()


###################################### inputdata
assert isinstance(cone.inputdata(), vtk.vtkPolyData)


###################################### mapper
assert isinstance(cone.mapper(), vtk.vtkPolyDataMapper)


###################################### pickable
cone.pickable(False)
cone.pickable(True)
assert cone.pickable()


###################################### pos
cone.SetPosition(1,2,3)
assert np.allclose([1,2,3], cone.pos())
cone.pos(5,6)
assert np.allclose([5,6,0], cone.pos())


###################################### addPos
cone.pos(5,6,7).addPos(3,0,0)
assert np.allclose([8,6,7], cone.pos())


###################################### x y z
cone.pos(10,11,12)
cone.x(1.1)
assert np.allclose([1.1,11,12], cone.pos())
cone.y(1.2)
assert np.allclose([1.1,1.2,12], cone.pos())
cone.z(1.3)
assert np.allclose([1.1,1.2,1.3], cone.pos())


###################################### rotate
cr = cone.pos(0,0,0).clone().rotate(90, axis=(0, 1, 0))
assert np.max(cr.points()[:,2]) < 1.01


###################################### orientation
cr = cone.pos(0,0,0).clone().orientation(newaxis=(1, 1, 0))
assert np.max(cr.points()[:,2]) < 1.01

# scale
cr.scale(5)
assert np.max(cr.points()[:,2]) > 4.99


###################################### orientation
bx = cone.box()
assert bx.N() == 24
assert bx.clean().N() == 8

###################################### getTransform
ct = cone.clone().rotateX(10).rotateY(10).rotateY(10)
assert isinstance(ct.getTransform(), vtk.vtkTransform)
ct.setTransform(ct.getTransform())
assert ct.getTransform().GetNumberOfConcatenatedTransforms()


###################################### getArrayNames
print('getArrayNames:', cone.getArrayNames())
arrnames = cone.getArrayNames()
assert arrnames['PointData'][0] == 'parr'
assert arrnames['CellData'][0] == 'carr'


###################################### getPointArray
print('Test getPointArray')
arr = sphere.getPointArray('parr')
assert len(arr)
assert np.max(arr) > .99

arr = sphere.getCellArray('carr')
assert len(arr)
assert np.max(arr) > .99


######################################__add__
print('Test __add__')
assert isinstance(cone+sphere, vtk.vtkAssembly)


###################################### points()
print('Test points')

s2 = sphere.clone()
pts = sphere.points()
pts2 = pts + [1,2,3]
pts3 = s2.points(pts2).points()
assert np.allclose(pts2, pts3)


###################################### faces
print('Test faces', np.array(sphere.faces()).shape )
assert np.array(sphere.faces()).shape == (2112, 3)


###################################### texture
print('Test texture')
st = sphere.clone().texture('wood2')
assert isinstance(st.GetTexture(), vtk.vtkTexture)


###################################### deletePoints
print('Test deletePoints')
sd = sphere.clone().deletePoints(range(100))
assert sd.N() == sphere.N()
assert sd.NCells() < sphere.NCells()


###################################### reverse
print('Test reverse')
sr = sphere.clone().reverse().cutWithPlane()
assert sr.N() == 576


###################################### quantize
print('Test quantize')
sq = sphere.clone().quantize(0.1)
assert sq.N() == 834


###################################### bounds
print('Test bounds')
ss = sphere.clone().scale([1,2,3])
assert np.allclose(ss.xbounds(), [-1,1], atol=0.01)
assert np.allclose(ss.ybounds(), [-2,2], atol=0.01)
assert np.allclose(ss.zbounds(), [-3,3], atol=0.01)


###################################### averageSize
print('Test sizes et al')
assert 0.9 < sphere.averageSize() < 1.0
assert 3.3 < sphere.diagonalSize() < 3.5
assert 1.9 < sphere.maxBoundSize() < 2.1
assert np.allclose(sphere.centerOfMass(), [0,0,0])
assert 4.1 < sphere.volume() < 4.2
assert 12.5 < sphere.area() < 12.6


###################################### closestPoint
print('Test closestPoint')
pt = [12,34,52]
assert np.allclose(sphere.closestPoint(pt),
                   [0.19883616, 0.48003298, 0.85441941])


###################################### findCellsWithin
print('Test findCellsWithin')
ics = sphere.findCellsWithin(xbounds=(-0.5, 0.5))
assert len(ics) == 1404


######################################transformMesh
print('Test transformMesh')
T = cone.clone().pos(35,67,87).getTransform()
s3 = sphere.clone().setTransform(T)
assert np.allclose(s3.centerOfMass(), (35,67,87))


######################################normalize
print('Test normalize')
c3 = cone.clone().normalize()
#print('centerOfMass =', c3.centerOfMass())
assert np.allclose(c3.centerOfMass(), [0,0,-1.41262311])
assert 0.9 < c3.averageSize() < 1.1


###################################### stretch
print('Test stretch')
c2 = cone.clone().stretch([0,0,0], [3,4,5])
assert c2.maxBoundSize() > 5


###################################### crop
print('Test crop')
c2 = cone.clone().crop(left=0.5)
assert np.min(c2.points()[:,0]) > -0.001


###################################### subdivide
print('Test subdivide')
s2 = sphere.clone().subdivide(4)
assert s2.N() == 270338


###################################### decimate
print('Test decimate')
s2 = sphere.clone().decimate(0.2)
assert s2.N() == 213


###################################### pointGaussNoise
print('Test pointGaussNoise')
s2 = sphere.clone().pointGaussNoise(2)
assert s2.maxBoundSize() > 1.1


###################################### normalAt
print('Test normalAt')
assert np.allclose(sphere.normalAt(12), [9.97668684e-01, 1.01513637e-04, 6.82437494e-02])

###################################### isInside
print('Test isInside')
assert sphere.isInside([0.1,0.2,0.3])

###################################### intersectWithLine
print('Test intersectWithLine')
pts = sphere.intersectWithLine([-2,-2,-2], [2,3,4])

assert np.allclose(pts[0], [-0.8179885149002075, -0.522485613822937, -0.2269827425479889])
assert np.allclose(pts[1], [-0.06572723388671875, 0.41784095764160156, 0.9014091491699219])


############################################################################ Assembly
asse = cone+sphere

###################################### getActors
print('Test getMeshes')
assert len(asse.unpack()) ==2
assert asse.unpack(0) == cone
assert asse.unpack(1) == sphere

assert 4.1 < asse.diagonalSize() < 4.2


############################################################################ Volume
X, Y, Z = np.mgrid[:30, :30, :30]
scalar_field = ((X-15)**2 + (Y-15)**2 + (Z-15)**2)/225
print('Test Volume, scalar min, max =', np.min(scalar_field), np.max(scalar_field))

vol = Volume(scalar_field)
volarr = vol.getPointArray()

assert volarr.shape[0] == 27000
assert np.max(volarr) == 3
assert np.min(volarr) == 0

###################################### isosurface
print('Test isosurface')
iso = vol.isosurface(threshold=1.0)
print('area', iso.area())
assert 2540 < iso.area() <  3000

#lego = vol.legosurface(vmin=0.3, vmax=0.5)
#show(lego)
#print('lego.N()', lego.N())
#assert 2610 < lego.N() < 2630
