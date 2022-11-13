from vedo import Cone, Sphere, merge, Volume, dataurl, utils
import numpy as np
import vtk

print('---------------------------------')
print('vtkVersion', vtk.vtkVersion().GetVTKVersion())
print('---------------------------------')


#####################################
cone = Cone(res=48)
sphere = Sphere(res=24)

carr = cone.cell_centers()[:, 2]
parr = cone.points()[:, 0]

cone.pointdata["parr"] = parr
cone.celldata["carr"] = carr

carr = sphere.cell_centers()[:, 2]
parr = sphere.points()[:, 0]

sphere.pointdata["parr"] = parr
sphere.celldata["carr"] = carr

sphere.pointdata["pvectors"] = np.sin(sphere.points())

sphere.compute_elevation()

cone.compute_normals()
sphere.compute_normals()


###################################### test clone()
c2 = cone.clone()
print('clone()', cone.npoints, c2.npoints)
assert cone.npoints == c2.npoints
print('clone()', cone.ncells, c2.ncells)
assert cone.ncells == c2.ncells


###################################### test merge()
m = merge(sphere, cone)
print('merge()', m.npoints, cone.npoints + sphere.npoints)
assert m.npoints == cone.npoints + sphere.npoints
print('merge()', m.ncells, cone.ncells + sphere.ncells)
assert m.ncells == cone.ncells + sphere.ncells


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


###################################### shift
cone.pos(5,6,7).shift(3,0,0)
print('shift',[8,6,7], cone.pos())
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
print('box',bx.npoints, 24)
assert bx.npoints == 24
print('box',bx.clean().npoints , 8)
assert bx.clean().npoints == 8

###################################### get_transform
ct = cone.clone().rotate_x(10).rotate_y(10).rotate_z(10)
print('get_transform', [ct.get_transform()], [vtk.vtkTransform])
assert isinstance(ct.get_transform(), vtk.vtkTransform)
ct.apply_transform(ct.get_transform())
print('get_transform',ct.get_transform().GetNumberOfConcatenatedTransforms())
assert ct.get_transform().GetNumberOfConcatenatedTransforms()


###################################### pointdata and celldata
arrnames = cone.pointdata.keys()
print('pointdata', arrnames, 'parr')
assert 'parr' in arrnames
arrnames = cone.celldata.keys()
print('celldata.keys', arrnames, 'carr')
assert 'carr' in arrnames


###################################### Get Point Data
arr = sphere.pointdata['parr']
print('pointdata',len(arr))
assert len(arr)
print('pointdata',np.max(arr) ,'>', .99)
assert np.max(arr) > .99

arr = sphere.celldata['carr']
print('celldata',[arr])
assert len(arr)
print('celldata',np.max(arr) ,'>', .99)
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
st = sphere.clone().texture(dataurl+'textures/wood2.jpg')
print('texture test')
assert isinstance(st.GetTexture(), vtk.vtkTexture)


###################################### delete_cells_by_point_index
sd = sphere.clone().delete_cells_by_point_index(range(100))
print('delete_cells_by_point_index',sd.npoints , sphere.npoints)
assert sd.npoints == sphere.npoints
print('delete_cells_by_point_index',sd.ncells ,'<', sphere.ncells)
assert sd.ncells < sphere.ncells


###################################### reverse
# this fails on some archs (see issue #185)
# lets comment it out temporarily
sr = sphere.clone().reverse().cut_with_plane()
print('DISABLED: reverse test', sr.npoints, 576)
rev = vtk.vtkReverseSense()
rev.SetInputData(sr.polydata())
rev.Update()
print('DISABLED: reverse vtk nr.pts, nr.cells')
print(rev.GetOutput().GetNumberOfPoints(),sr.polydata().GetNumberOfPoints(),
      rev.GetOutput().GetNumberOfCells(), sr.polydata().GetNumberOfCells())
# assert sr.npoints == 576


###################################### quantize
sq = sphere.clone().quantize(0.1)
print('quantize',sq.npoints , 834)
assert sq.npoints == 834


###################################### bounds
ss = sphere.clone().scale([1,2,3])
print('bounds',ss.xbounds())
assert np.allclose(ss.xbounds(), [-1,1], atol=0.01)
print('bounds',ss.ybounds())
assert np.allclose(ss.ybounds(), [-2,2], atol=0.01)
print('bounds',ss.zbounds())
assert np.allclose(ss.zbounds(), [-3,3], atol=0.01)


###################################### average_size
print('average_size', Sphere().scale(10).pos(1,3,7).average_size())
assert 9.9 < Sphere().scale(10).pos(1,3,7).average_size() < 10.1

print('diagonal_size',sphere.diagonal_size())
assert 3.3 < sphere.diagonal_size() < 3.5

print('center_of_mass',sphere.center_of_mass())
assert np.allclose(sphere.center_of_mass(), [0,0,0])

print('volume',sphere.volume())
assert 4.1 < sphere.volume() < 4.2

print('area',sphere.area())
assert 12.5 < sphere.area() < 12.6


###################################### closest_point
pt = [12,34,52]
print('closest_point',sphere.closest_point(pt), [0.19883616, 0.48003298, 0.85441941])
assert np.allclose(sphere.closest_point(pt),
                   [0.19883616, 0.48003298, 0.85441941])


###################################### findCellsWithin
ics = sphere.find_cells_in(xbounds=(-0.5, 0.5))
print('findCellsWithin',len(ics) , 1404)
assert len(ics) == 1404


######################################transformMesh
T = cone.clone().pos(35,67,87).get_transform()
s3 = sphere.clone().apply_transform(T)
print('transformMesh',s3.center_of_mass(), (35,67,87))
assert np.allclose(s3.center_of_mass(), (35,67,87))


######################################normalize
s3 = sphere.clone().pos(10,20,30).scale([7,8,9]).normalize()
print('normalize',s3.center_of_mass(), (10,20,30))
assert np.allclose(s3.center_of_mass(), (10,20,30))
print('normalize',s3.average_size())
assert 0.9 < s3.average_size() < 1.1


###################################### crop
c2 = cone.clone().crop(left=0.5)
print('crop',np.min(c2.points()[:,0]), '>', -0.001)
assert np.min(c2.points()[:,0]) > -0.001


###################################### subdivide
s2 = sphere.clone().subdivide(4)
print('subdivide',s2.npoints , 270338)
assert s2.npoints == 270338


###################################### decimate
s2 = sphere.clone().decimate(0.2)
print('decimate',s2.npoints , 213)
assert s2.npoints == 213

###################################### normal_at
print('normal_at',sphere.normal_at(12), [9.97668684e-01, 1.01513637e-04, 6.82437494e-02])
assert np.allclose(sphere.normal_at(12), [9.97668684e-01, 1.01513637e-04, 6.82437494e-02])

###################################### isInside
print('isInside',)
assert sphere.is_inside([0.1,0.2,0.3])

###################################### intersectWithLine (fails vtk7..)
# pts = sphere.intersectWithLine([-2,-2,-2], [2,3,4])
# print('intersectWithLine',pts[0])
# assert np.allclose(pts[0], [-0.8179885149002075, -0.522485613822937, -0.2269827425479889])
# print('intersectWithLine',pts[1])
# assert np.allclose(pts[1], [-0.06572723388671875, 0.41784095764160156, 0.9014091491699219])


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
print('unpack',asse.diagonal_size(), 4.15)
assert 4.1 < asse.diagonal_size() < 4.2


############################################################################ Volume
X, Y, Z = np.mgrid[:30, :30, :30]
scalar_field = ((X-15)**2 + (Y-15)**2 + (Z-15)**2)/225
print('Test Volume, scalar min, max =', np.min(scalar_field), np.max(scalar_field))

vol = Volume(scalar_field)
volarr = vol.pointdata[0]

print('Volume',volarr.shape[0] , 27000)
assert volarr.shape[0] == 27000
print('Volume',np.max(volarr) , 3)
assert np.max(volarr) == 3
print('Volume',np.min(volarr) , 0)
assert np.min(volarr) == 0

###################################### isosurface
iso = vol.isosurface(1.0)
print('isosurface', iso.area())
assert 2540 < iso.area() <  3000


###################################### utils change of coords
q = [5,2,3]
q = utils.cart2spher(*q)
q = utils.spher2cart(*q)
print("cart2spher spher2cart", q)
assert np.allclose(q, [5,2,3])
q = utils.cart2cyl(*q)
q = utils.cyl2cart(*q)
print("cart2cyl cyl2cart", q)
assert np.allclose(q, [5,2,3])
q = utils.cart2cyl(*q)
q = utils.cyl2spher(*q)
q = utils.spher2cart(*q)
print("cart2cyl cyl2spher spher2cart", q)
assert np.allclose(q, [5,2,3])
q = utils.cart2spher(*q)
q = utils.spher2cyl(*q)
q = utils.cyl2cart(*q)
print("cart2spher spher2cyl cyl2cart", q)
assert np.allclose(q, [5,2,3])

######################################
print("OK with test_actors")

