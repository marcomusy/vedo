# -*- coding: utf-8 -*-
#
from __future__ import division, print_function
import os, sys, time, types
import numpy as np
import vtk
from vtk.util import numpy_support
import vtkcolors


##############################################################################
vtkMV = vtk.vtkVersion().GetVTKMajorVersion() > 5
def setInput(vtkobj, p, port=0):
    if isinstance(p, vtk.vtkAlgorithmOutput):
        vtkobj.SetInputConnection(port, p) # passing port
        return    
    if vtkMV: vtkobj.SetInputData(p)
    else: vtkobj.SetInput(p)

def isSequence(arg): 
    if hasattr(arg, "strip"): return False
    if hasattr(arg, "__getslice__"): return True
    if hasattr(arg, "__iter__"): return True
    return False

def arange(start,stop, step=1): 
    return np.arange(start, stop, step)

def vector(x, y ,z=0):
    return np.array([x,y,z])

def mag(z):
    if isinstance(z[0], np.ndarray): 
        return np.array(list(map(np.linalg.norm, z)))
    else: 
        return np.linalg.norm(z)

def mag2(z):
    return np.dot(z,z)

def norm(v):
    if isinstance(v[0], np.ndarray):
        return np.divide(v, mag(v)[:,None])
    else: 
        return v/mag(v)


def makeActor(poly, c='gold', alpha=0.5, 
              wire=False, bc=None, edges=False, legend=None, texture=None):
    '''
    Return a vtkActor from an input vtkPolyData, optional args:
        c,       color in RGB format, hex, symbol or name
        alpha,   transparency (0=invisible)
        wire,    show surface as wireframe
        bc,      backface color of internal surface
        edges,   show edges as line on top of surface
        legend   optional string
        texture  jpg file name of surface texture, eg. 'metalfloor1'
    '''
    clp = vtk.vtkCleanPolyData()
    setInput(clp, poly)
    clp.Update()
    pdnorm = vtk.vtkPolyDataNormals()
    setInput(pdnorm, clp.GetOutput())
    pdnorm.ComputePointNormalsOn()
    pdnorm.ComputeCellNormalsOn()
    pdnorm.FlipNormalsOff()
    pdnorm.ConsistencyOn()
    pdnorm.Update()

    mapper = vtk.vtkPolyDataMapper()

    # check if color string contains a float, in this case ignore alpha
    al = vtkcolors.getAlpha(c)
    if al: alpha = al

    setInput(mapper, pdnorm.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prp = actor.GetProperty()
    if vtk.vtkVersion().GetVTKMajorVersion()>7: 
        prp.RenderPointsAsSpheresOn()
    if c is None: 
        mapper.ScalarVisibilityOn()
    else:
        mapper.ScalarVisibilityOff()
        c = vtkcolors.getColor(c)
        prp.SetColor(c)
        prp.SetOpacity(alpha)
    
        prp.SetSpecular(0.1)
        prp.SetSpecularColor(c)
        prp.SetSpecularPower(1)
    
        prp.SetAmbient(0.1)
        prp.SetAmbientColor(c)
    
        prp.SetDiffuse(1)
        prp.SetDiffuseColor(c)

    if edges: prp.EdgeVisibilityOn()
    if wire: prp.SetRepresentationToWireframe()
    if texture: 
        mapper.ScalarVisibilityOff()
        assignTexture(actor, texture)
    if bc: # defines a specific color for the backface
        backProp = vtk.vtkProperty()
        backProp.SetDiffuseColor(vtkcolors.getColor(bc))
        backProp.SetOpacity(alpha)
        actor.SetBackfaceProperty(backProp)

    assignPhysicsMethods(actor)    
    assignConvenienceMethods(actor, legend)    
    return actor


def makeAssembly(actors, legend=None):
    '''Group many actors as a single new actor'''
    assembly = vtk.vtkAssembly()
    for a in actors: assembly.AddPart(a)
    setattr(assembly, 'legend', legend) 
    assignPhysicsMethods(assembly)
    assignConvenienceMethods(assembly, legend)
    if hasattr(actors[0], 'base'):
        setattr(assembly, 'base', actors[0].base)
        setattr(assembly, 'top',  actors[0].top)
    return assembly


def assignTexture(actor, name, scale=1, falsecolors=False, mapTo=1):
    '''Assign a texture to actor from file or name in /textures directory'''
    if   mapTo == 1: tmapper = vtk.vtkTextureMapToCylinder()
    elif mapTo == 2: tmapper = vtk.vtkTextureMapToSphere()
    elif mapTo == 3: tmapper = vtk.vtkTextureMapToPlane()
    
    setInput(tmapper, polydata(actor))
    if mapTo == 1: tmapper.PreventSeamOn()
    
    xform = vtk.vtkTransformTextureCoords()
    xform.SetInputConnection(tmapper.GetOutputPort())
    xform.SetScale(scale,scale,scale)
    if mapTo == 1: xform.FlipSOn()
    xform.Update()
    
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(xform.GetOutputPort())
    
    cdir = os.path.dirname(__file__)
    if cdir == '': cdir = '.'  
    fn = cdir + '/textures/' + name + ".jpg"
    if os.path.exists(name): 
        fn = name
    elif not os.path.exists(fn):
        printc(('Texture', name, 'not found in', cdir+'/textures'), 'r')
        printc('Available textures:', c='m', end=' ')
        for ff in os.listdir(cdir + '/textures'):
            printc(ff.split('.')[0], end=' ', c='m')
        print()
        return 
        
    jpgReader = vtk.vtkJPEGReader()
    jpgReader.SetFileName(fn)
    atext = vtk.vtkTexture()
    atext.RepeatOn()
    atext.EdgeClampOff()
    atext.InterpolateOn()
    if falsecolors: atext.MapColorScalarsThroughLookupTableOn()
    atext.SetInputConnection(jpgReader.GetOutputPort())
    actor.GetProperty().SetColor(1,1,1)
    actor.SetMapper(mapper)
    actor.SetTexture(atext)


# ###########################################################################
def assignConvenienceMethods(actor, legend):
    if not hasattr(actor, 'legend'):
        setattr(actor, 'legend', legend)

    def _frotate(self, angle, axis, axis_point=[0,0,0], rad=False): 
        if rad: angle *= 57.3
        return rotate(self, angle, axis, axis_point, rad)
    actor.rotate = types.MethodType( _frotate, actor )

    def _frotateX(self, angle, axis_point=[0,0,0], rad=False): 
        if rad: angle *= 57.3
        return rotate(self, angle, [1,0,0], axis_point, rad)
    actor.rotateX = types.MethodType( _frotateX, actor )

    def _frotateY(self, angle, axis_point=[0,0,0], rad=False): 
        if rad: angle *= 57.3
        return rotate(self, angle, [0,1,0], axis_point, rad)
    actor.rotateY = types.MethodType( _frotateY, actor )

    def _frotateZ(self, angle, axis_point=[0,0,0], rad=False): 
        if rad: angle *= 57.3
        return rotate(self, angle, [0,0,1], axis_point, rad)
    actor.rotateZ = types.MethodType( _frotateZ, actor )

    def _forientation(self, newaxis=None, rotation=0): 
        return orientation(self, newaxis, rotation)
    actor.orientation = types.MethodType( _forientation, actor )

    def _fclone(self, c=None, alpha=None, wire=False, bc=None,
                edges=False, legend=None, texture=None, rebuild=True): 
        return clone(self, c, alpha, wire, bc, edges, legend, texture, rebuild)
    actor.clone = types.MethodType( _fclone, actor )

    def _fpoint(self, i, p=None): 
        if p is None : 
            poly = polydata(self, True, 0)
            p = [0,0,0]
            poly.GetPoints().GetPoint(i, p)
            return np.array(p)
        else:
            poly = polydata(self, False, 0)
            poly.GetPoints().SetPoint(i, p)
            TI = vtk.vtkTransform()
            actor.SetUserMatrix(TI.GetMatrix()) # reset
        return self
    actor.point = types.MethodType( _fpoint, actor )

    def _fN(self, index=0): 
        return polydata(self, False, index).GetNumberOfPoints()
    actor.N = types.MethodType( _fN, actor )

    def _fnormalize(self): return normalize(self)
    actor.normalize = types.MethodType( _fnormalize, actor )

    def _fshrink(self, fraction=0.85): return shrink(self, fraction)
    actor.shrink = types.MethodType( _fshrink, actor )

    def _fcutterw(self): return cutterWidget(self)
    actor.cutterWidget = types.MethodType( _fcutterw, actor )
     
    def _fpolydata(self, rebuild=True, index=0): 
        return polydata(self, rebuild, index)
    actor.polydata = types.MethodType( _fpolydata, actor )

    def _fcoordinates(self, rebuild=True): 
        return coordinates(self, rebuild)
    actor.coordinates = types.MethodType( _fcoordinates, actor )

    def _fxbounds(self): 
        b = polydata(actor, True).GetBounds()
        return (b[0],b[1])
    actor.xbounds = types.MethodType( _fxbounds, actor )
    def _fybounds(self): 
        b = polydata(actor, True).GetBounds()
        return (b[2],b[3])
    actor.ybounds = types.MethodType( _fybounds, actor )
    def _fzbounds(self): 
        b = polydata(actor, True).GetBounds()
        return (b[4],b[5])
    actor.zbounds = types.MethodType( _fzbounds, actor )


    def _fnormalAt(self, index): 
        normals = polydata(self, True).GetPointData().GetNormals()
        return np.array(normals.GetTuple(index))
    actor.normalAt = types.MethodType( _fnormalAt, actor )

    def _fnormals(self): 
        vtknormals = polydata(self, True).GetPointData().GetNormals()
        as_numpy = numpy_support.vtk_to_numpy(vtknormals)
        return as_numpy
    actor.normals = types.MethodType( _fnormals, actor )

    def _fstretch(self, startpt, endpt): 
        return stretch(self, startpt, endpt)
    actor.stretch = types.MethodType( _fstretch, actor)

    def _fsubdivide(self, N=1, method=0, legend=None): 
        return subdivide(self, N, method, legend)
    actor.subdivide = types.MethodType( _fsubdivide, actor)

    def _fcolor(self, c=None):
        if c: 
            self.GetProperty().SetColor(vtkcolors.getColor(c))
            return self
        else: return np.array(self.GetProperty().GetColor())
    actor.color = types.MethodType( _fcolor, actor)

    def _falpha(self, a=None):
        if a: 
            self.GetProperty().SetOpacity(a)
            return self
        else: return self.GetProperty().GetOpacity()
    actor.alpha = types.MethodType( _falpha, actor)

    def _fclosestPoint(self, pt, N=1, radius=None):
        return closestPoint(self, pt, N, radius)
    actor.closestPoint = types.MethodType( _fclosestPoint, actor)

    def _fintersectWithLine(self, p0, p1):
        return intersectWithLine(self, p0,p1)
    actor.intersectWithLine = types.MethodType(_fintersectWithLine , actor)

    def _fisInside(self, point, tol=0.0001):
        return isInside(self, point, tol)
    actor.isInside = types.MethodType(_fisInside , actor)
   
    def _finsidePoints(self, points, invert=False, tol=1e-05):
        return insidePoints(self, points, invert, tol)
    actor.insidePoints = types.MethodType(_finsidePoints , actor)

    def _fflipNormals(self):
        return flipNormals(self)
    actor.flipNormals = types.MethodType(_fflipNormals , actor)


# ###########################################################################
def assignPhysicsMethods(actor):
    
    def _fpos(self, p=None): 
        if p is None: 
            return np.array(self.GetPosition())
        self.SetPosition(p)
        return self # return itself to concatenate methods
    actor.pos = types.MethodType( _fpos, actor )

    def _faddpos(self, dp): 
        self.SetPosition(np.array(self.GetPosition()) +dp )
        return self
    actor.addpos = types.MethodType( _faddpos, actor )

    def _fpx(self, px=None):               # X  
        _pos = self.GetPosition()
        if px is None: 
            return _pos[0]
        newp = [px, _pos[1], _pos[2]]
        self.SetPosition(newp)
        return self
    actor.x = types.MethodType( _fpx, actor )

    def _fpy(self, py=None):               # Y  
        _pos = self.GetPosition()
        if py is None: 
            return _pos[1]
        newp = [_pos[0], py, _pos[2]]
        self.SetPosition(newp)
        return self
    actor.y = types.MethodType( _fpy, actor )

    def _fpz(self, pz=None):               # Z  
        _pos = self.GetPosition()
        if pz is None: 
            return _pos[2]
        newp = [_pos[0], _pos[1], pz]
        self.SetPosition(newp)
        return self
    actor.z = types.MethodType( _fpz, actor )

    def _fscale(self, p=None): 
        if p is None: 
            return np.array(self.GetScale())
        self.SetScale(p)
        return self # return itself to concatenate methods
    actor.scale = types.MethodType( _fscale, actor )

    def _fcenterOfMass(self): return centerOfMass(self)
    actor.centerOfMass = types.MethodType(_fcenterOfMass, actor)

    def _fvolume(self): return volume(self)
    actor.volume = types.MethodType(_fvolume, actor)

    def _farea(self): return area(self)
    actor.area = types.MethodType(_farea, actor)

######################################################### 
def clone(actor, c=None, alpha=None, wire=False, bc=None,
          edges=False, legend=None, texture=None, rebuild=True):
    '''
    Clone a vtkActor.
        If rebuild is True build its polydata in its current position in space
    '''
    poly = polydata(actor, rebuild)
    if not poly.GetNumberOfPoints():
        printc('Limitation: cannot clone textured obj. Returning input.',1)
        return actor
    polyCopy = vtk.vtkPolyData()
    polyCopy.DeepCopy(poly)

    if legend  is True and hasattr(actor, 'legend'): legend = actor.legend
    if alpha   is None: alpha = actor.GetProperty().GetOpacity()
    if c       is None: c = actor.GetProperty().GetColor()
    if texture is None and hasattr(actor, 'texture'): texture = actor.texture
    return makeActor(polyCopy, c, alpha, wire, bc, edges, legend, texture)


def flipNormals(actor): # N.B. input argument gets modified
    rs = vtk.vtkReverseSense()
    setInput(rs, polydata(actor, True))
    rs.ReverseNormalsOn()
    rs.Update()
    poly = rs.GetOutput()
    mapper = actor.GetMapper()
    setInput(mapper, poly)
    mapper.Update()
    actor.Modified()
    if hasattr(actor, 'poly'): actor.poly=None #clean cache
    return actor  # return same obj for concatenation


def normalize(actor): # N.B. input argument gets modified
    '''
    Shift actor's center of mass at origin and scale its average size to unit.
    '''
    cm = centerOfMass(actor)
    coords = coordinates(actor)
    if not len(coords) : return
    pts = coords - cm
    xyz2 = np.sum(pts * pts, axis=0)
    scale = 1/np.sqrt(np.sum(xyz2)/len(pts))
    t = vtk.vtkTransform()
    t.Scale(scale, scale, scale)
    t.Translate(-cm)
    tf = vtk.vtkTransformPolyDataFilter() 
    setInput(tf, actor.GetMapper().GetInput())
    tf.SetTransform(t)
    tf.Update()
    mapper = actor.GetMapper()
    setInput(mapper, tf.GetOutput())
    mapper.Update()
    actor.Modified()
    if hasattr(actor, 'poly'): actor.poly=None #clean cache
    return actor  # return same obj for concatenation

    
def rotate(actor, angle, axis, axis_point=[0,0,0], rad=False): 
    '''Rotate an actor around an arbitrary axis passing through axis_point'''
    anglerad = angle
    if not rad: anglerad = angle/57.3
    axis = norm(axis)
    a = np.cos(anglerad / 2)
    b, c, d = -axis * np.sin(anglerad / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    R = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    rv = np.dot(R, actor.GetPosition()-np.array(axis_point)) + axis_point
    
    if rad: angle *= 57.3
    # this stupid vtk method only rotates in the origin of the actor:
    actor.RotateWXYZ(angle, axis[0], axis[1], axis[2] )
    actor.SetPosition(rv)
    return actor
 

def orientation(actor, newaxis=None, rotation=0):
    '''
    Set/Get actor orientation.
        If rotation != 0 rotate actor around newaxis (in degree units)
    '''
    initaxis = norm(actor.top - actor.base)
    if newaxis is None: return initaxis
    newaxis = norm(newaxis)
    TI = vtk.vtkTransform()
    actor.SetUserMatrix(TI.GetMatrix()) # reset
    pos = np.array(actor.GetPosition())
    crossvec = np.cross(initaxis, newaxis)
    angle = np.arccos(np.dot(initaxis, newaxis))
    T = vtk.vtkTransform()
    T.PostMultiply()
    T.Translate(-pos)
    if rotation: T.RotateWXYZ(rotation, initaxis)
    T.RotateWXYZ(angle*57.3, crossvec)
    T.Translate(pos)
    actor.SetUserMatrix(T.GetMatrix())
    return actor


############################################################################
def shrink(actor, fraction=0.85):   # N.B. input argument gets modified
    '''Shrink the triangle polydata in the representation of actor'''

    poly = polydata(actor, True)
    shrink = vtk.vtkShrinkPolyData()
    setInput(shrink, poly)
    shrink.SetShrinkFactor(fraction)
    shrink.Update()
    mapper = actor.GetMapper()
    setInput(mapper, shrink.GetOutput())
    mapper.Update()
    actor.Modified()
    return actor   # return same obj for concatenation


def stretch(actor, q1, q2): 
    '''Stretch actor between points q1 and q2'''

    if not hasattr(actor, 'base'):
        printc('Please define vectors actor.base and actor.top at creation. Exit.','r')
        exit(0)

    TI = vtk.vtkTransform()
    actor.SetUserMatrix(TI.GetMatrix()) # reset

    p1, p2 = actor.base, actor.top
    q1,q2,z = np.array(q1), np.array(q2), np.array([0,0,1])
    plength = np.linalg.norm(p2-p1)
    qlength = np.linalg.norm(q2-q1)
    T = vtk.vtkTransform()
    T.PostMultiply()
    T.Translate(-p1)
    cosa = np.dot(p2-p1, z)/plength
    n  = np.cross(p2-p1, z)
    T.RotateWXYZ(np.arccos(cosa)*57.3, n)
    
    T.Scale(1,1, qlength/plength)

    cosa = np.dot(q2-q1, z)/qlength
    n  = np.cross(q2-q1, z)
    T.RotateWXYZ(-np.arccos(cosa)*57.3, n)
    T.Translate(q1)
    
    actor.SetUserMatrix(T.GetMatrix())
    return actor


def decimate(actor, fraction=0.5, N=None, verbose=True, boundaries=True):
    '''
    Downsample the number of vertices in a mesh.
        fraction gives the desired target of reduction. 
        E.g. fraction=0.1
             leaves 10% of the original nr of vertices.
    '''
    poly = polydata(actor)
    if N: # N = desired number of points
        Np = poly.GetNumberOfPoints()
        fraction = float(N)/Np
        if fraction >= 1: return actor   
        
    decimate = vtk.vtkDecimatePro()
    setInput(decimate, poly)
    decimate.SetTargetReduction(1.-fraction)
    decimate.PreserveTopologyOff()
    if boundaries: decimate.BoundaryVertexDeletionOn()
    else: decimate.BoundaryVertexDeletionOff()
    decimate.Update()
    if verbose:
        print ('Input nr. of pts:',poly.GetNumberOfPoints(),end='')
        print (' output:',decimate.GetOutput().GetNumberOfPoints())
    mapper = actor.GetMapper()
    setInput(mapper, decimate.GetOutput())
    mapper.Update()
    actor.Modified()
    if hasattr(actor, 'poly'): actor.poly=None #clean cache
    return actor  # return same obj for concatenation


def booleanOperation(actor1, actor2, operation='plus', c=None, alpha=1, 
                     wire=False, bc=None, edges=False, legend=None, texture=None):
    '''Volumetric union, intersection and subtraction of surfaces'''
    try:
        bf = vtk.vtkBooleanOperationPolyDataFilter()
    except AttributeError:
        printc('Boolean operation only possible for vtk version >= 8','r')
        return None
    poly1 = polydata(actor1, True)
    poly2 = polydata(actor2, True)
    if operation.lower() == 'plus':
        bf.SetOperationToUnion()
    elif operation.lower() == 'intersect':
        bf.SetOperationToIntersection()
    elif operation.lower() == 'minus':
        bf.SetOperationToDifference()
        bf.ReorientDifferenceCellsOn()
    if vtkMV:
        bf.SetInputData(0, poly1)
        bf.SetInputData(1, poly2)
    else:
        bf.SetInputConnection(0, poly1.GetProducerPort())
        bf.SetInputConnection(1, poly2.GetProducerPort())
    bf.Update()
    actor = makeActor(bf.GetOutput(), 
                      c, alpha, wire, bc, edges, legend, texture)
    return actor


def surfaceIntersection(actor1, actor2, tol=1e-06, lw=3,
                        c=None, alpha=1, legend=None):
    '''Intersect 2 surfaces and return a line actor'''
    try:
        bf = vtk.vtkIntersectionPolyDataFilter()
    except AttributeError:
        printc('surfaceIntersection only possible for vtk version > 6','r')
        return None
    poly1 = polydata(actor1, True)
    poly2 = polydata(actor2, True)
    bf.SetInputData(0, poly1)
    bf.SetInputData(1, poly2)
    bf.Update()
    if c is None: c = actor1.GetProperty().GetColor()
    actor = makeActor(bf.GetOutput(), c, alpha, 0, legend=legend)
    actor.GetProperty().SetLineWidth(lw)
    return actor


#########################################################
# Useful Functions
######################################################### 
def makePolyData(spoints, addLines=True):
    """Try to workout a polydata from ordered points"""
    sourcePoints = vtk.vtkPoints()
    sourceVertices = vtk.vtkCellArray()
    for pt in spoints:
        if len(pt)==3: #it's 3D!
            aid = sourcePoints.InsertNextPoint(pt[0], pt[1], pt[2])
        else:
            aid = sourcePoints.InsertNextPoint(pt[0], pt[1], 0)
        sourceVertices.InsertNextCell(1)
        sourceVertices.InsertCellPoint(aid)
    source = vtk.vtkPolyData()
    source.SetPoints(sourcePoints)
    source.SetVerts(sourceVertices)
    if addLines:
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(spoints))
        for i in range(len(spoints)): lines.InsertCellPoint(i)
        source.SetLines(lines)
    return source


def isInside(actor, point, tol=0.0001):
    """Return True if point is inside a polydata closed surface"""
    poly = polydata(actor, True)
    points = vtk.vtkPoints()
    points.InsertNextPoint(point)
    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)
    sep = vtk.vtkSelectEnclosedPoints()
    sep.SetTolerance(tol)
    sep.CheckSurfaceOff()
    setInput(sep, pointsPolydata)
    if vtkMV: sep.SetSurfaceData(poly)
    else: sep.SetSurface(poly)
    sep.Update()
    return sep.IsInside(0)


def insidePoints(actor, points, invert=False, tol=1e-05):
    """Return list of points that are inside a polydata closed surface"""
    poly = polydata(actor, True)
    # check if the stl file is closed
    featureEdge = vtk.vtkFeatureEdges()
    featureEdge.FeatureEdgesOff()
    featureEdge.BoundaryEdgesOn()
    featureEdge.NonManifoldEdgesOn()
    setInput(featureEdge, poly)
    featureEdge.Update()
    openEdges = featureEdge.GetOutput().GetNumberOfCells()
    if openEdges != 0:
        printc("Warning: polydata is not a closed surface",5)
    
    vpoints = vtk.vtkPoints()
    for p in points: vpoints.InsertNextPoint(p)
    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(vpoints)
    sep = vtk.vtkSelectEnclosedPoints()
    sep.SetTolerance(tol) 
    setInput(sep, pointsPolydata)
    if vtkMV: sep.SetSurfaceData(poly)
    else: sep.SetSurface(poly)
    sep.Update()
    
    mask1, mask2 = [], []
    for i,p in enumerate(points):
        if sep.IsInside(i) :
            mask1.append(p)
        else:
            mask2.append(p)
    if invert: 
        return mask2
    else:
        return mask1


def fillHoles(actor, size=None, legend=None): # not tested properly
    fh = vtk.vtkFillHolesFilter()
    if not size:
        mb = maxBoundSize(actor)
        size = mb/20
    fh.SetHoleSize(size)
    poly = polydata(actor)
    setInput(fh, poly)
    fh.Update()
    fpoly = fh.GetOutput()
    factor = makeActor(fpoly, legend=legend)
    factor.SetProperty(actor.GetProperty())
    return factor    


def cellCenters(actor):
    '''Get the list of cell centers of the mesh surface'''
    vcen = vtk.vtkCellCenters()
    setInput(vcen, polydata(actor, True))
    vcen.Update()
    return coordinates(vcen.GetOutput())


def isIdentity(M, tol=1e-06):
    '''Check if vtkMatrix4x4 is Identity'''
    for i in [0,1,2,3]: 
        for j in [0,1,2,3]: 
            e = M.GetElement(i,j)
            if i==j: 
                if np.abs(e-1) > tol: return False
            elif np.abs(e) > tol: return False
    return True


#################################################################### get stuff
def polydata(obj, rebuild=True, index=0): 
    '''
    Returns the vtkPolyData of a vtkActor or vtkAssembly.
        If rebuild=True returns a copy of polydata
        that corresponds to the current actor's position in space.
        If a vtkAssembly is passed, return the polydata of component index.
    '''
   
    if isinstance(obj, vtk.vtkActor):   
        if not rebuild: 
            if hasattr(obj, 'poly') :
                if obj.poly: return obj.poly
            else: 
                setattr(obj, 'poly', None)
            obj.poly = obj.GetMapper().GetInput() #cache it for speed
            return obj.poly
        M = obj.GetMatrix()
        if isIdentity(M):
            if hasattr(obj, 'poly') :
                if obj.poly: return obj.poly
            else: 
                setattr(obj, 'poly', None)
            obj.poly = obj.GetMapper().GetInput() #cache it for speed
            return obj.poly
        # if identity return the original polydata
        # otherwise make a copy that corresponds to 
        # the actual position in space of the actor
        transform = vtk.vtkTransform()
        transform.SetMatrix(M)
        tp = vtk.vtkTransformPolyDataFilter()
        tp.SetTransform(transform)
        if vtkMV: tp.SetInputData(obj.GetMapper().GetInput())
        else: tp.SetInput(obj.GetMapper().GetInput())
        tp.Update()
        return tp.GetOutput()

    elif isinstance(obj, vtk.vtkAssembly):
        cl = vtk.vtkPropCollection()
        obj.GetActors(cl)
        cl.InitTraversal()
        for i in range(index+1):
            act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
        pd = act.GetMapper().GetInput() #not optimized
        if not rebuild: return pd 
        M = act.GetMatrix()
        if isIdentity(M): return pd 
        # if identity return the original polydata
        # otherwise make a copy that corresponds to 
        # the actual position in space of the actor
        transform = vtk.vtkTransform()
        transform.SetMatrix(M)
        tp = vtk.vtkTransformPolyDataFilter()
        tp.SetTransform(transform)
        if vtkMV: tp.SetInputData(pd)
        else: tp.SetInput(pd)
        tp.Update()
        return tp.GetOutput()
    
    elif isinstance(obj, vtk.vtkPolyData): return obj
    elif isinstance(obj, vtk.vtkActor2D):  return obj.GetMapper().GetInput()
    elif isinstance(obj, vtk.vtkImageActor):  return obj.GetMapper().GetInput()
    elif obj is None: return None
    
    printc("Fatal Error in polydata(): ", 'r', end='')
    printc(("input is neither a vtkActor nor vtkAssembly.", [obj]), 'r')
    exit(1)


def subdivide(actor, N=1, method=0, legend=None):
    '''
    Increase the number of points in actor surface
        N = number of subdivisions
        method = 0, Loop
        method = 1, Linear
        method = 2, Adaptive
        method = 3, Butterfly
    '''        
    triangles = vtk.vtkTriangleFilter()
    setInput(triangles, polydata(actor))
    triangles.Update()
    originalMesh = triangles.GetOutput()
    if   method==0: sdf = vtk.vtkLoopSubdivisionFilter()
    elif method==1: sdf = vtk.vtkLinearSubdivisionFilter()
    elif method==2: sdf = vtk.vtkAdaptiveSubdivisionFilter()
    elif method==3: sdf = vtk.vtkButterflySubdivisionFilter()
    else:
        printc('Error in subdivide: unknown method.', 'r')
        exit(1)
    if method != 2: sdf.SetNumberOfSubdivisions(N)
    setInput(sdf, originalMesh)
    sdf.Update()
    out = sdf.GetOutput()
    sactor = makeActor(out, legend=legend)
    sactor.GetProperty().SetOpacity(actor.GetProperty().GetOpacity())
    sactor.GetProperty().SetColor(actor.GetProperty().GetColor())
    sactor.GetProperty().SetRepresentation(actor.GetProperty().GetRepresentation())
    return sactor


def coordinates(actor, rebuild=True):
    """Return a merged list of coordinates of actors or polys"""
    pts = []
    poly = polydata(actor, rebuild)
    for j in range(poly.GetNumberOfPoints()):
        p = [0, 0, 0]
        poly.GetPoint(j, p)
        pts.append(p)
    return np.array(pts)


def xbounds(actor):
    '''Get the the actor bounding [xmin,xmax] '''
    b = polydata(actor, True).GetBounds()
    return (b[0],b[1])

def ybounds(actor):
    '''Get the the actor bounding [ymin,ymax] '''
    b = polydata(actor, True).GetBounds()
    return (b[2],b[3])

def zbounds(actor):
    '''Get the the actor bounding [zmin,zmax] '''
    b = polydata(actor, True).GetBounds()
    return (b[4],b[5])

def centerOfMass(actor):
    '''Get the Center of Mass of the actor'''
    if vtkMV: #faster
        cmf = vtk.vtkCenterOfMass()
        setInput(cmf, polydata(actor, True))
        #cmf.UseScalarsAsWeightsOff()
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)
    else:
        pts = coordinates(actor, True)
        if not len(pts): return np.array([0,0,0])
        return np.mean(pts, axis=0)       

def volume(actor):
    '''Get the volume occupied by actor'''
    mass = vtk.vtkMassProperties()
    setInput(mass, polydata(actor))
    mass.Update() 
    return mass.GetVolume()

def area(actor):
    '''Get the surface area of actor'''
    mass = vtk.vtkMassProperties()
    setInput(mass, polydata(actor))
    mass.Update() 
    return mass.GetSurfaceArea()

def averageSize(actor):
    cm = centerOfMass(actor)
    coords = coordinates(actor, True)
    if not len(coords) : return
    pts = coords - cm
    xyz2 = np.sum(pts * pts, axis=0)
    return np.sqrt(np.sum(xyz2)/len(pts))

def diagonalSize(actor):
    '''Get the length of the diagonal of actor bounding box'''
    b = polydata(actor).GetBounds()
    return np.sqrt((b[1]-b[0])**2, (b[3]-b[2])**2, (b[5]-b[4])**2)

def maxBoundSize(actor):
    '''Get the maximum dimension in x, y or z of the actor bounding box'''
    b = polydata(actor, True).GetBounds()
    return max(abs(b[1]-b[0]), abs(b[3]-b[2]), abs(b[5]-b[4]))

    
def write(obj, fileoutput):
    '''
    Write 3D object to file.
    Possile extensions are: .vtk, .ply, .obj, .stl, .byu, .vtp
    '''
    fr = fileoutput.lower()
    if   '.vtk' in fr: w = vtk.vtkPolyDataWriter()
    elif '.ply' in fr: w = vtk.vtkPLYWriter()
    elif '.obj' in fr: 
        w = vtk.vtkOBJExporter()
        w.SetFilePrefix(fileoutput.replace('.obj',''))
        printc('Please use write(vp.renderWin)',3)
        w.SetInput(obj)
        w.Update()
        printc("Saved file: "+fileoutput, 'g')
        return
    elif '.stl' in fr: w = vtk.vtkSTLWriter()
    elif '.byu' in fr or '.g' in fr: w = vtk.vtkBYUWriter()
    elif '.vtp' in fr: w = vtk.vtkXMLPolyDataWriter()
    else:
        printc('Unavailable format in file '+fileoutput, c='r')
        exit(1)
    try:
        setInput(w, polydata(obj, True))
        w.SetFileName(fileoutput)
        w.Write()
        printc("Saved file: "+fileoutput, 'g')
    except:
        printc("Error saving: "+fileoutput, 'r')


########################################################################
def closestPoint(surf, pt, N=1, radius=None):
    """
    Find the closest point on a polydata given an other point.
        If N>1, return a list of N ordered closest points.
        If radius is given, get all points within.
    """
    poly = polydata(surf, True)
    trgp  = [0,0,0]
    cid   = vtk.mutable(0)
    dist2 = vtk.mutable(0)

    locexists = hasattr(surf, 'pointlocator')
    if not locexists or (locexists and surf.pointlocator is None):
        if N>1: pointlocator = vtk.vtkPointLocator()
        else: pointlocator = vtk.vtkCellLocator()
        pointlocator.SetDataSet(poly)
        pointlocator.BuildLocator()
        setattr(surf, 'pointlocator', pointlocator)
    if N>1:
        vtklist = vtk.vtkIdList()
        vmath = vtk.vtkMath()
        surf.pointlocator.FindClosestNPoints(N, pt, vtklist)
        trgp, dists2  = [], []
        for i in range(vtklist.GetNumberOfIds()):
            trgp_ = [0,0,0]
            vi = vtklist.GetId(i)
            poly.GetPoints().GetPoint(vi, trgp_ )
            trgp.append( trgp_ )
            dists2.append(vmath.Distance2BetweenPoints(trgp_, pt))
        dist2 = dists2
    elif radius:
        cell = vtk.mutable(0)
        r = surf.pointlocator.FindClosestPointWithinRadius(pt, radius, trgp, cell, cid, dist2)
        if not r: 
            trgp = pt
            dist2 = 0.0
    else: 
        subid = vtk.mutable(0)
        surf.pointlocator.FindClosestPoint(pt, trgp, cid, subid, dist2)
    return np.array(trgp)


def intersectWithLine(act, p0, p1):
    '''Return a list of points between p0 and p1 intersecting the actor'''
    if not hasattr(act, 'linelocator'):
        linelocator = vtk.vtkOBBTree()
        linelocator.SetDataSet(polydata(act, True))
        linelocator.BuildLocator()
        setattr(act, 'linelocator', linelocator)

    intersectPoints = vtk.vtkPoints()
    intersection = [0, 0, 0]
    act.linelocator.IntersectWithLine(p0, p1, intersectPoints, None)
    pts=[]
    for i in range(intersectPoints.GetNumberOfPoints()):
        intersectPoints.GetPoint(i, intersection)
        pts.append(list(intersection))
    return pts


def cutterWidget(obj, outputname='clipped.vtk', c=(0.2, 0.2, 1), alpha=1,
                 bc=(0.7, 0.8, 1), legend=None):
    '''Pop up a box widget to cut parts of actor. Return largest part.'''

    apd = polydata(obj)
    
    planes = vtk.vtkPlanes()
    planes.SetBounds(apd.GetBounds())

    clipper = vtk.vtkClipPolyData()
    setInput(clipper, apd)
    clipper.SetClipFunction(planes)
    clipper.InsideOutOn()
    clipper.GenerateClippedOutputOn()

    # check if color string contains a float, in this case ignore alpha
    al = vtkcolors.getAlpha(c)
    if al: alpha = al

    act0Mapper = vtk.vtkPolyDataMapper() # the part which stays
    act0Mapper.SetInputConnection(clipper.GetOutputPort())
    act0 = vtk.vtkActor()
    act0.SetMapper(act0Mapper)
    act0.GetProperty().SetColor(vtkcolors.getColor(c))
    act0.GetProperty().SetOpacity(alpha)
    backProp = vtk.vtkProperty()
    backProp.SetDiffuseColor(vtkcolors.getColor(bc))
    backProp.SetOpacity(alpha)
    act0.SetBackfaceProperty(backProp)
    #act0 = makeActor(clipper.GetOutputPort())
    
    act0.GetProperty().SetInterpolationToFlat()
    assignPhysicsMethods(act0)    
    assignConvenienceMethods(act0, legend)    

    act1Mapper = vtk.vtkPolyDataMapper() # the part which is cut away
    act1Mapper.SetInputConnection(clipper.GetClippedOutputPort())
    act1 = vtk.vtkActor()
    act1.SetMapper(act1Mapper)
    act1.GetProperty().SetColor(vtkcolors.getColor(c))
    act1.GetProperty().SetOpacity(alpha/10.)
    act1.GetProperty().SetRepresentationToWireframe()
    act1.VisibilityOn()
    
    ren = vtk.vtkRenderer()
    ren.SetBackground(1,1,1)
    
    ren.AddActor(act0)
    ren.AddActor(act1)
    
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600, 700)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    istyl = vtk.vtkInteractorStyleSwitch()
    istyl.SetCurrentStyleToTrackballCamera()
    iren.SetInteractorStyle(istyl)
    
    def SelectPolygons(vobj, event): vobj.GetPlanes(planes)
    
    boxWidget = vtk.vtkBoxWidget()
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1,0,1)
    boxWidget.GetOutlineProperty().SetColor(0.1,0.1,0.1)
    boxWidget.GetOutlineProperty().SetOpacity(0.8)
    boxWidget.SetPlaceFactor(1.05)
    boxWidget.SetInteractor(iren)
    setInput(boxWidget, apd)
    boxWidget.PlaceWidget()    
    boxWidget.AddObserver("InteractionEvent", SelectPolygons)
    boxWidget.On()
    
    printc('\nCutterWidget:\n Move handles to cut parts of the actor','m')
    printc(' Press q to continue, Escape to exit','m')
    printc((" Press X to save file to", outputname), 'm')
    def cwkeypress(obj, event):
        key = obj.GetKeySym()
        if   key == "q" or key == "space" or key == "Return":
            iren.ExitCallback()
        elif key == "X": 
            confilter = vtk.vtkPolyDataConnectivityFilter()
            setInput(confilter, clipper.GetOutput())
            confilter.SetExtractionModeToLargestRegion()
            confilter.Update()
            cpd = vtk.vtkCleanPolyData()
            setInput(cpd, confilter.GetOutput())
            cpd.Update()
            write(cpd.GetOutput(), outputname)
        elif key == "Escape": 
            exit(0)
    
    iren.Initialize()
    iren.AddObserver("KeyPressEvent", cwkeypress)
    iren.Start()
    boxWidget.Off()

    return act0


def reconstructSurface(points, neighbors=20, spacing=None,
                       c='gold', alpha=0.5, wire=False, bc=None, edges=False, legend=None):
    
    def doIterativeClosestPoint(source, target, rigid=False, locator=None, iters=20):

        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(source)
        icp.SetTarget(target)
        if locator: icp.SetLocator(locator)
        if rigid: icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(iters)
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()
        icpTransformFilter = vtk.vtkTransformPolyDataFilter()
        icpTransformFilter.SetInput(source)
        icpTransformFilter.SetTransform(icp)
        icpTransformFilter.Update()
        return icpTransformFilter.GetOutput()

    pointSource = vtk.vtkPointSource()
    pointSource.SetNumberOfPoints(len(points))
    pointSource.Update()
    for i,p in enumerate(points): 
        pointSource.GetOutput().GetPoints().SetPoint(i, p)

    surf = vtk.vtkSurfaceReconstructionFilter()
    if spacing: surf.SetSampleSpacing(spacing)
    if neighbors: surf.SetNeighborhoodSize(neighbors)
    surf.SetInputConnection(pointSource.GetOutputPort())    
    cf = vtk.vtkContourFilter()
    cf.SetInputConnection(surf.GetOutputPort())
    cf.SetValue(0, 0.0)
    
    reverse = vtk.vtkReverseSense()
    reverse.SetInputConnection(cf.GetOutputPort())
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()

    pt = pointSource.GetOutput().GetPoints()
    pt_bounds = pt.GetBounds()
    pd_bounds = reverse.GetOutput().GetBounds()
    scale = (pt_bounds[1] - pt_bounds[0])/(pd_bounds[1] - pd_bounds[0])
    
    transp = vtk.vtkTransform()
    transp.Translate(pt_bounds[0], pt_bounds[2], pt_bounds[4])
    transp.Scale(scale, scale, scale)
    transp.Translate(-pd_bounds[0], -pd_bounds[2], -pd_bounds[4])
    tpd = vtk.vtkTransformPolyDataFilter()
    tpd.SetInput(reverse.GetOutput())
    tpd.SetTransform(transp)
    tpd.Update()
    fpd = doIterativeClosestPoint(tpd.GetOutput(), pointSource.GetOutput())
    actor = makeActor(fpd, c, alpha, wire, bc, edges, legend)
    return actor


###########################################################################
class ProgressBar: 
    '''Class to print a progress bar with optional text on its right
    
    Usage example:
        import time                        
        pb = ProgressBar(0,400, c='red')
        for i in pb.range():
            time.sleep(.1)
            pb.print('some message') # or pb.print(counts=i)
    ''' 

    def __init__(self, start, stop, step=1, c=None, ETA=True, width=25):
        self.start  = start
        self.stop   = stop
        self.step   = step
        self.color  = c
        self.width  = width
        self.bar    = ""  
        self.percent= 0
        self.clock0 = 0
        self.ETA    = ETA
        self.clock0 = time.time()
        self._update(0)
        self._counts= 0
        self._oldbar= ""
        self._lentxt= 0
        self._range = arange(start, stop, step)
        self._len   = len(self._range)
        
    def print(self, txt='', counts=None):
        if counts: self._update(counts)
        else:      self._update(self._counts + self.step)
        if self.bar != self._oldbar:
            self._oldbar = self.bar
            eraser = [' ']*self._lentxt + ['\b']*self._lentxt 
            eraser = ''.join(eraser)
            if self.ETA:
                vel  = self._counts/(time.time() - self.clock0)
                remt =  (self.stop-self._counts)/vel
                if remt>60:
                    mins = int(remt/60)
                    secs = remt - 60*mins
                    mins = str(mins)+'m'
                    secs = str(int(secs))+'s '
                else:
                    mins = ''
                    secs= str(int(remt))+'s '
                vel = str(round(vel,1))
                eta = 'ETA: '+mins+secs+'('+vel+' it/s) '
            else: eta = ''
            txt = eta + str(txt) 
            s = self.bar + ' ' + eraser + txt + '\r'
            if self.color: 
                printc(s, c=self.color, end='')
            else: 
                sys.stdout.write(s)
                sys.stdout.flush()
            if self.percent==100: print ('')
            self._lentxt = len(txt)

    def range(self): return self._range
    def len(self): return self._len
 
    def _update(self, counts):
        if counts < self.start: counts = self.start
        elif counts > self.stop: counts = self.stop
        self._counts = counts
        self.percent = (self._counts - self.start)*100
        self.percent /= self.stop - self.start
        self.percent = int(round(self.percent))
        af = self.width - 2
        nh = int(round( self.percent/100 * af ))
        if   nh==0:  self.bar = "[>%s]" % (' '*(af-1))
        elif nh==af: self.bar = "[%s]" % ('='*af)
        else:        self.bar = "[%s>%s]" % ('='*(nh-1), ' '*(af-nh))
        ps = str(self.percent) + "%"
        self.bar = ' '.join([self.bar, ps])
        

################################################################### color print
def printc(strings, c='black', bold=True, separator=' ', end='\n'):
    '''
    Print to terminal in color. 
    
    Available colors:
        black, red, green, yellow, blue, magenta, cyan, white
    Usage example:
        cprint( 'anything', c='red', bold=False, end='' )
        cprint( ['anything', 455.5, vtkObject], 'green')
        cprint(299792.48, c=4) # 4 is blue
    '''
    if isinstance(strings, tuple): strings = list(strings)
    elif not isinstance(strings, list): strings = [str(strings)]
    txt = str()
    for i,s in enumerate(strings):
        if i == len(strings)-1: separator=''
        txt = txt + str(s) + separator
    
    if _terminal_has_colors:
        try:
            if isinstance(c, int): 
                ncol = c % 8
            else: 
                cols = {'black':0, 'red':1, 'green':2, 'yellow':3, 
                        'blue':4, 'magenta':5, 'cyan':6, 'white':7,
                        'k':0, 'r':1, 'g':2, 'y':3,
                        'b':4, 'm':5, 'c':6, 'w':7}
                ncol = cols[c.lower()]
            if bold: seq = "\x1b[1;%dm" % (30+ncol)
            else:    seq = "\x1b[0;%dm" % (30+ncol)
            sys.stdout.write(seq + txt + "\x1b[0m" +end)
            sys.stdout.flush()
        except: print (txt, end=end)
    else:
        print (txt, end=end)
        
def _has_colors(stream):
    if not hasattr(stream, "isatty"): return False
    if not stream.isatty(): return False # auto color only on TTYs
    try:
        import curses
        curses.setupterm()
        return curses.tigetnum("colors") > 2
    except:
        return False
_terminal_has_colors = _has_colors(sys.stdout)

