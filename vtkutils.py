from __future__ import division, print_function
import os, types
import numpy as np
import vtk
from vtk.util import numpy_support
import vtkcolors, vtkanalysis, vtkio 


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
    mapper.ScalarVisibilityOff()
    
    cdir = os.path.dirname(__file__)
    if cdir == '': cdir = '.'  
    fn = cdir + '/textures/' + name + ".jpg"
    if os.path.exists(name): 
        fn = name
    elif not os.path.exists(fn):
        vtkio.printc(('Texture', name, 'not found in', cdir+'/textures'), 'r')
        vtkio.printc('Available textures:', c='m', end=' ')
        for ff in os.listdir(cdir + '/textures'):
            vtkio.printc(ff.split('.')[0], end=' ', c='m')
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

    def _fcutterw(self): return vtkanalysis.cutterWidget(self)
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
        return vtkanalysis.subdivide(self, N, method, legend)
    actor.subdivide = types.MethodType( _fsubdivide, actor)

    def _fdecimate(self, fraction=0.5, N=None, verbose=True, boundaries=True): 
        return vtkanalysis.decimate(self, fraction, N, verbose, boundaries)
    actor.decimate = types.MethodType( _fdecimate, actor)

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

    def _fwire(self, a):
        if a: 
            self.GetProperty().SetRepresentationToWireframe()
        else:
            self.GetProperty().SetRepresentationToSurface()
        return self
    actor.wire = types.MethodType( _fwire, actor)

    def _flegend(self, a=None):
        if not hasattr(self, 'legend'):
            setattr(self, 'legend', a)
        return self
    actor.legend = types.MethodType( _flegend, actor)

    def _fclosestPoint(self, pt, N=1, radius=None):
        return closestPoint(self, pt, N, radius)
    actor.closestPoint = types.MethodType( _fclosestPoint, actor)

    def _fintersectWithLine(self, p0, p1):
        return vtkanalysis.intersectWithLine(self, p0,p1)
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
        vtkio.printc('Limitation: cannot clone textured obj. Returning input.',1)
        return actor
    polyCopy = vtk.vtkPolyData()
    polyCopy.DeepCopy(poly)

    if legend  is True and hasattr(actor, 'legend'): legend = actor.legend
    if alpha   is None: alpha = actor.GetProperty().GetOpacity()
    if c       is None: c = actor.GetProperty().GetColor()
    if texture is None and hasattr(actor, 'texture'): texture = actor.texture
    cact = makeActor(polyCopy, c, alpha, wire, bc, edges, legend, texture)
    cact.GetProperty().SetPointSize(actor.GetProperty().GetPointSize())
    return cact


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
    if hasattr(actor, 'poly'): actor.poly=poly
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
    if hasattr(actor, 'poly'): actor.poly=tf.GetOutput()
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
    # this vtk method only rotates in the origin of the actor:
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
        vtkio.printc('Please define vectors actor.base and actor.top at creation. Exit.','r')
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


def mergeActors(actors, c=None, alpha=1, 
                wire=False, bc=None, edges=False, legend=None, texture=None):
    '''
    Build a new actor formed by the sum of the polydata of the input objects.
    '''
    polylns = vtk.vtkAppendPolyData()
    for a in actors:
        polylns.AddInputData(polydata(a, True))
    polylns.Update()
    actor = makeActor(polylns.GetOutput(), 
                      c, alpha, wire, bc, edges, legend, texture)
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
        vtkio.printc("Warning: polydata is not a closed surface",5)
    
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


def cleanPolydata(actor, tol=None):
    '''
    Clean actor's polydata.
        tol paramenter defines how far should be the points from each other
        in terms of fraction of bounding box length.
    '''
    poly = polydata(actor, False)
    cleanPolyData = vtk.vtkCleanPolyData()
    setInput(cleanPolyData, poly)
    if tol: cleanPolyData.SetTolerance(tol)
    cleanPolyData.PointMergingOn()
    cleanPolyData.Update()
    mapper = actor.GetMapper()
    setInput(mapper, cleanPolyData.GetOutput())
    mapper.Update()
    actor.Modified()
    if hasattr(actor, 'poly'): actor.poly = cleanPolyData.GetOutput()
    return actor # NB: polydata is being changed
    

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
    
    vtkio.printc("Fatal Error in polydata(): ", 'r', end='')
    vtkio.printc(("input is neither a vtkActor nor vtkAssembly.", [obj]), 'r')
    exit(1)



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
    return np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2)

def maxBoundSize(actor):
    '''Get the maximum dimension in x, y or z of the actor bounding box'''
    b = polydata(actor, True).GetBounds()
    return max(abs(b[1]-b[0]), abs(b[3]-b[2]), abs(b[5]-b[4]))

   

########################################################################
def closestPoint(actor, pt, N=1, radius=None):
    """
    Find the closest point on a polydata given an other point.
        If N>1, return a list of N ordered closest points.
        If radius is given, get all points within.
    """
    poly = polydata(actor, True)

    if N>1 or radius: 
        plocexists = hasattr(actor, 'point_locator')
        if not plocexists or (plocexists and actor.point_locator is None):
            point_locator = vtk.vtkPointLocator()
            point_locator.SetDataSet(poly)
            point_locator.BuildLocator()
            setattr(actor, 'point_locator', point_locator)
        
        vtklist = vtk.vtkIdList()
        if N>1: actor.point_locator.FindClosestNPoints(N, pt, vtklist)
        else: actor.point_locator.FindPointsWithinRadius(radius, pt, vtklist)
        trgp  = []
        for i in range(vtklist.GetNumberOfIds()):
            trgp_ = [0,0,0]
            vi = vtklist.GetId(i)
            poly.GetPoints().GetPoint(vi, trgp_ )
            trgp.append( trgp_ )
        return np.array(trgp)

    clocexists = hasattr(actor, 'cell_locator')
    if not clocexists or (clocexists and actor.cell_locator is None):
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(poly)
        cell_locator.BuildLocator()
        setattr(actor, 'cell_locator', cell_locator)

    trgp  = [0,0,0]
    cid   = vtk.mutable(0)
    dist2 = vtk.mutable(0)
    subid = vtk.mutable(0)
    actor.cell_locator.FindClosestPoint(pt, trgp, cid, subid, dist2)
    return np.array(trgp)

