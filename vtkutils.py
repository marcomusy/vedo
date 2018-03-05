# -*- coding: utf-8 -*-
#
from __future__ import division, print_function
import os, sys, types
import numpy as np
import vtkcolors
import vtk
import time


##############################################################################
vtkMV = vtk.vtkVersion().GetVTKMajorVersion() > 5
def setInput(vtkobj, p, port=0):
    if isinstance(p, vtk.vtkAlgorithmOutput):
        vtkobj.SetInputConnection(port, p) # passing port
        return    
    if vtkMV: vtkobj.SetInputData(p)
    else: vtkobj.SetInput(p)


def arange(start,stop, step=1): 
    return np.arange(start, stop, step)

def vector(x,y,z=None):
    if z is None: return np.array([x,y,0]) #assume 2D
    return np.array([x,y,z])

def mag(z):
    if isinstance(z[0], np.ndarray): 
        return np.array(list(map(np.linalg.norm, z)))
    else: 
        return np.linalg.norm(z)

def norm(v):
    if isinstance(v[0], np.ndarray):
        return np.divide(v, mag(v)[:,None])
    else: 
        return v/mag(v)


def makeActor(poly, c='gold', alpha=0.5, 
              wire=False, bc=None, edges=False, legend=None, texture=None):
    '''Return a vtkActor from an input vtkPolyData, optional args:
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
    #triangles = vtk.vtkTriangleFilter()
    #setInput(triangles, clp.GetOutput())
    #triangles.Update() 
    pdnorm = vtk.vtkPolyDataNormals()
    setInput(pdnorm, clp.GetOutput())
    pdnorm.SetFeatureAngle(60.0)
    pdnorm.ComputePointNormalsOn()
    pdnorm.ComputeCellNormalsOn()
    pdnorm.FlipNormalsOff()
    pdnorm.ConsistencyOn()
    pdnorm.Update()

    mapper = vtk.vtkPolyDataMapper()

    setInput(mapper, pdnorm.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if c is None: 
        mapper.ScalarVisibilityOn()
    else:
        mapper.ScalarVisibilityOff()
        c = vtkcolors.getColor(c)
        actor.GetProperty().SetColor(c)
        actor.GetProperty().SetOpacity(alpha)
    
        actor.GetProperty().SetSpecular(0.1)
        actor.GetProperty().SetSpecularColor(c)
        actor.GetProperty().SetSpecularPower(1)
    
        actor.GetProperty().SetAmbient(0.1)
        actor.GetProperty().SetAmbientColor(c)
    
        actor.GetProperty().SetDiffuse(1)
        actor.GetProperty().SetDiffuseColor(c)

    if edges: actor.GetProperty().EdgeVisibilityOn()
    if wire: actor.GetProperty().SetRepresentationToWireframe()
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
    '''Treat many actors as a single new actor'''
    assembly = vtk.vtkAssembly()
    for a in actors: assembly.AddPart(a)
    setattr(assembly, 'legend', legend) 
    assignPhysicsMethods(assembly)
    assignConvenienceMethods(assembly, legend)
    return assembly


def assignTexture(actor, name, scale=1, falsecolors=False, mapTo=1):
    '''Assign a texture to actor from file or name in /textures directory'''
    if   mapTo == 1: tmapper = vtk.vtkTextureMapToCylinder()
    elif mapTo == 2: tmapper = vtk.vtkTextureMapToSphere()
    elif mapTo == 3: tmapper = vtk.vtkTextureMapToPlane()
    
    setInput(tmapper, polydata(actor))
    if mapTo == 1:  tmapper.PreventSeamOn()
    
    xform = vtk.vtkTransformTextureCoords()
    xform.SetInputConnection(tmapper.GetOutputPort())
    xform.SetScale(scale,scale,scale)
    if mapTo == 1: xform.FlipSOn()
    xform.Update()
    
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(xform.GetOutputPort())
    
    cdir = os.path.dirname(__file__)     
    fn = cdir + '/textures/'+name+".jpg"
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


def assignConvenienceMethods(actor, legend):
    if not hasattr(actor, 'legend'):
        setattr(actor, 'legend', legend)

    def _frotate(self, angle, axis, rad=False): 
        if rad: angle *= 57.3
        return rotate(self, angle, axis, rad)
    actor.rotate = types.MethodType( _frotate, actor )

    def _frotateX(self, angle, rad=False): 
        if rad: angle *= 57.3
        return rotate(self, angle, [1,0,0], rad)
    actor.rotateX = types.MethodType( _frotateX, actor )

    def _frotateY(self, angle, rad=False): 
        if rad: angle *= 57.3
        return rotate(self, angle, [0,1,0], rad)
    actor.rotateY = types.MethodType( _frotateY, actor )

    def _frotateZ(self, angle, rad=False): 
        if rad: angle *= 57.3
        return rotate(self, angle, [0,0,1], rad)
    actor.rotateZ = types.MethodType( _frotateZ, actor )

    def _fclone(self, c='gold', alpha=1, wire=False, bc=None,
                edges=False, legend=None, texture=None): 
        return clone(self, c, alpha, wire, bc, edges, legend, texture)
    actor.clone = types.MethodType( _fclone, actor )

    def _fpoint(self, i, p=None): 
        poly = polydata(self)
        if p is None:
            p = [0,0,0]
            poly.GetPoints().GetPoint(i, p)
            return np.array(p)
        else:
            poly.GetPoints().SetPoint(i, p)
        return 
    actor.point = types.MethodType( _fpoint, actor )

    def _fnormalize(self): return normalize(self)
    actor.normalize = types.MethodType( _fnormalize, actor )

    def _fshrink(self, fraction=0.85): return shrink(self, fraction)
    actor.shrink = types.MethodType( _fshrink, actor )

    def _fcutterw(self): return cutterWidget(self)
    actor.cutterWidget = types.MethodType( _fcutterw, actor )
    
    def _fvisible(self, alpha=1): self.GetProperty().SetOpacity(alpha)
    actor.visible = types.MethodType( _fvisible, actor )
    
    def _fgpoly(self): return polydata(self)
    actor.polydata = types.MethodType( _fgpoly, actor )

    def _fcoords(self): return coordinates(self)
    actor.coordinates = types.MethodType( _fcoords, actor )

    def _fstretch(self, startpt, endpt): 
        return stretch(self, startpt, endpt)
    actor.stretch = types.MethodType( _fstretch, actor)


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
     
    setattr(actor, '_vel',  np.array([0,0,0]))  # velocity
    def _fvel(self, v=None): 
        if v is None: return self._vel
        self._vel = v
    actor.vel = types.MethodType( _fvel, actor )
    
    def _fvx(self, vx=None):               # VX  
        if vx is None: return self._vel[0]
        newp = [vx, self._vel[1], self._vel[2]]
        self._vel = newp
    actor.vx = types.MethodType( _fvx, actor )

    def _fvy(self, vy=None):               # VY  
        if vy is None: return self._vel[1]
        newp = [self._vel[0], vy, self._vel[2]]
        self._vel = newp
    actor.vy = types.MethodType( _fvy, actor )

    def _fvz(self, vz=None):               # VZ  
        if vz is None: return self._vel[2]
        newp = [self._vel[0], self._vel[1], vz]
        self._vel = newp
    actor.vz = types.MethodType( _fvz, actor )
     
    setattr(actor, '_mass',  1.0)               # mass
    def _fmass(self, m=None): 
        if m is None: return self._mass
        self._mass = m
    actor.mass = types.MethodType( _fmass, actor )

    setattr(actor, '_omega', 0.0)                # angular velocity
    def _fomega(self, o=None): 
        if o is None: return self._omega
        self._omega = o
    actor.omega = types.MethodType( _fomega, actor )

    def _fmomentum(self): 
        return self._mass * self._vel
    actor.momentum = types.MethodType( _fmomentum, actor )

    def _fgamma(self):                 # Lorentz factor
        v2 = np.sum( self._vel*self._vel )
        return 1./np.sqrt(1. - v2/299792.48**2)
    actor.gamma = types.MethodType( _fgamma, actor )



######################################################### 
def clone(actor, c='gold', alpha=None, wire=False, bc=None,
          edges=False, legend=None, texture=None): 
    poly = polydata(actor)
    if not len(coordinates(actor)):
        printc('Limitation: cannot clone textured obj. Returning input.',1)
        return actor
    polyCopy = vtk.vtkPolyData()
    polyCopy.DeepCopy(poly)
    if not legend is None and hasattr(actor.legend): 
        legend = actor.legend
        
    if alpha is None: alpha = actor.GetProperty().GetOpacity()
    if hasattr(actor, 'texture'): texture = actor.texture
    a = makeActor(polyCopy, c, alpha, wire, bc, edges, legend, texture)
  
    assignPhysicsMethods(a)    
    assignConvenienceMethods(a, legend)    
    return a
    

def normalize(actor, s=1): # N.B. input argument gets modified
    # s= scale
    cm = centerOfMass(actor)
    coords = coordinates(actor)
    if not len(coords) : return
    pts = coordinates(actor) - cm
    xyz2 = np.sum(pts * pts, axis=0)
    scale = s/np.sqrt(np.sum(xyz2)/len(pts))
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
    return actor  # return same obj for concatenation


def rotate(actor, angle, axis, point=[0,0,0], rad=False): 
    if rad: angle *= 57.3

    if isinstance(actor, list):
        axis = np.asarray(axis)
        theta = np.asarray(angle)
        ax2 = np.sqrt(np.dot(axis, axis))
        if ax2: axis /= ax2
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        R = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
        rv = np.dot(R, actor)
        return rv

#    cm= np.array(point)     ## point not working ??
#    actor.SetPosition(-cm)    
    actor.RotateWXYZ(angle, axis[0], axis[1], axis[2] )
#    actor.SetPosition(cm)
    return actor   # return same obj for concatenation


############################################################################
def shrink(actor, fraction=0.85): # N.B. input argument is modified
    poly = polydata(actor)
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
    TI = vtk.vtkTransform()
    actor.SetUserMatrix(TI.GetMatrix()) # reset

    p1, p2 = actor.axis()
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
    return actor  # return same obj for concatenation


def boolActors(actor1, actor2, operation='plus', c=None, alpha=1, 
               wire=False, bc=None, edges=False, legend=None, texture=None):
    try:
        bf = vtk.vtkBooleanOperationPolyDataFilter()
    except AttributeError:
        printc('Boolean operation only possible for vtk version > 6','r')
        return None
    poly1 = polydata(actor1)
    poly2 = polydata(actor2)
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


#########################################################
# Useful Functions
######################################################### 
def makePolyData(spoints, addLines=True):
    """Try to workout a polydata from points"""
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


def isInside(actor, point):
    """Return True if point is inside a polydata closed surface"""
    poly = polydata(actor)
    points = vtk.vtkPoints()
    points.InsertNextPoint(point)
    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)
    sep = vtk.vtkSelectEnclosedPoints()
    setInput(sep, pointsPolydata)
    if vtkMV: sep.SetSurfaceData(poly)
    else: sep.SetSurface(poly)
    sep.Update()
    return sep.IsInside(0)


def insidePoints(actor, points, invert=False, tol=1e-05):
    """Return list of points that are inside a polydata closed surface"""
    poly = polydata(actor)
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


def fillHoles(actor, size=None, legend=None):
    fh = vtk.vtkFillHolesFilter()
    if not size:
        mb = maxOfBounds(actor)
        size = mb/20
    fh.SetHoleSize(size)
    poly = polydata(actor)
    setInput(fh, poly)
    fh.Update()
    fpoly = fh.GetOutput()
    factor = makeActor(fpoly, legend=legend)
    factor.SetProperty(actor.GetProperty())
    return factor    


#################################################################### get stuff
def polydata(obj, index=0): 
    '''
    Returns vtkPolyData from an other object (vtkActor, vtkAssembly)
    '''
    if isinstance(obj, list) and len(obj)>0: obj = obj[index]
   
    if isinstance(obj, vtk.vtkActor):   
        pd = obj.GetMapper().GetInput()
        transform = vtk.vtkTransform()
        transform.SetMatrix(obj.GetMatrix())
        tp = vtk.vtkTransformPolyDataFilter()
        tp.SetTransform(transform)
        if vtkMV: tp.SetInputData(pd)
        else: tp.SetInput(pd)
        tp.Update()
        return tp.GetOutput()

    elif isinstance(obj, vtk.vtkAssembly):
        cl = vtk.vtkPropCollection()
        obj.GetActors(cl)
        cl.InitTraversal()
        for i in range(index+1):
            act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
        pd = act.GetMapper().GetInput()
        transform = vtk.vtkTransform()
        transform.SetMatrix(act.GetMatrix())
        tp = vtk.vtkTransformPolyDataFilter()
        tp.SetTransform(transform)
        if vtkMV: tp.SetInputData(pd)
        else: tp.SetInput(pd)
        tp.Update()
        return tp.GetOutput()
    
    elif isinstance(obj, vtk.vtkPolyData): return obj
    elif isinstance(obj, vtk.vtkActor2D):  return obj.GetMapper().GetInput()
    elif isinstance(obj, vtk.vtkImageActor):  return obj.GetMapper().GetInput()

    printc("Fatal Error in polydata(): ", 'r', end='')
    printc(("input is neither a poly nor an actor int or assembly.", [obj]), 'r')
    exit(1)


def coordinates(actors):
    """Return a merged list of coordinates of actors or polys"""
    if not isinstance(actors, list): actors = [actors]
    pts = []
    for i in range(len(actors)):
        apoly = polydata(actors[i])
        for j in range(apoly.GetNumberOfPoints()):
            p = [0, 0, 0]
            apoly.GetPoint(j, p)
            pts.append(p)
    return np.array(pts)


def maxOfBounds(actor):
    '''Get the maximum dimension of the actor bounding box'''
    poly = polydata(actor)
    b = poly.GetBounds()
    maxb = max(abs(b[1]-b[0]), abs(b[3]-b[2]), abs(b[5]-b[4]))
    return maxb


def centerOfMass(actor):
    '''Get the Center of Mass of the actor'''
    if vtkMV: #faster
        cmf = vtk.vtkCenterOfMass()
        setInput(cmf, polydata(actor))
        #cmf.UseScalarsAsWeightsOff()
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)
    else:
        pts = coordinates(actor)
        if not len(pts): return np.array([0,0,0])
        return np.mean(pts, axis=0)       


def volume(actor):
    '''Get the volume occupied by actor'''
    mass = vtk.vtkMassProperties()
    setInput(mass, polydata(actor))
    mass.Update() 
    return mass.GetVolume()


def surfaceArea(actor):
    '''Get the surface area of actor'''
    mass = vtk.vtkMassProperties()
    setInput(mass, polydata(actor))
    mass.Update() 
    return mass.GetSurfaceArea()

    
def write(obj, fileoutput):
    fr = fileoutput.lower()
    if   '.vtk' in fr: w = vtk.vtkPolyDataWriter()
    elif '.ply' in fr: w = vtk.vtkPLYWriter()
    elif '.obj' in fr: 
        w = vtk.vtkOBJExporter()
        w.SetFilePrefix(fileoutput.replace('.obj',''))
        printc('input must be set manually to vp.renderWin',3)
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
        setInput(w, polydata(obj))
        w.SetFileName(fileoutput)
        w.Write()
        printc("Saved file: "+fileoutput, 'g')
    except:
        printc("Error saving: "+fileoutput, 'r')


########################################################################
def closestPoint(surf, pt, locator=None, N=None, radius=None):
    """
    Find the closest point on a polydata given an other point.
    If N is given, return a list of N ordered closest points.
    If radius is given, pick only within specified radius.
    """
    poly = polydata(surf)
    trgp  = [0,0,0]
    cid   = vtk.mutable(0)
    dist2 = vtk.mutable(0)
    if not locator:
        if N: locator = vtk.vtkPointLocator()
        else: locator = vtk.vtkCellLocator()
        locator.SetDataSet(poly)
        locator.BuildLocator()
    if N:
        vtklist = vtk.vtkIdList()
        vmath = vtk.vtkMath()
        locator.FindClosestNPoints(N, pt, vtklist)
        trgp_, trgp, dists2 = [0,0,0], [], []
        for i in range(vtklist.GetNumberOfIds()):
            vi = vtklist.GetId(i)
            poly.GetPoints().GetPoint(vi, trgp_ )
            trgp.append( trgp_ )
            dists2.append(vmath.Distance2BetweenPoints(trgp_, pt))
        dist2 = dists2
    elif radius:
        cell = vtk.mutable(0)
        r = locator.FindClosestPointWithinRadius(pt, radius, trgp, cell, cid, dist2)
        if not r: 
            trgp = pt
            dist2 = 0.0
    else: 
        subid = vtk.mutable(0)
        locator.FindClosestPoint(pt, trgp, cid, subid, dist2)
    return trgp


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
    
    printc(("Press X to save file:", outputname), 'm')
    def cwkeypress(obj, event):
        if obj.GetKeySym() == "X": 
            confilter = vtk.vtkPolyDataConnectivityFilter()
            setInput(confilter, clipper.GetOutput())
            confilter.SetExtractionModeToLargestRegion()
            confilter.Update()
            cpd = vtk.vtkCleanPolyData()
            setInput(cpd, confilter.GetOutput())
            cpd.Update()
            write(cpd.GetOutput(), outputname)
        elif obj.GetKeySym() == "Escape": 
            exit()
    
    iren.Initialize()
    iren.AddObserver("KeyPressEvent", cwkeypress)
    iren.Start()
    boxWidget.Off()

    return act0


###########################################################################
class ProgressBar: 
    '''Class to print a progress bar with optional text on its right'''
    # import time                        ### Usage example:
    # pb = ProgressBar(0,400, c='red')
    # for i in pb.range():
    #     time.sleep(.1)
    #     pb.print('some message')       # or pb.print(counts=i) 
    def __init__(self, start, stop, step=1, c=None, ETA=True, width=25):
        self.start  = start
        self.stop   = stop
        self.step   = step
        self.color  = c
        self.width  = width
        self.bar    = ""  
        self.percent= 0
        self._counts= 0
        self._oldbar= ""
        self._lentxt= 0
        self._range = arange(start, stop, step)
        self._len   = len(self._range)
        self.clock0 = 0
        self.ETA    = ETA
        self.clock0 = time.time()
        self._update(0)
        
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
    '''Print to terminal in color. Available colors:
    black, red, green, yellow, blue, magenta, cyan, white
    E.g.:
    cprint( 'anything', c='red', bold=False, end='' )
    cprint( ['anything', 455.5, vtkObject], 'green', separator='-')
    cprint(299792.48, c=4) #blue
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

