"""
Utilities submodule. 
Contains methods to perform simple operations with meshes.
"""

from __future__ import division, print_function
import os
import types
import numpy as np
from numpy import arange
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import vtk_to_numpy
import vtkplotter.colors as colors


##############################################################################
vtkMV = vtk.vtkVersion().GetVTKMajorVersion() > 5

_cdir = os.path.dirname(__file__)
if _cdir == '':
    _cdir = '.'
textures_path = _cdir + '/textures/'

textures = []
for _f in os.listdir(textures_path):
    textures.append(_f.split('.')[0])
textures.remove('earth')
textures = list(sorted(textures))


##############################################################################
def setInput(vtkobj, p, port=0):
    if isinstance(p, vtk.vtkAlgorithmOutput):
        vtkobj.SetInputConnection(port, p)  # passing port
        return
    if vtkMV:
        vtkobj.SetInputData(p)
    else:
        vtkobj.SetInput(p)


def isSequence(arg):
    '''Check if input is iterable.'''
    if hasattr(arg, "strip"):
        return False
    if hasattr(arg, "__getslice__"):
        return True
    if hasattr(arg, "__iter__"):
        return True
    return False


def vector(x, y=None, z=0.):
    '''Return a 2D or 3D numpy array.'''
    if y is None:  # assume x is already [x,y,z]
        return np.array(x, dtype=np.float64)
    return np.array([x, y, z], dtype=np.float64)


def mag(z):
    '''Get the magnitude of a vector.'''
    if isinstance(z[0], np.ndarray):
        return np.array(list(map(np.linalg.norm, z)))
    else:
        return np.linalg.norm(z)


def mag2(z):
    '''Get the squared magnitude of a vector.'''
    return np.dot(z, z)


def norm(v):
    '''Return the unit vector.'''
    if isinstance(v[0], np.ndarray):
        return np.divide(v, mag(v)[:, None])
    else:
        return v/mag(v)


def to_precision(x, p):
    """
    Returns a string representation of x formatted with precision p.

    *Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/
    kjs/number_object.cpp.
    Implemented in https://github.com/randlet/to-precision*
    """
    import math
    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []
    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e - 1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens - x):
        n = n + 1

    if n >= math.pow(10, p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)
    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p - 1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)
    return "".join(out)


#########################################################################
def makeActor(poly, c='gold', alpha=0.5,
              wire=False, bc=None, edges=False, legend=None, texture=None):
    '''
    Return a vtkActor from an input vtkPolyData.

    Options:

        c,       color in RGB format, hex, symbol or name

        alpha,   opacity value

        wire,    show surface as wireframe

        bc,      backface color of internal surface

        edges,   show edges as line on top of surface

        legend,  optional string

        texture, jpg file name or surface texture name
    '''
    clp = vtk.vtkCleanPolyData()
    clp.SetTolerance(0.0)
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
    setInput(mapper, pdnorm.GetOutput())

    # check if color string contains a float, in this case ignore alpha
    if alpha is None:
        alpha = 0.5
    al = colors.getAlpha(c)
    if al:
        alpha = al

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prp = actor.GetProperty()

    #########################################################################
    # On some vtk versions/platforms points are redered as ugly squares
    # in such a case uncomment this line:
    if vtk.vtkVersion().GetVTKMajorVersion() > 7:
        prp.RenderPointsAsSpheresOn()
    #########################################################################

    if c is None:
        mapper.ScalarVisibilityOn()
    else:
        mapper.ScalarVisibilityOff()
        c = colors.getColor(c)
        prp.SetColor(c)
        prp.SetOpacity(alpha)

        prp.SetSpecular(0.1)
        prp.SetSpecularColor(c)
        prp.SetSpecularPower(1)

        prp.SetAmbient(0.1)
        prp.SetAmbientColor(c)

        prp.SetDiffuse(1)
        prp.SetDiffuseColor(c)

    if edges:
        prp.EdgeVisibilityOn()
    if wire:
        prp.SetRepresentationToWireframe()
    if texture:
        mapper.ScalarVisibilityOff()
        assignTexture(actor, texture)
    if bc:  # defines a specific color for the backface
        backProp = vtk.vtkProperty()
        backProp.SetDiffuseColor(colors.getColor(bc))
        backProp.SetOpacity(alpha)
        actor.SetBackfaceProperty(backProp)

    assignPhysicsMethods(actor)
    assignConvenienceMethods(actor, legend)
    return actor


def makeAssembly(actors, legend=None):
    '''Group many actors as a single new actor.

    [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope1.py)    
    [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/icon.py)    
    '''
    if len(actors) == 0:
        return None
    assembly = vtk.vtkAssembly()
    for a in actors:
        assembly.AddPart(a)
    setattr(assembly, 'legend', legend)
    assignPhysicsMethods(assembly)
    assignConvenienceMethods(assembly, legend)
    if hasattr(actors[0], 'base'):
        setattr(assembly, 'base', actors[0].base)
        setattr(assembly, 'top',  actors[0].top)
    return assembly


def getActors(assembly):
    '''Unpack a list of vtkActor objects from a vtkAssembly'''
    cl = vtk.vtkPropCollection()
    assembly.GetActors(cl)
    actors = []
    cl.InitTraversal()
    for i in range(assembly.GetNumberOfPaths()):
        act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
        if isinstance(act, vtk.vtkCubeAxesActor):
            continue
        actors.append(act)
    return actors
        
        
def makeVolume(img, c=(0, 0, 0.6), alphas=[0, 0.4, 0.9, 1]):
    '''Make a vtkVolume actor'''

    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetBlendModeToMaximumIntensity()
    volumeMapper.SetInputData(img)
    colors.printc('scalar range is ', img.GetScalarRange(), c='b', bold=0)
    smin, smax = img.GetScalarRange()
    if smax > 1e10:
        print("Warning, high scalar range detected:", smax)
        smax = abs(10*smin)+.1
        print("         reset to:", smax)

    # Create transfer mapping scalar value to color
    r, g, b = colors.getColor(c)
    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(smin, 1.0, 1.0, 1.0)
    colorTransferFunction.AddRGBPoint((smax+smin)/3, r/2, g/2, b/2)
    colorTransferFunction.AddRGBPoint(smax, 0.0, 0.0, 0.0)

    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    for i, al in enumerate(alphas):
        xalpha = smin+(smax-smin)*i/(len(alphas)-1)
        # Create transfer mapping scalar value to opacity
        opacityTransferFunction.AddPoint(xalpha, al)
        colors.printc('\talpha at', round(xalpha, 1),
                      '\tset to', al, c='b', bold=0)

    # The property describes how the data will look
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.SetInterpolationTypeToLinear()

    # volume holds the mapper and the property and can be used to position/orient it
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    return volume


def makeIsosurface(image, c, alpha, wire, bc, edges, legend, texture,
                   smoothing, threshold, connectivity, scaling):
    '''Return a vtkActor isosurface from a vtkImageData object.'''

    if smoothing:
        print('  gaussian smoothing data with volume_smoothing =', smoothing)
        smImg = vtk.vtkImageGaussianSmooth()
        smImg.SetDimensionality(3)
        setInput(smImg, image)
        smImg.SetStandardDeviations(smoothing, smoothing, smoothing)
        smImg.Update()
        image = smImg.GetOutput()

    scrange = image.GetScalarRange()

    if not threshold:
        if scrange[1] > 1e10:
            threshold = (2*scrange[0]+abs(10*scrange[0]))/3.
            print("Warning, high scalar range detected:", scrange[1])
            print("         setting threshold to:", threshold)
        else:
            threshold = (2*scrange[0]+scrange[1])/3.
    cf = vtk.vtkContourFilter()
    setInput(cf, image)
    cf.UseScalarTreeOn()
    cf.ComputeScalarsOff()
    cf.SetValue(0, threshold)
    cf.Update()

    clp = vtk.vtkCleanPolyData()
    setInput(clp, cf.GetOutput())
    clp.Update()
    image = clp.GetOutput()

    if connectivity:
        print('  applying connectivity filter, select largest region')
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetExtractionModeToLargestRegion()
        setInput(conn, image)
        conn.Update()
        image = conn.GetOutput()

    if scaling:
        print('  scaling xyz by factors', scaling)
        tf = vtk.vtkTransformPolyDataFilter()
        setInput(tf, image)
        trans = vtk.vtkTransform()
        trans.Scale(scaling)
        tf.SetTransform(trans)
        tf.Update()
        image = tf.GetOutput()
    return makeActor(image, c, alpha, wire, bc, edges, legend, texture)


def assignTexture(actor, name, scale=1, falsecolors=False, mapTo=1):
    '''Assign a texture to actor from file or name in /textures directory.'''
    global textures_path
    if mapTo == 1:
        tmapper = vtk.vtkTextureMapToCylinder()
    elif mapTo == 2:
        tmapper = vtk.vtkTextureMapToSphere()
    elif mapTo == 3:
        tmapper = vtk.vtkTextureMapToPlane()

    setInput(tmapper, polydata(actor))
    if mapTo == 1:
        tmapper.PreventSeamOn()

    xform = vtk.vtkTransformTextureCoords()
    xform.SetInputConnection(tmapper.GetOutputPort())
    xform.SetScale(scale, scale, scale)
    if mapTo == 1:
        xform.FlipSOn()
    xform.Update()

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(xform.GetOutputPort())
    mapper.ScalarVisibilityOff()

    fn = textures_path + name + ".jpg"
    if os.path.exists(name):
        fn = name
    elif not os.path.exists(fn):
        colors.printc('Texture', name, 'not found in', textures_path, c='r')
        colors.printc('Available textures:', c='m', end=' ')
        for ff in os.listdir(textures_path):
            colors.printc(ff.split('.')[0], end=' ', c='m')
        print()
        return

    jpgReader = vtk.vtkJPEGReader()
    jpgReader.SetFileName(fn)
    atext = vtk.vtkTexture()
    atext.RepeatOn()
    atext.EdgeClampOff()
    atext.InterpolateOn()
    if falsecolors:
        atext.MapColorScalarsThroughLookupTableOn()
    atext.SetInputConnection(jpgReader.GetOutputPort())
    actor.GetProperty().SetColor(1, 1, 1)
    actor.SetMapper(mapper)
    actor.SetTexture(atext)


#########################################################
def clone(actor, c=None, alpha=None, wire=False, bc=None,
          edges=False, legend=None, texture=None, rebuild=True, mirror=''):
    '''
    Clone a vtkActor.
    If rebuild is True build its polydata in its current position in space.

    [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/carcrash.py)    
    [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mirror.py)    
    [**Example3**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/skeletonize.py)    
    [**Example4**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/recosurface.py)    
    '''
    poly = polydata(actor, rebuild)
    if not poly.GetNumberOfPoints():
        colors.printc(
            'Limitation: cannot clone textured obj. Returning input.', c=1)
        return actor
    polyCopy = vtk.vtkPolyData()
    polyCopy.DeepCopy(poly)

    if mirror:
        sx, sy, sz = 1, 1, 1
        dx, dy, dz = actor.GetPosition()
        if mirror.lower() == 'x':
            sx = -1
        elif mirror.lower() == 'y':
            sy = -1
        elif mirror.lower() == 'z':
            sz = -1
        else:
            colors.printc(
                "Error in mirror(): mirror must be set to x, y or z.", c=1)
            exit()
        for j in range(polyCopy.GetNumberOfPoints()):
            p = [0, 0, 0]
            polyCopy.GetPoint(j, p)
            polyCopy.GetPoints().SetPoint(j, p[0]*sx-dx*(sx-1),
                                          p[1]*sy-dy*(sy-1),
                                          p[2]*sz-dz*(sz-1))
        rs = vtk.vtkReverseSense()
        setInput(rs, polyCopy)
        rs.ReverseNormalsOn()
        rs.Update()
        polyCopy = rs.GetOutput()

    if legend is True and hasattr(actor, 'legend'):
        legend = actor.legend
    if alpha is None:
        alpha = actor.GetProperty().GetOpacity()
    if c is None:
        c = actor.GetProperty().GetColor()
    if texture is None and hasattr(actor, 'texture'):
        texture = actor.texture
    cact = makeActor(polyCopy, c, alpha, wire, bc, edges, legend, texture)
    cact.GetProperty().SetPointSize(actor.GetProperty().GetPointSize())
    return cact


def normalize(actor):  # N.B. input argument gets modified
    '''
    Shift actor's center of mass at origin and scale its average size to unit.

    [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/carcrash.py)    
    [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/keypress.py)    
    '''
    cm = centerOfMass(actor)
    coords = coordinates(actor)
    if not len(coords):
        return
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
    if hasattr(actor, 'poly'):
        actor.poly = tf.GetOutput()
    return actor  # return same obj for concatenation


def rotate(actor, angle, axis, axis_point=[0, 0, 0], rad=False):
    '''Rotate an actor around an arbitrary axis passing through axis_point.'''
    anglerad = angle
    if not rad:
        anglerad = angle/57.3
    axis = norm(axis)
    a = np.cos(anglerad / 2)
    b, c, d = -axis * np.sin(anglerad / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    R = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                  [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                  [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    rv = np.dot(R, actor.GetPosition()-np.array(axis_point)) + axis_point

    if rad:
        angle *= 57.3
    # this vtk method only rotates in the origin of the actor:
    actor.RotateWXYZ(angle, axis[0], axis[1], axis[2])
    actor.SetPosition(rv)
    return actor


def orientation(actor, newaxis=None, rotation=0):
    '''
    Set/Get actor orientation.
    If rotation != 0 rotate actor around newaxis (in degree units).

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope1.py)    
    '''
    initaxis = norm(actor.top - actor.base)
    if newaxis is None:
        return initaxis
    newaxis = norm(newaxis)
    TI = vtk.vtkTransform()
    actor.SetUserMatrix(TI.GetMatrix())  # reset
    pos = np.array(actor.GetPosition())
    crossvec = np.cross(initaxis, newaxis)
    angle = np.arccos(np.dot(initaxis, newaxis))
    T = vtk.vtkTransform()
    T.PostMultiply()
    T.Translate(-pos)
    if rotation:
        T.RotateWXYZ(rotation, initaxis)
    T.RotateWXYZ(angle*57.3, crossvec)
    T.Translate(pos)
    actor.SetUserMatrix(T.GetMatrix())
    return actor


def mirror(actor, axis='x'):
    '''Mirror the actor polydata along one of the cartesian axes.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mirror.py)    
    '''
    poly = polydata(actor, True)
    sx, sy, sz = 1, 1, 1
    dx, dy, dz = actor.GetPosition()
    if axis.lower() == 'x':
        sx = -1
    elif axis.lower() == 'y':
        sy = -1
    elif axis.lower() == 'z':
        sz = -1
    else:
        colors.printc("Error in mirror(): axis must be x, y or z.", c=1)
        exit()
    for j in range(poly.GetNumberOfPoints()):
        p = [0, 0, 0]
        poly.GetPoint(j, p)
        poly.GetPoints().SetPoint(j, p[0]*sx-2*dx, p[1]*sy-2*dy, p[2]*sz-2*dz)
    pnormals = poly.GetPointData().GetNormals()
    if pnormals:
        for j in range(pnormals.GetNumberOfTuples()):
            n = [0, 0, 0]
            pnormals.GetTuple(j, n)
            pnormals.SetTuple(j,  [n[0]*sx, n[1]*sy, n[2]*sz])
    cnormals = poly.GetCellData().GetNormals()
    if cnormals:
        for j in range(cnormals.GetNumberOfTuples()):
            n = [0, 0, 0]
            cnormals.GetTuple(j, n)
            cnormals.SetTuple(j, [n[0]*sx, n[1]*sy, n[2]*sz])

    actor.Modified()
    poly.GetPoints().Modified()
    return actor


############################################################################
def shrink(actor, fraction=0.85):   # N.B. input argument gets modified
    '''Shrink the triangle polydata in the representation of actor.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/shrink.py)    

    ![shrink](https://user-images.githubusercontent.com/32848391/46819143-41042280-cd83-11e8-9492-4f53679887fa.png)
    '''
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
    '''Stretch actor between points q1 and q2.

    [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/spring.py)    
    [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope1.py)    
    [**Example3**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/multiple_pendulum.py)    

    ![spring](https://user-images.githubusercontent.com/32848391/36788885-e97e80ae-1c8f-11e8-8b8f-ffc43dad1eb1.gif)
    '''
    if not hasattr(actor, 'base'):
        colors.printc(
            'Please define vectors actor.base and actor.top at creation. Exit.', c='r')
        exit(0)

    TI = vtk.vtkTransform()
    actor.SetUserMatrix(TI.GetMatrix())  # reset

    p1, p2 = actor.base, actor.top
    q1, q2, z = np.array(q1), np.array(q2), np.array([0, 0, 1])
    plength = np.linalg.norm(p2-p1)
    qlength = np.linalg.norm(q2-q1)
    T = vtk.vtkTransform()
    T.PostMultiply()
    T.Translate(-p1)
    cosa = np.dot(p2-p1, z)/plength
    n = np.cross(p2-p1, z)
    T.RotateWXYZ(np.arccos(cosa)*57.3, n)

    T.Scale(1, 1, qlength/plength)

    cosa = np.dot(q2-q1, z)/qlength
    n = np.cross(q2-q1, z)
    T.RotateWXYZ(-np.arccos(cosa)*57.3, n)
    T.Translate(q1)

    actor.SetUserMatrix(T.GetMatrix())
    return actor


def cutPlane(actor, origin=(0, 0, 0), normal=(1, 0, 0), showcut=False):
    '''
    Takes a vtkActor and cuts it with the plane defined by a point and a normal. 

    showcut = shows the cut away part as thin wireframe

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/trail.py)    

    ![tea1](https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png)
    '''
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    poly = polydata(actor)
    clipper = vtk.vtkClipPolyData()
    setInput(clipper, poly)
    clipper.SetClipFunction(plane)
    clipper.GenerateClippedOutputOn()
    clipper.SetValue(0.)
    clipper.Update()
    if hasattr(actor, 'GetProperty'):
        alpha = actor.GetProperty().GetOpacity()
        c = actor.GetProperty().GetColor()
        bf = actor.GetBackfaceProperty()
    else:
        alpha = 1
        c = 'gold'
        bf = None
    leg = None
    if hasattr(actor, 'legend'):
        leg = actor.legend
    clipActor = makeActor(clipper.GetOutput(), c=c, alpha=alpha, legend=leg)
    clipActor.SetBackfaceProperty(bf)

    acts = [clipActor]
    if showcut:
        cpoly = clipper.GetClippedOutput()
        restActor = makeActor(cpoly, c=c, alpha=0.05, wire=1)
        acts.append(restActor)

    if len(acts) > 1:
        asse = makeAssembly(acts)
        return asse
    else:
        return clipActor


def mergeActors(actors, c=None, alpha=1,
                wire=False, bc=None, edges=False, legend=None, texture=None):
    '''
    Build a new actor formed by the fusion of the polydata of the input objects.
    Similar to makeAssembly, but in this case the input objects become a single mesh.
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
def isInside(actor, point, tol=0.0001):
    """Return True if point is inside a polydata closed surface."""
    poly = polydata(actor, True)
    points = vtk.vtkPoints()
    points.InsertNextPoint(point)
    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)
    sep = vtk.vtkSelectEnclosedPoints()
    sep.SetTolerance(tol)
    sep.CheckSurfaceOff()
    setInput(sep, pointsPolydata)
    if vtkMV:
        sep.SetSurfaceData(poly)
    else:
        sep.SetSurface(poly)
    sep.Update()
    return sep.IsInside(0)


def insidePoints(actor, points, invert=False, tol=1e-05):
    """Return list of points that are inside a polydata closed surface."""
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
        colors.printc("Warning: polydata is not a closed surface", c=5)

    vpoints = vtk.vtkPoints()
    for p in points:
        vpoints.InsertNextPoint(p)
    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(vpoints)
    sep = vtk.vtkSelectEnclosedPoints()
    sep.SetTolerance(tol)
    setInput(sep, pointsPolydata)
    if vtkMV:
        sep.SetSurfaceData(poly)
    else:
        sep.SetSurface(poly)
    sep.Update()

    mask1, mask2 = [], []
    for i, p in enumerate(points):
        if sep.IsInside(i):
            mask1.append(p)
        else:
            mask2.append(p)
    if invert:
        return mask2
    else:
        return mask1


def pointIsInTriangle(p, p1, p2, p3):
    '''
    Return True if a point is inside (or above/below) a triangle defined by 3 points in space.
    '''
    p = np.array(p)
    u = np.array(p2) - p1
    v = np.array(p3) - p1
    n = np.cross(u, v)
    w = p - p1
    ln = np.dot(n, n)
    if not ln:
        return True  # degenerate triangle
    gamma = (np.dot(np.cross(u, w), n)) / ln
    beta = (np.dot(np.cross(w, v), n)) / ln
    alpha = 1-gamma-beta
    if 0 < alpha < 1 and 0 < beta < 1 and 0 < gamma < 1:
        return True
    return False


def pointToLineDistance(p, p1, p2):
    '''Compute the distance of a point to a line (not the segment) defined by p1 and p2.'''
    d = np.sqrt(vtk.vtkLine.DistanceToLine(p, p1, p2))
    return d


def fillHoles(actor, size=None, legend=None):  # not tested properly
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
    '''Get the list of cell centers of the mesh surface.

    [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/delaunay2d.py)    
    [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_coloring.py)    
    '''
    vcen = vtk.vtkCellCenters()
    setInput(vcen, polydata(actor, True))
    vcen.Update()
    return coordinates(vcen.GetOutput(), copy=True)


def isIdentity(M, tol=1e-06):
    '''Check if vtkMatrix4x4 is Identity.'''
    for i in [0, 1, 2, 3]:
        for j in [0, 1, 2, 3]:
            e = M.GetElement(i, j)
            if i == j:
                if np.abs(e-1) > tol:
                    return False
            elif np.abs(e) > tol:
                return False
    return True


def clean(actor, tol=None):
    '''
    Clean actor's polydata. Can also be used to decimate a mesh if tol is large.

    tol, defines how far should be the points from each other
    in terms of fraction of the bounding box length.

    [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares1D.py)    
    [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/recosurface.py)    
    '''
    poly = polydata(actor, False)
    cleanPolyData = vtk.vtkCleanPolyData()
    setInput(cleanPolyData, poly)
    if tol:
        cleanPolyData.SetTolerance(tol)
    cleanPolyData.PointMergingOn()
    cleanPolyData.Update()
    mapper = actor.GetMapper()
    setInput(mapper, cleanPolyData.GetOutput())
    mapper.Update()
    actor.Modified()
    if hasattr(actor, 'poly'):
        actor.poly = cleanPolyData.GetOutput()
    return actor  # NB: polydata is being changed


# get stuff
def polydata(obj, rebuild=True, index=0):
    '''
    Returns the vtkPolyData of a vtkActor or vtkAssembly.

    If rebuild=True returns a copy of polydata
    that corresponds to the current actor's position in space.

    If a vtkAssembly is passed, return the polydata of component index.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/quadratic_morphing.py)    
    '''
    if isinstance(obj, vtk.vtkActor):
        if not rebuild:
            if hasattr(obj, 'poly'):
                if obj.poly:
                    return obj.poly
            else:
                setattr(obj, 'poly', None)
            obj.poly = obj.GetMapper().GetInput()  # cache it for speed
            return obj.poly
        M = obj.GetMatrix()
        if isIdentity(M):
            if hasattr(obj, 'poly'):
                if obj.poly:
                    return obj.poly
            else:
                setattr(obj, 'poly', None)
            obj.poly = obj.GetMapper().GetInput()  # cache it for speed
            return obj.poly
        # if identity return the original polydata
        # otherwise make a copy that corresponds to
        # the actual position in space of the actor
        transform = vtk.vtkTransform()
        transform.SetMatrix(M)
        tp = vtk.vtkTransformPolyDataFilter()
        tp.SetTransform(transform)
        if vtkMV:
            tp.SetInputData(obj.GetMapper().GetInput())
        else:
            tp.SetInput(obj.GetMapper().GetInput())
        tp.Update()
        return tp.GetOutput()

    elif isinstance(obj, vtk.vtkAssembly):
        cl = vtk.vtkPropCollection()
        obj.GetActors(cl)
        cl.InitTraversal()
        for i in range(index+1):
            act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
        pd = act.GetMapper().GetInput()  # not optimized
        if not rebuild:
            return pd
        M = act.GetMatrix()
        if isIdentity(M):
            return pd
        # if identity return the original polydata
        # otherwise make a copy that corresponds to
        # the actual position in space of the actor
        transform = vtk.vtkTransform()
        transform.SetMatrix(M)
        tp = vtk.vtkTransformPolyDataFilter()
        tp.SetTransform(transform)
        if vtkMV:
            tp.SetInputData(pd)
        else:
            tp.SetInput(pd)
        tp.Update()
        return tp.GetOutput()

    elif isinstance(obj, vtk.vtkPolyData):
        return obj
    elif isinstance(obj, vtk.vtkActor2D):
        return obj.GetMapper().GetInput()
    elif isinstance(obj, vtk.vtkImageActor):
        return obj.GetMapper().GetInput()
    elif obj is None:
        return None

    colors.printc("Fatal Error in polydata(): ", c='r', end='')
    colors.printc("input is neither a vtkActor nor vtkAssembly.", [obj], c='r')
    exit(1)


def coordinates(actor, rebuild=True, copy=True):
    """
    Return the list of coordinates of an actor or polydata.

    Options:

        rebuild, if False ignore any previous trasformation applied to the mesh.

        copy, if False return the reference to the points so that they can be modified in place.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/align1.py)    
    """
    poly = polydata(actor, rebuild)
    if copy:
        return np.array(vtk_to_numpy(poly.GetPoints().GetData()))
    else:
        return vtk_to_numpy(poly.GetPoints().GetData())


def xbounds(actor):
    '''Get the actor bounds [xmin,xmax].

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/colormaps.py)    
    '''
    b = polydata(actor, True).GetBounds()
    return (b[0], b[1])


def ybounds(actor):
    '''Get the actor bounds [ymin,ymax].

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/colormaps.py)    
    '''
    b = polydata(actor, True).GetBounds()
    return (b[2], b[3])


def zbounds(actor):
    '''Get the actor bounds [zmin,zmax].

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/colormaps.py)    
    '''
    b = polydata(actor, True).GetBounds()
    return (b[4], b[5])


def centerOfMass(actor):
    '''Get the center of mass of actor.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fatlimb.py)    
    '''
    if vtkMV:  # faster
        cmf = vtk.vtkCenterOfMass()
        setInput(cmf, polydata(actor, True))
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)
    else:
        pts = coordinates(actor, copy=False)
        if not len(pts):
            return np.array([0, 0, 0])
        return np.mean(pts, axis=0)


def volume(actor):
    '''Get the volume occupied by actor.'''
    mass = vtk.vtkMassProperties()
    mass.SetGlobalWarningDisplay(0)
    setInput(mass, polydata(actor))
    mass.Update()
    return mass.GetVolume()


def area(actor):
    '''Get the surface area of actor.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/largestregion.py)    
    '''
    mass = vtk.vtkMassProperties()
    mass.SetGlobalWarningDisplay(0)
    setInput(mass, polydata(actor))
    mass.Update()
    return mass.GetSurfaceArea()


def averageSize(actor):
    '''Calculate the average size of a mesh.'''
    cm = centerOfMass(actor)
    coords = coordinates(actor, copy=False)
    if not len(coords):
        return 0
    s, c = 0.0, 0.0
    n = len(coords)
    step = int(n/10000.)+1
    for i in arange(0, n, step):
        s += mag(coords[i] - cm)
        c += 1
    return s/c


def diagonalSize(actor):
    '''Get the length of the diagonal of actor bounding box.'''
    b = polydata(actor).GetBounds()
    return np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2)


def maxBoundSize(actor):
    '''Get the maximum dimension in x, y or z of the actor bounding box.'''
    b = polydata(actor, True).GetBounds()
    return max(abs(b[1]-b[0]), abs(b[3]-b[2]), abs(b[5]-b[4]))


########################################################################
def closestPoint(actor, pt, N=1, radius=None, returnIds=False):
    """
    Find the closest point on a polydata given an other point.
    The appropriate locator is built on the fly and cached for speed.
        If N>1, return a list of N ordered closest points.
        If radius is given, get all points within.

    [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/align1.py)    
    [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitplanes.py)    
    [**Example3**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitspheres1.py)    
    [**Example4**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/quadratic_morphing.py)    
    """
    poly = polydata(actor, True)

    if N > 1 or radius:
        plocexists = hasattr(actor, 'point_locator')
        if not plocexists or (plocexists and actor.point_locator is None):
            point_locator = vtk.vtkPointLocator()
            point_locator.SetDataSet(poly)
            point_locator.BuildLocator()
            setattr(actor, 'point_locator', point_locator)

        vtklist = vtk.vtkIdList()
        if N > 1:
            actor.point_locator.FindClosestNPoints(N, pt, vtklist)
        else:
            actor.point_locator.FindPointsWithinRadius(radius, pt, vtklist)
        if returnIds:
            return [int(vtklist.GetId(k)) for k in range(vtklist.GetNumberOfIds())]
        else:
            trgp = []
            for i in range(vtklist.GetNumberOfIds()):
                trgp_ = [0, 0, 0]
                vi = vtklist.GetId(i)
                poly.GetPoints().GetPoint(vi, trgp_)
                trgp.append(trgp_)
            return np.array(trgp)

    clocexists = hasattr(actor, 'cell_locator')
    if not clocexists or (clocexists and actor.cell_locator is None):
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(poly)
        cell_locator.BuildLocator()
        setattr(actor, 'cell_locator', cell_locator)

    trgp = [0, 0, 0]
    cid = vtk.mutable(0)
    dist2 = vtk.mutable(0)
    subid = vtk.mutable(0)
    actor.cell_locator.FindClosestPoint(pt, trgp, cid, subid, dist2)
    if returnIds:
        return int(cid)
    else:
        return np.array(trgp)


def pointScalars(actor, scalars, name):
    """
    Set point scalars to the polydata.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_coloring.py)    
    """
    poly = polydata(actor, False)
    scalars = np.array(scalars) - np.min(scalars)
    scalars = scalars/np.max(scalars)
    if len(scalars) != poly.GetNumberOfPoints():
        colors.printc('Number of scalars != nr. of points', c=1)
        exit()
    arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    arr.SetName(name)
    poly.GetPointData().AddArray(arr)
    poly.GetPointData().SetActiveScalars(name)
    actor.GetMapper().ScalarVisibilityOn()


def pointColors(actor, scalars, cmap='jet', alpha=1):
    """
    Set individual point colors by setting an array of scalars.
    Scalars can be a string name.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_coloring.py)    
    """
    poly = polydata(actor, False)

    if isinstance(scalars, str):
        scalars = vtk_to_numpy(poly.GetPointData().GetArray(scalars))

    n = len(scalars)
    if n != poly.GetNumberOfPoints():
        colors.printc('Number of scalars != nr. of points', c=1)
        exit()

    vmin, vmax = np.min(scalars), np.max(scalars)
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(vmin, vmax)
    if n > 1000:
        n = 1000
    lut.SetNumberOfTableValues(n)
    lut.Build()
    for i in range(n):
        c = colors.colorMap(i, cmap, 0, n)
        lut.SetTableValue(i, c[0], c[1], c[2], 1-i/n*(1-alpha))

    arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    arr.SetName('pointcolors_'+cmap)
    poly.GetPointData().AddArray(arr)
    poly.GetPointData().SetActiveScalars('pointcolors_'+cmap)
    actor.GetMapper().SetScalarRange(vmin, vmax)
    actor.GetMapper().SetLookupTable(lut)
    actor.GetMapper().ScalarVisibilityOn()


def cellScalars(actor, scalars, name):
    """
    Set cell scalars to the polydata. Scalars can be a string name.
    """
    poly = polydata(actor, False)
    if isinstance(scalars, str):
        scalars = vtk_to_numpy(poly.GetPointData().GetArray(scalars))

    scalars = np.array(scalars) - np.min(scalars)
    scalars = scalars/np.max(scalars)
    if len(scalars) != poly.GetNumberOfCells():
        colors.printc('Number of scalars != nr. of cells', c=1)
        exit()
    arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    arr.SetName(name)
    poly.GetCellData().AddArray(arr)
    poly.GetCellData().SetActiveScalars(name)
    actor.GetMapper().ScalarVisibilityOn()


def cellColors(actor, scalars, cmap='jet', alpha=1):
    """
    Set individual cell colors by setting a scalar.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_coloring.py)    

    ![mcol](https://user-images.githubusercontent.com/32848391/46818965-c509da80-cd82-11e8-91fd-4c686da4a761.png)
    """
    poly = polydata(actor, False)
    if len(scalars) != poly.GetNumberOfCells():
        colors.printc('Number of scalars != nr. of cells', c=1)
        exit()

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(len(scalars))
    lut.Build()
    vmin, vmax = np.min(scalars), np.max(scalars)
    n = len(scalars)
    for i in range(n):
        c = colors.colorMap(i, cmap, 0, n)
        lut.SetTableValue(i, c[0], c[1], c[2], 1-i/n*(1-alpha))
    arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    arr.SetName('cellcolors_'+cmap)
    poly.GetCellData().AddArray(arr)
    poly.GetCellData().SetActiveScalars('cellcolors_'+cmap)
    actor.GetMapper().SetScalarRange(vmin, vmax)
    actor.GetMapper().SetLookupTable(lut)
    actor.GetMapper().ScalarVisibilityOn()


def scalars(actor, name=None):
    """
    Retrieve point or cell scalars using array name or index number.
    If no name is given return the list of names of existing arrays.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_coloring.py)    
    """
    poly = polydata(actor, False)

    if name is None:
        ncd = poly.GetCellData().GetNumberOfArrays()
        npd = poly.GetPointData().GetNumberOfArrays()
        arrs=[]
        for i in range(npd):
            arrs.append(poly.GetPointData().GetArrayName(i))
        for i in range(ncd):
            arrs.append(poly.GetCellData().GetArrayName(i))
        return arrs

    arr = poly.GetPointData().GetArray(name)
    if arr:
        if isinstance(name, int): 
            name = poly.GetPointData().GetArrayName(name)
        poly.GetPointData().SetActiveScalars(name)
        return vtk_to_numpy(arr)
    else:
        if isinstance(name, int): 
            name = poly.GetCellData().GetArrayName(name)
        arr = poly.GetCellData().GetArray(name)
        if arr:
            poly.GetCellData().SetActiveScalars(name)
            return vtk_to_numpy(arr)
    return None


def intersectWithLine(act, p0, p1):
    '''Return a list of points between p0 and p1 intersecting the actor.

    [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/spherical_harmonics1.py)    
    [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/spherical_harmonics2.py)    
    '''
    if not hasattr(act, 'line_locator'):
        line_locator = vtk.vtkOBBTree()
        line_locator.SetDataSet(polydata(act, True))
        line_locator.BuildLocator()
        setattr(act, 'line_locator', line_locator)

    intersectPoints = vtk.vtkPoints()
    intersection = [0, 0, 0]
    act.line_locator.IntersectWithLine(p0, p1, intersectPoints, None)
    pts = []
    for i in range(intersectPoints.GetNumberOfPoints()):
        intersectPoints.GetPoint(i, intersection)
        pts.append(list(intersection))
    return pts
   

def subdivide(actor, N=1, method=0, legend=None):
    '''Increase the number of points of actor surface mesh.

    Options:

        N = number of subdivisions

        method = 0, Loop

        method = 1, Linear

        method = 2, Adaptive

        method = 3, Butterfly

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py)    
    ![beeth](https://user-images.githubusercontent.com/32848391/46819341-ca1b5980-cd83-11e8-97b7-12b053d76aac.png)
    '''
    triangles = vtk.vtkTriangleFilter()
    setInput(triangles, polydata(actor))
    triangles.Update()
    originalMesh = triangles.GetOutput()
    if method == 0:
        sdf = vtk.vtkLoopSubdivisionFilter()
    elif method == 1:
        sdf = vtk.vtkLinearSubdivisionFilter()
    elif method == 2:
        sdf = vtk.vtkAdaptiveSubdivisionFilter()
    elif method == 3:
        sdf = vtk.vtkButterflySubdivisionFilter()
    else:
        colors.printc('Error in subdivide: unknown method.', c='r')
        exit(1)
    if method != 2:
        sdf.SetNumberOfSubdivisions(N)
    setInput(sdf, originalMesh)
    sdf.Update()
    out = sdf.GetOutput()
    if legend is None and hasattr(actor, 'legend'):
        legend = actor.legend
    sactor = makeActor(out, legend=legend)
    sactor.GetProperty().SetOpacity(actor.GetProperty().GetOpacity())
    sactor.GetProperty().SetColor(actor.GetProperty().GetColor())
    sactor.GetProperty().SetRepresentation(actor.GetProperty().GetRepresentation())
    return sactor


def decimate(actor, fraction=0.5, N=None, verbose=True, boundaries=True):
    '''
    Downsample the number of vertices in a mesh,

    fraction gives the desired target of reduction. 

    E.g. fraction=0.1 leaves 10% of the original nr of vertices.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/skeletonize.py)    
    '''
    poly = polydata(actor, True)
    if N:  # N = desired number of points
        Np = poly.GetNumberOfPoints()
        fraction = float(N)/Np
        if fraction >= 1:
            return actor

    decimate = vtk.vtkDecimatePro()
    setInput(decimate, poly)
    decimate.SetTargetReduction(1.-fraction)
    decimate.PreserveTopologyOff()
    if boundaries:
        decimate.BoundaryVertexDeletionOn()
    else:
        decimate.BoundaryVertexDeletionOff()
    decimate.Update()
    if verbose:
        print('Input nr. of pts:', poly.GetNumberOfPoints(), end='')
        print(' output:', decimate.GetOutput().GetNumberOfPoints())
    mapper = actor.GetMapper()
    setInput(mapper, decimate.GetOutput())
    mapper.Update()
    actor.Modified()
    if hasattr(actor, 'poly'):
        actor.poly = decimate.GetOutput()
    return actor  # return same obj for concatenation


def gaussNoise(actor, sigma):
    '''
    Add gaussian noise in percent of the diagonal size of actor.
    '''
    sz = diagonalSize(actor)
    pts = coordinates(actor)
    n = len(pts)
    ns = np.random.randn(n, 3)*sigma*sz/100
    vpts = vtk.vtkPoints()
    vpts.SetNumberOfPoints(n)
    vpts.SetData(numpy_to_vtk(pts+ns))
    actor.GetMapper().GetInput().SetPoints(vpts)
    actor.GetMapper().GetInput().GetPoints().Modified()
    return actor
    
    
def printInfo(obj):
    '''Print information about a vtkActor or vtkAssembly.'''

    def printvtkactor(actor, tab=''):
        poly = polydata(actor)
        pro = actor.GetProperty()
        pos = actor.GetPosition()
        bnds = actor.GetBounds()
        col = pro.GetColor()
        colr = to_precision(col[0], 3)
        colg = to_precision(col[1], 3)
        colb = to_precision(col[2], 3)
        alpha = pro.GetOpacity()
        npt = poly.GetNumberOfPoints()
        ncl = poly.GetNumberOfCells()

        print(tab, end='')
        colors.printc('vtkActor', c='g', bold=1, invert=1, dim=1, end=' ')

        if hasattr(actor, 'legend') and actor.legend:
            colors.printc('legend: ', c='g', bold=1, end='')
            colors.printc(actor.legend, c='g', bold=0)
        else:
            print()

        if hasattr(actor, 'filename'):
            colors.printc(tab+'           file: ', c='g', bold=1, end='')
            colors.printc(actor.filename, c='g', bold=0)

        colors.printc(tab+'          color: ', c='g', bold=1, end='')
        if actor.GetMapper().GetScalarVisibility():
            colors.printc('defined by point or cell data', c='g', bold=0)
        else:
            colors.printc(colors.getColorName(col) + ', rgb=('+colr+', '
                          + colg+', '+colb+'), alpha='+str(alpha), c='g', bold=0)

            if actor.GetBackfaceProperty():
                bcol = actor.GetBackfaceProperty().GetDiffuseColor()
                bcolr = to_precision(bcol[0], 3)
                bcolg = to_precision(bcol[1], 3)
                bcolb = to_precision(bcol[2], 3)
                colors.printc(tab+'     back color: ', c='g', bold=1, end='')
                colors.printc(colors.getColorName(bcol) + ', rgb=('+bcolr+', '
                              + bcolg+', ' + bcolb+')', c='g', bold=0)

        colors.printc(tab+'         points: ', c='g', bold=1, end='')
        colors.printc(npt, c='g', bold=0)

        colors.printc(tab+'          cells: ', c='g', bold=1, end='')
        colors.printc(ncl, c='g', bold=0)

        colors.printc(tab+'       position: ', c='g', bold=1, end='')
        colors.printc(pos, c='g', bold=0)

        colors.printc(tab+'     c. of mass: ', c='g', bold=1, end='')
        colors.printc(centerOfMass(poly), c='g', bold=0)

        colors.printc(tab+'      ave. size: ', c='g', bold=1, end='')
        colors.printc(to_precision(averageSize(poly), 4), c='g', bold=0)

        colors.printc(tab+'     diag. size: ', c='g', bold=1, end='')
        colors.printc(diagonalSize(poly), c='g', bold=0)

        colors.printc(tab+'         bounds: ', c='g', bold=1, end='')
        bx1, bx2 = to_precision(bnds[0], 3), to_precision(bnds[1], 3)
        colors.printc('x=('+bx1+', '+bx2+')', c='g', bold=0, end='')
        by1, by2 = to_precision(bnds[2], 3), to_precision(bnds[3], 3)
        colors.printc(' y=('+by1+', '+by2+')', c='g', bold=0, end='')
        bz1, bz2 = to_precision(bnds[4], 3), to_precision(bnds[5], 3)
        colors.printc(' z=('+bz1+', '+bz2+')', c='g', bold=0)

        colors.printc(tab+'           area: ', c='g', bold=1, end='')
        colors.printc(to_precision(area(poly), 8), c='g', bold=0)

        colors.printc(tab+'         volume: ', c='g', bold=1, end='')
        colors.printc(to_precision(volume(poly), 8), c='g', bold=0)

        arrtypes = dict()
        arrtypes[vtk.VTK_UNSIGNED_CHAR] = 'VTK_UNSIGNED_CHAR'
        arrtypes[vtk.VTK_UNSIGNED_INT] = 'VTK_UNSIGNED_INT'
        arrtypes[vtk.VTK_FLOAT] = 'VTK_FLOAT'
        arrtypes[vtk.VTK_DOUBLE] = 'VTK_DOUBLE'

        if poly.GetPointData():
            ptdata = poly.GetPointData()
            for i in range(ptdata.GetNumberOfArrays()):
                name = ptdata.GetArrayName(i)
                if name:
                    colors.printc(tab+'     point data: ',
                                  c='g', bold=1, end='')
                    try:
                        tt = arrtypes[ptdata.GetArray(i).GetDataType()]
                        colors.printc('name='+name, 'type='+tt, c='g', bold=0)
                    except:
                        tt = ptdata.GetArray(i).GetDataType()
                        colors.printc('name='+name, 'type=', tt, c='g', bold=0)

        if poly.GetCellData():
            cldata = poly.GetCellData()
            for i in range(cldata.GetNumberOfArrays()):
                name = cldata.GetArrayName(i)
                if name:
                    colors.printc(tab+'      cell data: ',
                                  c='g', bold=1, end='')
                    try:
                        tt = arrtypes[cldata.GetArray(i).GetDataType()]
                        colors.printc('name='+name, 'type='+tt, c='g', bold=0)
                    except:
                        tt = cldata.GetArray(i).GetDataType()
                        colors.printc('name='+name, 'type=', tt, c='g', bold=0)

    if not obj:
        colors.printc('Click an object and press i', c='y')
        return

    elif isinstance(obj, vtk.vtkActor):
        colors.printc('_'*60, c='g', bold=0)
        printvtkactor(obj)

    elif isinstance(obj, vtk.vtkAssembly):
        colors.printc('_'*60, c='g', bold=0)
        colors.printc('vtkAssembly', c='g', bold=1, invert=1, end=' ')
        if hasattr(obj, 'legend'):
            colors.printc('legend: ', c='g', bold=1, end='')
            colors.printc(obj.legend, c='g', bold=0)
        else:
            print()

        pos = obj.GetPosition()
        bnds = obj.GetBounds()
        colors.printc('          position: ', c='g', bold=1, end='')
        colors.printc(pos, c='g', bold=0)

        colors.printc('            bounds: ', c='g', bold=1, end='')
        bx1, bx2 = to_precision(bnds[0], 3), to_precision(bnds[1], 3)
        colors.printc('x=('+bx1+', '+bx2+')', c='g', bold=0, end='')
        by1, by2 = to_precision(bnds[2], 3), to_precision(bnds[3], 3)
        colors.printc(' y=('+by1+', '+by2+')', c='g', bold=0, end='')
        bz1, bz2 = to_precision(bnds[4], 3), to_precision(bnds[5], 3)
        colors.printc(' z=('+bz1+', '+bz2+')', c='g', bold=0)

        cl = vtk.vtkPropCollection()
        obj.GetActors(cl)
        cl.InitTraversal()
        for i in range(obj.GetNumberOfPaths()):
            act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
            if isinstance(act, vtk.vtkActor):
                printvtkactor(act, tab='     ')
    else:
        colors.printc('_'*60, c='g', bold=0)
        colors.printc(obj, c='g')
        colors.printc(type(obj), c='g', invert=1)


# ###########################################################################
def add_actor(f):
    '''decorator, internal use only'''
    def wrapper(*args, **kwargs):
        actor = f(*args, **kwargs)
        args[0].actors.append(actor)
        return actor
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper


# ###########################################################################
def assignConvenienceMethods(actor, legend):
    '''Set convenience methods to vtkActor object.'''
    if not hasattr(actor, 'legend'):
        setattr(actor, 'legend', legend)

    def _fclone(self, c=None, alpha=None, wire=False, bc=None,
                edges=False, legend=None, texture=None, rebuild=True, mirror=''):
        return clone(self, c, alpha, wire, bc, edges, legend, texture, rebuild, mirror)
    actor.clone = types.MethodType(_fclone, actor)

    def _fpoint(self, i, p=None):
        if p is None:
            poly = polydata(self, True, 0)
            p = [0, 0, 0]
            poly.GetPoints().GetPoint(i, p)
            return np.array(p)
        else:
            poly = polydata(self, False, 0)
            poly.GetPoints().SetPoint(i, p)
            TI = vtk.vtkTransform()
            actor.SetUserMatrix(TI.GetMatrix())  # reset
        return self
    actor.point = types.MethodType(_fpoint, actor)

    def _fN(self, index=0):
        return polydata(self, False, index).GetNumberOfPoints()
    actor.N = types.MethodType(_fN, actor)

    def _fnormalize(self): return normalize(self)
    actor.normalize = types.MethodType(_fnormalize, actor)

    def _fshrink(self, fraction=0.85): return shrink(self, fraction)
    actor.shrink = types.MethodType(_fshrink, actor)

    def _fcutPlane(self, origin=(0, 0, 0), normal=(1, 0, 0), showcut=False):
        return cutPlane(self, origin, normal, showcut)
    actor.cutPlane = types.MethodType(_fcutPlane, actor)

    def _fpolydata(self, rebuild=True, index=0):
        return polydata(self, rebuild, index)
    actor.polydata = types.MethodType(_fpolydata, actor)

    def _fcoordinates(self, rebuild=True, copy=True):
        return coordinates(self, rebuild, copy)
    actor.coordinates = types.MethodType(_fcoordinates, actor)

    def _fxbounds(self):
        b = polydata(actor, True).GetBounds()
        return (b[0], b[1])
    actor.xbounds = types.MethodType(_fxbounds, actor)

    def _fybounds(self):
        b = polydata(actor, True).GetBounds()
        return (b[2], b[3])
    actor.ybounds = types.MethodType(_fybounds, actor)

    def _fzbounds(self):
        b = polydata(actor, True).GetBounds()
        return (b[4], b[5])
    actor.zbounds = types.MethodType(_fzbounds, actor)

    def _fnormalAt(self, index):
        normals = polydata(self, True).GetPointData().GetNormals()
        return np.array(normals.GetTuple(index))
    actor.normalAt = types.MethodType(_fnormalAt, actor)

    def _fnormals(self):
        vtknormals = polydata(self, True).GetPointData().GetNormals()
        as_numpy = vtk_to_numpy(vtknormals)
        return as_numpy
    actor.normals = types.MethodType(_fnormals, actor)

    def _fstretch(self, startpt, endpt):
        return stretch(self, startpt, endpt)
    actor.stretch = types.MethodType(_fstretch, actor)

    def _fsubdivide(self, N=1, method=0, legend=None):
        return subdivide(self, N, method, legend)
    actor.subdivide = types.MethodType(_fsubdivide, actor)

    def _fdecimate(self, fraction=0.5, N=None, verbose=True, boundaries=True):
        return decimate(self, fraction, N, verbose, boundaries)
    actor.decimate = types.MethodType(_fdecimate, actor)

    def _fcolor(self, c=None):
        if c is not None:
            self.GetProperty().SetColor(colors.getColor(c))
            return self
        else:
            return np.array(self.GetProperty().GetColor())
    actor.color = types.MethodType(_fcolor, actor)

    def _falpha(self, a=None):
        if a is not None:
            self.GetProperty().SetOpacity(a)
            return self
        else:
            return self.GetProperty().GetOpacity()
    actor.alpha = types.MethodType(_falpha, actor)

    def _fwire(self, a=True):
        if a:
            self.GetProperty().SetRepresentationToWireframe()
        else:
            self.GetProperty().SetRepresentationToSurface()
        return self
    actor.wire = types.MethodType(_fwire, actor)

    def _fclosestPoint(self, pt, N=1, radius=None):
        return closestPoint(self, pt, N, radius)
    actor.closestPoint = types.MethodType(_fclosestPoint, actor)

    def _fintersectWithLine(self, p0, p1):
        return intersectWithLine(self, p0, p1)
    actor.intersectWithLine = types.MethodType(_fintersectWithLine, actor)

    def _fclean(self, tol=None):
        return clean(self, tol)
    actor.clean = types.MethodType(_fclean, actor)

    def _fisInside(self, point, tol=0.0001):
        return isInside(self, point, tol)
    actor.isInside = types.MethodType(_fisInside, actor)

    def _finsidePoints(self, points, invert=False, tol=1e-05):
        return insidePoints(self, points, invert, tol)
    actor.insidePoints = types.MethodType(_finsidePoints, actor)

    def _fcellCenters(self):
        return cellCenters(self)
    actor.cellCenters = types.MethodType(_fcellCenters, actor)

    def _fpointScalars(self, scalars, name):
        return pointScalars(self, scalars, name)
    actor.pointScalars = types.MethodType(_fpointScalars, actor)

    def _fpointColors(self, scalars, cmap='jet', alpha=1):
        return pointColors(self, scalars, cmap, alpha)
    actor.pointColors = types.MethodType(_fpointColors, actor)

    def _fcellScalars(self, scalars, name):
        return cellScalars(self, scalars, name)
    actor.cellScalars = types.MethodType(_fcellScalars, actor)

    def _fcellColors(self, scalars, cmap='jet', alpha=1):
        return cellColors(self, scalars, cmap, alpha)
    actor.cellColors = types.MethodType(_fcellColors, actor)

    def _fscalars(self, name=None):
        return scalars(self, name)
    actor.scalars = types.MethodType(_fscalars, actor)

    def _fpointSize(self, s):
        if isinstance(self, vtk.vtkAssembly):
            cl = vtk.vtkPropCollection()
            self.GetActors(cl)
            cl.InitTraversal()
            a = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
            a.GetProperty().SetRepresentationToPoints()
            a.GetProperty().SetPointSize(s)
        else:
            self.GetProperty().SetRepresentationToPoints()
            self.GetProperty().SetPointSize(s)
        return self
    actor.pointSize = types.MethodType(_fpointSize, actor)

    def _flineWidth(self, lw):
        self.GetProperty().SetLineWidth(lw)
        return self
    actor.lineWidth = types.MethodType(_flineWidth, actor)

# ###########################################################################
def assignPhysicsMethods(actor):
    '''Set convenient physics methods to vtkActor object.'''

    def _fpos(self, p=None):
        if p is None:
            return np.array(self.GetPosition())
        self.SetPosition(p)
        return self  # return itself to concatenate methods
    actor.pos = types.MethodType(_fpos, actor)

    def _faddPos(self, dp):
        self.SetPosition(np.array(self.GetPosition()) + dp)
        return self
    actor.addPos = types.MethodType(_faddPos, actor)

    def _fpx(self, px=None):               # X
        _pos = self.GetPosition()
        if px is None:
            return _pos[0]
        newp = [px, _pos[1], _pos[2]]
        self.SetPosition(newp)
        return self
    actor.x = types.MethodType(_fpx, actor)

    def _fpy(self, py=None):               # Y
        _pos = self.GetPosition()
        if py is None:
            return _pos[1]
        newp = [_pos[0], py, _pos[2]]
        self.SetPosition(newp)
        return self
    actor.y = types.MethodType(_fpy, actor)

    def _fpz(self, pz=None):               # Z
        _pos = self.GetPosition()
        if pz is None:
            return _pos[2]
        newp = [_pos[0], _pos[1], pz]
        self.SetPosition(newp)
        return self
    actor.z = types.MethodType(_fpz, actor)

    if not hasattr(actor, '_time'):
        setattr(actor, '_time', 0.0)
    def _ftime(self, t=None):
        if t is None:
            return self._time
        self._time = t
        return self  # return itself to concatenate methods
    actor.time = types.MethodType(_ftime, actor)

    def _fscale(self, p=None):
        if p is None:
            return np.array(self.GetScale())
        self.SetScale(p)
        return self  # return itself to concatenate methods
    actor.scale = types.MethodType(_fscale, actor)

    def _frotate(self, angle, axis, axis_point=[0, 0, 0], rad=False):
        if rad:
            angle *= 57.3
        return rotate(self, angle, axis, axis_point, rad)
    actor.rotate = types.MethodType(_frotate, actor)

    def _frotateX(self, angle, axis_point=[0, 0, 0], rad=False):
        if rad:
            angle *= 57.3
        return rotate(self, angle, [1, 0, 0], axis_point, rad)
    actor.rotateX = types.MethodType(_frotateX, actor)

    def _frotateY(self, angle, axis_point=[0, 0, 0], rad=False):
        if rad:
            angle *= 57.3
        return rotate(self, angle, [0, 1, 0], axis_point, rad)
    actor.rotateY = types.MethodType(_frotateY, actor)

    def _frotateZ(self, angle, axis_point=[0, 0, 0], rad=False):
        if rad:
            angle *= 57.3
        return rotate(self, angle, [0, 0, 1], axis_point, rad)
    actor.rotateZ = types.MethodType(_frotateZ, actor)

    def _forientation(self, newaxis=None, rotation=0):
        return orientation(self, newaxis, rotation)
    actor.orientation = types.MethodType(_forientation, actor)

    def _fcenterOfMass(self): return centerOfMass(self)
    actor.centerOfMass = types.MethodType(_fcenterOfMass, actor)

    def _fvolume(self): return volume(self)
    actor.volume = types.MethodType(_fvolume, actor)

    def _farea(self): return area(self)
    actor.area = types.MethodType(_farea, actor)

    def _fdiagonalSize(self): return diagonalSize(self)
    actor.diagonalSize = types.MethodType(_fdiagonalSize, actor)

    def _fgaussNoise(self, sigma): return gaussNoise(self, sigma)
    actor.gaussNoise = types.MethodType(_fgaussNoise, actor)












