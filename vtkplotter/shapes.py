from __future__ import division, print_function
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
import vtkplotter.utils as utils
import vtkplotter.colors as colors
from vtkplotter.actors import Actor, Assembly
import vtkplotter.docs as docs

__doc__="""
Submodule to generate basic geometric shapes.
"""+docs._defs

__all__ = [
    'point',
    'points',
    'line',
    'tube',
    'lines',
    'ribbon',
    'arrow',
    'arrows',
    'polygon',
    'rectangle',
    'disc',
    'sphere',
    'spheres',
    'earth',
    'ellipsoid',
    'grid',
    'plane',
    'box',
    'cube',
    'helix',
    'cylinder',
    'cone',
    'pyramid',
    'torus',
    'paraboloid',
    'hyperboloid',
    'text',
    'glyph',
]


########################################################################
def point(pos=[0,0,0], r=10, c='gray', alpha=1):
    '''Create a simple point actor.'''
    return points([pos], r, c, alpha)

def points(plist, r=5, c='gray', alpha=1):
    '''
    Build a point ``Actor`` for a list of points.

    :param float r: point radius.
    :param c: color name, number, or list of [R,G,B] colors of same length as plist.
    :type c: int, str, list
    :param float alpha: transparency in range [0,1].

    .. hint:: |lorenz| |lorenz.py|_
    '''
    n = len(plist)
    if n == 0:
        return None
    elif n == 3: # assume plist is in the format [all_x, all_y, all_z]
        if utils.isSequence(plist[0]) and len(plist[0]) > 3:
            plist = list(zip(plist[0], plist[1], plist[2]))
    elif n == 2: # assume plist is in the format [all_x, all_y, 0]
        if utils.isSequence(plist[0]) and len(plist[0]) > 3:
            plist = list(zip(plist[0], plist[1], [0]*len(plist[0])))

    if utils.isSequence(c) and utils.isSequence(c[0]) and len(c[0])==3:
        return _colorPoints(plist, c, r, alpha)

    n = len(plist) # refresh
    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(n)
    src.Update()
    pd = src.GetOutput()
    if n == 1:  # passing just one point
        pd.GetPoints().SetPoint(0, [0, 0, 0])
    else:
        pd.GetPoints().SetData(numpy_to_vtk(plist, deep=True))
    actor = Actor(pd, c, alpha)
    actor.GetProperty().SetPointSize(r)
    if n == 1:
        actor.SetPosition(plist[0])
    return actor

def _colorPoints(plist, cols, r, alpha):
    n = len(plist)
    if n > len(cols):
        colors.printc("Error: mismatch in colorPoints()", n, len(cols), c=1)
        exit()
    if n != len(cols):
        colors.printc("Warning: mismatch in colorPoints()", n, len(cols))
    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(n)
    src.Update()
    vgf = vtk.vtkVertexGlyphFilter()
    vgf.SetInputData(src.GetOutput())
    vgf.Update()
    pd = vgf.GetOutput()
    ucols = vtk.vtkUnsignedCharArray()
    ucols.SetNumberOfComponents(3)
    ucols.SetName("pointsRGB")
    for i, p in enumerate(plist):
        c = np.array(colors.getColor(cols[i]))*255
        ucols.InsertNextTuple3(c[0], c[1], c[2])
    pd.GetPoints().SetData(numpy_to_vtk(plist, deep=True))
    pd.GetPointData().SetScalars(ucols)
    actor = Actor(pd, c, alpha) 
    actor.mapper.ScalarVisibilityOn()
    actor.GetProperty().SetInterpolationToFlat()
    actor.GetProperty().SetPointSize(r)
    return actor


def glyph(actor, glyphObj, orientationArray='', scaleByVectorSize=False, 
          c='gold', alpha=1):
    """
    At each vertex of a mesh, another mesh - a `'glyph'` - is shown with
    various orientation options and coloring.
    
    :param orientationArray: list of vectors, ``vtkAbstractArray`` 
        or the name of an already existing points array. 
    :type orientationArray: list, str, vtkAbstractArray
    :param bool scaleByVectorSize: glyph mesh is scaled by the size of
        the vectors.
    
    .. hint:: |glyphs| |glyphs.py|_
    """    
    if isinstance(glyphObj, Actor):
        glyphObj = glyphObj.polydata()
    
    gly = vtk.vtkGlyph3D()
    gly.SetInputData(actor.polydata())
    gly.SetSourceData(glyphObj)
    gly.SetColorModeToColorByScalar()

    if orientationArray!='':
        gly.OrientOn()
        gly.SetScaleFactor(1)

        if scaleByVectorSize:
            gly.SetScaleModeToScaleByVector()
        else:
            gly.SetScaleModeToDataScalingOff()

        if orientationArray == 'normals' or orientationArray == 'Normals':
            gly.SetVectorModeToUseNormal()
        elif isinstance(orientationArray, vtk.vtkAbstractArray):
            actor.GetMapper().GetInput().GetPointData().AddArray(orientationArray)
            actor.GetMapper().GetInput().GetPointData().SetActiveVectors('glyph_vectors')
            gly.SetInputArrayToProcess(0,0,0, 0, 'glyph_vectors')
            gly.SetVectorModeToUseVector()
        elif utils.isSequence(orientationArray): # passing a list
            actor.addPointField(orientationArray, 'glyph_vectors')
            gly.SetInputArrayToProcess(0,0,0, 0, 'glyph_vectors') 
            gly.SetVectorModeToUseVector()
        else: # passing a name
            gly.SetInputArrayToProcess(0,0,0, 0, orientationArray) 
            gly.SetVectorModeToUseVector()

    gly.Update()
    pd = gly.GetOutput()

    #    if utils.isSequence(c) and len(c) != 3:
    #        ucols = vtk.vtkUnsignedCharArray()
    #        ucols.SetNumberOfComponents(3)
    #        ucols.SetName("glyphRGB")
    #        for col in c:
    #            cl = np.array(colors.getColor(col))*255
    #            ucols.InsertNextTuple3(cl[0], cl[1], cl[2])
    #        pd.GetPointData().SetScalars(ucols)
    #        c = None
        
    actor = Actor(pd, c, alpha) 
    actor.GetProperty().SetInterpolationToFlat()
    return actor



def line(p0, p1=None, lw=1, c='r', alpha=1, dotted=False):
    '''
    Build the line segment between points `p0` and `p1`.
    If `p0` is a list of points returns the line connecting them.

    :param lw: line width.
    :param c: color name, number, or list of [R,G,B] colors.
    :type c: int, str, list
    :param float alpha: transparency in range [0,1].
    :param bool dotted: draw a dotted line
    '''
    # detect if user is passing a list of points:
    if utils.isSequence(p0[0]):
        ppoints = vtk.vtkPoints()  # Generate the polyline
        dim = len((p0[0]))
        if dim == 2:
            for i in range(len(p0)):
                p = p0[i]
                ppoints.InsertPoint(i, p[0], p[1], 0)
        else:
            ppoints.SetData(numpy_to_vtk(p0, deep=True))
        lines = vtk.vtkCellArray()  # Create the polyline.
        lines.InsertNextCell(len(p0))
        for i in range(len(p0)):
            lines.InsertCellPoint(i)
        poly = vtk.vtkPolyData()
        poly.SetPoints(ppoints)
        poly.SetLines(lines)
    else:  # or just 2 points to link
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(p0)
        lineSource.SetPoint2(p1)
        lineSource.Update()
        poly = lineSource.GetOutput()

    actor = Actor(poly, c, alpha)
    actor.GetProperty().SetLineWidth(lw)
    if dotted:
        actor.GetProperty().SetLineStipplePattern(0xf0f0)
        actor.GetProperty().SetLineStippleRepeatFactor(1)
    actor.base = np.array(p0)
    actor.top = np.array(p1)
    return actor


def tube(points, r=1, c='r', alpha=1, res=12):
    '''Build a tube along the line defined by a set of points.

    :param r: constant radius or list of radii.
    :type r: float, list
    :param c: constant color or list of colors for each point.
    :type c: float, list
    
    .. hint:: |ribbon| |ribbon.py|_
    
        |tube| |tube.py|_
    '''
    ppoints = vtk.vtkPoints()  # Generate the polyline
    ppoints.SetData(numpy_to_vtk(points, deep=True))
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(points))
    for i in range(len(points)):
        lines.InsertCellPoint(i)
    polyln = vtk.vtkPolyData()
    polyln.SetPoints(ppoints)
    polyln.SetLines(lines)

    tuf = vtk.vtkTubeFilter()
    tuf.CappingOn()
    tuf.SetNumberOfSides(res)
    tuf.SetInputData(polyln)
    if utils.isSequence(r):
        arr = numpy_to_vtk(np.ascontiguousarray(r), deep=True)
        arr.SetName('TubeRadius')
        polyln.GetPointData().AddArray(arr)
        polyln.GetPointData().SetActiveScalars('TubeRadius')
        tuf.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    else:
        tuf.SetRadius(r)

    usingColScals = False
    if utils.isSequence(c) and len(c)!=3:
        usingColScals = True
        cc = vtk.vtkUnsignedCharArray()
        cc.SetName("TubeColors")
        cc.SetNumberOfComponents(3)
        cc.SetNumberOfTuples(len(c))
        for i,ic in enumerate(c):
            r,g,b = colors.getColor(ic)
            cc.InsertTuple3(i, int(255*r), int(255*g), int(255*b))
        polyln.GetPointData().AddArray(cc)
        c = None

    tuf.Update()
    polytu = tuf.GetOutput()
 
    actor = Actor(polytu, c=c, alpha=alpha, computeNormals=0)
    if usingColScals:
        actor.mapper.SetScalarModeToUsePointFieldData()
        actor.mapper.ScalarVisibilityOn()
        actor.mapper.SelectColorArray("TubeColors")
        actor.mapper.Modified()
        
    actor.base = np.array(points[0])
    actor.top  = np.array(points[-1])
    return actor


def lines(plist0, plist1=None, lw=1, c='r', alpha=1, dotted=False):
    '''
    Build the line segments between two lists of points `plist0` and `plist1`.
    `plist0` can be also passed in the form ``[[point1, point2], ...]``.

    .. hint:: |fitspheres2.py|_    
    '''
    if plist1 is not None:
        plist0 = list(zip(plist0, plist1))

    polylns = vtk.vtkAppendPolyData()
    for twopts in plist0:
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(twopts[0])
        lineSource.SetPoint2(twopts[1])
        polylns.AddInputConnection(lineSource.GetOutputPort())
    polylns.Update()

    actor = Actor(polylns.GetOutput(), c, alpha)
    actor.GetProperty().SetLineWidth(lw)
    if dotted:
        actor.GetProperty().SetLineStipplePattern(0xf0f0)
        actor.GetProperty().SetLineStippleRepeatFactor(1)
    return actor


def ribbon(line1, line2, c='m', alpha=1, res=(200,5)):
    '''Connect two lines to generate the surface inbetween.

    .. hint:: |ribbon| |ribbon.py|_    
    '''
    if isinstance(line1, Actor):
        line1 = line1.coordinates()
    if isinstance(line2, Actor):
        line2 = line2.coordinates()

    ppoints1 = vtk.vtkPoints()  # Generate the polyline1
    ppoints1.SetData(numpy_to_vtk(line1, deep=True))
    lines1 = vtk.vtkCellArray()
    lines1.InsertNextCell(len(line1))
    for i in range(len(line1)):
        lines1.InsertCellPoint(i)
    poly1 = vtk.vtkPolyData()
    poly1.SetPoints(ppoints1)
    poly1.SetLines(lines1)

    ppoints2 = vtk.vtkPoints()  # Generate the polyline2
    ppoints2.SetData(numpy_to_vtk(line2, deep=True))
    lines2 = vtk.vtkCellArray()
    lines2.InsertNextCell(len(line2))
    for i in range(len(line2)):
        lines2.InsertCellPoint(i)
    poly2 = vtk.vtkPolyData()
    poly2.SetPoints(ppoints2)
    poly2.SetLines(lines2)

    # build the lines
    lines1 = vtk.vtkCellArray()
    lines1.InsertNextCell(poly1.GetNumberOfPoints())
    for i in range(poly1.GetNumberOfPoints()):
        lines1.InsertCellPoint(i)

    polygon1 = vtk.vtkPolyData()
    polygon1.SetPoints(ppoints1)
    polygon1.SetLines(lines1)

    lines2 = vtk.vtkCellArray()
    lines2.InsertNextCell(poly2.GetNumberOfPoints())
    for i in range(poly2.GetNumberOfPoints()):
        lines2.InsertCellPoint(i)

    polygon2 = vtk.vtkPolyData()
    polygon2.SetPoints(ppoints2)
    polygon2.SetLines(lines2)

    mergedPolyData = vtk.vtkAppendPolyData()
    mergedPolyData.AddInputData(polygon1)
    mergedPolyData.AddInputData(polygon2)
    mergedPolyData.Update()

    rsf = vtk.vtkRuledSurfaceFilter()
    rsf.CloseSurfaceOff()
    rsf.SetRuledModeToResample()
    rsf.SetResolution(res[0], res[1])
    rsf.SetInputData(mergedPolyData.GetOutput())
    rsf.Update()
    return Actor(rsf.GetOutput(), c=c, alpha=alpha)


def arrow(startPoint, endPoint, s=None, c='r', alpha=1,
          res=12, rwSize=(800,800)):
    '''
    Build a 3D arrow from `startPoint` to `endPoint` of section size `s`,
    expressed as the fraction of the window size.
    
    .. note:: If ``s=None`` the arrow is scaled proportionally to its length,
              otherwise it represents the fraction of the window size.
              
    |OrientedArrow|
    '''
    axis = np.array(endPoint) - np.array(startPoint)
    length = np.linalg.norm(axis)
    if length:        
        axis = axis/length
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    arr = vtk.vtkArrowSource()
    arr.SetShaftResolution(res)
    arr.SetTipResolution(res)
    if s:
        sz = 0.02
        arr.SetTipRadius(sz)
        arr.SetShaftRadius(sz/1.75)
        arr.SetTipLength(sz*15)
    arr.Update()
    t = vtk.vtkTransform()
    t.RotateZ(phi*57.3)
    t.RotateY(theta*57.3)
    t.RotateY(-90)  # put it along Z
    if s:
        w, h = rwSize
        sz = (w+h)/2*s
        t.Scale(length, sz, sz)
    else:
        t.Scale(length, length, length)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(arr.GetOutput())
    tf.SetTransform(t)
    tf.Update()

    actor = Actor(tf.GetOutput(), c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(startPoint)
    actor.DragableOff()
    actor.PickableOff()
    actor.base = np.array(startPoint)
    actor.top = np.array(endPoint)
    return actor


def arrows(startPoints, endPoints=None,
           s=None, c='r', alpha=1, res=8, rwSize=(800,800)):
    '''
    Build arrows between two lists of points `startPoints` and `endPoints`.
    `startPoints` can be also passed in the form ``[[point1, point2], ...]``.
    '''
    if endPoints is not None:
        startPoints = list(zip(startPoints, endPoints))

    polyapp = vtk.vtkAppendPolyData()
    for twopts in startPoints:
        startPoint, endPoint = twopts
        axis = np.array(endPoint) - np.array(startPoint)
        length = np.linalg.norm(axis)
        if length:
            axis = axis/length
        theta = np.arccos(axis[2])
        phi = np.arctan2(axis[1], axis[0])
        arr = vtk.vtkArrowSource()
        arr.SetShaftResolution(res)
        arr.SetTipResolution(res)
        if s:
            sz = 0.02
            arr.SetTipRadius(sz)
            arr.SetShaftRadius(sz/1.75)
            arr.SetTipLength(sz*15)
        t = vtk.vtkTransform()
        t.Translate(startPoint)
        t.RotateZ(phi*57.3)
        t.RotateY(theta*57.3)
        t.RotateY(-90)  # put it along Z
        if s and rwSize:
            w, h = rwSize
            sz = (w+h)/2*s
            t.Scale(length, sz, sz)
        else:
            t.Scale(length, length, length)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(arr.GetOutputPort())
        tf.SetTransform(t)
        polyapp.AddInputConnection(tf.GetOutputPort())

    polyapp.Update()
    actor = Actor(polyapp.GetOutput(), c, alpha)
    return actor


def polygon(pos=[0, 0, 0], normal=[0, 0, 1], nsides=6, r=1,
            c='coral', bc='darkgreen', lw=1, alpha=1, followcam=False):
    '''
    Build a 2D polygon of `nsides` of radius `r` oriented as `normal`.

    :param followcam: if `True` the text will auto-orient itself to the active camera.
        A ``vtkCamera`` object can also be passed.
    :type followcam: bool, vtkCamera  
    
    |Polygon|
    '''
    ps = vtk.vtkRegularPolygonSource()
    ps.SetNumberOfSides(nsides)
    ps.SetRadius(r)
    ps.SetNormal(-np.array(normal))
    ps.Update()

    tf = vtk.vtkTriangleFilter()
    tf.SetInputConnection(ps.GetOutputPort())
    tf.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tf.GetOutputPort())
    if followcam:
        import vtkplotter.plotter as plt
        actor = vtk.vtkFollower()
        if isinstance(followcam, vtk.vtkCamera):
            actor.SetCamera(followcam)
        else:
            actor.SetCamera(plt._plotter_instance.camera)
    else:
        actor = Actor()

    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.getColor(c))
    # check if color string contains a float, in this case ignore alpha
    al = colors._getAlpha(c)
    if al:
        alpha = al
    actor.GetProperty().SetOpacity(alpha)
    actor.GetProperty().SetLineWidth(lw)
    actor.GetProperty().SetInterpolationToFlat()
    if bc:  # defines a specific color for the backface
        backProp = vtk.vtkProperty()
        backProp.SetDiffuseColor(colors.getColor(bc))
        backProp.SetOpacity(alpha)
        actor.SetBackfaceProperty(backProp)
    actor.SetPosition(pos)
    return actor


def rectangle(p1=[0,0,0], p2=[2,1,0],
              c='k', bc='dg', lw=1, alpha=1, texture=None):
    '''Build a rectangle in the xy plane identified by two corner points.'''
    p1 = np.array(p1)
    p2 = np.array(p2)
    pos = (p1+p2)/2
    length = abs(p2[0]-p1[0])
    height = abs(p2[1]-p1[1])
    rec = plane(pos, [0,0,-1], length, height, c, bc, alpha, None, texture)
    return rec
    
    
def disc(pos=[0, 0, 0], normal=[0, 0, 1], r1=0.5, r2=1,
         c='coral', bc='darkgreen', lw=1, alpha=1,res=12):
    '''
    Build a 2D disc of internal radius `r1` and outer radius `r2`,
    oriented perpendicular to `normal`.
    
    |Disk|
    '''
    ps = vtk.vtkDiskSource()
    ps.SetInnerRadius(r1)
    ps.SetOuterRadius(r2)
    ps.SetRadialResolution(res)
    ps.SetCircumferentialResolution(res*6) # ~2pi
    ps.Update()

    axis = np.array(normal)/np.linalg.norm(normal)
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(ps.GetOutput())
    tf.SetTransform(t)
    tf.Update()

    pd = tf.GetOutput()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)

    actor = Actor()# vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.getColor(c))
    # check if color string contains a float, in this case ignore alpha
    al = colors._getAlpha(c)
    if al:
        alpha = al
    actor.GetProperty().SetOpacity(alpha)
    actor.GetProperty().SetLineWidth(lw)
    actor.GetProperty().SetInterpolationToFlat()
    if bc:  # defines a specific color for the backface
        backProp = vtk.vtkProperty()
        backProp.SetDiffuseColor(colors.getColor(bc))
        backProp.SetOpacity(alpha)
        actor.SetBackfaceProperty(backProp)
    actor.SetPosition(pos)
    return actor


def sphere(pos=[0, 0, 0], r=1, c='r', alpha=1, res=24):
    '''Build a sphere at position `pos` of radius `r`.
    
    |Sphere|
    '''
    ss = vtk.vtkSphereSource()
    ss.SetRadius(r)
    ss.SetThetaResolution(2*res)
    ss.SetPhiResolution(res)
    ss.Update()
    pd = ss.GetOutput()
    actor = Actor(pd, c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    return actor


def spheres(centers, r=1, c='r', alpha=1, res=8):
    '''
    Build a (possibly large) set of spheres at `centers` of radius `r`.

    Either `c` or `r` can be a list of RGB colors or radii.

    .. hint:: |manyspheres| |manyspheres.py|_    
    '''

    cisseq = False
    if utils.isSequence(c):
        cisseq = True

    if cisseq:
        if len(centers) > len(c):
            colors.printc("Mismatch in spheres() colors", len(centers), len(c), c=1)
            exit()
        if len(centers) != len(c):
            colors.printc("Warning: mismatch in spheres() colors", len(centers), len(c))

    risseq = False
    if utils.isSequence(r):
        risseq = True

    if risseq:
        if len(centers) > len(r):
            colors.printc("Mismatch in spheres() radius", len(centers), len(r), c=1)
            exit()
        if len(centers) != len(r):
            colors.printc("Warning: mismatch in spheres() radius", len(centers), len(r))
    if cisseq and risseq:
        colors.printc("Limitation: c and r cannot be both sequences.", c=1)
        exit()

    src = vtk.vtkSphereSource()
    if not risseq:
        src.SetRadius(r)
    src.SetPhiResolution(res)
    src.SetThetaResolution(2*res)
    src.Update()
    
    psrc = vtk.vtkPointSource()
    psrc.SetNumberOfPoints(len(centers))
    psrc.Update()
    pd = psrc.GetOutput()
    vpts = pd.GetPoints()

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(src.GetOutputPort())

    if cisseq:
        glyph.SetColorModeToColorByScalar()
        ucols = vtk.vtkUnsignedCharArray()
        ucols.SetNumberOfComponents(3)
        ucols.SetName("colors")
        for i, p in enumerate(centers):
            vpts.SetPoint(i, p)
            cc = np.array(colors.getColor(c[i]))*255
            ucols.InsertNextTuple3(cc[0], cc[1], cc[2])
            pd.GetPointData().SetScalars(ucols)
            glyph.ScalingOff()
    elif risseq:
        glyph.SetScaleModeToScaleByScalar()
        urads = vtk.vtkFloatArray()
        urads.SetName("scales")
        for i, p in enumerate(centers):
            vpts.SetPoint(i, p)
            urads.InsertNextValue(r[i])
        pd.GetPointData().SetScalars(urads)
    else:
        for i, p in enumerate(centers):
            vpts.SetPoint(i, p)

    glyph.SetInputData(pd)
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(glyph.GetOutput())

    actor = Actor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToPhong()
    if cisseq:
        mapper.ScalarVisibilityOn()
    else:
        mapper.ScalarVisibilityOff()
        actor.GetProperty().SetColor(colors.getColor(c))
    # check if color string contains a float, in this case ignore alpha
    al = colors._getAlpha(c)
    if al:
        alpha = al
    actor.GetProperty().SetOpacity(alpha)
    return actor


def earth(pos=[0, 0, 0], r=1, lw=1):
    '''Build a textured actor representing the Earth.

    .. hint:: |geodesic| |geodesic.py|_    
    '''
    import os
    tss = vtk.vtkTexturedSphereSource()
    tss.SetRadius(r)
    tss.SetThetaResolution(72)
    tss.SetPhiResolution(36)
    earthMapper = vtk.vtkPolyDataMapper()
    earthMapper.SetInputConnection(tss.GetOutputPort())
    earthActor = Actor(c='w')
    earthActor.SetMapper(earthMapper)
    atext = vtk.vtkTexture()
    pnmReader = vtk.vtkPNMReader()
    cdir = os.path.dirname(__file__)
    if cdir == '':
        cdir = '.'
    fn = cdir + '/textures/earth.ppm'
    pnmReader.SetFileName(fn)
    atext.SetInputConnection(pnmReader.GetOutputPort())
    atext.InterpolateOn()
    earthActor.SetTexture(atext)
    if not lw:
        earthActor.SetPosition(pos)
        return earthActor
    es = vtk.vtkEarthSource()
    es.SetRadius(r/.995)
    earth2Mapper = vtk.vtkPolyDataMapper()
    earth2Mapper.SetInputConnection(es.GetOutputPort())
    earth2Actor = Actor()# vtk.vtkActor()
    earth2Actor.SetMapper(earth2Mapper)
    earth2Mapper.ScalarVisibilityOff()
    earth2Actor.GetProperty().SetLineWidth(lw)
    ass = Assembly([earthActor, earth2Actor])
    ass.SetPosition(pos)
    return ass


def ellipsoid(pos=[0, 0, 0], axis1=[1, 0, 0], axis2=[0, 2, 0], axis3=[0, 0, 3],
              c='c', alpha=1, res=24):
    """
    Build a 3D ellipsoid centered at position `pos`.
    
    .. note:: `axis1` and `axis2` are only used to define sizes and one azimuth angle.
    """
    elliSource = vtk.vtkSphereSource()
    elliSource.SetThetaResolution(res)
    elliSource.SetPhiResolution(res)
    elliSource.Update()
    l1 = np.linalg.norm(axis1)
    l2 = np.linalg.norm(axis2)
    l3 = np.linalg.norm(axis3)
    axis1 = np.array(axis1)/l1
    axis2 = np.array(axis2)/l2
    axis3 = np.array(axis3)/l3
    angle = np.arcsin(np.dot(axis1, axis2))
    theta = np.arccos(axis3[2])
    phi = np.arctan2(axis3[1], axis3[0])

    t = vtk.vtkTransform()
    t.PostMultiply()
    t.Scale(l1, l2, l3)
    t.RotateX(angle*57.3)
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(elliSource.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = Actor(pd, c=c, alpha=alpha)
    actor.GetProperty().BackfaceCullingOn()
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    actor.base = -np.array(axis1)/2 + pos
    actor.top  =  np.array(axis1)/2 + pos
    return actor


def grid(pos=[0, 0, 0], normal=[0, 0, 1], sx=1, sy=1, c='g', bc='darkgreen',
         lw=1, alpha=1, resx=10, resy=10):
    '''Return a grid plane.

    .. hint:: |brownian2D| |brownian2D.py|_    
    '''
    ps = vtk.vtkPlaneSource()
    ps.SetResolution(resx, resy)
    ps.Update()
    poly0 = ps.GetOutput()
    t0 = vtk.vtkTransform()
    t0.Scale(sx, sy, 1)
    tf0 = vtk.vtkTransformPolyDataFilter()
    tf0.SetInputData(poly0)
    tf0.SetTransform(t0)
    tf0.Update()
    poly = tf0.GetOutput()
    axis = np.array(normal)/np.linalg.norm(normal)
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(poly)
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()
    actor = Actor(pd, c=c, bc=bc, alpha=alpha)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(lw)
    actor.SetPosition(pos)
    actor.PickableOff()
    return actor


def plane(pos=[0, 0, 0], normal=[0, 0, 1], sx=1, sy=None, c='g', bc='darkgreen',
          alpha=1, texture=None):
    '''
    Draw a plane of size `sx` and `sy` oriented perpendicular to vector `normal`
    and so that it passes through point `pos`.
    
    |Plane| 
    '''
    if sy is None:
        sy = sx
    ps = vtk.vtkPlaneSource()
    ps.SetResolution(1, 1)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(ps.GetOutputPort())
    tri.Update()
    poly = tri.GetOutput()
    axis = np.array(normal)/np.linalg.norm(normal)
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.Scale(sx, sy, 1)
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(poly)
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()
    actor = Actor(pd, c=c, bc=bc, alpha=alpha, texture=texture)
    actor.SetPosition(pos)
    return actor


def box(pos=[0, 0, 0], length=1, width=2, height=3, normal=(0, 0, 1),
        c='g', alpha=1, texture=None):
    '''Build a box of dimensions `x=length, y=width and z=height` oriented along vector `normal`.

    .. hint:: |aspring| |aspring.py|_    
    '''
    src = vtk.vtkCubeSource()
    src.SetXLength(length)
    src.SetYLength(width)
    src.SetZLength(height)
    src.Update()
    poly = src.GetOutput()

    axis = np.array(normal)/np.linalg.norm(normal)
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(poly)
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = Actor(pd, c, alpha, texture=texture)
    actor.SetPosition(pos)
    return actor


def cube(pos=[0, 0, 0], length=1, normal=(0, 0, 1),
         c='g', alpha=1., texture=None):
    '''Build a cube of size `length` oriented along vector `normal`.

    .. hint:: |colorcubes| |colorcubes.py|_    
    '''
    return box(pos, length, length, length, normal, c, alpha, texture)


def helix(startPoint=[0, 0, 0], endPoint=[1, 0, 0], coils=20, r=0.1, r2=None,
          thickness=None, c='grey', alpha=1):
    '''
    Build a spring of specified nr of `coils` between `startPoint` and `endPoint`.

    :param int coils: number of coils
    :param float r: radius at start point
    :param float r2: radius at end point
    :param float thickness: thickness of the coil section

    .. hint:: |aspring| |aspring.py|_    
    '''
    diff = endPoint-np.array(startPoint)
    length = np.linalg.norm(diff)
    if not length:
        return None
    if not r:
        r = length/20
    trange = np.linspace(0, length, num=50*coils)
    om = 6.283*(coils-.5)/length
    if not r2:
        r2=r 
    pts= []
    for t in trange:
        f = (length-t)/length
        rd = r*f + r2*(1-f)
        pts.append([rd*np.cos(om*t), rd*np.sin(om*t), t])
    
    pts = [[0, 0, 0]] + pts + [[0, 0, length]]
    diff = diff/length
    theta = np.arccos(diff[2])
    phi = np.arctan2(diff[1], diff[0])
    sp = line(pts).polydata(False)
    t = vtk.vtkTransform()
    t.RotateZ(phi*57.3)
    t.RotateY(theta*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(sp)
    tf.SetTransform(t)
    tf.Update()
    tuf = vtk.vtkTubeFilter()
    tuf.SetNumberOfSides(12)
    tuf.CappingOn()
    tuf.SetInputData(tf.GetOutput())
    if not thickness:
        thickness = r/10
    tuf.SetRadius(thickness)
    tuf.Update()
    poly = tuf.GetOutput()
    actor = Actor(poly, c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(startPoint)
    actor.base = np.array(startPoint)
    actor.top = np.array(endPoint)
    return actor


def cylinder(pos=[0,0,0], r=1, height=1, axis=[0, 0, 1], c='teal', alpha=1, res=24):
    '''
    Build a cylinder of specified height and radius `r`, centered at `pos`.

    If `pos` is a list of 2 points, e.g. `pos=[v1,v2]`, build a cylinder with base
    centered at `v1` and top at `v2`.
    
    |Cylinder|
    '''

    if utils.isSequence(pos[0]):  # assume user is passing pos=[base, top]
        base = np.array(pos[0])
        top = np.array(pos[1])
        pos = (base+top)/2
        height = np.linalg.norm(top-base)
        axis = top-base
        axis = utils.norm(axis)
    else:
        axis = utils.norm(axis)
        base = pos - axis*height/2
        top = pos + axis*height/2

    cyl = vtk.vtkCylinderSource()
    cyl.SetResolution(res)
    cyl.SetRadius(r)
    cyl.SetHeight(height)
    cyl.Update()

    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateX(90)  # put it along Z
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(cyl.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = Actor(pd, c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    actor.base = base + pos
    actor.top = top + pos
    return actor


def cone(pos=[0, 0, 0], r=1, height=3, axis=[0, 0, 1],
         c='dg', alpha=1, res=48):
    '''
    Build a cone of specified radius `r` and `height`, centered at `pos`.
    
    |Cone|
    '''
    con = vtk.vtkConeSource()
    con.SetResolution(res)
    con.SetRadius(r)
    con.SetHeight(height)
    con.SetDirection(axis)
    con.Update()
    actor = Actor(con.GetOutput(), c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    v = utils.norm(axis)*height/2
    actor.base = pos - v
    actor.top = pos + v
    return actor

def pyramid(pos=[0, 0, 0], s=1, height=1, axis=[0, 0, 1],
            c='dg', alpha=1):
    '''
    Build a pyramid of specified base size `s` and `height`, centered at `pos`.
    '''
    return cone(pos, s, height, axis, c, alpha, 4)


def torus(pos=[0,0,0], r=1, thickness=0.1, axis=[0,0,1], c='khaki', alpha=1, res=30):
    '''
    Build a torus of specified outer radius `r` internal radius `thickness`, centered at `pos`.

    .. hint:: |gas| |gas.py|_    
    '''
    rs = vtk.vtkParametricTorus()
    rs.SetRingRadius(r)
    rs.SetCrossSectionRadius(thickness)
    pfs = vtk.vtkParametricFunctionSource()
    pfs.SetParametricFunction(rs)
    pfs.SetUResolution(res*3)
    pfs.SetVResolution(res)
    pfs.Update()

    nax = np.linalg.norm(axis)
    if nax:
        axis = np.array(axis)/nax
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(pfs.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = Actor(pd, c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    return actor


def paraboloid(pos=[0, 0, 0], r=1, height=1, axis=[0, 0, 1], c='cyan', alpha=1, res=50):
    '''
    Build a paraboloid of specified height and radius `r`, centered at `pos`.
    
    .. note::
        Full volumetric expression is:
            :math:`F(x,y,z)=a_0x^2+a_1y^2+a_2z^2+a_3xy+a_4yz+a_5xz+ a_6x+a_7y+a_8z+a_9`

            |paraboloid|  
    '''
    quadric = vtk.vtkQuadric()
    quadric.SetCoefficients(1, 1, 0, 0, 0, 0, 0, 0, height/4, 0)
    # F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2
    #         + a3*x*y + a4*y*z + a5*x*z
    #         + a6*x   + a7*y   + a8*z  +a9
    sample = vtk.vtkSampleFunction()
    sample.SetSampleDimensions(res, res, res)
    sample.SetImplicitFunction(quadric)

    contours = vtk.vtkContourFilter()
    contours.SetInputConnection(sample.GetOutputPort())
    contours.GenerateValues(1, .01, .01)
    contours.Update()

    axis = np.array(axis)/np.linalg.norm(axis)
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    t.Scale(r, r, r)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(contours.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = Actor(pd, c, alpha)
    actor.mirror('n')
    actor.GetProperty().SetInterpolationToPhong()
    actor.mapper.ScalarVisibilityOff()
    actor.SetPosition(pos)
    return actor


def hyperboloid(pos=[0, 0, 0], a2=1, value=0.5, height=1, axis=[0, 0, 1],
                c='magenta', alpha=1, res=100):
    '''
    Build a hyperboloid of specified aperture `a2` and `height`, centered at `pos`.
    
    Full volumetric expression is:
        :math:`F(x,y,z)=a_0x^2+a_1y^2+a_2z^2+a_3xy+a_4yz+a_5xz+ a_6x+a_7y+a_8z+a_9`

    .. hint:: |mesh_bands| |mesh_bands.py|_    
    '''
    q = vtk.vtkQuadric()
    q.SetCoefficients(2, 2, -1/a2, 0, 0, 0, 0, 0, 0, 0)
    # F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2
    #         + a3*x*y + a4*y*z + a5*x*z
    #         + a6*x   + a7*y   + a8*z  +a9
    sample = vtk.vtkSampleFunction()
    sample.SetSampleDimensions(res, res, res)
    sample.SetImplicitFunction(q)

    contours = vtk.vtkContourFilter()
    contours.SetInputConnection(sample.GetOutputPort())
    contours.GenerateValues(1, value, value)
    contours.Update()

    axis = np.array(axis)/np.linalg.norm(axis)
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    t.Scale(1, 1, height)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(contours.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = Actor(pd, c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.mapper.ScalarVisibilityOff()
    actor.SetPosition(pos)
    return actor


def text(txt, pos=3, normal=(0, 0, 1), s=1, depth=0.1, justify='bottom-left',
         c=(0.7,0.7,0.7), alpha=1, bc=None, bg=None, font='arial', followcam=False):
    '''
    Returns a ``vtkActor`` that shows a 3D text.

    :param pos: position in 3D space, 
                if an integer is passed [1,8], place a 2D text in one of the 4 corners.
    :type pos: list, int
    :param float s: size of text.
    :param float depth: text thickness.
    :param str justify: text justification 
        (bottom-left, bottom-right, top-left, top-right, centered).
    :param bg: background color of corner annotations. Only applies of `pos` is ``int``.
    :param str font: either `arial`, `courier` or `times`. Only applies of `pos` is ``int``.
    :param followcam: if `True` the text will auto-orient itself to the active camera.
        A ``vtkCamera`` object can also be passed.
    :type followcam: bool, vtkCamera  
    
    .. hint:: |colorcubes| |colorcubes.py|_ 
    
        |markpoint| |markpoint.py|_
    
        |annotations.py|_ Allows to read a text file and show it in the rendering window.
    '''
    if isinstance(pos, int):
        if pos>8: pos=8
        if pos<1: pos=1
        ca = vtk.vtkCornerAnnotation()
        ca.SetNonlinearFontScaleFactor(s/3)
        ca.SetText(pos-1, str(txt))
        ca.PickableOff()
        cap = ca.GetTextProperty()
        cap.SetColor(colors.getColor(c))
        if font.lower() == 'courier':
            cap.SetFontFamilyToCourier()
        elif font.lower() == 'times':
            cap.SetFontFamilyToTimes()
        else:
            cap.SetFontFamilyToArial()
        if bg:
            bgcol = colors.getColor(bg)
            cap.SetBackgroundColor(bgcol)
            cap.SetBackgroundOpacity(alpha*0.5)
            cap.SetFrameColor(bgcol)
            cap.FrameOn()
        return ca

    tt = vtk.vtkVectorText()
    tt.SetText(str(txt))
    tt.Update()
    ttmapper = vtk.vtkPolyDataMapper()
    if followcam:
        depth = 0
        normal = (0, 0, 1)
    if depth:
        extrude = vtk.vtkLinearExtrusionFilter()
        extrude.SetInputConnection(tt.GetOutputPort())
        extrude.SetExtrusionTypeToVectorExtrusion()
        extrude.SetVector(0, 0, 1)
        extrude.SetScaleFactor(depth)
        ttmapper.SetInputConnection(extrude.GetOutputPort())
    else:
        ttmapper.SetInputConnection(tt.GetOutputPort())
    if followcam:
        import vtkplotter.plotter as plt
        ttactor = vtk.vtkFollower()
        if isinstance(followcam, vtk.vtkCamera):
            ttactor.SetCamera(followcam)
        else:
            ttactor.SetCamera(plt._plotter_instance.camera)
    else:
        ttactor = Actor()
    ttactor.SetMapper(ttmapper)
    ttactor.GetProperty().SetColor(colors.getColor(c))
    ttmapper.Update()
    
    bb = tt.GetOutput().GetBounds()
    dx, dy = (bb[1]-bb[0])/2*s, (bb[3]-bb[2])/2*s
    cm = np.array([(bb[1]+bb[0])/2,(bb[3]+bb[2])/2,(bb[5]+bb[4])/2])*s
    shift = -cm
    if 'cent' in justify:
        pass
    elif 'bottom-left' in justify:
        shift += np.array([dx,dy,0])
    elif 'top-left' in justify:
        shift += np.array([dx,-dy,0])
    elif 'bottom-right' in justify:
        shift += np.array([-dx,dy,0])
    elif 'top-right' in justify:
        shift += np.array([-dx,-dy,0])
    else:
        colors.printc("text(): Unknown justify type", justify, c=1)
         
    # check if color string contains a float, in this case ignore alpha
    al = colors._getAlpha(c)
    if al:
        alpha = al
    ttactor.GetProperty().SetOpacity(alpha)

    nax = np.linalg.norm(normal)
    if nax:
        normal = np.array(normal)/nax
    theta = np.arccos(normal[2])
    phi = np.arctan2(normal[1], normal[0])
    ttactor.SetScale(s, s, s)
    ttactor.RotateZ(phi*57.3)
    ttactor.RotateY(theta*57.3)
    ttactor.SetPosition(pos+shift)
    if bc:  # defines a specific color for the backface
        backProp = vtk.vtkProperty()
        backProp.SetDiffuseColor(colors.getColor(bc))
        backProp.SetOpacity(alpha)
        ttactor.SetBackfaceProperty(backProp)
    ttactor.PickableOff()
    return ttactor
