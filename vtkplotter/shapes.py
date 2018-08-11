from __future__ import division, print_function
import vtk
import numpy as np

import vtkplotter.utils as vu
import vtkplotter.colors as vc
import vtkplotter.vtkio as vio


########################################################################
def points(plist, c='b', tags=[], r=5, alpha=1, legend=None):
    '''
    Build a vtkActor for a list of points.

    c can be a list of [R,G,B] colors of same length as plist
    
    If tags (a list of strings) is specified, is displayed along 
    with the points.
    '''

    if len(plist) == 0: return None
    if vu.isSequence(c) and vu.isSequence(c[0]):
        return _colorPoints(plist, c, r, alpha, legend)

    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(len(plist))
    src.Update()
    pd = src.GetOutput()
    if len(plist) == 1: #passing just one point
        pd.GetPoints().SetPoint(0, [0,0,0])
    else:
        for i,p in enumerate(plist): 
            pd.GetPoints().SetPoint(i, p)
    actor = vu.makeActor(pd, c, alpha)
    actor.GetProperty().SetPointSize(r)
    if len(plist) == 1: actor.SetPosition(plist[0])

    if legend: setattr(actor, 'legend', legend)
    return actor

def _colorPoints(plist, cols, r, alpha, legend):
    if len(plist) > len(cols):
        vio.printc(("Mismatch in colorPoints()", len(plist), len(cols)), 1)
        exit()
    if len(plist) != len(cols):
        vio.printc(("Warning: mismatch in colorPoints()", len(plist), len(cols)))
    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(len(plist))
    src.Update()
    vertexFilter = vtk.vtkVertexGlyphFilter()
    vu.setInput(vertexFilter, src.GetOutput())
    vertexFilter.Update()
    pd = vertexFilter.GetOutput()
    ucols = vtk.vtkUnsignedCharArray()
    ucols.SetNumberOfComponents(3)
    ucols.SetName("RGB")
    for i,p in enumerate(plist):
        pd.GetPoints().SetPoint(i, p)
        c = np.array(vc.getColor(cols[i]))*255
        if vu.vtkMV:
            ucols.InsertNextTuple3(c[0],c[1],c[2])
        else:
            ucols.InsertNextTupleValue(c)
    pd.GetPointData().SetScalars(ucols)
    mapper = vtk.vtkPolyDataMapper()
    vu.setInput(mapper, pd)
    mapper.ScalarVisibilityOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToFlat()
    # check if color string contains a float, in this case ignore alpha
    al = vc.getAlpha(c)
    if al: alpha = al
    actor.GetProperty().SetOpacity(alpha)
    actor.GetProperty().SetPointSize(r)
    return actor


def line(p0, p1=None, lw=1, tube=False, dotted=False,
            c='r', alpha=1., legend=None):
    '''Build the line segment between points p0 and p1.
        
        if p0 is a list of points returns the line connecting them.
        
        if tube=True, lines are rendered as tubes of radius lw
    '''

    #detect if user is passing a list of points:
    if vu.isSequence(p0[0]):
        ppoints = vtk.vtkPoints() # Generate the polyline
        poly = vtk.vtkPolyData()
        for i in range(len(p0)):
            p = p0[i]
            ppoints.InsertPoint(i, p[0],p[1],p[2])
        lines = vtk.vtkCellArray() # Create the polyline.
        lines.InsertNextCell(len(p0))
        for i in range(len(p0)): lines.InsertCellPoint(i)
        poly.SetPoints(ppoints)
        poly.SetLines(lines)
    else: # or just 2 points to link
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(p0)
        lineSource.SetPoint2(p1)
        lineSource.Update()
        poly = lineSource.GetOutput()

    if tube:
        tuf = vtk.vtkTubeFilter()
        tuf.SetNumberOfSides(12)
        #tuf.CappingOn()
        vu.setInput(tuf, poly)
        tuf.SetRadius(lw)
        tuf.Update()
        poly= tuf.GetOutput()
        actor = vu.makeActor(poly, c, alpha, legend=legend)
        actor.GetProperty().SetInterpolationToPhong()
    else:
        actor = vu.makeActor(poly, c, alpha, legend=legend)
        actor.GetProperty().SetLineWidth(lw)
        if dotted:
            actor.GetProperty().SetLineStipplePattern(0xf0f0)
            actor.GetProperty().SetLineStippleRepeatFactor(1)
    setattr(actor, 'base', np.array(p0))
    setattr(actor, 'top',  np.array(p1))
    return actor


def lines(plist0, plist1=None, lw=1, dotted=False,
            c='r', alpha=1, legend=None):
    '''Build the line segments between two lists of points plist0 and plist1.
        plist0 can be also passed in the form [[point1, point2], ...]
    '''        
    if plist1 is not None:
        plist0 = list(zip(plist0,plist1))
        
    polylns = vtk.vtkAppendPolyData()
    for twopts in plist0:
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(twopts[0])
        lineSource.SetPoint2(twopts[1])
        polylns.AddInputConnection(lineSource.GetOutputPort())
    polylns.Update()
    
    actor = vu.makeActor(polylns.GetOutput(), c, alpha, legend=legend)
    actor.GetProperty().SetLineWidth(lw)
    if dotted:
        actor.GetProperty().SetLineStipplePattern(0xf0f0)
        actor.GetProperty().SetLineStippleRepeatFactor(1)
    return actor



def arrow(startPoint, endPoint, c, s=None, alpha=1,
          legend=None, texture=None, res=12, rwSize=None):
    '''Build a 3D arrow from startPoint to endPoint of section size s,
    expressed as the fraction of the window size.
    If s=None the arrow is scaled proportionally to its length.'''

    axis = np.array(endPoint) - np.array(startPoint)
    length = np.linalg.norm(axis)
    if not length: return None
    axis = axis/length
    theta = np.arccos(axis[2])
    phi   = np.arctan2(axis[1], axis[0])
    arr = vtk.vtkArrowSource()
    arr.SetShaftResolution(res) 
    arr.SetTipResolution(res)
    if s: 
        sz=0.02
        arr.SetTipRadius(sz)
        arr.SetShaftRadius(sz/1.75)
        arr.SetTipLength(sz*15)
    arr.Update()
    t = vtk.vtkTransform()
    t.RotateZ(phi*57.3)
    t.RotateY(theta*57.3)
    t.RotateY(-90) #put it along Z
    if s: 
        w,h = rwSize
        sz = (w+h)/2*s
        t.Scale(length,sz,sz)
    else:
        t.Scale(length,length,length)
    tf = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf, arr.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    
    actor = vu.makeActor(tf.GetOutput(),
                        c, alpha, legend=legend, texture=texture)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(startPoint)
    actor.DragableOff()
    actor.PickableOff()
    setattr(actor, 'base', np.array(startPoint))
    setattr(actor, 'top',  np.array(endPoint))
    return actor


def arrows(startPoints, endPoints=None,
            c='r', s=None, alpha=1, legend=None, res=8, rwSize=None):
    '''Build arrows between two lists of points startPoints and endPoints.
        startPoints can be also passed in the form [[point1, point2], ...]
    '''        
    if endPoints is not None:
        startPoints = list(zip(startPoints,endPoints))
    
    polyapp = vtk.vtkAppendPolyData()
    for twopts in startPoints:
        startPoint, endPoint = twopts
        axis = np.array(endPoint) - np.array(startPoint)
        length = np.linalg.norm(axis)
        if not length: return None
        axis = axis/length
        theta = np.arccos(axis[2])
        phi   = np.arctan2(axis[1], axis[0])
        arr = vtk.vtkArrowSource()
        arr.SetShaftResolution(res) 
        arr.SetTipResolution(res)
        if s: 
            sz=0.02
            arr.SetTipRadius(sz)
            arr.SetShaftRadius(sz/1.75)
            arr.SetTipLength(sz*15)
        t = vtk.vtkTransform()
        t.Translate(startPoint)###
        t.RotateZ(phi*57.3)
        t.RotateY(theta*57.3)
        t.RotateY(-90) #put it along Z
        if s: 
            w,h = rwSize
            sz = (w+h)/2*s
            t.Scale(length,sz,sz)
        else:
            t.Scale(length,length,length)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(arr.GetOutputPort())
        tf.SetTransform(t)
        polyapp.AddInputConnection(tf.GetOutputPort())
    
    polyapp.Update()
    actor = vu.makeActor(polyapp.GetOutput(), c, alpha, legend=legend)
    return actor   


def polygon(pos=[0,0,0], normal=[0,0,1], nsides=6, r=1,
            c='coral', bc='darkgreen', lw=1, alpha=1,
            legend=None, texture=None, followcam=False, camera=None):
    '''Build a 2D polygon of nsides of radius r oriented as normal
    
    If followcam=True the polygon will always reorient itself to current camera.
    '''
    ps = vtk.vtkRegularPolygonSource()
    ps.SetNumberOfSides(nsides)
    ps.SetRadius(r)
    ps.SetNormal(-np.array(normal))
    ps.Update()

    tf = vtk.vtkTriangleFilter()
    vu.setInput(tf, ps.GetOutputPort())
    tf.Update()

    mapper = vtk.vtkPolyDataMapper()
    vu.setInput(mapper, tf.GetOutputPort())
    if followcam: #follow cam
        actor = vtk.vtkFollower()
        actor.SetCamera(camera)
        if not camera:
            vio.printc('Warning: vtkCamera does not yet exist for polygon',5)
    else:
        actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(vc.getColor(c))
    # check if color string contains a float, in this case ignore alpha
    al = vc.getAlpha(c)
    if al: alpha = al
    actor.GetProperty().SetOpacity(alpha)
    actor.GetProperty().SetLineWidth(lw)
    actor.GetProperty().SetInterpolationToFlat()
    if bc: # defines a specific color for the backface
        backProp = vtk.vtkProperty()
        backProp.SetDiffuseColor(vc.getColor(bc))
        backProp.SetOpacity(alpha)
        actor.SetBackfaceProperty(backProp)
    if texture: vu.assignTexture(actor, texture)
    vu.assignPhysicsMethods(actor)
    vu.assignConvenienceMethods(actor, legend)
    actor.SetPosition(pos)
    return actor


def disc(pos=[0,0,0], normal=[0,0,1], r1=0.5, r2=1,
         c='coral', bc='darkgreen', lw=1, alpha=1, 
         legend=None, texture=None, res=12):
    '''Build a 2D disc of internal radius r1 and outer radius r2,
    oriented perpendicular to normal'''
    ps = vtk.vtkDiskSource()
    ps.SetInnerRadius(r1)
    ps.SetOuterRadius(r2)
    ps.SetRadialResolution(res)
    ps.SetCircumferentialResolution(res*4)
    ps.Update()
    tr = vtk.vtkTriangleFilter()
    vu.setInput(tr, ps.GetOutputPort())
    tr.Update()

    axis  = np.array(normal)/np.linalg.norm(normal)
    theta = np.arccos(axis[2])
    phi   = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf, tr.GetOutput())
    tf.SetTransform(t)
    tf.Update()

    pd = tf.GetOutput()
    mapper = vtk.vtkPolyDataMapper()
    vu.setInput(mapper, pd)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(vc.getColor(c))
    # check if color string contains a float, in this case ignore alpha
    al = vc.getAlpha(c)
    if al: alpha = al
    actor.GetProperty().SetOpacity(alpha)
    actor.GetProperty().SetLineWidth(lw)
    actor.GetProperty().SetInterpolationToFlat()
    if bc: # defines a specific color for the backface
        backProp = vtk.vtkProperty()
        backProp.SetDiffuseColor(vc.getColor(bc))
        backProp.SetOpacity(alpha)
        actor.SetBackfaceProperty(backProp)
    if texture: vu.assignTexture(actor, texture)
    vu.assignPhysicsMethods(actor)
    vu.assignConvenienceMethods(actor, legend)
    actor.SetPosition(pos)
    return actor


def sphere(pos=[0,0,0], r=1,
            c='r', alpha=1, wire=False, legend=None, texture=None, res=24):
    '''Build a sphere at position pos of radius r.'''
    ss = vtk.vtkSphereSource()
    ss.SetRadius(r)
    ss.SetThetaResolution(res)
    ss.SetPhiResolution(res)
    ss.Update()
    pd = ss.GetOutput()
    actor = vu.makeActor(pd, c, alpha, wire, legend=legend, texture=texture)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    return actor


def spheres(centers, r=1,
            c='r', alpha=1, wire=False, legend=None, texture=None, res=8):
    '''
    Build a (possibly large) set of spheres at centers of radius r.
    
    Either c or r can be a list of RGB colors or radii.
    '''

    cisseq=False
    if vu.isSequence(c): cisseq=True

    if cisseq:
        if len(centers) > len(c):
            vio.printc(("Mismatch in spheres() colors", len(centers), len(c)), 1)
            exit()
        if len(centers) != len(c):
            vio.printc(("Warning: mismatch in spheres() colors", len(centers), len(c)))
            
    risseq=False
    if vu.isSequence(r): risseq=True

    if risseq:
        if len(centers) > len(r):
            vio.printc(("Mismatch in spheres() radius", len(centers), len(r)), 1)
            exit()
        if len(centers) != len(r):
            vio.printc(("Warning: mismatch in spheres() radius", len(centers), len(r)))
    if cisseq and risseq:
        vio.printc("Limitation: c and r cannot be both sequences.",1)
        exit()

    src = vtk.vtkSphereSource()
    if not risseq: src.SetRadius(r)
    src.SetPhiResolution(res)
    src.SetThetaResolution(res)
    src.Update()
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(src.GetOutputPort())

    psrc = vtk.vtkPointSource()
    psrc.SetNumberOfPoints(len(centers))
    psrc.Update()
    pd = psrc.GetOutput()
    vpts = pd.GetPoints()
    
    if cisseq:
        glyph.SetColorModeToColorByScalar()
        ucols = vtk.vtkUnsignedCharArray()
        ucols.SetNumberOfComponents(3)
        ucols.SetName("colors")
        for i,p in enumerate(centers):
            vpts.SetPoint(i, p)
            cc = np.array(vc.getColor(c[i]))*255
            if vu.vtkMV:
                ucols.InsertNextTuple3(cc[0],cc[1],cc[2])
            else:
                ucols.InsertNextTupleValue(cc)            
            pd.GetPointData().SetScalars(ucols)
            glyph.ScalingOff()
    elif risseq:
        glyph.SetScaleModeToScaleByScalar()
        urads = vtk.vtkFloatArray()
        urads.SetName("scales")
        for i,p in enumerate(centers):
            vpts.SetPoint(i, p)
            urads.InsertNextValue(r[i])            
        pd.GetPointData().SetScalars(urads)
    else:
        for i,p in enumerate(centers): vpts.SetPoint(i, p)        

    vu.setInput(glyph, pd)
    glyph.Update()
    
    mapper = vtk.vtkPolyDataMapper()
    vu.setInput(mapper, glyph.GetOutput())
    if cisseq: mapper.ScalarVisibilityOn()
    else: mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToPhong()
    # check if color string contains a float, in this case ignore alpha
    al = vc.getAlpha(c)
    if al: alpha = al
    actor.GetProperty().SetOpacity(alpha)
    if not cisseq: 
        if texture is not None:
            vu.assignTexture(actor, texture)
            mapper.ScalarVisibilityOff()
        else:
            actor.GetProperty().SetColor(vc.getColor(c))
    vu.assignConvenienceMethods(actor, legend)
    return actor


def ellipsoid(pos=[0,0,0], axis1=[1,0,0], axis2=[0,2,0], axis3=[0,0,3],
              c='c', alpha=1, legend=None, texture=None, res=24):
    """
    Build a 3D ellipsoid centered at position pos.
    Axis1 and axis2 are only used to define sizes and one azimuth angle
    """
    elliSource = vtk.vtkSphereSource()
    elliSource.SetThetaResolution(res)
    elliSource.SetPhiResolution(res)
    elliSource.Update()
    l1 = np.linalg.norm(axis1)
    l2 = np.linalg.norm(axis2)
    l3 = np.linalg.norm(axis3)
    axis1  = np.array(axis1)/l1
    axis2  = np.array(axis2)/l2
    axis3  = np.array(axis3)/l3
    angle = np.arcsin(np.dot(axis1,axis2))
    theta = np.arccos(axis3[2])
    phi   = np.arctan2(axis3[1], axis3[0])

    t = vtk.vtkTransform()
    t.PostMultiply()
    t.Scale(l1,l2,l3)
    t.RotateX(angle*57.3)
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf, elliSource.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor= vu.makeActor(pd, c=c, alpha=alpha, legend=legend, texture=texture)
    actor.GetProperty().BackfaceCullingOn()
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    return actor


def grid(pos=[0,0,0], normal=[0,0,1], sx=1, sy=1, c='g', bc='darkgreen',
         lw=1, alpha=1, legend=None, resx=10, resy=10):
    '''Return a grid plane'''
    ps = vtk.vtkPlaneSource()
    ps.SetResolution(resx, resy)
    ps.Update()
    poly0 = ps.GetOutput()
    t0 = vtk.vtkTransform()
    t0.Scale(sx,sy,1)
    tf0 = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf0, poly0)
    tf0.SetTransform(t0)
    tf0.Update()
    poly = tf0.GetOutput()
    axis  = np.array(normal)/np.linalg.norm(normal)
    theta = np.arccos(axis[2])
    phi   = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf, poly)
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()
    actor = vu.makeActor(pd, c=c, bc=bc, alpha=alpha, legend=legend)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(lw)
    actor.SetPosition(pos)
    actor.PickableOff()
    return actor


def plane(pos=[0,0,0], normal=[0,0,1], sx=1, sy=None, c='g', bc='darkgreen',
          alpha=1, legend=None, texture=None):
    '''
    Draw a plane of size sx and sy oriented perpendicular to vector normal  
    and so that it passes through point pos.
    '''
    if sy is None: sy=sx
    ps = vtk.vtkPlaneSource()
    ps.SetResolution(1, 1)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(ps.GetOutputPort())
    tri.Update()
    poly = tri.GetOutput()
    axis  = np.array(normal)/np.linalg.norm(normal)
    theta = np.arccos(axis[2])
    phi   = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.Scale(sx,sy,1)
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf, poly)
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()
    actor = vu.makeActor(pd, c=c, bc=bc, alpha=alpha, legend=legend, texture=texture)
    actor.SetPosition(pos)
    actor.PickableOff()
    return actor


def box(pos=[0,0,0], length=1, width=2, height=3, normal=(0,0,1),
        c='g', alpha=1, wire=False, legend=None, texture=None):
    '''Build a box of dimensions x=length, y=width and z=height
    oriented along vector normal'''
    src = vtk.vtkCubeSource()
    src.SetXLength(length)
    src.SetYLength(width)
    src.SetZLength(height)
    src.Update()
    poly = src.GetOutput()

    axis  = np.array(normal)/np.linalg.norm(normal)
    theta = np.arccos(axis[2])
    phi   = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)

    tf = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf, poly)
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = vu.makeActor(pd, c, alpha, wire, legend=legend, texture=texture)
    actor.SetPosition(pos)
    return actor


def helix(startPoint=[0,0,0], endPoint=[1,1,1], coils=20, r=None,
          thickness=None, c='grey', alpha=1, legend=None, texture=None):
    '''
    Build a spring actor of specified nr of coils between startPoint and endPoint
    '''
    diff = endPoint-np.array(startPoint)
    length = np.linalg.norm(diff)
    if not length: return None
    if not r: r=length/20
    trange = np.linspace(0, length, num=50*coils)
    om = 6.283*(coils-.5)/length
    pts = [ [r*np.cos(om*t),r*np.sin(om*t),t] for t in trange ]
    pts = [ [0,0,0] ] + pts + [ [0, 0, length] ]
    diff = diff/length
    theta = np.arccos(diff[2])
    phi   = np.arctan2(diff[1], diff[0])
    sp = vu.polydata(line(pts), False)
    t = vtk.vtkTransform()
    t.RotateZ(phi*57.3)
    t.RotateY(theta*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf, sp)
    tf.SetTransform(t)
    tf.Update()
    tuf = vtk.vtkTubeFilter()
    tuf.SetNumberOfSides(12)
    tuf.CappingOn()
    vu.setInput(tuf, tf.GetOutput())
    if not thickness: thickness = r/10
    tuf.SetRadius(thickness)
    tuf.Update()
    poly = tuf.GetOutput()
    actor = vu.makeActor(poly, c, alpha, legend=legend, texture=texture)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(startPoint)
    setattr(actor, 'base',np.array(startPoint))
    setattr(actor, 'top', np.array(endPoint))
    return actor


def cylinder(pos=[0,0,0], r=1, height=1, axis=[0,0,1],
             c='teal', wire=0, alpha=1, edges=False, 
             legend=None, texture=None, res=24):
    '''
    Build a cylinder of specified height and radius r, centered at pos.
    
    If pos is a list of 2 points, e.g. pos=[v1,v2], build a cylinder with base
    centered at v1 and top at v2.
    '''
    
    if vu.isSequence(pos[0]): # assume user is passing pos=[base, top]
        base = np.array(pos[0])
        top  = np.array(pos[1])
        pos = (base+top)/2
        height = np.linalg.norm(top-base)
        axis = top-base
        axis  = vu.norm(axis)
    else:
        axis  = vu.norm(axis)
        base = pos - axis*height/2
        top  = pos + axis*height/2

    cyl = vtk.vtkCylinderSource()
    cyl.SetResolution(res)
    cyl.SetRadius(r)
    cyl.SetHeight(height)
    cyl.Update()

    theta = np.arccos(axis[2])
    phi   = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateX(90) #put it along Z
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf, cyl.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = vu.makeActor(pd, c, alpha, wire, edges=edges,
                         legend=legend, texture=texture)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    setattr(actor,'base',base)
    setattr(actor,'top', top)
    return actor


def cone(pos=[0,0,0], r=1, height=1, axis=[0,0,1],
         c='dg', alpha=1, legend=None, texture=None, res=48):
    '''
    Build a cone of specified radius r and height, centered at pos.
    '''
    con = vtk.vtkConeSource()
    con.SetResolution(res)
    con.SetRadius(r)
    con.SetHeight(height)
    con.SetDirection(axis)
    con.Update()
    actor = vu.makeActor(con.GetOutput(), c, alpha, legend=legend, texture=texture)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    v = vu.norm(axis)*height/2
    setattr(actor,'base', pos - v)
    setattr(actor,'top',  pos + v)
    return actor


def ring(pos=[0,0,0], r=1, thickness=0.1, axis=[0,0,1],
         c='khaki', alpha=1, wire=False, legend=None, texture=None, res=30):
    '''
    Build a torus of specified outer radius r internal radius thickness, centered at pos.
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
    if nax: axis  = np.array(axis)/nax
    theta = np.arccos(axis[2])
    phi   = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    tf = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf, pfs.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = vu.makeActor(pd, c=c, alpha=alpha, wire=wire, legend=legend, texture=texture)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    return actor



###################################################################
def paraboloid(pos=[0,0,0], r=1, height=1, axis=[0,0,1],
               c='cyan', alpha=1, legend=None, texture=None, res=50):
    '''
    Build a paraboloid of specified height and radius r, centered at pos.
    '''
    quadric = vtk.vtkQuadric()
    quadric.SetCoefficients(1, 1, 0, 0, 0, 0, 0, 0, height/4, 0)
    #F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2
    #         + a3*x*y + a4*y*z + a5*x*z
    #         + a6*x   + a7*y   + a8*z  +a9
    sample = vtk.vtkSampleFunction()
    sample.SetSampleDimensions(res,res,res)
    sample.SetImplicitFunction(quadric)

    contours = vtk.vtkContourFilter()
    contours.SetInputConnection(sample.GetOutputPort())
    contours.GenerateValues(1, .01, .01)
    contours.Update()

    axis  = np.array(axis)/np.linalg.norm(axis)
    theta = np.arccos(axis[2])
    phi   = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    t.Scale(r, r, r)
    tf = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf, contours.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = vu.makeActor(pd, c=c, alpha=alpha, legend=legend, texture=texture)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetMapper().ScalarVisibilityOff()
    actor.SetPosition(pos)
    return actor


def hyperboloid(pos=[0,0,0], a2=1, value=0.5, height=1, axis=[0,0,1],
                c='magenta', alpha=1, legend=None, texture=None, res=50):
    '''
    Build a hyperboloid of specified aperture a2 and height, centered at pos.
    '''
    q = vtk.vtkQuadric()
    q.SetCoefficients(2, 2, -1/a2, 0, 0, 0, 0, 0, 0, 0)
    #F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2
    #         + a3*x*y + a4*y*z + a5*x*z
    #         + a6*x   + a7*y   + a8*z  +a9
    sample = vtk.vtkSampleFunction()
    sample.SetSampleDimensions(res,res,res)
    sample.SetImplicitFunction(q)

    contours = vtk.vtkContourFilter()
    contours.SetInputConnection(sample.GetOutputPort())
    contours.GenerateValues(1, value, value)
    contours.Update()

    axis  = np.array(axis)/np.linalg.norm(axis)
    theta = np.arccos(axis[2])
    phi   = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(theta*57.3)
    t.RotateZ(phi*57.3)
    t.Scale(1,1,height)
    tf = vtk.vtkTransformPolyDataFilter()
    vu.setInput(tf, contours.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = vu.makeActor(pd, c=c, alpha=alpha, legend=legend, texture=texture)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetMapper().ScalarVisibilityOff()
    actor.SetPosition(pos)
    return actor


def text(txt, pos=(0,0,0), normal=(0,0,1), s=1, depth=0.1,
         c='k', alpha=1, bc=None, texture=None, followcam=False, cam=None):
    '''
    Returns a vtkActor that shows a text in 3D.
        
        pos = position in 3D space
              if an integer is passed [1 -> 8], places text in a corner
        s = size of text 
        depth = text thickness
        followcam = True, the text will auto-orient itself to it
    '''
    if isinstance(pos, int):
        cornerAnnotation = vtk.vtkCornerAnnotation()
        cornerAnnotation.SetNonlinearFontScaleFactor(s/3)
        cornerAnnotation.SetText(pos-1, txt )
        cornerAnnotation.GetTextProperty().SetColor( vc.getColor(c) )
        return cornerAnnotation
        
    tt = vtk.vtkVectorText()
    tt.SetText(txt)
    tt.Update()
    ttmapper = vtk.vtkPolyDataMapper()
    if depth:
        extrude = vtk.vtkLinearExtrusionFilter()
        extrude.SetInputConnection(tt.GetOutputPort())
        extrude.SetExtrusionTypeToVectorExtrusion()
        extrude.SetVector(0, 0, 1)    
        extrude.SetScaleFactor(depth)
        ttmapper.SetInputConnection(extrude.GetOutputPort())
    else:
        ttmapper.SetInputConnection(tt.GetOutputPort())
    if followcam: #follow cam
        ttactor = vtk.vtkFollower()
        ttactor.SetCamera(cam)
    else:
        ttactor = vtk.vtkActor()
    ttactor.SetMapper(ttmapper)
    ttactor.GetProperty().SetColor(vc.getColor(c))

    # check if color string contains a float, in this case ignore alpha
    al = vc.getAlpha(c)
    if al: alpha = al
    ttactor.GetProperty().SetOpacity(alpha)

    nax = np.linalg.norm(normal)
    if nax: normal  = np.array(normal)/nax
    theta = np.arccos(normal[2])
    phi   = np.arctan2(normal[1], normal[0])
    ttactor.SetScale(s,s,s)
    ttactor.RotateZ(phi*57.3)
    ttactor.RotateY(theta*57.3)
    ttactor.SetPosition(pos)
    if bc: # defines a specific color for the backface
        backProp = vtk.vtkProperty()
        backProp.SetDiffuseColor(vc.getColor(bc))
        backProp.SetOpacity(alpha)
        ttactor.SetBackfaceProperty(backProp)
    if texture: vu.assignTexture(ttactor, texture)
    vu.assignConvenienceMethods(ttactor, None)
    vu.assignPhysicsMethods(ttactor)
    return ttactor















