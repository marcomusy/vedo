from __future__ import division, print_function
import vtk
import numpy as np

import vtkplotter.utils as vu
import vtkplotter.colors as vc
import vtkplotter.vtkio as vio
import vtkplotter.shapes as vs


def spline(points, smooth=0.5, degree=2, 
           s=2, c='b', alpha=1., nodes=False, legend=None, res=20):
    '''
    Return a vtkActor for a spline that doesnt necessarly 
    pass exactly throught all points.
        smooth = smoothing factor:
            0 = interpolate points exactly, 
            1 = average point positions
        degree = degree of the spline (1<degree<5)
        nodes = True shows also original the points 
    '''
    try:
        from scipy.interpolate import splprep, splev
    except ImportError:
        vc.printc('Warning: ..scipy not installed, using vtkCardinalSpline instead.',c=5)
        return _vtkspline(points, s, c, alpha, nodes, legend, res)

    Nout = len(points)*res # Number of points on the spline
    points = np.array(points)

    minx, miny, minz = np.min(points, axis=0)
    maxx, maxy, maxz = np.max(points, axis=0)
    maxb = max(maxx-minx, maxy-miny, maxz-minz)
    smooth *= maxb/2 # must be in absolute units
    
    x,y,z = points[:,0], points[:,1], points[:,2]
    tckp, _ = splprep([x,y,z], task=0, s=smooth, k=degree) # find the knots
    # evaluate spline, including interpolated points:
    xnew,ynew,znew = splev(np.linspace(0,1, Nout), tckp)

    ppoints = vtk.vtkPoints() # Generate the polyline for the spline
    profileData = vtk.vtkPolyData()
    for i in range(Nout):
        ppoints.InsertPoint(i, xnew[i],ynew[i],znew[i])
    lines = vtk.vtkCellArray() # Create the polyline
    lines.InsertNextCell(Nout)
    for i in range(Nout): lines.InsertCellPoint(i)
    profileData.SetPoints(ppoints)
    profileData.SetLines(lines)
    actline = vu.makeActor(profileData, c=c, alpha=alpha, legend=legend)
    actline.GetProperty().SetLineWidth(s)
    if nodes:
        actnodes = vs.points(points, r=5, c=c, alpha=alpha)
        ass = vu.makeAssembly([actline, actnodes], legend=legend)
        return ass
    else:
        return actline

def _vtkspline(points, s, c, alpha, nodes, legend, res):
    numberOfOutputPoints = len(points)*res # Number of points on the spline
    numberOfInputPoints  = len(points) # One spline for each direction.
    aSplineX = vtk.vtkCardinalSpline() #  interpolate the x values
    aSplineY = vtk.vtkCardinalSpline() #  interpolate the y values
    aSplineZ = vtk.vtkCardinalSpline() #  interpolate the z values

    inputPoints = vtk.vtkPoints()
    for i in range(0, numberOfInputPoints):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        aSplineX.AddPoint(i, x)
        aSplineY.AddPoint(i, y)
        aSplineZ.AddPoint(i, z)
        inputPoints.InsertPoint(i, x, y, z)

    inputData = vtk.vtkPolyData() 
    inputData.SetPoints(inputPoints)
    points = vtk.vtkPoints() 
    profileData = vtk.vtkPolyData()
    for i in range(0, numberOfOutputPoints):
        t = (numberOfInputPoints-1.)/(numberOfOutputPoints-1.)*i
        x,y,z = aSplineX.Evaluate(t), aSplineY.Evaluate(t), aSplineZ.Evaluate(t)
        points.InsertPoint(i, x,y,z)

    lines = vtk.vtkCellArray() # Create the polyline.
    lines.InsertNextCell(numberOfOutputPoints)
    for i in range(0, numberOfOutputPoints): lines.InsertCellPoint(i)

    profileData.SetPoints(points)
    profileData.SetLines(lines)
    actline = vu.makeActor(profileData, c=c, alpha=alpha, legend=legend)
    actline.GetProperty().SetLineWidth(s)
    actline.GetProperty().SetInterpolationToPhong()
    if nodes:
        pts = vu.coordinates(inputData)
        actnodes = vs.points(pts, r=s*1.5, c=c, alpha=alpha)
        ass = vu.makeAssembly([actline, actnodes], legend=legend)
        return ass
    else:
        return actline


def xyplot(points, title='', c='b', corner=1, lines=False):
    """
    Return a vtkActor that is a plot of 2D points in x and y.

    Use corner to assign its position:
        1=topleft, 
        2=topright, 
        3=bottomleft, 
        4=bottomright.
    """
    c = vc.getColor(c) # allow different codings
    array_x = vtk.vtkFloatArray()
    array_y = vtk.vtkFloatArray()
    array_x.SetNumberOfTuples(len(points))
    array_y.SetNumberOfTuples(len(points))
    for i,p in enumerate(points):
        array_x.InsertValue(i,p[0])
        array_y.InsertValue(i,p[1])
    field = vtk.vtkFieldData()
    field.AddArray(array_x)
    field.AddArray(array_y)
    data = vtk.vtkDataObject()
    data.SetFieldData(field)
    plot = vtk.vtkXYPlotActor()
    plot.AddDataObjectInput(data)
    plot.SetDataObjectXComponent(0,0)
    plot.SetDataObjectYComponent(0,1)
    plot.SetXValuesToValue()
    plot.SetXTitle(title)
    plot.SetYTitle('')
    plot.ExchangeAxesOff()
    plot.PlotPointsOn()
    if not lines: plot.PlotLinesOff()
    plot.GetProperty().SetPointSize(5)
    plot.GetProperty().SetLineWidth(2)
    plot.SetNumberOfXLabels(3) #not working
    plot.GetProperty().SetColor(0,0,0)
    plot.GetProperty().SetOpacity(0.7)
    plot.SetPlotColor(0,c[0],c[1],c[2])
    tprop = plot.GetAxisLabelTextProperty()
    tprop.SetColor(0,0,0)
    tprop.SetOpacity(0.7)
    tprop.SetFontFamily(0)
    tprop.BoldOff()
    tprop.ItalicOff()
    tprop.ShadowOff()
    tprop.SetFontSize(3) #not working
    plot.SetAxisTitleTextProperty(tprop)
    plot.SetAxisLabelTextProperty(tprop)
    plot.SetTitleTextProperty(tprop)
    if corner==1: plot.GetPositionCoordinate().SetValue(.0, .8, 0)
    if corner==2: plot.GetPositionCoordinate().SetValue(.7, .8, 0)
    if corner==3: plot.GetPositionCoordinate().SetValue(.0, .0, 0)
    if corner==4: plot.GetPositionCoordinate().SetValue(.7, .0, 0)
    plot.GetPosition2Coordinate().SetValue(.3, .2, 0)
    return plot


def fxy(z='sin(3*x)*log(x-y)/3', x=[0,3], y=[0,3],
        zlimits=[None,None], showNan=True, zlevels=10, wire=False,
        c='b', bc='aqua', alpha=1, legend=True, texture=None, res=100):
    '''
    Build a surface representing the 3D function specified as a string
    or as a reference to an external function.
    Red points indicate where the function does not exist (showNan).

    zlevels will draw the specified number of z-levels contour lines.

    Examples:
        vp = vtkplotter.Plotter()
        vp.fxy('sin(3*x)*log(x-y)/3')
        or
        def z(x,y): return math.sin(x*y)
        vp.fxy(z) # or equivalently:
        vp.fxy(lambda x,y: math.sin(x*y))
    '''
    if isinstance(z, str):
        try:
            z = z.replace('math.','').replace('np.','')
            namespace = locals()
            code  = "from math import*\ndef zfunc(x,y): return "+z
            exec(code, namespace)
            z = namespace['zfunc']
        except:
            vc.printc('Syntax Error in fxy()',c=1)
            return None

    ps = vtk.vtkPlaneSource()
    ps.SetResolution(res, res)
    ps.SetNormal([0,0,1])
    ps.Update()
    poly = ps.GetOutput()
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    todel, nans = [], []

    if zlevels:
        tf = vtk.vtkTriangleFilter()
        vu.setInput(tf, poly)
        tf.Update()
        poly = tf.GetOutput()

    for i in range(poly.GetNumberOfPoints()):
        px,py,_ = poly.GetPoint(i)
        xv = (px+.5)*dx+x[0]
        yv = (py+.5)*dy+y[0]
        try:
            zv = z(xv, yv)
            poly.GetPoints().SetPoint(i, [xv,yv,zv])
        except:
            todel.append(i)
            nans.append([xv,yv,0])

    if len(todel):
        cellIds = vtk.vtkIdList()
        poly.BuildLinks()

        for i in todel:
            poly.GetPointCells(i, cellIds)
            for j in range(cellIds.GetNumberOfIds()):
                poly.DeleteCell(cellIds.GetId(j)) #flag cell

        poly.RemoveDeletedCells()
        cl = vtk.vtkCleanPolyData()
        vu.setInput(cl, poly)
        cl.Update()
        poly = cl.GetOutput()
    
    if not poly.GetNumberOfPoints(): 
        vc.printc('Function is not real in the domain',c=1)
        return vtk.vtkActor()
    
    if zlimits[0]:
        a = vu.cutPlane(poly, (0,0,zlimits[0]), (0,0,1))
        poly = vu.polydata(a)
    if zlimits[1]:
        a = vu.cutPlane(poly, (0,0,zlimits[1]), (0,0,-1))
        poly = vu.polydata(a)

    if c is None:
        elev = vtk.vtkElevationFilter()
        vu.setInput(elev,poly)
        elev.Update()
        poly = elev.GetOutput()

    actor = vu.makeActor(poly, c=c, bc=bc, alpha=alpha, wire=wire,
                         legend=legend, texture=texture)
    acts=[actor]
    if zlevels:
       elevation = vtk.vtkElevationFilter()
       vu.setInput(elevation, poly)
       bounds = poly.GetBounds()
       elevation.SetLowPoint( 0,0,bounds[4])
       elevation.SetHighPoint(0,0,bounds[5])
       elevation.Update()
       bcf = vtk.vtkBandedPolyDataContourFilter()
       vu.setInput(bcf, elevation.GetOutput())
       bcf.SetScalarModeToValue()
       bcf.GenerateContourEdgesOn()
       bcf.GenerateValues(zlevels, elevation.GetScalarRange())
       bcf.Update()
       zpoly = bcf.GetContourEdgesOutput()
       zbandsact = vu.makeActor(zpoly, c='k', alpha=alpha)
       zbandsact.GetProperty().SetLineWidth(1.5)
       acts.append(zbandsact)

    if showNan and len(todel):
        bb = actor.GetBounds()
        zm = (bb[4]+bb[5])/2
        nans = np.array(nans)+[0,0,zm]
        nansact = vs.points(nans, c='red', alpha=alpha/2)
        acts.append(nansact)

    if len(acts)>1:
        asse = vu.makeAssembly(acts)
        return asse
    else:
        return actor


def delaunay2D(plist, tol=None, c='gold', alpha=0.5, wire=False, bc=None, edges=False, 
               legend=None, texture=None):
    '''
    Create a mesh from points in the XY plane.
    '''
    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(len(plist))
    src.Update()
    pd = src.GetOutput()
    for i,p in enumerate(plist): pd.GetPoints().SetPoint(i, p)
    delny = vtk.vtkDelaunay2D()
    vu.setInput(delny, pd)
    if tol: delny.SetTolerance(tol)
    delny.Update()
    return vu.makeActor(delny.GetOutput(), c, alpha, wire, bc, edges, legend, texture)


def normals(actor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8, legend=None):
    '''
    Build a vtkActor made of the normals at vertices shown as arrows
    '''
    maskPts = vtk.vtkMaskPoints()
    maskPts.SetOnRatio(ratio)
    maskPts.RandomModeOff()
    src = vu.polydata(actor)
    vu.setInput(maskPts, src)
    arrow = vtk.vtkArrowSource()
    arrow.SetTipRadius(0.075)
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetInputConnection(maskPts.GetOutputPort())
    glyph.SetVectorModeToUseNormal()
    b = src.GetBounds()
    sc = max( [ b[1]-b[0], b[3]-b[2], b[5]-b[4] ] )/20.
    glyph.SetScaleFactor(sc)
    glyph.SetColorModeToColorByVector()
    glyph.SetScaleModeToScaleByVector()
    glyph.OrientOn()
    glyph.Update()
    glyphMapper = vtk.vtkPolyDataMapper()
    glyphMapper.SetInputConnection(glyph.GetOutputPort())
    glyphMapper.SetScalarModeToUsePointFieldData()
    glyphMapper.SetColorModeToMapScalars()
    glyphMapper.ScalarVisibilityOn()
    glyphMapper.SelectColorArray("Elevation")
    glyphActor = vtk.vtkActor()
    glyphActor.SetMapper(glyphMapper)
    glyphActor.GetProperty().EdgeVisibilityOff()
    glyphActor.GetProperty().SetColor(vc.getColor(c))
    # check if color string contains a float, in this case ignore alpha
    al = vc.getAlpha(c)
    if al: alpha = al
    glyphActor.GetProperty().SetOpacity(alpha)
    aactor = vu.makeAssembly([actor, glyphActor], legend=legend)
    return aactor


def curvature(actor, method=1, r=1, alpha=1, lut=None, legend=None):
    '''
    Build a copy of vtkActor that contains the color coded surface
    curvature following four different ways to calculate it:
        method =  0-gaussian, 1-mean, 2-max, 3-min
    '''
    poly = vu.polydata(actor)
    cleaner = vtk.vtkCleanPolyData()
    vu.setInput(cleaner, poly)
    curve = vtk.vtkCurvatures()
    curve.SetInputConnection(cleaner.GetOutputPort())
    curve.SetCurvatureType(method)
    curve.InvertMeanCurvatureOn()
    curve.Update()
    print('CurvatureType set to:', method)
    if not lut:
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(256)
        lut.SetHueRange(0.15, 1)
        lut.SetSaturationRange(1, 1)
        lut.SetValueRange(1, 1)
        lut.SetAlphaRange(alpha, 1)
        b = poly.GetBounds()
        sc = max( [ b[1]-b[0], b[3]-b[2], b[5]-b[4] ] )
        lut.SetRange(-0.01/sc*r, 0.01/sc*r)
    cmapper = vtk.vtkPolyDataMapper()
    cmapper.SetInputConnection(curve.GetOutputPort())
    cmapper.SetLookupTable(lut)
    cmapper.SetUseLookupTableScalarRange(1)
    cactor = vtk.vtkActor()
    cactor.SetMapper(cmapper)
    return cactor


def boundaries(actor, c='p', lw=5, legend=None):
    '''Build a copy of actor that shows the boundary lines of its surface.'''
    fe = vtk.vtkFeatureEdges()
    vu.setInput(fe, vu.polydata(actor))
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOn()
    fe.ManifoldEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.ColoringOff()
    fe.Update()
    bactor = vu.makeActor(fe.GetOutput(), c=c, alpha=1, legend=legend)
    bactor.GetProperty().SetLineWidth(lw)
    return bactor


def extractLargestRegion(actor, c=None, alpha=None, wire=False, bc=None,
                         edges=False, legend=None, texture=None):
    '''Split actor into a set of disconnected polydata'''
    conn = vtk.vtkConnectivityFilter()
    conn.SetExtractionModeToLargestRegion()
    conn.ScalarConnectivityOff()
    poly = vu.polydata(actor, True)
    vu.setInput(conn, poly)
    conn.Update()
    epoly = conn.GetOutput()
    if legend  is True and hasattr(actor, 'legend'): legend = actor.legend
    if alpha   is None: alpha = actor.GetProperty().GetOpacity()
    if c       is None: c = actor.GetProperty().GetColor()
    if texture is None and hasattr(actor, 'texture'): texture = actor.texture
    eact = vu.makeActor(epoly, c, alpha, wire, bc, edges, legend, texture)
    eact.GetProperty().SetPointSize(actor.GetProperty().GetPointSize())
    return eact
    

################# working with point clouds
def fitLine(points, c='orange', lw=1, alpha=0.6, legend=None):
    '''
    Fits a line through points.

    Extra info is stored in actor.slope, actor.center, actor.variances
    '''
    data = np.array(points)
    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)
    vv = vv[0]/np.linalg.norm(vv[0])
    # vv contains the first principal component, i.e. the direction
    # vector of the best fit line in the least squares sense.
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    a = np.linalg.norm(xyz_min - datamean)
    b = np.linalg.norm(xyz_max - datamean)
    p1 = datamean -a*vv
    p2 = datamean +b*vv
    l = vs.line(p1, p2, c=c, lw=lw, alpha=alpha)
    setattr(l, 'slope', vv)
    setattr(l, 'center', datamean)
    setattr(l, 'variances', dd)
    return l


def fitPlane(points, c='g', bc='darkgreen', alpha=0.8, legend=None):
    '''
    Fits a plane to a set of points.

    Extra info is stored in actor.normal, actor.center, actor.variance
    '''
    data = np.array(points)
    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    s= np.linalg.norm(xyz_max - xyz_min)
    n = np.cross(vv[0],vv[1])
    pla = vs.plane(datamean, n, s, s, c, bc, alpha, legend, None)
    setattr(pla, 'normal', n)
    setattr(pla, 'center', datamean)
    setattr(pla, 'variance', dd[2])
    return pla


def fitSphere(coords, c='r', alpha=1, wire=1, legend=None):
    '''
    Fits a sphere to a set of points.
    
    Extra info is stored in actor.radius, actor.center, actor.residue
    '''
    coords = np.array(coords)
    n = len(coords)
    A = np.zeros((n,4))
    A[:,:-1] = coords*2
    A[:,  3] = 1
    f = np.zeros((n,1))
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    f[:,0] = x*x+ y*y +z*z
    C, residue, rank, sv = np.linalg.lstsq(A,f, rcond=None) # solve AC=f
    if rank<4: return None
    t = (C[0]*C[0]) + (C[1]*C[1]) + (C[2]*C[2]) +C[3]
    radius = np.sqrt(t)[0]
    center = np.array([C[0][0], C[1][0], C[2][0]])
    if len(residue): residue = np.sqrt(residue[0])/n
    else: residue=0
    s = vs.sphere(center, radius, c, alpha, wire=wire, legend=legend)
    setattr(s, 'radius', radius)
    setattr(s, 'center', center)
    setattr(s, 'residue', residue)
    return s



def pca(points, pvalue=.95, c='c', alpha=0.5, pcaAxes=False, legend=None):
    '''
    Show the oriented PCA ellipsoid that contains fraction pvalue of points.
        axes = True, show the 3 PCA semi axes
    Extra info is stored in actor.sphericity, actor.va, actor.vb, actor.vc
    (sphericity = 1 for a perfect sphere)
    '''
    try:
        from scipy.stats import f
    except:
        vc.printc("Error in ellipsoid(): scipy not installed. Skip.",c=1)
        return None
    if isinstance(points, vtk.vtkActor): points=vu.coordinates(points)
    if len(points) == 0: return None
    P = np.array(points, ndmin=2, dtype=float)
    cov = np.cov(P, rowvar=0)      # covariance matrix
    U, s, R = np.linalg.svd(cov)   # singular value decomposition
    p, n = s.size, P.shape[0]
    fppf = f.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p) # f % point function
    ua,ub,uc = np.sqrt(s*fppf)*2   # semi-axes (largest first)
    center = np.mean(P, axis=0)    # centroid of the hyperellipsoid
    sphericity = ( ((ua-ub)/(ua+ub))**2
                 + ((ua-uc)/(ua+uc))**2
                 + ((ub-uc)/(ub+uc))**2 )/3. *4.
    elliSource = vtk.vtkSphereSource()
    elliSource.SetThetaResolution(48)
    elliSource.SetPhiResolution(48)
    matri = vtk.vtkMatrix4x4()
    matri.DeepCopy((R[0][0] *ua, R[1][0] *ub, R[2][0] *uc, center[0],
                    R[0][1] *ua, R[1][1] *ub, R[2][1] *uc, center[1],
                    R[0][2] *ua, R[1][2] *ub, R[2][2] *uc, center[2], 0,0,0,1))
    vtra = vtk.vtkTransform()
    vtra.SetMatrix(matri)
    ftra = vtk.vtkTransformFilter()
    ftra.SetTransform(vtra)
    ftra.SetInputConnection(elliSource.GetOutputPort())
    ftra.Update()
    actor_elli = vu.makeActor(ftra.GetOutput(), c, alpha, legend=legend)
    actor_elli.GetProperty().BackfaceCullingOn()
    actor_elli.GetProperty().SetInterpolationToPhong()
    if pcaAxes:
        axs = []
        for ax in ([1,0,0], [0,1,0], [0,0,1]):
            l = vtk.vtkLineSource()
            l.SetPoint1([0,0,0])
            l.SetPoint2(ax)
            l.Update()
            t = vtk.vtkTransformFilter()
            t.SetTransform(vtra)
            vu.setInput(t, l.GetOutput())
            t.Update()
            axs.append(vu.makeActor(t.GetOutput(), c, alpha))
        finact = vu.makeAssembly([actor_elli]+axs, legend=legend)
    else : 
        finact = actor_elli
    setattr(finact, 'sphericity', sphericity)
    setattr(finact, 'va', ua)
    setattr(finact, 'vb', ub)
    setattr(finact, 'vc', uc)
    return finact


def smoothMLS2D(actor, f=0.2, decimate=1, recursive=0, showNPlanes=0):
    '''
    Smooth actor or points with a Moving Least Squares variant.
    The list actor.variances contain the residue calculated for each point.
    Input actor's polydata is modified.
    
        f, smoothing factor - typical range s [0,2]
        
        decimate, decimation factor (an integer number) 
        
        recursive, move points while algorithm proceedes
        
        showNPlanes, build an actor showing the fitting plane for N random points            
    '''        
    coords  = vu.coordinates(actor)
    ncoords = len(coords)
    Ncp     = int(ncoords*f/100)
    nshow   = int(ncoords/decimate)
    if showNPlanes: ndiv = int(nshow/showNPlanes*decimate)
    
    if Ncp<5:
        vc.printc('Please choose a higher fraction than '+str(f), c=1)
        Ncp=5
    print('smoothMLS: Searching #neighbours, #pt:', Ncp, ncoords)
    
    poly = vu.polydata(actor, True)
    vpts = poly.GetPoints()
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    vtklist = vtk.vtkIdList()        
    variances, newsurf, acts = [], [], []
    pb = vio.ProgressBar(0, ncoords)
    for i, p in enumerate(coords):
        pb.print('smoothing...')
        if i%decimate: continue
        
        locator.FindClosestNPoints(Ncp, p, vtklist)
        points  = []
        for j in range(vtklist.GetNumberOfIds()):
            trgp = [0,0,0]
            vpts.GetPoint(vtklist.GetId(j), trgp )
            points.append( trgp )
        if len(points)<5: continue
        
        points = np.array(points)
        pointsmean = points.mean(axis=0) # plane center
        uu, dd, vv = np.linalg.svd(points-pointsmean)
        a,b,c = np.cross(vv[0],vv[1]) # normal
        d,e,f = pointsmean # plane center
        x,y,z = p
        t = (a*d -a*x +b*e- b*y +c*f -c*z)#/(a*a+b*b+c*c)
        newp = [x+t*a, y+t*b, z+t*c] 
        variances.append(dd[2])
        newsurf.append(newp)
        if recursive: vpts.SetPoint(i, newp)
    
        if showNPlanes and not i%ndiv: 
            plane = fitPlane(points, alpha=0.3) # fitting plane
            iapts = vs.points(points)  # blue points
            acts += [plane, iapts]
                    
    if decimate==1 and not recursive:
        for i in range(ncoords): vpts.SetPoint(i, newsurf[i])

    setattr(actor, 'variances', np.array(variances))

    if showNPlanes:
        apts = vs.points(newsurf, c='r 0.6', r=2)
        ass = vu.makeAssembly([apts]+acts)
        return ass #NB: a demo actor is returned

    return actor #NB: original actor is modified
   
    
def smoothMLS1D(actor, f=0.2, showNLines=0):
    '''
    Smooth actor or points with a Moving Least Squares variant.
    The list actor.variances contain the residue calculated for each point.
    Input actor's polydata is modified.
    
        f, smoothing factor - typical range s [0,2]
                      
        showNLines, build an actor showing the fitting line for N random points            
    '''        
    coords  = vu.coordinates(actor)
    ncoords = len(coords)
    Ncp     = int(ncoords*f/10)
    nshow   = int(ncoords)
    if showNLines: ndiv = int(nshow/showNLines)
    
    if Ncp<3:
        vc.printc('Please choose a higher fraction than '+str(f), c=1)
        Ncp=3
    
    poly = vu.polydata(actor, True)
    vpts = poly.GetPoints()
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    vtklist = vtk.vtkIdList()        
    variances, newline, acts = [], [], []
    for i, p in enumerate(coords):
        
        locator.FindClosestNPoints(Ncp, p, vtklist)
        points  = []
        for j in range(vtklist.GetNumberOfIds()):
            trgp = [0,0,0]
            vpts.GetPoint(vtklist.GetId(j), trgp )
            points.append( trgp )
        if len(points)<2: continue
        
        points = np.array(points)
        pointsmean = points.mean(axis=0) # plane center
        uu, dd, vv = np.linalg.svd(points-pointsmean)
        newp = np.dot(p-pointsmean, vv[0])*vv[0] + pointsmean
        variances.append(dd[1]+dd[2])
        newline.append(newp)
    
        if showNLines and not i%ndiv: 
            fline = fitLine(points, lw=4,alpha=1) # fitting plane
            iapts = vs.points(points)  # blue points
            acts += [fline, iapts]
                    
    for i in range(ncoords): vpts.SetPoint(i, newline[i])

    if showNLines:
        apts = vs.points(newline, c='r 0.6', r=2)
        ass = vu.makeAssembly([apts]+acts)
        return ass #NB: a demo actor is returned

    setattr(actor, 'variances', np.array(variances))
    return actor #NB: original actor is modified
   

def extractLines(actor, n=5):
    
    coords  = vu.coordinates(actor)
    poly = vu.polydata(actor, True)
    vpts = poly.GetPoints()
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    vtklist = vtk.vtkIdList()  
    spts=[]
    for i, p in enumerate(coords):
        locator.FindClosestNPoints(n, p, vtklist)
        points  = []
        for j in range(vtklist.GetNumberOfIds()):
            trgp = [0,0,0]
            vpts.GetPoint(vtklist.GetId(j), trgp )
            if (p-trgp).any(): points.append( trgp )
        p0 = points.pop()-p
        dots = [np.dot(p0, ps-p) for ps in points]
        if len(np.unique(np.sign(dots)))==1:
            spts.append(p)
    return np.array(spts)



def align(source, target, iters=100, legend=None):
    '''
    Return a copy of source actor which is aligned to
    target actor through vtkIterativeClosestPointTransform() method.
    '''
    sprop = source.GetProperty()
    source = vu.polydata(source)
    target = vu.polydata(target)
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.SetMaximumNumberOfIterations(iters)
    icp.StartByMatchingCentroidsOn()
    icp.Update()
    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    vu.setInput(icpTransformFilter, source)
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()
    poly = icpTransformFilter.GetOutput()
    actor = vu.makeActor(poly, legend=legend)
    actor.SetProperty(sprop)
    setattr(actor, 'transform', icp.GetLandmarkTransform())
    return actor


def booleanOperation(actor1, actor2, operation='plus', c=None, alpha=1, 
                     wire=False, bc=None, edges=False, legend=None, texture=None):
    '''Volumetric union, intersection and subtraction of surfaces'''
    try:
        bf = vtk.vtkBooleanOperationPolyDataFilter()
    except AttributeError:
        vc.printc('Boolean operation only possible for vtk version >= 8', c='r')
        return None
    poly1 = vu.polydata(actor1, True)
    poly2 = vu.polydata(actor2, True)
    if operation.lower() == 'plus':
        bf.SetOperationToUnion()
    elif operation.lower() == 'intersect':
        bf.SetOperationToIntersection()
    elif operation.lower() == 'minus':
        bf.SetOperationToDifference()
        bf.ReorientDifferenceCellsOn()
    if vu.vtkMV:
        bf.SetInputData(0, poly1)
        bf.SetInputData(1, poly2)
    else:
        bf.SetInputConnection(0, poly1.GetProducerPort())
        bf.SetInputConnection(1, poly2.GetProducerPort())
    bf.Update()
    actor = vu.makeActor(bf.GetOutput(), 
                         c, alpha, wire, bc, edges, legend, texture)
    return actor



def surfaceIntersection(actor1, actor2, tol=1e-06, lw=3,
                        c=None, alpha=1, legend=None):
    '''Intersect 2 surfaces and return a line actor'''
    try:
        bf = vtk.vtkIntersectionPolyDataFilter()
    except AttributeError:
        vc.printc('surfaceIntersection only possible for vtk version > 6',c='r')
        return None
    poly1 = vu.polydata(actor1, True)
    poly2 = vu.polydata(actor2, True)
    bf.SetInputData(0, poly1)
    bf.SetInputData(1, poly2)
    bf.Update()
    if c is None: c = actor1.GetProperty().GetColor()
    actor = vu.makeActor(bf.GetOutput(), c, alpha, 0, legend=legend)
    actor.GetProperty().SetLineWidth(lw)
    return actor

def recoSurface(points, bins=256,
                c='gold', alpha=1, wire=False, bc='t', edges=False, legend=None):
    '''
    Surface reconstruction from sparse points.
    '''
    
    if isinstance(points, vtk.vtkActor): points = vu.coordinates(points)
    N=len(points)
    if N<50: 
        print('recoSurface: Use at least 50 points.')
        return None
    points = np.array(points)

    ptsSource = vtk.vtkPointSource()
    ptsSource.SetNumberOfPoints(N)
    ptsSource.Update()
    vpts = ptsSource.GetOutput().GetPoints()
    for i,p in enumerate(points): vpts.SetPoint(i, p)
    polyData = ptsSource.GetOutput()

    distance = vtk.vtkSignedDistance()
    f=0.1
    x0,x1,y0,y1,z0,z1 = polyData.GetBounds()
    distance.SetBounds(x0-(x1-x0)*f, x1+(x1-x0)*f,
                       y0-(y1-y0)*f, y1+(y1-y0)*f,
                       z0-(z1-z0)*f, z1+(z1-z0)*f)
    if polyData.GetPointData().GetNormals():
        distance.SetInputData(polyData)
        vu.setInput(distance, polyData)
    else:
        normals = vtk.vtkPCANormalEstimation()
        vu.setInput(normals, polyData)
        normals.SetSampleSize(int(N/50))
        normals.SetNormalOrientationToGraphTraversal()
        distance.SetInputConnection(normals.GetOutputPort())
        print ('Recalculating normals for', N, 'points, sample size=',int(N/50))
    radius = vu.diagonalSize(polyData)/bins*5
    distance.SetRadius(radius)
    distance.SetDimensions(bins, bins, bins)
    distance.Update()  

    print ('Calculating mesh from points with R =', radius)
    surface = vtk.vtkExtractSurface()
    surface.SetRadius(radius * .99)
    surface.HoleFillingOn()
    surface.ComputeNormalsOff()
    surface.ComputeGradientsOff()
    surface.SetInputConnection(distance.GetOutputPort())
    surface.Update()  
    return vu.makeActor(surface.GetOutput(), c, alpha, wire, bc, edges, legend)


def cluster(points, radius, legend=None):
    '''
    Clustering of points in space.
    radius, is the radius of local search.
    Individual subsets can be accessed through actor.clusters
    '''
    if isinstance(points, vtk.vtkActor): 
        poly = vu.polydata(points)
    else:
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(points))
        src.Update()
        vpts = src.GetOutput().GetPoints()
        for i,p in enumerate(points): vpts.SetPoint(i, p)
        poly = src.GetOutput()
        
    cluster = vtk.vtkEuclideanClusterExtraction()
    vu.setInput(cluster, poly)
    cluster.SetExtractionModeToAllClusters()
    cluster.SetRadius(radius)
    cluster.ColorClustersOn()
    cluster.Update()
    
    idsarr = cluster.GetOutput().GetPointData().GetArray('ClusterId')
    Nc = cluster.GetNumberOfExtractedClusters()
    
    sets = [ [] for i in range(Nc)]
    for i,p in enumerate(points): sets[idsarr.GetValue(i)].append(p)
    
    acts = []
    for i,aset in enumerate(sets): 
        acts.append(vs.points(aset, c=i))

    actor = vu.makeAssembly(acts, legend=legend)
    setattr(actor, 'clusters', sets)
    print('Nr. of extracted clusters', Nc)
    if Nc>10: print('First ten:')
    for i in range(Nc):
        if i>9: 
            print('...')
            break
        print('Cluster #'+str(i)+',  N =', len(sets[i]))
    print('Access individual clusters through attribute: actor.cluster')
    return actor
    
    
def removeOutliers(points, radius, c='k', alpha=1, legend=None):
    '''
    Remove outliers from a cloud of points within radius search
    '''
    isactor=False
    if isinstance(points, vtk.vtkActor): 
        isactor=True
        poly = vu.polydata(points)
    else:
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(points))
        src.Update()
        vpts = src.GetOutput().GetPoints()
        for i,p in enumerate(points): vpts.SetPoint(i, p)
        poly = src.GetOutput()
    
    removal = vtk.vtkRadiusOutlierRemoval()
    vu.setInput(removal, poly)
    
    removal.SetRadius(radius)
    removal.SetNumberOfNeighbors(5)
    removal.GenerateOutliersOff()
    removal.Update()
    rpoly = removal.GetOutput()
    print("# of removed outlier points: ", 
          removal.GetNumberOfPointsRemoved(),'/', poly.GetNumberOfPoints())
    outpts=[]
    for i in range(rpoly.GetNumberOfPoints()): 
        outpts.append(list(rpoly.GetPoint(i)))
    outpts = np.array(outpts)
    if not isactor: return outpts

    actor = vs.points(outpts, c=c, alpha=alpha, legend=legend)
    return actor  # return same obj for concatenation

          

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


