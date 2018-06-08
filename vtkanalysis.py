from __future__ import division, print_function
import vtk
import numpy as np
import vtkutils as vu
import vtkcolors as vc
import vtkshapes as vs
import vtkio as vio



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
        vio.printc('Warning: ..scipy not installed, using vtkCardinalSpline instead.',5)
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
        actnodes = vs.points(points, r=s*1.5, c=c, alpha=alpha)
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
        vp = plotter.vtkPlotter()
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
            vio.printc('Syntax Error in fxy()',1)
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
        vio.printc('Function is not real in the domain',1)
        return vtk.vtkActor()
    
    if zlimits[0]:
        a = cutPlane(poly, (0,0,zlimits[0]), (0,0,1), False)
        poly = vu.polydata(a)
    if zlimits[1]:
        a = cutPlane(poly, (0,0,zlimits[1]), (0,0,-1), False)
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


def cutPlane(actor, origin=(0,0,0), normal=(1,0,0),
             showcut=True):
    '''
    Takes actor and cuts it with the plane defined by a point
    and a normal. 
        showcut  = shows the cut away part as thin wireframe
        showline = marks with a thick line the cut
    '''
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)
    poly = vu.polydata(actor)
    clipper = vtk.vtkClipPolyData()
    vu.setInput(clipper, poly)
    clipper.SetClipFunction(plane)
    clipper.GenerateClippedOutputOn()
    clipper.SetValue(0.)
    clipper.Update()
    if hasattr(actor, 'GetProperty'):
        alpha = actor.GetProperty().GetOpacity()
        c = actor.GetProperty().GetColor()
        bf = actor.GetBackfaceProperty()
    else:
        alpha=1
        c='gold'
        bf=None
    leg = None
    if hasattr(actor, 'legend'): leg = actor.legend
    clipActor = vu.makeActor(clipper.GetOutput(),c=c,alpha=alpha, legend=leg)
    clipActor.SetBackfaceProperty(bf)

    acts = [clipActor]
    if showcut:
        cpoly = clipper.GetClippedOutput()
        restActor = vu.makeActor(cpoly, c=c, alpha=0.05, wire=1)
        acts.append(restActor)

    if len(acts)>1:
        asse = vu.makeAssembly(acts)
        return asse
    else:
        return clipActor


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
    pla = vs.grid(datamean, n, s, c, bc, 3, alpha, False, legend, None, 1)
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
        vio.printc("Error in ellipsoid(): scipy not installed. Skip.",1)
        return None
    if len(points) == 0: return None
    P = np.array(points, ndmin=2, dtype=float)
    cov = np.cov(P, rowvar=0)      # covariance matrix
    U, s, R = np.linalg.svd(cov)   # singular value decomposition
    p, n = s.size, P.shape[0]
    fppf = f.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p) # f % point function
    ua,ub,uc = np.sqrt(s*fppf)*2   # semi-axes (largest first)
    center = np.mean(P, axis=0)    # centroid of the hyperellipsoid
    sphericity =  (((ua-ub)/(ua+ub))**2
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


def smoothMLS(actor, f=0.2, decimate=1, recursive=0, showNPlanes=0):
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
        vio.printc('Please choose a higher fraction than'+str(f), 1)
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
    vu.setInput(triangles, vu.polydata(actor))
    triangles.Update()
    originalMesh = triangles.GetOutput()
    if   method==0: sdf = vtk.vtkLoopSubdivisionFilter()
    elif method==1: sdf = vtk.vtkLinearSubdivisionFilter()
    elif method==2: sdf = vtk.vtkAdaptiveSubdivisionFilter()
    elif method==3: sdf = vtk.vtkButterflySubdivisionFilter()
    else:
        vio.printc('Error in subdivide: unknown method.', 'r')
        exit(1)
    if method != 2: sdf.SetNumberOfSubdivisions(N)
    vu.setInput(sdf, originalMesh)
    sdf.Update()
    out = sdf.GetOutput()
    if legend is None and hasattr(actor, 'legend'): legend=actor.legend
    sactor = vu.makeActor(out, legend=legend)
    sactor.GetProperty().SetOpacity(actor.GetProperty().GetOpacity())
    sactor.GetProperty().SetColor(actor.GetProperty().GetColor())
    sactor.GetProperty().SetRepresentation(actor.GetProperty().GetRepresentation())
    return sactor


def decimate(actor, fraction=0.5, N=None, verbose=True, boundaries=True):
    '''
    Downsample the number of vertices in a mesh.
        fraction gives the desired target of reduction. 
        E.g. fraction=0.1
             leaves 10% of the original nr of vertices.
    '''
    poly = vu.polydata(actor, True)
    if N: # N = desired number of points
        Np = poly.GetNumberOfPoints()
        fraction = float(N)/Np
        if fraction >= 1: return actor   
        
    decimate = vtk.vtkDecimatePro()
    vu.setInput(decimate, poly)
    decimate.SetTargetReduction(1.-fraction)
    decimate.PreserveTopologyOff()
    if boundaries: decimate.BoundaryVertexDeletionOn()
    else: decimate.BoundaryVertexDeletionOff()
    decimate.Update()
    if verbose:
        print ('Input nr. of pts:',poly.GetNumberOfPoints(),end='')
        print (' output:',decimate.GetOutput().GetNumberOfPoints())
    mapper = actor.GetMapper()
    vu.setInput(mapper, decimate.GetOutput())
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
        vio.printc('Boolean operation only possible for vtk version >= 8','r')
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
       


def intersectWithLine(act, p0, p1):
    '''Return a list of points between p0 and p1 intersecting the actor'''
    if not hasattr(act, 'linelocator'):
        linelocator = vtk.vtkOBBTree()
        linelocator.SetDataSet(vu.polydata(act, True))
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

    apd = vu.polydata(obj)
    
    planes = vtk.vtkPlanes()
    planes.SetBounds(apd.GetBounds())

    clipper = vtk.vtkClipPolyData()
    vu.setInput(clipper, apd)
    clipper.SetClipFunction(planes)
    clipper.InsideOutOn()
    clipper.GenerateClippedOutputOn()

    # check if color string contains a float, in this case ignore alpha
    al = vc.getAlpha(c)
    if al: alpha = al

    act0Mapper = vtk.vtkPolyDataMapper() # the part which stays
    act0Mapper.SetInputConnection(clipper.GetOutputPort())
    act0 = vtk.vtkActor()
    act0.SetMapper(act0Mapper)
    act0.GetProperty().SetColor(vc.getColor(c))
    act0.GetProperty().SetOpacity(alpha)
    backProp = vtk.vtkProperty()
    backProp.SetDiffuseColor(vc.getColor(bc))
    backProp.SetOpacity(alpha)
    act0.SetBackfaceProperty(backProp)
    #act0 = makeActor(clipper.GetOutputPort())
    
    act0.GetProperty().SetInterpolationToFlat()
    vu.assignPhysicsMethods(act0)    
    vu.assignConvenienceMethods(act0, legend)    

    act1Mapper = vtk.vtkPolyDataMapper() # the part which is cut away
    act1Mapper.SetInputConnection(clipper.GetClippedOutputPort())
    act1 = vtk.vtkActor()
    act1.SetMapper(act1Mapper)
    act1.GetProperty().SetColor(vc.getColor(c))
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
    vu.setInput(boxWidget, apd)
    boxWidget.PlaceWidget()    
    boxWidget.AddObserver("InteractionEvent", SelectPolygons)
    boxWidget.On()
    
    vio.printc('\nCutterWidget:\n Move handles to cut parts of the actor','m')
    vio.printc(' Press q to continue, Escape to exit','m')
    vio.printc((" Press X to save file to", outputname), 'm')
    def cwkeypress(obj, event):
        key = obj.GetKeySym()
        if   key == "q" or key == "space" or key == "Return":
            iren.ExitCallback()
        elif key == "X": 
            confilter = vtk.vtkPolyDataConnectivityFilter()
            vu.setInput(confilter, clipper.GetOutput())
            confilter.SetExtractionModeToLargestRegion()
            confilter.Update()
            cpd = vtk.vtkCleanPolyData()
            vu.setInput(cpd, confilter.GetOutput())
            cpd.Update()
            vio.write(cpd.GetOutput(), outputname)
        elif key == "Escape": 
            exit(0)
    
    iren.Initialize()
    iren.AddObserver("KeyPressEvent", cwkeypress)
    iren.Start()
    boxWidget.Off()
    return act0


def surfaceIntersection(actor1, actor2, tol=1e-06, lw=3,
                        c=None, alpha=1, legend=None):
    '''Intersect 2 surfaces and return a line actor'''
    try:
        bf = vtk.vtkIntersectionPolyDataFilter()
    except AttributeError:
        vio.printc('surfaceIntersection only possible for vtk version > 6','r')
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
        print ('Recalculating normals for', N, 'points')
        normals = vtk.vtkPCANormalEstimation()
        vu.setInput(normals, polyData)
        normals.SetSampleSize(int(N/50))
        normals.SetNormalOrientationToGraphTraversal()
        distance.SetInputConnection(normals.GetOutputPort())
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



def recoSurface2(points, bins=256,
                 c='gold', alpha=1, wire=False, bc='t', edges=False, legend=None):

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
        print ('Recalculating normals for', N, 'points')
        normals = vtk.vtkPCANormalEstimation()
        vu.setInput(normals, polyData)
        normals.SetSampleSize(int(N/50))
        normals.SetNormalOrientationToGraphTraversal()
        distance.SetInputConnection(normals.GetOutputPort())
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






