#!/usr/bin/env python
# A helper tool for visualizing vtk objects
from __future__ import print_function
__author__ = "Marco Musy"
__license__ = "MIT"
__version__ = "2.2"
__maintainer__ = "Marco Musy"
__email__ = "marco.musy@embl.es"
__status__ = "stable"

import os, vtk, numpy as np

vtkMV = vtk.vtkVersion().GetVTKMajorVersion() > 5
def setInput(vtkobj, p):
        if vtkMV: vtkobj.SetInputData(p)
        else:     vtkobj.SetInput(p)

#############################################################################
class vtkPlotter:

    def help(self):
        print ("""\n
        A python helper class to easily draw VTK tridimensional objects.
        Please follow instructions at:
        https://github.com/marcomusy/vtkPlotter
        Useful commands on graphic window:\n  """)
        print ("VTK version:", vtk.vtkVersion().GetVTKVersion())
        self.tips()

    def tips(self):
        print ("""Press ----------------------------------------
        m   to minimise opacity
        /   to maximize opacity
        .,  to increase/reduce opacity
        w/s to toggle wireframe/solid style
        D   to toggle edges visibility
        F   to flip normals
        C   to print current camera info
        O   to show vertices only
        123 to change color scheme
        V   to toggle verbose mode
        S   to save a screenshot
        q   to continue
        e   to close window and continue
        Esc to abort and exit
        Ctrl-mouse  to rotate scene
        Shift-mouse to shift scene
        Right-mouse click to zoom in/out
        --------------------------------------""")


    def __init__(self, shape=(1,1), size=(800,800), 
                bg=(1,1,1), bg2=None, balloon=False, 
                verbose=True, interactive=True):
        self.shape      = shape #nr of rows and columns
        self.size       = size
        self.balloon    = balloon
        self.verbose    = verbose
        self.renderer   = None  #current renderer
        self.renderers  = []
        self.interactive= interactive
        self.initialized= False
        self.axes       = True
        self.camera     = None
        self.commoncam  = True
        self.resetcam   = True
        self.parallelcam  = True
        self.camThickness = 2000
        self.actors     = []
        self.legend     = []
        self.names      = []
        self.tetmeshes  = []    # vtkUnstructuredGrid
        self.flat       = True
        self.phong      = False
        self.gouraud    = False
        self.bculling   = False
        self.fculling   = False
        self.legendSize = 0.25
        self.legendBG   = (.96,.96,.9)
        self.legendPosition = 2   # 1=topright
        self.result     = dict()  # stores extra output information
        self.caxes_exist = False

        if balloon:
            self.balloonWidget = vtk.vtkBalloonWidget()
            self.balloonRep = vtk.vtkBalloonRepresentation()
            self.balloonRep.SetBalloonLayoutToImageRight()
            self.balloonWidget.SetRepresentation(self.balloonRep)

        self.videoname = None
        self.videoformat = None
        self.fps       = 12
        self.frames    = []

        #######################################
        # build the renderers scene:
        for i in reversed(range(shape[0])):
            for j in range(shape[1]):
                arenderer = vtk.vtkRenderer()
                arenderer.SetBackground(bg)
                if bg2:
                    arenderer.GradientBackgroundOn()
                    arenderer.SetBackground2(bg2)
                x0 = i/float(shape[0])
                y0 = j/float(shape[1])
                x1 = (i+1)/float(shape[0])
                y1 = (j+1)/float(shape[1])
                arenderer.SetViewport(y0,x0, y1,x1)
                self.renderers.append(arenderer)
        self.renderWin = vtk.vtkRenderWindow()
        self.renderWin.BordersOn()
        self.renderWin.PolygonSmoothingOn()
        self.renderWin.LineSmoothingOn()
        self.renderWin.PointSmoothingOn()
        self.renderWin.SetSize(list(reversed(self.size)))
        for r in self.renderers: self.renderWin.AddRenderer(r)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renderWin)
        vsty = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(vsty)


    #######################################
    def loadXml(self, filename):
        if not os.path.exists(filename): return False
        if vtkMV:
            print ('Not yet tested on vtk 6.0 or higher.')
            return False
        try:
            import xml.etree.ElementTree as et
            if '.gz' in filename:
                import gzip
                inF = gzip.open(filename, 'rb')
                outF = open('/tmp/filename.xml', 'wb')
                outF.write( inF.read() )
                outF.close()
                inF.close()
                tree = et.parse('/tmp/filename.xml')
            else: tree = et.parse(filename)
            coords, connectivity = [], []
            for mesh in tree.getroot():
                for elem in mesh:
                    if self.verbose: print ('reading',elem.tag)
                    for e in elem.findall('vertex'):
                        x = float(e.get('x'))
                        y = float(e.get('y'))
                        z = float(e.get('z'))
                        coords.append([x,y,z])
                    for e in elem.findall('tetrahedron'):
                        v0 = int(e.get('v0'))
                        v1 = int(e.get('v1'))
                        v2 = int(e.get('v2'))
                        v3 = int(e.get('v3'))
                        connectivity.append([v0,v1,v2,v3])
            points = vtk.vtkPoints()
            for p in coords: points.InsertNextPoint(p)

            ugrid = vtk.vtkUnstructuredGrid()
            ugrid.SetPoints(points)
            cellArray = vtk.vtkCellArray()
            for itet in range(len(connectivity)):
                tetra = vtk.vtkTetra()
                for k,j in enumerate(connectivity[itet]):
                    tetra.GetPointIds().SetId(k, j)
                cellArray.InsertNextCell(tetra)
            ugrid.SetCells(vtk.VTK_TETRA, cellArray)
            self.tetmeshes.append(ugrid)
            # Create a mapper and actor this way
            #  3D cells are mapped only if they are used by only one cell,
            #  i.e., on the boundary of the data set
            #mapper = vtk.vtkDataSetMapper()
            #mapper.SetInputConnection(ugrid.GetProducerPort())
            #actor = vtk.vtkActor()
            #actor.SetMapper(mapper)
            if self.verbose:
                print ('Appending vtkUnstructuredGrid to vtkPlotter.tetmeshes')
        except:
            print ("Cannot parse xml file. Skip.", filename)
        try:
            if self.verbose:
                print ('Trying to convert fenics mesh file')
            import dolfin as dlf

            mesh = dlf.Mesh(filename)
            boundarysurf = dlf.BoundaryMesh(mesh, "exterior")
            dlf.File("/tmp/mesh.pvd") << boundarysurf
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName("/tmp/mesh000000.vtu")
            reader.Update()
            gf = vtk.vtkGeometryFilter()
            setInput(gf, reader.GetOutput())
            gf.Update()
            cl = vtk.vtkCleanPolyData()
            cl.SetInput(gf.GetOutput())
            cl.Update()
            poly = cl.GetOutput()
            b = poly.GetBounds()
            maxb = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
            V  = dlf.FunctionSpace(mesh, 'CG', 1)
            u  = dlf.Function(V)
            bc = dlf.DirichletBC(V, 1, dlf.DomainBoundary())
            bc.apply(u.vector())
            d2v = dlf.dof_to_vertex_map(V)
            idxs = d2v[u.vector() == 0.0] #indeces
            coords = mesh.coordinates()
            if self.verbose:
                print ('Appending tetrahedral vertices to vtkPlotter.actors')
            self.points(coords[idxs], r=maxb/400, c=(.8,0,.2), alpha=.2)
            self.names.append(filename)
            return poly
        except: return False


    def loadPoly(self, filename, reader=None):
        '''Return a vtkPolyData object'''
        if not os.path.exists(filename): return False
        fl = filename.lower()
        if '.vtk' in fl: reader = vtk.vtkPolyDataReader()
        if '.vtp' in fl: reader = vtk.vtkXMLPolyDataReader()
        if '.ply' in fl: reader = vtk.vtkPLYReader()
        if '.obj' in fl: reader = vtk.vtkOBJReader()
        if '.stl' in fl: reader = vtk.vtkSTLReader()
        if not reader: reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        if not reader.GetOutput(): return False
        mergeTriangles = vtk.vtkTriangleFilter()
        setInput(mergeTriangles, reader.GetOutput())
        mergeTriangles.Update()
        poly = mergeTriangles.GetOutput()
        self.names.append(filename)
        return poly


    def loadDir(self, mydir, tag='.'):
        '''Return a list of vtkActors from files in mydir of any formats'''
        if not os.path.exists(mydir): return False
        acts = []
        for ifile in sorted(os.listdir(mydir)):
            if tag in ifile:
                a = self.load(mydir + '/' + ifile)
                if a:
                    acts.append( a )
                    self.names.append( mydir + '/' + ifile )
        return acts


    def load(self, filename, reader=None, c='gold', alpha=0.2, 
             wire=False, bc=False, edges=False):
        '''Return a vtkActor, optional args:
           c,     color in RGB format, hex, symbol or name
           alpha, transparency (0=invisible)
           wire,  show surface as wireframe
           bc,    backface color of internal surface
        '''
        c = getcolor(c) # allow different codings
        if bc: bc = getcolor(bc)
        fl = filename.lower()
        if '.xml' in fl or '.xml.gz' in fl: # Fenics tetrahedral mesh file
            poly = self.loadXml(filename)
        elif '.pcd' in fl:                  # PCL point-cloud format
            f = open(filename, 'r')
            lines = f.readlines()
            f.close()
            start = False
            pts = []
            N, expN = 0, 0
            for text in lines:
                if start:
                    if N >= expN: break
                    l = text.split()
                    pts.append([float(l[0]),float(l[1]),float(l[2])])
                    N += 1
                if not start and 'POINTS' in text:
                    expN= int(text.split()[1])
                if not start and 'DATA ascii' in text:
                    start = True
            if expN != N:
                print ('Mismatch in pcd file', expN, len(pts))
            src = vtk.vtkPointSource()
            src.SetNumberOfPoints(len(pts))
            src.Update()
            poly = src.GetOutput()
            for i,p in enumerate(pts): poly.GetPoints().SetPoint(i, p)
        else:
            poly = self.loadPoly(filename, reader=reader)
        if not poly:
            print ('Unable to load', filename)
            return False
        actor = self.makeActor(poly, c, alpha, wire, bc, edges)
        self.actors.append(actor)
        return actor


    def getPD(self, obj, index=0):
        '''
        Returns polydata from an other object (vtkActor, vtkAssembly, int)
         e.g.: vp.getPD(3) #gets fourth's actor polydata
        '''
        if   isinstance(obj, vtk.vtkPolyData): return obj
        elif isinstance(obj, vtk.vtkActor):    return obj.GetMapper().GetInput()
        elif isinstance(obj, vtk.vtkActor2D):  return obj.GetMapper().GetInput()
        elif isinstance(obj, int): return self.actors[obj].GetMapper().GetInput()
        elif isinstance(obj, vtk.vtkAssembly):
            cl = vtk.vtkPropCollection()
            obj.GetActors(cl)
            cl.InitTraversal()
            for i in range(index+1):
                act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
            return act.GetMapper().GetInput()
        print ("Error: input is neither a poly nor an actor int or assembly.", obj)
        return False


    def getActors(self, r=None):
        '''Return the actors list in renderer number r
        if None, use current renderer'''
        if r is None: acs = self.renderer.GetActors()
        else: 
            if r>=len(self.renderers):
                print ("Error in getActors: non existing renderer",r)
                exit(0)
            acs = self.renderers[r].GetActors()
        acs.InitTraversal()
        acts=[]
        for i in range(acs.GetNumberOfItems()):
            a = acs.GetNextItem()
            if isinstance(a, vtk.vtkCubeAxesActor): continue
            if isinstance(a, vtk.vtkLegendBoxActor): continue
            acts.append(a)
        return acts


    def makeActor(self, poly, c='gold', alpha=0.5, 
                  wire=False, bc=False, edges=False):
        '''Return a vtkActor from an input vtkPolyData, optional args:
           c,     color in RGB format, hex, symbol or name
           alpha, transparency (0=invisible)
           wire,  show surface as wireframe
           bc,    backface color of internal surface
           edges, show edges as line on top of surface
        '''
        c = getcolor(c)
        if bc: bc = getcolor(bc)
        dataset = vtk.vtkPolyDataNormals()
        setInput(dataset, poly)
        dataset.SetFeatureAngle(60.0)
        dataset.ComputePointNormalsOn()
        dataset.ComputeCellNormalsOn()
        dataset.FlipNormalsOff()
        dataset.ConsistencyOn()
        dataset.Update()
        mapper = vtk.vtkPolyDataMapper()
        setInput(mapper, dataset.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetSpecular(0.025)
        if self.flat:    
            actor.GetProperty().SetInterpolationToFlat()
            self.phong = self.gouraud = False
            actor.GetProperty().SetSpecular(0)
        if self.phong:   
            actor.GetProperty().SetInterpolationToPhong()
            self.flat = self.gouraud = False
        if self.gouraud: 
            actor.GetProperty().SetInterpolationToGouraud()
            self.flat = self.phong = False
        if edges: actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetColor(c)
        actor.GetProperty().SetOpacity(alpha)
        if self.bculling:
            actor.GetProperty().BackfaceCullingOn()
        else:
            actor.GetProperty().BackfaceCullingOff()
        if self.fculling:
            actor.GetProperty().FrontfaceCullingOn()
        else:
            actor.GetProperty().FrontfaceCullingOff()
        if wire: actor.GetProperty().SetRepresentationToWireframe()
        if bc: # defines a specific color for the backface
            backProp = vtk.vtkProperty()
            backProp.SetDiffuseColor(bc)
            backProp.SetOpacity(alpha)
            actor.SetBackfaceProperty(backProp)
        return actor


    def assembly(self, actors):
        '''Treat many actors as a single new actor'''
        assembly = vtk.vtkAssembly()
        for a in actors: assembly.AddPart(a)
        return assembly


    def move_camera(self, camstart, camstop, fraction):
        '''Takes as input two vtkCamera objects and returns
        a new vtkCamera that is at intermediate position:
        fraction=0 -> camstart,  fraction=1 -> camstop.
        Press c key in interactive mode to dump a vtkCamera 
        parameter for the current camera view.
        '''
        cam = vtk.vtkCamera()
        cam.DeepCopy(camstart)
        p1 = np.array(camstart.GetPosition())
        f1 = np.array(camstart.GetFocalPoint())
        v1 = np.array(camstart.GetViewUp())
        s1 = np.array(camstart.GetParallelScale())
        p2 = np.array(camstop.GetPosition())
        f2 = np.array(camstop.GetFocalPoint())
        v2 = np.array(camstop.GetViewUp())
        s2 = np.array(camstop.GetParallelScale())
        cam.SetPosition(     p2*fraction+p1*(1.-fraction))
        cam.SetFocalPoint(   f2*fraction+f1*(1.-fraction))
        cam.SetViewUp(       v2*fraction+v1*(1.-fraction))
        cam.SetParallelScale(s2*fraction+s1*(1.-fraction))
        self.camera = cam
        self.show()


    #############################################################################
    def colorpoints(self, plist, cols, r=10., alpha=0.8):
        '''Return a vtkActor for a list of points.
        Input cols is a list of RGB colors of same length as plist
        '''
        if len(plist) != len(cols):
            print ("Mismatch in colorpoints()", len(plist), len(cols))
            quit()
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(plist))
        src.Update()
        vertexFilter = vtk.vtkVertexGlyphFilter()
        vertexFilter.SetInputData(src.GetOutput())
        vertexFilter.Update()
        pd = vertexFilter.GetOutput()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("RGB")
        for i,p in enumerate(plist):
            pd.GetPoints().SetPoint(i, p)
            c = np.array(getcolor(cols[i]))*255
            colors.InsertNextTupleValue(np.clip(c, 0, 255))
        pd.GetPointData().SetScalars(colors)
        mapper = vtk.vtkPolyDataMapper()
        setInput(mapper, pd)
        mapper.ScalarVisibilityOn()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToFlat()
        actor.GetProperty().SetOpacity(alpha)
        actor.GetProperty().SetPointSize(r)
        return actor


    def points(self, plist, c='b', r=10., alpha=1.):
        if isinstance(c, list):
            return self.colorpoints(plist, c, r, alpha)
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(plist))
        src.Update()
        pd = src.GetOutput()
        for i,p in enumerate(plist): pd.GetPoints().SetPoint(i, p)
        actor = self.makeActor(pd, c, alpha)
        actor.GetProperty().SetPointSize(r)
        self.actors.append(actor)
        return actor


    def line(self, p0,p1, lw=1, c='r', alpha=1.):
        '''Returns the line segment between points p0 and p1'''
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(p0)
        lineSource.SetPoint2(p1)
        lineSource.Update()
        actor = self.makeActor(lineSource.GetOutput(), c, alpha)
        actor.GetProperty().SetLineWidth(lw)
        self.actors.append(actor)
        return actor


    def sphere(self, pt, r=1, c='r', alpha=1.):
        src = vtk.vtkSphereSource()
        src.SetThetaResolution(24)
        src.SetPhiResolution(24)
        src.SetRadius(r)
        src.SetCenter(pt)
        src.Update()
        actor = self.makeActor(src.GetOutput(), c, alpha)
        actor.GetProperty().SetInterpolationToPhong()
        self.actors.append(actor)
        return actor


    def cube(self, pt, r=1, c='g', alpha=1.):
        src = vtk.vtkCubeSource()
        src.SetXLength(r)
        src.SetYLength(r)
        src.SetZLength(r)
        src.SetCenter(pt)
        src.Update()
        actor = self.makeActor(src.GetOutput(), c, alpha)
        self.actors.append(actor)
        return actor


    def grid(self, center=(0,0,0), normal=(0,0,1), s=10, N=10, 
             c='g', bc='darkgreen', lw=1, alpha=1, wire=True):
        '''Return a grid plane'''
        ps = vtk.vtkPlaneSource()
        ps.SetResolution(N, N)
        ps.SetCenter(np.array(center)/float(s))
        ps.SetNormal(normal)
        ps.Update()
        actor = self.makeActor(ps.GetOutput(), c=c, alpha=alpha)
        actor.SetScale(s,s,s)
        if wire: actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(lw)
        actor.PickableOff()
        self.actors.append(actor)
        if bc: # defines a specific color for the backface
            backProp = vtk.vtkProperty()
            backProp.SetDiffuseColor(getcolor(bc))
            backProp.SetOpacity(alpha)
            actor.SetBackfaceProperty(backProp)
        return actor
        
        
    def plane(self, center=(0,0,0), normal=(0,0,1), s=10, N=10, 
              c='g', bc='darkgreen', lw=1, alpha=1):
        return self.grid(center, normal, s, 1, c, bc, lw, alpha, 0)


    def arrow(self, startPoint, endPoint, c='r', alpha=1):
        arrowSource = vtk.vtkArrowSource()
        arrowSource.SetShaftResolution(24)
        arrowSource.SetTipResolution(24)
        arrowSource.SetTipRadius(0.06)
        nx = [0, 0, 0]
        ny = [0, 0, 0]
        nz = [0, 0, 0]
        math = vtk.vtkMath()
        math.Subtract(endPoint, startPoint, nx)
        length = math.Norm(nx)
        math.Normalize(nx)
        math.Cross(nx, [7.4,3.1,-1.2], nz)
        math.Normalize(nz)
        math.Cross(nz, nx, ny)
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, nx[i])
            matrix.SetElement(i, 1, ny[i])
            matrix.SetElement(i, 2, nz[i])
        transform = vtk.vtkTransform()
        transform.Translate(startPoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(arrowSource.GetOutputPort())
        transformPD.Update()
        actor = self.makeActor(transformPD.GetOutput(), c, alpha)
        actor.GetProperty().SetInterpolationToPhong()
        self.actors.append(actor)
        return actor


    def spline(self, points, s=10, c='navy', alpha=1., nodes=True):
        '''Return a vtkActor for a spline that goes exactly trought all
        list of points.
        nodes = True shows the points and therefore returns a vtkAssembly
        '''
        ## the spline passes through all points exactly
        numberOfOutputPoints = len(points)*20 # Number of points on the spline
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

        inputData = vtk.vtkPolyData() # Create a polydata to be glyphed.
        inputData.SetPoints(inputPoints)
        points = vtk.vtkPoints() # Generate the polyline for the spline.
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
        profileTubes = vtk.vtkTubeFilter() # Add thickness to the resulting line.
        profileTubes.SetNumberOfSides(12)
        setInput(profileTubes, profileData)
        profileTubes.SetRadius(s)
        profileTubes.Update()
        acttube = self.makeActor(profileTubes.GetOutput(), c=c, alpha=alpha)
        if nodes:
            balls = vtk.vtkSphereSource() # Use sphere as glyph source.
            balls.SetRadius(s*1.2)
            balls.SetPhiResolution(12)
            balls.SetThetaResolution(12)
            glyphPoints = vtk.vtkGlyph3D()
            setInput(glyphPoints, inputData)
            glyphPoints.SetSource(balls.GetOutput())
            actnodes = self.makeActor(glyphPoints.GetOutput(), c=c, alpha=alpha)
            acttube  = self.assembly([acttube, actnodes])
        self.actors.append(acttube)
        return acttube


    def bspline(self, points, nknots=-1,
                s=1, c=(0,0,0.8), alpha=1., nodes=True):
        '''Return a vtkActor for a spline that goes exactly trought all
        list of points.
        nknots= number of nodes used by the bspline. A small nr implies 
                a smoother interpolation. Default -1 gives max precision.
        nodes = True shows the points and therefore returns a vtkAssembly
        '''
        try:
            from scipy.interpolate import splprep, splev
        except ImportError:
            print ("Error in bspline(): scipy not installed. Skip.")
            return vtk.vtkActor()

        Nout = len(points)*20 # Number of points on the spline
        points = np.array(points)
        x,y,z = points[:,0], points[:,1], points[:,2]
        tckp, _ = splprep([x,y,z], nest=nknots) # find the knot points
        # evaluate spline, including interpolated points:
        xnew,ynew,znew = splev(np.linspace(0,1, Nout), tckp)
        ppoints = vtk.vtkPoints() # Generate the polyline for the spline.
        profileData = vtk.vtkPolyData()
        for i in range(Nout):
            ppoints.InsertPoint(i, xnew[i],ynew[i],znew[i])
        lines = vtk.vtkCellArray() # Create the polyline.
        lines.InsertNextCell(Nout)
        for i in range(Nout): lines.InsertCellPoint(i)
        profileData.SetPoints(ppoints)
        profileData.SetLines(lines)
        profileTubes = vtk.vtkTubeFilter() # Add thickness to the resulting line.
        profileTubes.SetNumberOfSides(12)
        setInput(profileTubes, profileData)
        profileTubes.SetRadius(s)
        profileTubes.Update()
        poly = profileTubes.GetOutput()
        acttube = self.makeActor(poly, c=c, alpha=alpha)
        if nodes:
            actnodes = self.points(points, r=s*50, c=c, alpha=alpha)
            self.actors.pop()
            acttube = self.assembly([acttube, actnodes])
        self.actors.append(acttube)
        return acttube


    def text(self, txt, pos=(0,0,0), s=1, c='k', alpha=1, cam=True, bc=False):
        '''Returns a vtkActor that shows a text 3D
           if cam is True the text will auto-orient to it
        '''
        c = getcolor(c) 
        if bc: bc = getcolor(bc)
        tt = vtk.vtkVectorText()
        tt.SetText(txt)
        ttmapper = vtk.vtkPolyDataMapper()
        ttmapper.SetInputConnection(tt.GetOutputPort())
        if cam: #follow cam
            ttactor = vtk.vtkFollower()
            ttactor.SetCamera(self.camera)
        else:
            ttactor = vtk.vtkActor()
        ttactor.SetMapper(ttmapper)
        ttactor.GetProperty().SetColor(c)
        ttactor.GetProperty().SetOpacity(alpha)
        ttactor.AddPosition(pos)
        ttactor.SetScale(s,s,s)
        if bc: # defines a specific color for the backface
            backProp = vtk.vtkProperty()
            backProp.SetDiffuseColor(bc)
            backProp.SetOpacity(alpha)
            ttactor.SetBackfaceProperty(backProp)
        self.actors.append(ttactor)
        return ttactor


    def xyplot(self, points, title='', c='r', pos=1, lines=False):
        """Return a vtkActor that is a plot of 2D points in x and y
           pos assignes the position: 
           1=topleft, 2=topright, 3=bottomleft, 4=bottomright
        """
        c = getcolor(c) # allow different codings
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
        tprop = plot.GetTitleTextProperty()
        tprop.SetColor(0,0,0)
        tprop.SetOpacity(0.7)
        tprop.SetFontFamily(1)
        tprop.BoldOff()
        tprop.ItalicOff()
        tprop.ShadowOff()
        tprop.SetFontSize(6) #not working
        plot.SetAxisTitleTextProperty(tprop)
        plot.SetAxisLabelTextProperty(tprop)
        plot.SetTitleTextProperty(tprop)
        if pos==1: plot.GetPositionCoordinate().SetValue(.0, .8, 0)
        if pos==2: plot.GetPositionCoordinate().SetValue(.7, .8, 0)
        if pos==3: plot.GetPositionCoordinate().SetValue(.0, .0, 0)
        if pos==4: plot.GetPositionCoordinate().SetValue(.7, .0, 0)
        plot.GetPosition2Coordinate().SetValue(.3, .2, 0)
        self.actors.append(plot)
        return plot


    def normals(self, pactor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8):
        '''Returns a vtkActor that contains the normals at vertices,
           these are shown as arrows.
        '''
        maskPts = vtk.vtkMaskPoints()
        maskPts.SetOnRatio(ratio)
        maskPts.RandomModeOff()
        src = self.getPD(pactor)
        setInput(maskPts, src)
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
        glyphActor.GetProperty().SetColor(getcolor(c))
        glyphActor.GetProperty().SetOpacity(alpha)
        self.actors.append(glyphActor)
        return glyphActor


    def curvature(self, pactor, ctype=1, r=1, alpha=1, lut=None):
        '''Returns a vtkActor that contains the color coded surface
           curvature following four different ways to calculate it:
           ctype =  0-gaussian, 1-mean, 2-max, 3-min
        '''
        poly = self.getPD(pactor)
        cleaner = vtk.vtkCleanPolyData()
        setInput(cleaner, poly)
        curve = vtk.vtkCurvatures()
        curve.SetInputConnection(cleaner.GetOutputPort())
        curve.SetCurvatureType(ctype)
        curve.InvertMeanCurvatureOn()
        curve.Update()
        if self.verbose: print ('CurvatureType set to:',ctype)
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
        cActor = vtk.vtkActor()
        cActor.SetMapper(cmapper)
        self.actors.append(cActor)
        return cActor


    def boundaries(self, pactor, c='p', lw=5):
        '''Returns a vtkActor that shows the boundary lines
           of a surface.
        '''
        fe = vtk.vtkFeatureEdges()
        setInput(fe, self.getPD(pactor))
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOn()
        fe.ManifoldEdgesOn()
        fe.NonManifoldEdgesOn()
        fe.ColoringOff()
        fe.Update()
        act = self.makeActor(fe.GetOutput(), c=c, alpha=1)
        act.GetProperty().SetLineWidth(lw)
        self.actors.append(act)
        return act


    ################# working with point clouds
    def fitline(self, points, c='orange', lw=1, alpha=0.6, tube=False):
        '''Fits a line through points.
           tube = show a rough estimate of error band at 2 sigma level
           Extra info is stored in vp.results['slope','center','variances']
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
        l = self.line(p1, p2, c=c, lw=lw, alpha=alpha)
        self.result['slope']  = vv
        self.result['center'] = datamean
        self.result['variances'] = dd
        if self.verbose:
            print ("Extra info saved in vp.results['slope','center','variances']")
        if tube: # show a rough estimate of error band at 2 sigma level
            tb = vtk.vtkTubeFilter()
            tb.SetNumberOfSides(48)
            setInput(tb, self.getPD(l))
            r = np.sqrt((dd[1]+dd[2])/2./len(points))
            tb.SetRadius(r)
            a = self.makeActor(tb.GetOutput(), c=c, alpha=alpha/4.)
            l = self.assembly([l,a])
            self.actors[-1] = l # replace
        return l


    def fitplane(self, points, c='g', bc='darkgreen'):
        '''Fits a plane to a set of points.
           Extra info is stored in vp.results['normal','center','variance']
        '''
        data = np.array(points)
        datamean = data.mean(axis=0)
        uu, dd, vv = np.linalg.svd(data - datamean)
        xyz_min = points.min(axis=0)
        xyz_max = points.max(axis=0)
        s= np.linalg.norm(xyz_max - xyz_min)
        n = np.cross(vv[0],vv[1])
        pla = self.plane(datamean, n, c=c, bc=bc, s=s, lw=2, alpha=0.8)
        self.result['normal']  = n
        self.result['center']  = datamean
        self.result['variance']= dd[2]
        if self.verbose:
            print ("Extra info saved in vp.results['normal','center','variance']")
        return pla


    def ellipsoid(self, points, pvalue=.95, c='c', alpha=0.5, pcaaxes=False):
        '''Show the oriented PCA ellipsoid that contains 95% of points.
           axes = True, show the 3 PCA semi axes
           Extra info is stored in vp.results['sphericity','a','b','c']
           sphericity = 1 for a perfect sphere
        '''
        try:
            from scipy.stats import f
        except:
            print ("Error in ellipsoid(): scipy not installed. Skip.")
            return vtk.vtkActor()
        P = np.array(points, ndmin=2, dtype=float)
        cov = np.cov(P, rowvar=0)    # covariance matrix
        U, s, R = np.linalg.svd(cov) # singular value decomposition
        p, n = s.size, P.shape[0]
        fppf = f.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p) # f % point function
        va,vb,vc = np.sqrt(s*fppf)   # semi-axes (largest first)
        center = np.mean(P, axis=0)  # centroid of the hyperellipsoid
        self.result['sphericity'] =  1-np.sqrt(((va-vb)/(va+vb))**2
                                             + ((va-vc)/(va+vc))**2
                                             + ((vb-vc)/(vb+vc))**2 )/1.7321*2
        self.result['a'] = va
        self.result['b'] = vb
        self.result['c'] = vc
        if self.verbose:
            print ("Extra info saved in vp.results['sphericity','a','b','c']")
        elliSource = vtk.vtkSphereSource()
        elliSource.SetThetaResolution(48)
        elliSource.SetPhiResolution(48)
        matri = vtk.vtkMatrix4x4()
        matri.DeepCopy((R[0][0] *va, R[1][0] *vb, R[2][0] *vc, center[0],
                        R[0][1] *va, R[1][1] *vb, R[2][1] *vc, center[1],
                        R[0][2] *va, R[1][2] *vb, R[2][2] *vc, center[2], 0,0,0,1))
        vtra = vtk.vtkTransform()
        vtra.SetMatrix(matri)
        ftra = vtk.vtkTransformFilter()
        ftra.SetTransform(vtra)
        ftra.SetInputConnection(elliSource.GetOutputPort())
        ftra.Update()
        actor_elli = self.makeActor(ftra.GetOutput(), c, alpha)
        actor_elli.GetProperty().BackfaceCullingOn()
        actor_elli.GetProperty().SetInterpolationToPhong()
        if pcaaxes:
            axs = []
            for ax in ([1,0,0], [0,1,0], [0,0,1]):
                l = vtk.vtkLineSource()
                l.SetPoint1([0,0,0])
                l.SetPoint2(ax)
                l.Update()
                t = vtk.vtkTransformFilter()
                t.SetTransform(vtra)
                setInput(t, l.GetOutput())
                t.Update()
                axs.append(self.makeActor(t.GetOutput(), c, alpha))
            self.actors.append( self.assembly([actor_elli]+axs) )
        else : self.actors.append(actor_elli)
        return self.lastactor()


    def align(self, source, target, rigid=False, iters=100):
        '''Return a vtkActor which is the same as source but
           aligned to target though IterativeClosestPoint method
           rigid = True, then no scaling is allowed.
        '''
        sprop = source.GetProperty()
        source = self.getPD(source)
        target = self.getPD(target)
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(source)
        icp.SetTarget(target)
        if rigid: icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(iters)
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()
        icpTransformFilter = vtk.vtkTransformPolyDataFilter()
        setInput(icpTransformFilter, source)
        icpTransformFilter.SetTransform(icp)
        icpTransformFilter.Update()
        poly = icpTransformFilter.GetOutput()
        actor = self.makeActor(poly)
        actor.SetProperty(sprop)
        return actor


    ##########################################
    def draw_cubeaxes(self, c=(.2, .2, .6)):
        if self.caxes_exist and not self.axes: return
        ca = vtk.vtkCubeAxesActor()
        if self.renderer:
            ca.SetBounds(self.renderer.ComputeVisiblePropBounds())
        if self.camera: ca.SetCamera(self.camera)
        else:  ca.SetCamera(self.renderer.GetActiveCamera())
        if vtkMV:
            ca.GetXAxesLinesProperty().SetColor(c)
            ca.GetYAxesLinesProperty().SetColor(c)
            ca.GetZAxesLinesProperty().SetColor(c)
            for i in range(3):
                ca.GetLabelTextProperty(i).SetColor(c)
                ca.GetTitleTextProperty(i).SetColor(c)
            ca.XAxisLabelVisibilityOn()
            ca.YAxisLabelVisibilityOff()
            ca.ZAxisLabelVisibilityOff()
            ca.SetXTitle('x-axis')
            ca.SetTitleOffset(10)
        else:
            ca.GetProperty().SetColor(c)
        ca.SetFlyMode(3)
        ca.XAxisMinorTickVisibilityOff()
        ca.YAxisMinorTickVisibilityOff()
        ca.ZAxisMinorTickVisibilityOff()
        self.caxes_exist = True
        self.renderer.AddActor(ca)

    def draw_ruler(self):
        #draws a simple ruler at the bottom
        ls = vtk.vtkLegendScaleActor()
        ls.RightAxisVisibilityOff()
        ls.TopAxisVisibilityOff()
        ls.LegendVisibilityOff()
        ls.LeftAxisVisibilityOff()
        ls.GetBottomAxis().SetNumberOfMinorTicks(1)
        ls.GetBottomAxis().GetProperty().SetColor(0,0,0)
        ls.GetBottomAxis().GetLabelTextProperty().SetColor(0,0,0)
        ls.GetBottomAxis().GetLabelTextProperty().BoldOff()
        ls.GetBottomAxis().GetLabelTextProperty().ItalicOff()
        ls.GetBottomAxis().GetLabelTextProperty().ShadowOff()
        self.renderer.AddActor(ls)

    def draw_legend(self):
        texts = []
        for t in self.legend:
            if t=='': continue
            texts.append(t)
        N = len(texts)
        if N > len(self.actors):
            print ('Mismatch in Legend:')
            print ('only', len(self.actors), 'actors but', N, 'legend lines.')
            return
        legend = vtk.vtkLegendBoxActor()
        legend.SetNumberOfEntries(N)
        legend.UseBackgroundOn()
        legend.SetBackgroundColor(self.legendBG)
        legend.SetBackgroundOpacity(0.6)
        legend.LockBorderOn()
        for i in range(N):
            a = self.actors[i]
            if isinstance(a, vtk.vtkAssembly):
                cl = vtk.vtkPropCollection()
                a.GetActors(cl)
                cl.InitTraversal()
                act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
                c = act.GetProperty().GetColor()
                if c==(1,1,1): c=(0.5,0.5,0.5)
                legend.SetEntry(i, self.getPD(a), "  "+texts[i], c)
            else:
                c = a.GetProperty().GetColor()
                if c==(1,1,1): c=(0.5,0.5,0.5)
                legend.SetEntry(i, self.getPD(i), "  "+texts[i], c)
        pos = self.legendPosition
        width = self.legendSize
        legend.SetWidth(width)
        legend.SetHeight(width/5.*N)
        sx, sy = 1-width, 1-width/5.*N
        if pos==1: legend.GetPositionCoordinate().SetValue(  0, sy) #x,y from bottomleft
        if pos==2: legend.GetPositionCoordinate().SetValue( sx, sy) #default
        if pos==3: legend.GetPositionCoordinate().SetValue(  0,  0)
        if pos==4: legend.GetPositionCoordinate().SetValue( sx,  0)
        self.renderer.AddActor(legend)


    ###############################################################################
    def show(self, actors=None, legend=None, at=0, #at=render wind. nr.
             axes=None, ruler=False, interactive=None, outputimage=None,
             c='gold', alpha=0.2, wire=False, bc=False, edges=False):
        '''
        Input: a mixed list of vtkActors, vtkPolydata and filename strings
        legend = a string or list of string for each actor. Empty string skips.
        at     = number of the renderer to plot to.
        axes   = show xyz axes
        ruler  = draws a simple ruler at the bottom
        interactive = pause and interact w/ window or continue execution
        outputimage = filename to dump a screenshot without asking
        wire   = show in wireframe representation
        edges  = show the edges on top of surface
        bc     = background color, set a color for the back surface face
        '''
        
        # override what was stored internally with passed input
        if not actors is None:
            if not isinstance(actors, list): self.actors = [actors]
            else: self.actors = actors
        if not legend is None:
            if isinstance(legend, list): self.legend = list(legend)
            if isinstance(legend,  str): self.legend = [str(legend)]
        if not axes        is None: self.axes = axes
        if not interactive is None: self.interactive = interactive
        if self.verbose:
            print ('Drawing', len(self.actors),'actors ', end='')
            if self.shape != (1,1): print ('on window',at,'-', end='')
            else: print (' - ', end='')
            if self.interactive: print ('Interactive mode: On.')
            else: print ('Interactive mode: Off.')

        if at<len(self.renderers):
            self.renderer = self.renderers[at]
        else:
            print ("Error in show(): wrong renderer index",at)
            return
        if not self.camera: self.camera = self.renderer.GetActiveCamera()
        else: self.camera.SetThickness(self.camThickness)
        if self.parallelcam: self.camera.ParallelProjectionOn()
        if self.commoncam:
            for r in self.renderers: r.SetActiveCamera(self.camera)

        for i in range(len(self.actors)): # scan for filepaths
            a = self.actors[i]
            if isinstance(a, str): #assume a filepath was given
                ok = self.load(a, c=c, bc=bc, alpha=alpha, wire=wire, edges=edges)
                if ok:
                    self.actors[i] = self.actors.pop() #put it in right position

        for i in range(len(self.actors)): # scan for polydata
            a = self.actors[i]
            if isinstance(a, vtk.vtkPolyData): #assume a filepath was given
                act = self.makeActor(a, c=c, bc=bc, alpha=alpha, 
                                     wire=wire, edges=edges)
                self.actors[i] = act #put it in right position

        acts = self.getActors()
        for ia in self.actors: 
            if not ia in acts: self.renderer.AddActor(ia)

        if ruler: self.draw_ruler()
        if self.axes: self.draw_cubeaxes()
        if self.legend or len(self.legend): self.draw_legend()

        if self.resetcam: self.renderer.ResetCamera()

        if not self.initialized:
            self.interactor.Initialize()
            self.initialized = True
            self.interactor.AddObserver("LeftButtonPressEvent", self.mouseleft)
            self.interactor.AddObserver("KeyPressEvent", self.keypress)
            if self.verbose: self.tips()
            if self.balloon:
                self.balloonWidget.SetInteractor(self.interactor)
                self.balloonWidget.EnabledOn()
        if self.balloon:
            for i, ia in enumerate(self.actors):
                if i<len(self.names):
                    self.balloonWidget.AddBalloon(ia, self.names[i])

        if self.renderer: self.renderWin.Render()
        if outputimage: screenshot(outputimage)
        if self.renderer and self.interactive: self.interact()


    ############################### events
    def mouseleft(self, obj, event):
        x,y = self.interactor.GetEventPosition()
        self.renderer = obj.FindPokedRenderer(x,y)
        self.renderWin = obj.GetRenderWindow()
        #print ('Renderer clicked:', self.renderers.index(self.renderer))

    def keypress(self, obj, event):
        key = obj.GetKeySym()
        #print (key)
        if   key == "q" or key == "space":
            self.interactor.ExitCallback()
        elif key == "e":
            if self.verbose:
                print ("Closing window and return control to python.")
            rw = self.interactor.GetRenderWindow()
            rw.Finalize()
            self.interactor.TerminateApp()
            del self.renderWin, self.interactor
            return
        elif key == "Escape":
            if self.verbose: print ("Quitting, Bye.")
            exit(0)
        elif key == "S":
            print ('Saving window as screenshot.png')
            screenshot()
        elif key == "C":
            cam = self.renderer.GetActiveCamera()
            print ('\ncam = vtk.vtkCamera() #example code')
            print ('cam.SetPosition(',  [round(e,3) for e in cam.GetPosition()],  ')')
            print ('cam.SetFocalPoint(',[round(e,3) for e in cam.GetFocalPoint()],')')
            print ('cam.SetParallelScale(',round(cam.GetParallelScale(),3),')')
            print ('cam.SetViewUp(', [round(e,3) for e in cam.GetViewUp()],')')
            return
        elif key == "m":
            for a in self.getActors():
                a.GetProperty().SetOpacity(0.05)
        elif key == "comma":
            for a in self.getActors():
                ap = a.GetProperty()
                ap.SetOpacity(max([ap.GetOpacity()-0.05, 0.05]))
        elif key == "period":
            for a in self.getActors():
                ap = a.GetProperty()
                ap.SetOpacity(min([ap.GetOpacity()+0.05, 1.0]))
        elif key == "slash":
            for a in self.getActors():
                a.GetProperty().SetOpacity(1)
        elif key == "V":
            if not(self.verbose): self.tips()
            self.verbose = not(self.verbose)
            print ("Verbose: ", self.verbose)
        elif key in ["1", "KP_End", "KP_1"]:
            for i,ia in enumerate(self.getActors()):
                ia.GetProperty().SetColor(colors1[i])
        elif key in ["2", "KP_Down", "KP_2"]:
            for i,ia in enumerate(self.getActors()):
                ia.GetProperty().SetColor(colors2[i])
        elif key in ["3", "KP_Next", "KP_3"]:
            for i,ia in enumerate(self.getActors()):
                ia.GetProperty().SetColor(colors3[i])
        elif key == "o":
            for ia in self.getActors():
                ps = ia.GetProperty().GetPointSize()
                ia.GetProperty().SetPointSize(ps-1)
                ia.GetProperty().SetRepresentationToPoints()
        elif key == "O":
            for ia in self.getActors():
                try:
                    ps = ia.GetProperty().GetPointSize()
                    ia.GetProperty().SetPointSize(ps+2)
                    ia.GetProperty().SetRepresentationToPoints()
                except AttributeError: pass
        elif key == "D":
            for ia in self.getActors():
                try:
                    ev = ia.GetProperty().GetEdgeVisibility()
                    ia.GetProperty().SetEdgeVisibility(not(ev))
                    c = ia.GetProperty().GetColor()
                    ia.GetProperty().SetEdgeColor(c)
                except AttributeError: pass
        elif key == "N":
            for ia in self.getActors():
                try:
                    rs = vtk.vtkReverseSense()
                    rs.ReverseNormalsOn()
                    setInput(rs, self.getPD(ia))
                    rs.Update()
                    ns = rs.GetOutput().GetPointData().GetNormals()
                    rna = vtk.vtkFloatArray.SafeDownCast(ns)
                    ia.GetMapper().GetInput().GetPointData().SetNormals(rna)
                    del rs
                except: 
                    print ("Cannot flip normals.")
        self.interactor.Render()


    def interact(self):
        if hasattr(self, 'interactor'):
            self.interactor.Render()
            self.interactor.Start()

    def lastactor(self): return self.actors[-1]

    
    def open_video(self, name='movie.avi', fps=12, duration=None, format="XVID"):
        try:
            import cv2, glob
        except:
            print ("open_video: cv2 not installed? Skip.")
            return
        self.videoname = name
        self.videoformat = format
        self.videoduration = duration
        self.fps = float(fps) # if duration is given, will be recalculated
        self.frames = []
        if not os.path.exists('/tmp/v'): os.mkdir('/tmp/v')
        for fl in glob.glob("/tmp/v/*.png"): os.remove(fl)
        print ("Video", name, "is open. Press q to continue.")
        itr = bool(self.interactive)
        self.show(interactive=True)
        self.interactive = itr
        
    def addframe_video(self):
        if not self.videoname: return
        fr = '/tmp/v/'+str(len(self.frames))+'.png'
        screenshot(fr)
        self.frames.append(fr)

    def pause_video(self, pause):
        '''insert a pause, in seconds'''
        if not self.videoname: return
        fr = self.frames[-1]
        n = int(self.fps*pause)
        for i in range(n): 
            fr2='/tmp/v/'+str(len(self.frames))+'.png'
            self.frames.append(fr2)
            os.system("cp -f %s %s" % (fr, fr2))
            
    def release_video(self):        
        if not self.videoname: return
        import cv2
        if self.videoduration:
            self.fps = len(self.frames)/float(self.videoduration)
            print ("Recalculated video FPS to", round(self.fps,3))
        fourcc = cv2.cv.CV_FOURCC(*self.videoformat)
        vid = None
        size = None
        for image in self.frames:
            if not os.path.exists(image):
                print ('Image not found:', image)
                continue
            img = cv2.imread(image)
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = cv2.VideoWriter(self.videoname, 
                                      fourcc, self.fps, size, True)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = cv2.resize(img, size)
            vid.write(img)
        if self.verbose: print ("Video saved to:", self.videoname)
        vid.release()
    
    def release_gif(self):
        if not self.videoname: return
        try: import imageio
        except: 
            print ("open_video: imageio not installed? Skip.")
            return
        images = []
        for fl in self.frames:
            images.append(imageio.imread(fl))
        imageio.mimsave('animation.gif', images)


#########################################################
# basic color schemes
######################################################### 
colors= [[.9,0.05,0.05], [0,.95,0], [0, .2,.9],[0,1,1], 
        [.94,.2,.9],[1,1,0],[0,0,0],[1,1,1],[1,.388,.278],
        [.5,.5,0],[.5,0,.5], [.5,0,0], [.67,.43,.157],
        [0,.5,.5],[0,.5,0], [0,0,.5], [1,.627,0.478],
        [.92,.757,0],[.9,.745,1], [1,.5,.19], [1,.745,.7], 
        [1,.98,.75],[.5,1,.831],[.5,.5,.5] ]
color_names= ['red','green','blue','cyan',
              'magenta','yellow', 'black', 'white','tomato',
              'olive', 'purple', 'maroon', 'brown',
              'teal','darkgreen','navy','salmon',
              'gold','lavender','orange','pink',
              'beige','aqua','grey']
color_nicks= ['r','g','b','c','m','y','k','w','t','o','p']

colors1=[]
colors1.append((1.0,0.647,0.0))     # orange
colors1.append((0.59,0.0,0.09))     # dark red
colors1.append((0.5,1.0,0.0))       # green
colors1.append((0.5,0.5,0))         # yellow-green
colors1.append((0.0, 0.66,0.42))    # green blue
colors1.append((0.0,0.18,0.65))     # blue
colors1.append((0.4,0.0,0.4))       # plum
colors1.append((0.4,0.0,0.6))
colors1.append((0.2,0.4,0.6))
colors1.append((0.1,0.3,0.2))
colors1 = colors1 * 100

colors2=[]
colors2.append((0.99,0.83,0))       # gold
colors2.append((0.59, 0.0,0.09))    # dark red
colors2.append((.984,.925,.354))    # yellow
colors2.append((0.5,  0.5,0))       # yellow-green
colors2.append((0.5,  1.0,0.0))     # green
colors2.append((0.0, 0.66,0.42))    # green blue
colors2.append((0.0, 0.18,0.65))    # blue
colors2.append((0.4,  0.0,0.4))     # plum
colors2 = colors2 * 100

colors3=[]
for i in range(10):
    pc = (i+0.5)/10.
    sig = 0.2
    r = np.exp(-((pc-0.0)/sig)**2/2.)
    g = np.exp(-((pc-0.5)/sig)**2/2.)
    b = np.exp(-((pc-1.0)/sig)**2/2.)
    colors3.append((r,g,b))
colors3 = colors3 * 100


#########################################################
# Useful Functions
######################################################### 
def screenshot(filename='screenshot.png'):
    try:
        import gtk.gdk
        w = gtk.gdk.get_default_root_window().get_screen().get_active_window()
        sz = w.get_size()
        pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, sz[0], sz[1])
        pb = pb.get_from_drawable(w,w.get_colormap(),0,0,0,0, sz[0], sz[1])
        if pb is not None:
            pb.save(filename, "png")
            #print ("Screenshot saved to", filename)
        else: print ("Unable to save the screenshot. Skip.")
    except:
        print ("Unable to take the screenshot. Skip.")


####################################
def makeList(vtksrcs):
    if not isinstance(vtksrcs, list):
        vtksrcs = [vtksrcs]
    pts = []
    for ipoly in range(len(vtksrcs)):
        apoly = vtksrcs[ipoly]
        for i in range(apoly.GetNumberOfPoints()):
            p = [0, 0, 0]
            apoly.GetPoint(i, p)
            pts.append(p)
    return pts


####################################
def makeSource(spoints, addLines=True):
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
    source.Update()
    return source


####################################
def isinside(poly, point):
    """Return True if point is inside a polydata closed surface"""
    points = vtk.vtkPoints()
    points.InsertNextPoint(point)
    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)
    selectEnclosedPoints = vtk.vtkSelectEnclosedPoints()
    selectEnclosedPoints.SetInput(pointsPolydata)
    selectEnclosedPoints.SetSurface(poly)
    selectEnclosedPoints.Update()
    return selectEnclosedPoints.IsInside(0)
    
    
####################################
def closestPoint(polydata, pt, locator=None):
    """Find the closest point on a surface given an other point"""
    trgp  = [0,0,0]
    cid   = vtk.mutable(0)
    subid = vtk.mutable(0)
    dist2 = vtk.mutable(0)
    if not locator:
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(polydata)
        locator.BuildLocator()
    locator.FindClosestPoint(pt, trgp, cid, subid, dist2)
    return (trgp, dist2)


####################################
def write(poly, fileoutput):
    wt = vtk.vtkPolyDataWriter()
    setInput(wt, poly)
    wt.SetFileName(fileoutput)
    print ("Writing", fileoutput, v.GetNumberOfPoints(),"points.")
    wt.Write()

####################################
def getcolor(c):
    #Convert a color to (r,g,b) format from many input formats
    if isinstance(c,list) or isinstance(c,tuple) : #RGB
        if c[0]<=1 and c[1]<=1 and c[2]<=1:
            return c
        else:
            return list(np.array(c)/255.)
    if isinstance(c,str):
        if '#' in c: #hex to rgb
            h = c.lstrip('#')
            rgb255 = list(int(h[i:i+2], 16) for i in (0, 2 ,4))
            rgb = np.array(rgb255)/255.
            if np.sum(rgb)>1: return [0,0,0]
            return list(rgb)
        if len(c)==1: #single letter color
            cc = color_nicks
        else:         #full name color
            cc = color_names
        try: 
            ic = cc.index(c.lower())
            return colors[ic]        
        except ValueError:
            # ToDo: add vtk6 defs for colors
            print ("Unknow color name", c, 'is not in:\n', cc)
            if len(c)==1: print ("Available colors:\n", cols_names)
            return [0,0,0]
    if isinstance(c, int): 
        return colors1[c]
    return [0,0,0]
    

###########################################################################
if __name__ == '__main__':
###########################################################################
    try:
        import sys
        fs = sys.argv[1:]
        if len(fs)==1: 
            leg=None
        else: 
            leg=fs
            print ('Loading',len(fs),'files:', fs)
        vp = vtkPlotter(bg2=(.94,.94,1), balloon=False)
        acts =[]
        alpha = 1./len(fs)
        for f in fs:
            vp.load(f, alpha=alpha)
        vp.show(legend=leg)
    except:
        print ("Something went wrong.")
        print ("Usage: plotter.py file*.vtk  # [vtp,ply,obj,stl,xml,pcd]")
###########################################################################









