#!/usr/bin/env python
# A helper tool for visualizing vtk objects
__author__ = "Marco Musy"
__license__ = "MIT"
__version__ = "2.0"
__maintainer__ = "Marco Musy"
__email__ = "marco.musy@embl.es"
__status__ = "Production"

import vtk, numpy as np


#############################################################################
class vtkPlotter:
    
    def help(self):
        print """\n 
        A python helper class to easily draw VTK tridimensional objects.
        Please follow instructions at:
        https://github.com/marcomusy/vtkPlotter
        Useful commands on graphic window:
        """
        self.tips()        

    def tips(self):
        print "Press --------------------------------"
        print " m   to minimise opacity"
        print " /   to maximize opacity"
        print " .,  to increase/reduce opacity"
        print " w/s to toggle wireframe/solid style"
        print " v   to toggle verbose mode"
        print " S   to save a screenshot"
        print " c   to print current camera info"
        print " q   to return to python session"
        print " e   to close window and return"
        print " Esc to abort and exit python "
        print " Move mouse to change 3D point of view"
        print "      Ctrl-mouse to rotate scene"
        print "      Shift-mouse to shift scene"
        print "      Right-mouse to zoom in/out"
        print "--------------------------------------"

        
    def __init__(self, shape=(1,1), size=(800,800), bg=(1,1,1)):
        self.shape      = shape #nr of rows and columns
        self.windowsize = size
        self.renderer   = None  #current renderer
        self.renderers  = []
        self.interactive= True
        self.initialized= False
        self.axes       = True
        self.camera     = None
        self.commoncam  = True
        self.resetcam   = True
        self.parallelcam  = True
        self.camThickness = 2000
        self.actors     = []
        self.legend     = []
        self.files      = []   
        self.tetmeshes  = []     # vtkUnstructuredGrid
        self.result     = dict() # contains extra output information
        self.verbose    = True
        self.phong      = True
        self.flat       = False
        self.gouraud    = False
        self.colors     = []
        self.colors1    = []
        self.colors2    = []
        self.legendSize = 0.8
        self.legendBG   = (.96,.96,.9)
        self.legendPosition = 2   # 1=topleft
        
        #######################################
        # build the renderers scene:
        for i in reversed(range(shape[0])): 
            for j in range(shape[1]): 
                arenderer = vtk.vtkRenderer()	
                arenderer.SetBackground(bg)
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
        self.renderWin.SetSize(list(reversed(self.windowsize)))
        for r in self.renderers: self.renderWin.AddRenderer(r)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renderWin)
        vsty = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(vsty)

        ######################################### color schemes
        self.colors.append((1.0,0.647,0.0))     # orange
        self.colors.append((0.59,0.0,0.09))     # dark red
        self.colors.append((0.5,1.0,0.0))       # green
        self.colors.append((0.5,0.5,0))         # yellow-green
        self.colors.append((0.0, 0.66,0.42))    # green blue
        self.colors.append((0.0,0.18,0.65))     # blue
        self.colors.append((0.4,0.0,0.4))       # plum
        self.colors.append((0.4,0.0,0.6))
        self.colors.append((0.2,0.4,0.6))
        self.colors.append((0.1,0.3,0.2))
        self.colors = self.colors * 100

        self.colors1.append((0.99,0.83,0))       # gold
        self.colors1.append((0.59, 0.0,0.09))    # dark red
        self.colors1.append((.984,.925,.354))    # yellow
        self.colors1.append((0.5,  0.5,0))       # yellow-green
        self.colors1.append((0.5,  1.0,0.0))     # green           
        self.colors1.append((0.0, 0.66,0.42))    # green blue
        self.colors1.append((0.0, 0.18,0.65))    # blue
        self.colors1.append((0.4,  0.0,0.4))     # plum
        self.colors1 = self.colors1 * 100      
        
        for i in range(10):
            pc = (i+0.5)/10.
            sig = 0.2
            r = np.exp(-((pc-0.0)/sig)**2/2.)
            g = np.exp(-((pc-0.5)/sig)**2/2.)
            b = np.exp(-((pc-1.0)/sig)**2/2.)
            self.colors2.append((r,g,b))
        self.colors2 = self.colors2 * 100
  

    #######################################
    def loadXml(self, filename):
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
                    if self.verbose: print 'reading',elem.tag
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
                print 'Appending vtkUnstructuredGrid to vtkPlotter.tetmeshes'
        except:
            print "Cannot parse xml file. Skip.", filename
        try:
            if self.verbose: 
                print 'Trying to convert fenics mesh file'
            import dolfin as dlf

            mesh = dlf.Mesh(filename)
            boundarysurf = dlf.BoundaryMesh(mesh, "exterior")
            dlf.File("/tmp/mesh.pvd") << boundarysurf
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName("/tmp/mesh000000.vtu")
            reader.Update()
            gf = vtk.vtkGeometryFilter()
            gf.SetInput(reader.GetOutput())
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
                print 'Appending tetrahedral vertices to vtkPlotter.actors'
            self.make_points(coords[idxs], r=maxb/400, c=(.8,0,.2), alpha=.2)
            self.files.append(filename)
            act = self.makeActor(poly, c=(.8,0,.2), alpha=.2)
            self.actors.append(act)
            return act
        except: 
            print 'Unable to read fenics mesh file', filename
            return False
        
             
    def loadPoly(self, filename, reader=None):
        fl = filename.lower()
        if '.vtk' in fl: reader = vtk.vtkPolyDataReader()
        if '.vtp' in fl: reader = vtk.vtkXMLPolyDataReader()
        if '.ply' in fl: reader = vtk.vtkPLYReader()
        if '.obj' in fl: reader = vtk.vtkOBJReader()
        if '.stl' in fl: reader = vtk.vtkSTLReader()
        if not reader: reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)    
        reader.Update()
        if not reader.GetOutput(): 
            print 'Unable to load', filename
            return False
        mergeTriangles = vtk.vtkTriangleFilter()
        mergeTriangles.SetInput(reader.GetOutput())
        mergeTriangles.Update()
        poly = mergeTriangles.GetOutput()
        self.files.append(filename)
        return poly
 
 
    def loadDir(self, mydir, tag='.'):
        import os
        acts = []
        self.files = []
        for ifile in sorted(os.listdir(mydir)):
            if tag in ifile:
                a = self.load(mydir + '/' + ifile)
                if a:
                    acts.append( a )
                    self.files.append( mydir + '/' + ifile )
        if not len(self.files): 
            print 'No files found containing tag:', tag, 'in', mydir
        return acts


    def load(self, filename, reader=None, c=(1,0.647,0), alpha=0.2):
        fl = filename.lower()
        if '.xml' in fl or '.xml.gz' in fl: # Fenics tetrahedral mesh file
            return self.loadXml(filename)
        poly = self.loadPoly(filename, reader=reader)
        actor  = self.makeActor(poly, c, alpha)
        self.actors.append(actor)
        return actor


    def getPD(self, polyOrActor):
        # returns polydata from an other object
        if isinstance(polyOrActor, vtk.vtkPolyData): return polyOrActor
        elif isinstance(polyOrActor, vtk.vtkActor):
            return polyOrActor.GetMapper().GetInput()
        elif isinstance(polyOrActor, vtk.vtkActor2D):
            return polyOrActor.GetMapper().GetInput()
        elif isinstance(polyOrActor, int):
            return self.actors[polyOrActor].GetMapper().GetInput()
        print "Error: input is neither a poly nor an actor.", polyOrActor
        quit()


    def make_screenshot(self, filename='screenshot.png'):
        try:
            import gtk.gdk
            w = gtk.gdk.get_default_root_window().get_screen().get_active_window()
            sz = w.get_size()
            if self.verbose: print "The size of active window is %d x %d" % sz
            pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, sz[0], sz[1])
            pb = pb.get_from_drawable(w,w.get_colormap(),0,0,0,0, sz[0], sz[1])
            if pb is not None:
                pb.save(filename, "png")
                print "Screenshot saved to", filename
            else: print "Unable to save the screenshot. Skip."
        except:
            print "Unable to take the screenshot. Skip."
            
    
    def makeActor(self, poly, c=(0.5, 0.5, 0.5), alpha=1):
        dataset = vtk.vtkPolyDataNormals()
        dataset.SetInput(poly)
        dataset.SetFeatureAngle(60.0)
        dataset.ComputePointNormalsOn()
        dataset.ComputeCellNormalsOn()
        dataset.FlipNormalsOff()
        dataset.ConsistencyOn()
        dataset.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(dataset.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if self.phong:   actor.GetProperty().SetInterpolationToPhong()
        if self.flat:    actor.GetProperty().SetInterpolationToFlat()
        if self.gouraud: actor.GetProperty().SetInterpolationToGouraud()
        actor.GetProperty().EdgeVisibilityOff()
        actor.GetProperty().SetColor(c)
        actor.GetProperty().SetOpacity(alpha)
        actor.GetProperty().SetSpecular(0.05)
        actor.GetProperty().BackfaceCullingOn()
        actor.GetProperty().FrontfaceCullingOff()
        return actor


    def make_assembly(self, actors): 
        # treat many actors as a single new actor
        assembly = vtk.vtkAssembly()
        for a in actors: assembly.AddPart(a)
        return assembly
       
    
    def move_camera(self, camstart, camstop, fraction):
        # frac=0 -> camstart,  frac=1 -> camstop
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
    def make_colorpoints(self, plist, cols, r=10., alpha=0.8):
        ### cols= (r,g,b) in range [0,1]
        if len(plist) != len(cols):
            print "Mismatch in make_colorpoints()", len(plist), len(cols)
            quit()
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(plist))
        src.Update()        
        vertexFilter = vtk.vtkVertexGlyphFilter()
        vertexFilter.SetInput(src.GetOutput())
        vertexFilter.Update()
        pd = vertexFilter.GetOutput()    
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("RGB")
        for i,p in enumerate(plist): 
            pd.GetPoints().SetPoint(i, p)
            c = np.array(cols[i])*255
            colors.InsertNextTupleValue(np.clip(c, 0, 255))
        pd.GetPointData().SetScalars(colors)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(pd)
        mapper.ScalarVisibilityOn()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToFlat()
        actor.GetProperty().SetOpacity(alpha)
        actor.GetProperty().SetPointSize(r)
        return actor
    
    
    def make_points(self, plist, c=(0, 0, 1), r=10., alpha=1.):
        if len(c)>3: 
            return self.make_colorpoints(plist, c, r, alpha)
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(plist))
        src.Update()
        pd = src.GetOutput()
        for i,p in enumerate(plist): pd.GetPoints().SetPoint(i, p)
        actor = self.makeActor(pd, c, alpha)
        actor.GetProperty().SetPointSize(r)
        self.actors.append(actor)
        return actor
    
    
    def make_line(self, p0,p1, lw=1, c=(1, 0, 0), alpha=1.):
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(p0)
        lineSource.SetPoint2(p1)
        lineSource.Update()
        actor = self.makeActor(lineSource.GetOutput(), c, alpha)
        actor.GetProperty().SetLineWidth(lw)
        self.actors.append(actor)
        return actor
    
    
    def make_sphere(self, pt, r=1, c=(1, 0, 0), alpha=1.):
        src = vtk.vtkSphereSource()
        src.SetThetaResolution(24)
        src.SetPhiResolution(24)
        src.SetRadius(r)
        src.SetCenter(pt)
        src.Update()
        actor = self.makeActor(src.GetOutput(), c, alpha) 
        self.actors.append(actor)
        return actor 
    
    
    def make_cube(self, pt, r=1, c=(0, 1, 0), alpha=1.):
        src = vtk.vtkCubeSource()
        src.SetXLength(r)
        src.SetYLength(r)
        src.SetZLength(r)
        src.SetCenter(pt)
        src.Update()
        actor = self.makeActor(src.GetOutput(), c, alpha) 
        self.actors.append(actor)
        return actor
    
    
    def make_grid(self, center=(0,0,0), normal=(0,0,1), s=10, N=10, c=(0,0,1), lw=1, alpha=1):
        ps = vtk.vtkPlaneSource()
        ps.SetResolution(N, N)
        ps.SetCenter(np.array(center)/float(s))
        ps.SetNormal(normal)
        ps.Update()
        actor = self.makeActor(ps.GetOutput(), c=c, alpha=alpha)
        actor.SetScale(s,s,s)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(lw)
        actor.PickableOff()
        backProp = vtk.vtkProperty()
        backProp.SetDiffuseColor([0,0,0])
        backProp.SetOpacity(alpha)
        actor.SetBackfaceProperty(backProp)
        self.actors.append(actor)
        return actor
    
    
    def make_arrow(self, startPoint, endPoint, c=(1, 0, 0), alpha=1):
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
        actor = self.makeActor(transformPD.GetOutput(), c, alpha)
        self.actors.append(actor)
        return actor
    
    
    def make_spline(self, points, s=10, c=(0,0,0.8), alpha=1., nodes=True):
        numberOfOutputPoints = len(points)*20 # Number of points on the spline
        numberOfInputPoints = len(points)  # One spline for each direction.
        aSplineX = vtk.vtkCardinalSpline() #interpolate the x values
        aSplineY = vtk.vtkCardinalSpline() #interpolate the y values
        aSplineZ = vtk.vtkCardinalSpline() #interpolate the z values
        
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
        profileTubes.SetInput(profileData)
        profileTubes.SetRadius(s)    
        acttube = self.makeActor(profileTubes.GetOutput(), c=(.1,.1,1), alpha=alpha)
        if nodes:
            balls = vtk.vtkSphereSource() # Use sphere as glyph source.
            balls.SetRadius(s*1.2)
            balls.SetPhiResolution(12)
            balls.SetThetaResolution(12)            
            glyphPoints = vtk.vtkGlyph3D()
            glyphPoints.SetInput(inputData)
            glyphPoints.SetSource(balls.GetOutput())    
            actnodes = self.makeActor(glyphPoints.GetOutput(), c=c, alpha=alpha)
            return self.make_assembly([acttube, actnodes])
        self.actors.append(acttube)
        return acttube

    
    def make_text(self, txt, pos=(0,0,0), s=1, c=(0,0,0), alpha=1, cam=True):
        tt = vtk.vtkVectorText()
        tt.SetText(txt)
        ttmapper = vtk.vtkPolyDataMapper()
        ttmapper.SetInput(tt.GetOutput())        
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
        self.actors.append(ttactor)
        return ttactor    
    
    
    def make_xyplot(self, points, title='', c=(1,0,0), pos=1, lines=False):
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
    
    
    def make_axes(self, origin=(0,0,0), s=1, lw=5):
        axes = vtk.vtkAxes()
        axes.SetOrigin(origin)
        axes.SetScaleFactor(s)
        axes.SymmetricOff()
        axesActor = self.makeActor(axes.GetOutput())
        axesActor.GetProperty().SetLineWidth(lw)
        self.actors.append(axesActor)
        return axesActor
        
        
    def make_normals(self, pactor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8):
        maskPts = vtk.vtkMaskPoints()
        maskPts.SetOnRatio(ratio)
        maskPts.RandomModeOff()
        src = self.getPD(pactor)
        maskPts.SetInput(src)
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
        glyphActor.GetProperty().SetColor(c)
        glyphActor.GetProperty().SetOpacity(alpha)
        self.actors.append(glyphActor)
        return glyphActor


    def make_curvatures(self, pactor, ctype=1, r=1, alpha=1, lut=None):
        #ctype=curvature type: 0-gaussian, 1-mean, 2-max, 3-min
        poly = self.getPD(pactor)
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInput(poly)
        curve = vtk.vtkCurvatures()
        curve.SetInput(cleaner.GetOutput())
        curve.SetCurvatureType(ctype)
        curve.InvertMeanCurvatureOn()
        if self.verbose: print 'CurvatureType set to:',ctype
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
        cmapper.SetInput(curve.GetOutput())
        cmapper.SetLookupTable(lut)
        cmapper.SetUseLookupTableScalarRange(1)
        cActor = vtk.vtkActor()
        cActor.SetMapper(cmapper)
        self.actors.append(cActor)
        return cActor
        
        
    def make_boundaries(self, pactor, c=(1, 0, 0.5), lw=5):
        ### shows the boundaries lines of the polydata
        fe = vtk.vtkFeatureEdges()
        fe.SetInput(self.getPD(pactor))
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
    def make_fitline(self, points, c=(.5,0,1), lw=1, alpha=0.6, tube=False):
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
        l = self.make_line(p1, p2, c=c, lw=lw, alpha=alpha)
        self.result['slope']  = vv
        self.result['center'] = datamean
        self.result['variances'] = dd
        if self.verbose:
            print "Extra info saved in vp.results['slope','center','variances']"
        if tube: # show a rough estimate of error band at 2 sigma level
            tb = vtk.vtkTubeFilter() 
            tb.SetNumberOfSides(48)
            tb.SetInput(self.getPD(l))
            r = np.sqrt((dd[1]+dd[2])/2./len(points))
            tb.SetRadius(r)    
            a = self.makeActor(tb.GetOutput(), c=c, alpha=alpha/4.)
            l = self.make_assembly([l,a])
            self.actors[-1] = l # replace
        return l


    def make_fitplane(self, points, c=(.5,0,1)):
        data = np.array(points)
        datamean = data.mean(axis=0)
        uu, dd, vv = np.linalg.svd(data - datamean)
        xyz_min = points.min(axis=0)
        xyz_max = points.max(axis=0)
        s= np.linalg.norm(xyz_max - xyz_min)
        n = np.cross(vv[0],vv[1])
        pla = self.make_grid(datamean, n, c=c, s=s, lw=2, alpha=0.8)
        self.result['normal']  = n
        self.result['center']  = datamean
        self.result['variance']= dd[2]
        if self.verbose:
            print "Extra info saved in vp.results['normal','center','variance']"
        return pla


    def make_ellipsoid(self, points, pvalue=.95, c=(0,1,1), alpha=0.5, axes=False):
        # build the ellpsoid that contains 95% of points
        try:
            from scipy.stats import f
        except:
            print "scipy not installed. Skip."
            return None
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
            print "Extra info saved in vp.results['sphericity','a','b','c']"
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
        ftra.SetInput(elliSource.GetOutput())
        ftra.Update()
        actor_elli = self.makeActor(ftra.GetOutput(), c, alpha)
        actor_elli.GetProperty().BackfaceCullingOn()
        if axes:
            axs = []
            for ax in ([1,0,0], [0,1,0], [0,0,1]):
                l = vtk.vtkLineSource()
                l.SetPoint1([0,0,0])
                l.SetPoint2(ax)
                l.Update()
                t = vtk.vtkTransformFilter()
                t.SetTransform(vtra)
                t.SetInput(l.GetOutput())
                t.Update()
                axs.append(self.makeActor(t.GetOutput(), c, alpha))
            self.actors.append( self.make_assembly(axs+[actor_elli]) )
        else : self.actors.append(actor_elli)
        return self.actors[-1]      
    

    ##########################################
    def draw_cubeaxes(self, c=(.2, .2, .6)):
        cubeAxesActor = vtk.vtkCubeAxesActor()
        if self.renderer: 
            cubeAxesActor.SetBounds(self.renderer.ComputeVisiblePropBounds())
        if self.camera: cubeAxesActor.SetCamera(self.camera)
        else:  cubeAxesActor.SetCamera(self.renderer.GetActiveCamera())
        cubeAxesActor.GetProperty().SetColor(c)
        cubeAxesActor.SetFlyMode(3)
        cubeAxesActor.XAxisMinorTickVisibilityOff()
        cubeAxesActor.YAxisMinorTickVisibilityOff()
        cubeAxesActor.ZAxisMinorTickVisibilityOff()
        self.renderer.AddActor(cubeAxesActor)

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
        if len(self.actors) < N:
            print 'Mismatch in Legend:', len(self.actors), 'actors'
            print 'but', N, 'legend lines. Skip.'
            return
        legend = vtk.vtkLegendBoxActor()
        legend.SetNumberOfEntries(N)
        for i in range(N):
            a = self.actors[i]
            c = a.GetProperty().GetColor()
            legend.SetEntry(i, a.GetMapper().GetInput(), " "+texts[i], c)
        s = self.legendSize
        pos = self.legendPosition
        if pos==1: legend.GetPositionCoordinate().SetValue(.0, s, 0)
        if pos==2: legend.GetPositionCoordinate().SetValue( s, s, 0)
        if pos==3: legend.GetPositionCoordinate().SetValue(.0,.0, 0)
        if pos==4: legend.GetPositionCoordinate().SetValue( s,.0, 0)
        legend.UseBackgroundOn()
        legend.SetBackgroundColor(self.legendBG)
        self.renderer.AddActor(legend)       
        
        
    ###############################################################################
    def show(self, actors=None, legend=None, at=0, #at=render wind. nr.
             axes=None, ruler=False, interactive=None, outputimage=None):
        
        # override what was stored internally with passed input
        if not actors is None:
            if not isinstance(actors, list): self.actors = [actors]
            else: self.actors = actors 
        if not legend is None: self.legend = legend 
        if not axes   is None: self.axes   = axes 
        if not interactive is None: self.interactive = interactive 
        if self.verbose: 
            print 'Drawing', len(self.actors),'actors',
            if self.shape != (1,1): print 'on window',at,'-',
            else: print '-',      
            if self.interactive: print 'Interactive: On.'
            else: print 'Interactive: Off.'
            
        self.renderer = self.renderers[at]
        if not self.camera: self.camera = self.renderer.GetActiveCamera()
        else: self.camera.SetThickness(self.camThickness)
        if self.parallelcam: self.camera.ParallelProjectionOn()
        if self.commoncam: 
            for r in self.renderers: r.SetActiveCamera(self.camera)
 
        for ia in self.actors:
            if isinstance(ia, vtk.vtkPolyData):
                ia = self.makeActor(ia, c=(1,0.647,0), alpha=0.1)
            self.renderer.AddActor(ia)   
    
        if ruler: self.draw_ruler()
        if self.axes: self.draw_cubeaxes()
        if self.legend or len(self.legend): self.draw_legend()

        if self.resetcam: self.renderer.ResetCamera()

        if not self.initialized: 
            self.interactor.Initialize()
            self.initialized = True
            self.interactor.AddObserver("KeyPressEvent", self.keypress)

        self.interactor.Render()
        if self.verbose: self.tips()
        if outputimage: self.make_screenshot(outputimage)
        if self.renderer and self.interactive: self.interact()


    def keypress(self, obj, event):
        key = obj.GetKeySym()
        if key == "q":
            if self.verbose: 
                print "Returning control to python script/command line."
                print "Use vp.interact() to go back to 3D scene."
            self.interactor.ExitCallback()
        if key == "e": 
            if self.verbose:
                print "Closing window and return control to python."
            rw = self.interactor.GetRenderWindow()
            rw.Finalize()
            self.interactor.TerminateApp()
            del self.renderWin, self.interactor
            return
        if key == "Escape":                 
            if self.verbose: print "Quitting now, Bye."
            exit(0)
        if key == "S":
            print 'Saving picture as screenshot.png'
            self.make_screenshot()
        if key == "c": 
            cam = self.renderer.GetActiveCamera()
            print '\ncam = vtk.vtkCamera() #example code'
            print 'cam.SetPosition(',  [round(e,3) for e in cam.GetPosition()],  ')'
            print 'cam.SetFocalPoint(',[round(e,3) for e in cam.GetFocalPoint()],')'
            print 'cam.SetParallelScale(',round(cam.GetParallelScale(),3),')'
            print 'cam.SetViewUp(', [round(e,3) for e in cam.GetViewUp()],')'
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        if key == "m":
            for i in range(actors.GetNumberOfItems()):
                actors.GetNextItem().GetProperty().SetOpacity(0.1)                    
        if key == "comma":
            for i in range(actors.GetNumberOfItems()):
                ap = actors.GetNextItem().GetProperty()
                ap.SetOpacity(max([ap.GetOpacity()-0.05, 0.1]))
        if key == "period":
            for i in range(actors.GetNumberOfItems()):
                ap = actors.GetNextItem().GetProperty()
                ap.SetOpacity(min([ap.GetOpacity()+0.05, 1.0]))
        if key == "slash":
            for i in range(actors.GetNumberOfItems()):
                actors.GetNextItem().GetProperty().SetOpacity(1)
        if key == "v": 
            self.verbose = not(self.verbose)
            print "Verbose: ", self.verbose
        
    def interact(self): self.interactor.Start()
        
    
        
    







