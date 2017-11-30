#!/usr/bin/env python
# A helper tool for visualizing vtk objects
from __future__ import print_function
__author__  = "Marco Musy"
__license__ = "MIT"
__version__ = "3.1"
__maintainer__ = __author__
__email__   = "marco.musy@embl.es"
__status__  = "dev"

import os, vtk, numpy as np
from glob import glob


#############################################################################
class vtkPlotter:

    def help(self):
        print ("""\n
        A python helper class to easily draw VTK tridimensional objects.
        Please follow instructions at:
        https://github.com/marcomusy/vtkPlotter
        Useful commands on graphic window:\n  """)
        print ("VTK version:", vtk.vtkVersion().GetVTKVersion())
        try:
            import platform
            print ("Python version:", platform.python_version())
        except: pass
        print('\nAvailable color names:\n', color_names)
        print('Colors abbreviations:\n', color_nicks,'\n')
        self.tips()

    def tips(self):
        msg = """Press: ------------------------------------------
        m   to minimise opacity of selected actor
        /   to maximize opacity of selected actor
        .,  to increase/reduce opacity
        w/s to toggle wireframe/solid style
        oO  to change point size of vertices
        lL  to change edge line width
        n   to show normals for selected actor
        x   to remove selected actor
        1-5 to change color scheme
        V   to toggle verbose mode
        C   to print current camera info
        S   to save a screenshot
        q   to continue
        e   to close current window
        Esc to abort and exit
        Ctrl-mouse  to rotate scene
        Shift-mouse to shift scene
        Right-mouse click to zoom in/out
        -----------------------------------------"""
        print (msg)


    def __init__(self, shape=(1,1), size='auto', screensize=(1000,1600),
                bg=(1,1,1), bg2=None, balloon=False, verbose=True, interactive=True):
        self.shape      = shape # nr of rows and columns in render window
        self.size       = size  # window size
        self.verbose    = verbose
        self.interactive= interactive # allows to interact with renderer
        self.renderer   = None  # current renderer
        self.renderers  = []
        self.axes       = True  # show or hide axes
        self.units      = ''    # axes units
        self.camera     = None  # current vtkCamera 
        self.commoncam  = True  # share the same camera in renderers
        self.resetcam   = True  # reset camera when calling show()
        self.parallelcam = True # parallel projection or perspective
        self.actors     = []    # list of actor to be shown
        self.legend     = []    # list of legend entries for actors
        self.flat       = True
        self.phong      = False
        self.gouraud    = False 
        self.bculling   = False # back face culling
        self.fculling   = False # front face culling
        self.legendSize = 0.2
        self.legendBG   = (.96,.96,.9)
        self.legendPosition = 2   # 1=topright
        self.result     = dict()  # stores extra output information
        
        # internal stuff:
        self.icol1      = 0
        self.icol2      = 0
        self.icol3      = 0
        self.clickedr   = 0 #clicked renderer number
        self.clickedx   = 0 
        self.clickedy   = 0
        self.clickedActor = None
        self.camThickness = 2000
        self.balloon    = balloon
        self.locator    = None
        self.initialized= False
        self.videoname  = None
        self.videoformat = None
        self.frames     = []
        self.fps        = 12
        self.caxes_exist = []
        
        if balloon: # tends to crash, so disabled.
            self.balloonWidget = vtk.vtkBalloonWidget()
            self.balloonRep = vtk.vtkBalloonRepresentation()
            self.balloonRep.SetBalloonLayoutToImageRight()
            self.balloonWidget.SetRepresentation(self.balloonRep)

        if size=='auto': # figure out reasonable window size
                maxs = screensize #max sizes allowed in y and x
                xs = maxs[0]/2.*shape[0]
                ys = maxs[0]/2.*shape[1]
                if xs>maxs[0]: # shrink
                    xs = maxs[0]
                    ys = maxs[0]/shape[0]*shape[1]
                if ys>maxs[1]: 
                    ys = maxs[1]
                    xs = maxs[1]/shape[1]*shape[0]
                self.size = (xs,ys)
                if shape==(1,1): 
                    self.size = (maxs[1]/2,maxs[1]/2)
                elif self.verbose:
                    print ('Window size set to:', self.size)
        
   
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
                self.caxes_exist.append(False)
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


    ####################################### LOADER
    def load(self, filesOrDirs, c='gold', alpha=0.2, 
              wire=False, bc=None, edges=False, legend=True):
        '''Return a vtkActor, optional args:
           c,     color in RGB format, hex, symbol or name
           alpha, transparency (0=invisible)
           wire,  show surface as wireframe
           bc,    backface color of internal surface
           legend, text to show on legend, if True picks filename.
        '''
        acts = []
        if isinstance(legend, int): legend = bool(legend)
        for fod in sorted(glob(filesOrDirs)):
            if os.path.isfile(fod): 
                a = self._loadFile(fod, c, alpha, wire, bc, edges, legend)
                acts.append(a)
            elif os.path.isdir(fod):
                acts = self._loadDir(fod, c, alpha, wire, bc, edges, legend)
        if not len(self.actors):
            print ('Cannot find:', filesOrDirs)
            exit(0) 
        if len(acts) == 1: 
            return acts[0]
        else: 
            return acts

    def _loadFile(self, filename, c, alpha, wire, bc, edges, legend):
        fl = filename.lower()
        if '.xml' in fl or '.xml.gz' in fl: # Fenics tetrahedral mesh file
            actor = self._loadXml(filename, c, alpha, wire, bc, edges, legend)
        elif '.pcd' in fl:                  # PCL point-cloud format
            actor = self._loadPCD(filename, c, alpha, legend)
        else:
            poly = self._loadPoly(filename)
            if not poly:
                print ('Unable to load', filename)
                return False
            if legend is True: legend = os.path.basename(filename)
            actor = self.makeActor(poly, c, alpha, wire, bc, edges, legend)
            if '.txt' in fl or '.xyz' in fl: 
                actor.GetProperty().SetPointSize(4)
        self.actors.append(actor)
        return actor
        
    def _loadDir(self, mydir, c, alpha, wire, bc, edges, legend):
        acts = []
        for ifile in sorted(os.listdir(mydir)):
            self._loadFile(mydir+'/'+ifile, c, alpha, wire, bc, edges)
        return acts

    def _loadPoly(self, filename):
        '''Return a vtkPolyData object, NOT a vtkActor
        '''
        if not os.path.exists(filename): 
            print ('Cannot find file', filename)
            exit(0)
        fl = filename.lower()
        if   '.vtk' in fl: reader = vtk.vtkPolyDataReader()
        elif '.ply' in fl: reader = vtk.vtkPLYReader()
        elif '.obj' in fl: reader = vtk.vtkOBJReader()
        elif '.stl' in fl: reader = vtk.vtkSTLReader()
        elif '.byu' in fl or '.g' in fl: reader = vtk.vtkBYUReader()
        elif '.vtp' in fl: reader = vtk.vtkXMLPolyDataReader()
        elif '.vts' in fl: reader = vtk.vtkXMLStructuredGridReader()
        elif '.vtu' in fl: reader = vtk.vtkXMLUnstructuredGridReader()
        elif '.txt' in fl: reader = vtk.vtkParticleReader() # (x y z scalar) 
        elif '.xyz' in fl: reader = vtk.vtkParticleReader()
        else: reader = vtk.vtkDataReader()
        reader.SetFileName(filename)
        reader.Update()
        if '.vts' in fl: # structured grid
            gf = vtk.vtkStructuredGridGeometryFilter()
            gf.SetInputConnection(reader.GetOutputPort())
            gf.Update()
            poly = gf.GetOutput()
        elif '.vtu' in fl: # unstructured grid
            gf = vtk.vtkGeometryFilter()
            gf.SetInputConnection(reader.GetOutputPort())
            gf.Update()    
            poly = gf.GetOutput()
        else: poly = reader.GetOutput()
        
        if not poly: 
            print ('Unable to load', filename)
            return False
        
        mergeTriangles = vtk.vtkTriangleFilter()
        setInput(mergeTriangles, poly)
        mergeTriangles.Update()
        poly = mergeTriangles.GetOutput()
        return poly

    def _loadXml(self, filename, c, alpha, wire, bc, edges, legend):
        '''Reads a Fenics/Dolfin file format
        '''
        if not os.path.exists(filename): 
            print ('Cannot find file', filename)
            exit(0)
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
                    if self.verbose: print ('Reading',elem.tag)
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
            # 3D cells are mapped only if they are used by only one cell,
            #  i.e., on the boundary of the data set
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputConnection(ugrid.GetProducerPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetInterpolationToFlat()
            actor.GetProperty().SetColor(getColor(c))
            actor.GetProperty().SetOpacity(alpha/2.)
            #actor.GetProperty().VertexVisibilityOn()
            if edges: actor.GetProperty().EdgeVisibilityOn()
            if wire:  actor.GetProperty().SetRepresentationToWireframe()
            vpts = vtk.vtkPointSource()
            vpts.SetNumberOfPoints(len(coords))
            vpts.Update()
            vpts.GetOutput().SetPoints(points)
            pts_act = self.makeActor(vpts.GetOutput(), c='b', alpha=alpha)
            pts_act.GetProperty().SetPointSize(3)
            pts_act.GetProperty().SetRepresentationToPoints()
            actor2 = self.assembly([pts_act, actor])
            if legend: setattr(actor2, 'legend', legend)
            if legend is True: 
                setattr(actor2, 'legend', os.path.basename(filename))
            return actor2
        except:
            print ("Cannot parse xml file. Skip.", filename)
            return False
 
    def _loadPCD(self, filename, c, alpha, legend):
        '''Return vtkActor from Point Cloud file format
        '''            
        if not os.path.exists(filename): 
            print ('Cannot find file', filename)
            exit(0)
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
        if not poly:
            print ('Unable to load', filename)
            return False
        actor = self.makeActor(poly, getColor(c), alpha)
        actor.GetProperty().SetPointSize(4)
        if legend: setattr(actor, 'legend', legend)
        if legend is True: setattr(actor, 'legend', os.path.basename(filename))
        return actor
        
    ############################################# getters
    def getPD(self, obj, index=0): # get PolyData
        '''
        Returns vtkPolyData from an other object (vtkActor, vtkAssembly, int)
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


    def getActors(self, obj=None):
        '''
        Return the actors list in renderer number obj (int). 
        If None, use current renderer.
        If obj is a vtkAssembly return the actors contained in it.
        If obj is a string, return actors with that legend name.
        '''
        if isinstance(obj , vtk.vtkAssembly):
            cl = vtk.vtkPropCollection()
            obj.GetActors(cl)
            actors=[]
            cl.InitTraversal()
            for i in range(obj.GetNumberOfPaths()):
                act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
                if isinstance(act, vtk.vtkCubeAxesActor): continue
                actors.append(act)
            return actors
            
        elif isinstance(obj, int) or obj is None:
            if obj is None: 
                acs = self.renderer.GetActors()
            elif obj>=len(self.renderers):
                print ("Error in getActors: non existing renderer",obj)
                return []
            else:
                print ("Error in getActors(): Unknown argument", obj)
                return []
                acs = self.renderers[obj].GetActors()
            actors=[]
            acs.InitTraversal()
            for i in range(acs.GetNumberOfItems()):
                a = acs.GetNextItem()
                if isinstance(a, vtk.vtkCubeAxesActor): continue
                actors.append(a)
            return actors
            
        elif isinstance(obj, str): # search the actor by the legend name
            actors=[]
            for a in self.actors:
                if hasattr(a, 'legend') and obj in a.legend:
                    actors.append(a)
                return actors
                
        if self.verbose: print ('Warning in getActors: unexpected input type',obj)
        return []


    def getPoint(self, i, actor):
        if isinstance(actor, int): actor = self.actors[actor]
        poly = self.getPD(actor)
        p = [0,0,0]
        poly.GetPoints().GetPoint(i, p)
        return np.array(p)


    def coordinates(self, actors):
        """Return a merged list of coordinates of actors or polys"""
        if not isinstance(actors, list):
            actors = [actors]
        pts = []
        for i in range(len(actors)):
            apoly = self.getPD(actors[i])
            for j in range(apoly.GetNumberOfPoints()):
                p = [0, 0, 0]
                apoly.GetPoint(j, p)
                pts.append(p)
        return pts
        

    #############
    def makeActor(self, poly, c='gold', alpha=0.5, 
                  wire=False, bc=None, edges=False, legend=None):
        '''Return a vtkActor from an input vtkPolyData, optional args:
           c,     color in RGB format, hex, symbol or name
           alpha, transparency (0=invisible)
           wire,  show surface as wireframe
           bc,    backface color of internal surface
           edges, show edges as line on top of surface
        '''
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
        actor.GetProperty().SetColor(getColor(c))
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
            backProp.SetDiffuseColor(getColor(bc))
            backProp.SetOpacity(alpha)
            actor.SetBackfaceProperty(backProp)
        if legend: setattr(actor, 'legend', legend) 
        return actor


    def assembly(self, actors, legend=None):
        '''Treat many actors as a single new actor'''
        assembly = vtk.vtkAssembly()
        for a in actors: assembly.AddPart(a)
        if legend:
            setattr(assembly, 'legend', legend) 
        elif hasattr(actors[0], 'legend'): 
            setattr(assembly, 'legend', actors[0].legend) 
        return assembly


    def moveCamera(self, camstart, camstop, fraction):
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


    ################################################################## vtk objects
    def points(self, plist, c='b', r=10., alpha=1., legend=None):
        '''
        Return a vtkActor for a list of points.
        Input cols is a list of RGB colors of same length as plist
        '''
        if isinstance(c, list):
            return self._colorPoints(plist, c, r, alpha, legend)
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(plist))
        src.Update()
        pd = src.GetOutput()
        for i,p in enumerate(plist): pd.GetPoints().SetPoint(i, p)
        actor = self.makeActor(pd, c, alpha)
        actor.GetProperty().SetPointSize(r)
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor
        
    def _colorPoints(self, plist, cols, r, alpha, legend):
        if len(plist) != len(cols):
            print ("Mismatch in colorPoints()", len(plist), len(cols))
            exit()
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(plist))
        src.Update()
        vertexFilter = vtk.vtkVertexGlyphFilter()
        setInput(vertexFilter, src.GetOutput())
        vertexFilter.Update()
        pd = vertexFilter.GetOutput()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("RGB")
        for i,p in enumerate(plist):
            pd.GetPoints().SetPoint(i, p)
            c = np.array(getColor(cols[i]))*255
            colors.InsertNextTupleValue(c)
        pd.GetPointData().SetScalars(colors)
        mapper = vtk.vtkPolyDataMapper()
        setInput(mapper, pd)
        mapper.ScalarVisibilityOn()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToFlat()
        actor.GetProperty().SetOpacity(alpha)
        actor.GetProperty().SetPointSize(r)
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor


    def line(self, p0,p1, lw=1, c='r', alpha=1., legend=None):
        '''Returns the line segment between points p0 and p1'''
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(p0)
        lineSource.SetPoint2(p1)
        lineSource.Update()
        actor = self.makeActor(lineSource.GetOutput(), c, alpha)
        actor.GetProperty().SetLineWidth(lw)
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor


    def sphere(self, pt, r=1, c='r', alpha=1., legend=None):
        src = vtk.vtkSphereSource()
        src.SetThetaResolution(24)
        src.SetPhiResolution(24)
        src.SetRadius(r)
        src.SetCenter(pt)
        src.Update()
        actor = self.makeActor(src.GetOutput(), c, alpha)
        actor.GetProperty().SetInterpolationToPhong()
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor


    def cube(self, pt, r=1, c='g', alpha=1., legend=None):
        src = vtk.vtkCubeSource()
        src.SetXLength(r)
        src.SetYLength(r)
        src.SetZLength(r)
        src.SetCenter(pt)
        src.Update()
        actor = self.makeActor(src.GetOutput(), c, alpha)
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor


    def plane(self, center=(0,0,0), normal=(0,0,1), s=10, N=10, 
              c='g', bc='darkgreen', lw=1, alpha=1, wire=False, legend=None):
        pl = self.grid(center, normal, s, 1, c, bc, lw, alpha, wire, legend)
        pl.GetProperty().SetEdgeVisibility(1)
        return pl
        
        
    def grid(self, center=(0,0,0), normal=(0,0,1), s=10, N=10, 
             c='g', bc='darkgreen', lw=1, alpha=1, wire=True, legend=None):
        '''Return a grid plane'''
        ps = vtk.vtkPlaneSource()
        ps.SetResolution(N, N)
        ps.SetCenter(np.array(center)/float(s))
        ps.SetNormal(normal)
        ps.Update()
        actor = self.makeActor(ps.GetOutput(), c=c, bc=bc, alpha=alpha)
        actor.SetScale(s,s,s)
        if wire: actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(lw)
        actor.PickableOff()
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor
    
    
    def arrow(self, startPoint, endPoint, c='r', alpha=1, legend=None):
        axis = np.array(endPoint) - np.array(startPoint)
        length = np.linalg.norm(axis)
        axis = axis/length
        theta = np.arccos(axis[2])
        phi   = np.arctan2(axis[1], axis[0])
        arr = vtk.vtkArrowSource()
        arr.SetShaftResolution(24)
        arr.SetTipResolution(24)
        arr.SetTipRadius(0.06)
        actor = self.makeActor(arr.GetOutput(), c, alpha)
        actor.GetProperty().SetInterpolationToPhong()
        actor.SetPosition(startPoint)
        actor.RotateZ(phi*57.3)
        actor.RotateY(theta*57.3)
        actor.SetScale(length,length,length)
        actor.RotateY(-90) #put it along Z
        actor.DragableOff()
        actor.PickableOff()
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor


    def cylinder(self, center, radius, height, axis=[1,1,1],
                 c='teal', alpha=1, legend=None):
        cyl = vtk.vtkCylinderSource()
        cyl.SetResolution(24)
        cyl.SetRadius(radius)
        cyl.SetHeight(height)
        #cyl.SetAxis(axis)
        axis  = np.array(axis)/np.linalg.norm(axis)
        theta = np.arccos(axis[2])
        phi   = np.arctan2(axis[1], axis[0])
        actor = self.makeActor(cyl.GetOutput(), c, alpha)
        actor.GetProperty().SetInterpolationToPhong()
        actor.SetPosition(center)
        actor.RotateZ(phi*57.3)
        actor.RotateY(theta*57.3)
        actor.RotateX(90) #put it along Z
        actor.DragableOff()
        actor.PickableOff()
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor


    def spline(self, points, s=10, c='navy', alpha=1., nodes=True, legend=None):
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
            gl = vtk.vtkGlyph3D()
            setInput(gl, inputData)
            gl.SetSource(balls.GetOutput())
            actnodes = self.makeActor(gl.GetOutput(), c=c, alpha=alpha)
            acttube  = self.assembly([acttube, actnodes])
        self.actors.append(acttube)
        if legend: setattr(acttube, 'legend', legend) 
        return acttube


    def bspline(self, points, nknots=-1,
                s=1, c=(0,0,0.8), alpha=1., nodes=True, legend=None):
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
            return None

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
        if legend: setattr(acttube, 'legend', legend) 
        return acttube


    def text(self, txt, pos=(0,0,0), s=1, c='k', alpha=1, bc=None, cam=True):
        '''Returns a vtkActor that shows a text 3D
           if cam is True the text will auto-orient to it
        '''
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
        ttactor.GetProperty().SetColor(getColor(c))
        ttactor.GetProperty().SetOpacity(alpha)
        ttactor.AddPosition(pos)
        ttactor.SetScale(s,s,s)
        if bc: # defines a specific color for the backface
            backProp = vtk.vtkProperty()
            backProp.SetDiffuseColor(getColor(bc))
            backProp.SetOpacity(alpha)
            ttactor.SetBackfaceProperty(backProp)
        self.actors.append(ttactor)
        return ttactor


    def xyplot(self, points, title='', c='r', pos=1, lines=False):
        """Return a vtkActor that is a plot of 2D points in x and y
           pos assignes the position: 
           1=topleft, 2=topright, 3=bottomleft, 4=bottomright
        """
        c = getColor(c) # allow different codings
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


    def normals(self, pactor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8, legend=None):
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
        glyphActor.GetProperty().SetColor(getColor(c))
        glyphActor.GetProperty().SetOpacity(alpha)
        actor = self.assembly([pactor,glyphActor])
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor


    def curvature(self, pactor, method=1, r=1, alpha=1, lut=None, legend=None):
        '''Returns a vtkActor that contains the color coded surface
           curvature following four different ways to calculate it:
           method =  0-gaussian, 1-mean, 2-max, 3-min
        '''
        poly = self.getPD(pactor)
        cleaner = vtk.vtkCleanPolyData()
        setInput(cleaner, poly)
        curve = vtk.vtkCurvatures()
        curve.SetInputConnection(cleaner.GetOutputPort())
        curve.SetCurvatureType(method)
        curve.InvertMeanCurvatureOn()
        curve.Update()
        if self.verbose: print ('CurvatureType set to:',method)
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
        actor = vtk.vtkActor()
        actor.SetMapper(cmapper)
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor


    def boundaries(self, pactor, c='p', lw=5, legend=None):
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
        actor = self.makeActor(fe.GetOutput(), c=c, alpha=1)
        actor.GetProperty().SetLineWidth(lw)
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor


    ################# working with point clouds
    def fitLine(self, points, c='orange', lw=1, alpha=0.6, tube=False, legend=None):
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
        if legend: setattr(l, 'legend', legend) 
        return l


    def fitPlane(self, points, c='g', bc='darkgreen', legend=None):
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
        if legend: setattr(pla, 'legend', legend) 
        return pla


    def ellipsoid(self, points, pvalue=.95, c='c', alpha=0.5, 
                  pcaaxes=False, legend=None):
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
        if legend: setattr(self.lastActor(), 'legend', legend) 
        return self.lastActor()


    def align(self, source, target, rigid=False, iters=100, legend=None):
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
        icp.Update()
        icpTransformFilter = vtk.vtkTransformPolyDataFilter()
        setInput(icpTransformFilter, source)
        icpTransformFilter.SetTransform(icp)
        icpTransformFilter.Update()
        poly = icpTransformFilter.GetOutput()
        actor = self.makeActor(poly)
        actor.SetProperty(sprop)
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend) 
        return actor


    def cutActor(self, actor, origin=(0,0,0), normal=(1,0,0),
                showcut=True, showline=False, showpts=True, legend=None):
        '''Takes actor and cuts it with the plane defined by a point 
           and a normal. Substitutes it to the original actor.
           showcut  = shows the cut away part as thin wireframe
           showline = marks with a thick line the cut
           showpts  = shows the vertices along the cut
        '''
        if not actor in self.actors:
            print ('Error in cutActor: actor not in vp.actors. Skip.')
            return
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        poly = self.getPD(actor)
        clipper = vtk.vtkClipPolyData()
        setInput(clipper, poly)
        clipper.SetClipFunction(plane)
        clipper.GenerateClippedOutputOn()
        clipper.SetValue(0.)
        alpha = actor.GetProperty().GetOpacity()
        c = actor.GetProperty().GetColor()
        bf = actor.GetBackfaceProperty()
        clipActor = self.makeActor(clipper.GetOutput(),c=c,alpha=alpha)
        clipActor.SetBackfaceProperty(bf)
        
        acts = [clipActor]
        if showcut:
            cpoly = clipper.GetClippedOutput()
            restActor = self.makeActor(cpoly, c=c, alpha=0.05, wire=1)
            acts.append(restActor)
        cutEdges = vtk.vtkCutter()
        setInput(cutEdges, poly)
        cutEdges.SetCutFunction(plane)
        cutEdges.SetValue(0, 0.)
        cutStrips = vtk.vtkStripper()
        cutStrips.SetInputConnection(cutEdges.GetOutputPort())
        cutStrips.Update()
        if showline: 
            cutPoly = vtk.vtkPolyData()
            cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
            cutPoly.SetPolys(cutStrips.GetOutput().GetLines())
            cutline = self.makeActor(cutPoly, c=c, alpha=np.sqrt(alpha))
            cutline.GetProperty().SetRepresentationToWireframe()
            cutline.GetProperty().SetLineWidth(4)
            acts.append(cutline)
        if showpts: 
            vpts = vtk.vtkPointSource()
            cspts = cutStrips.GetOutput().GetPoints()
            vpts.SetNumberOfPoints(cspts.GetNumberOfPoints())
            vpts.Update()
            vpts.GetOutput().SetPoints(cspts)
            points_act = self.makeActor(vpts.GetOutput(), c=c, alpha=np.sqrt(alpha))
            points_act.GetProperty().SetPointSize(4)
            points_act.GetProperty().SetRepresentationToPoints()
            acts.append(points_act)

        if len(acts)>1: 
            finact = self.assembly(acts)
        else: 
            finact = clipActor
        i = self.actors.index(actor)
        arem = self.actors[i]
        del arem
        if legend: setattr(finact, 'legend', legend) 
        self.actors[i] = finact # substitute
        # do not return actor
        

    ####################################
    def closestPoint(self, surf, pt, locator=None, N=None, radius=None):
        """Find the closest point on a polydata given an other point.
        If N is given, return a list of N ordered closest points.
        If radius is given, pick only within specified radius.
        """
        polydata = self.getPD(surf)
        trgp  = [0,0,0]
        cid   = vtk.mutable(0)
        subid = vtk.mutable(0)
        dist2 = vtk.mutable(0)
        self.result['closest_exists'] = False
        if locator: self.locator = locator
        elif not self.locator:
            if N: self.locator = vtk.vtkPointLocator()
            else: self.locator = vtk.vtkCellLocator()
            self.locator.SetDataSet(polydata)
            self.locator.BuildLocator()
        if N:
            vtklist = vtk.vtkIdList()
            vmath = vtk.vtkMath()
            self.locator.FindClosestNPoints(N, pt, vtklist)
            trgp_, trgp, dists2 = [0,0,0], [], []
            npt = vtklist.GetNumberOfIds()
            if npt: self.result['closest_exists'] = True
            for i in range(vtklist.GetNumberOfIds()):
                vi = vtklist.GetId(i)
                polydata.GetPoints().GetPoint(vi, trgp_ )
                trgp.append( trgp_ )
                dists2.append(vmath.Distance2BetweenPoints(trgp_, pt))
            dist2 = dists2
        elif radius:
            cell = vtk.mutable(0)
            r = self.locator.FindClosestPointWithinRadius(pt, radius, trgp, cell, cid, dist2)
            self.result['closest_exists'] = bool(r)
            if not r: 
                trgp = pt
                dist2 = 0.0
        else: 
            self.locator.FindClosestPoint(pt, trgp, cid, subid, dist2)
            self.result['closest_exists'] = True
        self.result['distance2'] = dist2
        return trgp
        

    ##########################################
    def _draw_cubeaxes(self, c=(.2, .2, .6)):
        r = self.renderers.index(self.renderer)
        if self.caxes_exist[r] or not self.axes: return
        ca = vtk.vtkCubeAxesActor()
        if self.renderer:
            ca.SetBounds(self.renderer.ComputeVisiblePropBounds())
        if self.camera: ca.SetCamera(self.camera)
        else: ca.SetCamera(self.renderer.GetActiveCamera())
        if vtkMV:
            ca.GetXAxesLinesProperty().SetColor(c)
            ca.GetYAxesLinesProperty().SetColor(c)
            ca.GetZAxesLinesProperty().SetColor(c)
            for i in range(3):
                ca.GetLabelTextProperty(i).SetColor(c)
                ca.GetTitleTextProperty(i).SetColor(c)
            ca.SetTitleOffset(10)
        else:
            ca.GetProperty().SetColor(c)
        ca.SetFlyMode(3)
        ca.XAxisLabelVisibilityOn()
        ca.YAxisLabelVisibilityOn()
        ca.ZAxisLabelVisibilityOff()
        ca.SetXTitle('x-axis '+self.units)
        ca.SetYTitle('y-axis '+self.units)
        ca.XAxisMinorTickVisibilityOff()
        ca.YAxisMinorTickVisibilityOff()
        ca.ZAxisMinorTickVisibilityOff()
        self.caxes_exist[r] = True
        self.renderer.AddActor(ca)


    def _draw_ruler(self):
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


    def _draw_legend(self):
        # remove old legend if present on current renderer:
        acs = self.renderer.GetActors2D()
        acs.InitTraversal()
        for i in range(acs.GetNumberOfItems()):
            a = acs.GetNextItem()
            if isinstance(a, vtk.vtkLegendBoxActor):
                self.renderer.RemoveActor(a)

        actors = self.getActors()
        NA, NL = len(actors), len(self.legend)
        if NL > NA:
            #print ('Mismatch in Legend:', end='')
            #print (NA, 'actors but', NL, 'legend entries.')
            pass

        acts, texts = [], []
        for i in range(len(actors)):
            a = actors[i]
            if i<len(self.legend) and self.legend[i]!='': 
                texts.append(self.legend[i])
                acts.append(a)
            elif hasattr(a, 'legend') and a.legend: 
                texts.append(a.legend)
                acts.append(a)
        
        NT = len(texts)
        if NT>25: NT=25
        vtklegend = vtk.vtkLegendBoxActor()
        vtklegend.SetNumberOfEntries(NT)
        for i in range(NT):
            ti = texts[i]
            a  = acts[i]
            if isinstance(a, vtk.vtkAssembly):
                cl = vtk.vtkPropCollection()
                a.GetActors(cl)
                cl.InitTraversal()
                act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
                c = act.GetProperty().GetColor()
                if c==(1,1,1): c=(0.7,0.7,0.7) # awoid white
                try:
                    vtklegend.SetEntry(i, self.getPD(a), "  "+ti, c)
                except:
                    sp = vtk.vtkSphereSource() #make a dummy sphere as icon
                    sp.Update()
                    vtklegend.SetEntry(i, sp.GetOutput(),"  "+ti, c)
            else:
                c = a.GetProperty().GetColor()
                if c==(1,1,1): c=(0.7,0.7,0.7)
                vtklegend.SetEntry(i, self.getPD(a), "  "+ti, c)
        pos = self.legendPosition
        width = self.legendSize
        vtklegend.SetWidth(width)
        vtklegend.SetHeight(width/5.*NT)
        sx, sy = 1-width, 1-width/5.*NT
        if pos==1: vtklegend.GetPositionCoordinate().SetValue(  0, sy) #x,y from bottomleft
        elif pos==2: vtklegend.GetPositionCoordinate().SetValue( sx, sy) #default
        elif pos==3: vtklegend.GetPositionCoordinate().SetValue(  0,  0)
        elif pos==4: vtklegend.GetPositionCoordinate().SetValue( sx,  0)
        vtklegend.UseBackgroundOn()
        vtklegend.SetBackgroundColor(self.legendBG)
        vtklegend.SetBackgroundOpacity(0.6)
        vtklegend.LockBorderOn()
        self.renderer.AddActor(vtklegend)


    #################################################################################
    def show(self, actors=None, at=0, # at=render wind. nr.
             legend=True, axes=None, ruler=False, interactive=None, outputimage=None,
             c='gold', alpha=0.2, wire=False, bc=None, edges=False, q=False):
        '''
        Input: a mixed list of vtkActors, vtkPolydata and filename strings
        at     = number of the renderer to plot to
        legend = a string or list of string for each actor, if False will not show it
        axes   = show xyz axes
        ruler  = draws a simple ruler at the bottom
        interactive = pause and interact w/ window or continue execution
        outputimage = filename to dump a screenshot without asking
        wire   = show in wireframe representation
        edges  = show the edges on top of surface
        bc     = background color, set a color for the back surface face
        q      = force exit after show() command
        '''
        
        # override what was stored internally with passed input
        if actors:
            if not isinstance(actors, list): self.actors = [actors]
            else: self.actors = actors
        if not legend is None:
            if   isinstance(legend, list): self.legend = list(legend)
            elif isinstance(legend,  str): self.legend = [str(legend)]
        if not axes        is None: self.axes = axes
        if not interactive is None: self.interactive = interactive
        if self.verbose:
            print ('Drawing', len(self.actors),'actors ', end='')
            if self.shape != (1,1) : print ('on window',at,'-', end='')
            else: print (' - ', end='')
            if self.interactive: print ('Interactive mode: On.')
            else: print ('Interactive mode: Off.')

        if at<len(self.renderers):
            self.renderer = self.renderers[at]
        else:
            print ("Error in show(): wrong renderer index",at)
            return
        if not self.camera: 
            self.camera = self.renderer.GetActiveCamera()
        else: 
            self.camera.SetThickness(self.camThickness)
        if self.parallelcam: self.camera.ParallelProjectionOn()
        if self.commoncam:
            for r in self.renderers: r.SetActiveCamera(self.camera)

        for i in range(len(self.actors)): # scan for filepaths
            a = self.actors[i]
            if isinstance(a, str): #assume a filepath was given
                out = self.load(a, c=c, bc=bc, alpha=alpha, wire=wire, edges=edges)
                if isinstance(out, str): 
                    print ('File not found:', out)
                    self.actors.pop() #something went wrong, purge list
                else:
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

        if ruler: self._draw_ruler()
        if self.axes: self._draw_cubeaxes()
        if legend: self._draw_legend()

        if self.resetcam: 
            self.renderer.ResetCamera()
            self.camera.Zoom(1.05)

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
            for ia in self.actors:
                if hasattr(ia, 'legend'):
                    self.balloonWidget.AddBalloon(ia, a.legend)

        if hasattr(self, 'interactor') and self.interactor: 
            self.interactor.Render()
        if outputimage: screenshot(outputimage)
        if self.interactive: self.interact()
        if q : # gracefully exit
            if self.verbose: print ('q flag set to True. Exit. Bye.')
            exit(0)


    ############################### events
    def mouseleft(self, obj, event):
        x,y = self.interactor.GetEventPosition()
        self.renderer = obj.FindPokedRenderer(x,y)
        self.renderWin = obj.GetRenderWindow()
        clickedr = self.renderers.index(self.renderer)
        picker = vtk.vtkPropPicker()
        picker.PickProp(x,y, self.renderer)
        clickedActor = picker.GetActor()
        if not clickedActor: clickedActor = picker.GetAssembly()
        if self.verbose and (len(self.renderers)>1 or clickedr>0):
            if self.clickedr != clickedr:
                print ('Current Renderer:', clickedr, end='')
                print (', nr. of actors =', len(self.getActors()))
        if self.verbose and clickedActor and hasattr(clickedActor,'legend'):
            bh = hasattr(self.clickedActor,'legend')
            if len(clickedActor.legend) > 1 :
                if not bh or (bh and self.clickedActor.legend!=clickedActor.legend): 
                    try:                    
                        mass = vtk.vtkMassProperties()
                        apoly = self.getPD(clickedActor)
                        mass.SetInput(apoly)
                        mass.Update() 
                        area = '{:.1e}'.format(float(mass.GetSurfaceArea()))
                        vol  = '{:.1e}'.format(float(mass.GetVolume()))
                    except:
                        area, vol = 'n.a.', 'n.a.'
                    try: 
                        indx = str(self.actors.index(clickedActor))
                    except ValueError: indx = None                        
                    try: 
                        rgb = list(clickedActor.GetProperty().GetColor())
                        cn = '('+getColorName(rgb)+'),'
                    except: cn = None                        
                    if (not indx is None) and isinstance(clickedActor, vtk.vtkAssembly): 
                        print ('-> assembly',indx+':',clickedActor.legend, end=' ')
                    elif not indx is None:
                        print ('-> actor',indx+':',clickedActor.legend, end=' ')
                        if cn: print (cn.ljust(12), end='')
                    npt = str(apoly.GetNumberOfPoints()).ljust(5)
                    print ('Npt='+npt,'area='+str(area),'vol='+str(vol))
        self.clickedActor = clickedActor
        self.clickedx, self.clickedy = x,y
        self.clickedr = clickedr


    def keypress(self, obj, event):
        key = obj.GetKeySym()
        #print ('Pressed key:', key)
        if   key == "q" or key == "space" or key == "Return":
            self.interactor.ExitCallback()
        elif key == "e":
            if self.verbose: print ("Closing window...")
            self.interactor.GetRenderWindow().Finalize()
            self.interactor.TerminateApp()
            del self.renderWin, self.interactor
            return
        elif key == "Escape":
            self.interactor.TerminateApp()
            self.interactor.GetRenderWindow().Finalize()
            self.interactor.TerminateApp()
            del self.renderWin, self.interactor
            exit(0)
        elif key == "S":
            print ('Saving window as screenshot.png')
            screenshot()
            return
        elif key == "C":
            cam = self.renderer.GetActiveCamera()
            print ('\ncam = vtk.vtkCamera() #example code')
            print ('cam.SetPosition(',  [round(e,3) for e in cam.GetPosition()],  ')')
            print ('cam.SetFocalPoint(',[round(e,3) for e in cam.GetFocalPoint()],')')
            print ('cam.SetParallelScale(',round(cam.GetParallelScale(),3),')')
            print ('cam.SetViewUp(', [round(e,3) for e in cam.GetViewUp()],')\n')
            return
        elif key == "m":
            if self.clickedActor in self.getActors():
                self.clickedActor.GetProperty().SetOpacity(0.05)
            else:
                for a in self.getActors(): a.GetProperty().SetOpacity(.05)
        elif key == "comma":
            if self.clickedActor in self.getActors():
                ap = self.clickedActor.GetProperty()
                ap.SetOpacity(max([ap.GetOpacity()-0.05, 0.05]))
            else:
                for a in self.getActors():
                    ap = a.GetProperty()
                    ap.SetOpacity(max([ap.GetOpacity()-0.05, 0.05]))
        elif key == "period":
            if self.clickedActor in self.getActors():
                ap = self.clickedActor.GetProperty()
                ap.SetOpacity(min([ap.GetOpacity()+0.05, 1.0]))
            else:
                for a in self.getActors():
                    ap = a.GetProperty()
                    ap.SetOpacity(min([ap.GetOpacity()+0.05, 1.0]))
        elif key == "slash":
            if self.clickedActor in self.getActors():
                self.clickedActor.GetProperty().SetOpacity(1) 
            else:
                for a in self.getActors(): a.GetProperty().SetOpacity(1)
        elif key == "V":
            if not(self.verbose): self.tips()
            self.verbose = not(self.verbose)
            print ("Verbose: ", self.verbose)
        elif key in ["1", "KP_End", "KP_1"]:
            for i,ia in enumerate(self.getActors()):
                ia.GetProperty().SetColor(colors1[i+self.icol1])
            self.icol1 += 1
            self._draw_legend()
        elif key in ["2", "KP_Down", "KP_2"]:
            for i,ia in enumerate(self.getActors()):
                ia.GetProperty().SetColor(colors2[i+self.icol2])
            self.icol2 += 1
            self._draw_legend()
        elif key in ["4", "KP_Left", "KP_4"]:
            for i,ia in enumerate(self.getActors()):
                ia.GetProperty().SetColor(colors3[i+self.icol3])
            self.icol3 += 1
            self._draw_legend()
        elif key in ["5", "KP_Begin", "KP_5"]:
            c = getColor('gold')
            acs = self.getActors()
            alpha = 1./len(acs)
            for ia in acs:
                ia.GetProperty().SetColor(c)
                ia.GetProperty().SetOpacity(alpha)
            self._draw_legend()
        elif key == "o":
            if self.clickedActor in self.getActors(): acts=[self.clickedActor]
            else: acts = self.getActors()
            for ia in acts:
                try:
                    ps = ia.GetProperty().GetPointSize()
                    ia.GetProperty().SetPointSize(ps-1)
                    ia.GetProperty().SetRepresentationToPoints()
                except AttributeError: pass
        elif key == "O":
            if self.clickedActor in self.getActors(): acts=[self.clickedActor]
            else: acts = self.getActors()
            for ia in acts:
                try:
                    ps = ia.GetProperty().GetPointSize()
                    ia.GetProperty().SetPointSize(ps+2)
                    ia.GetProperty().SetRepresentationToPoints()
                except AttributeError: pass
        elif key == "l":
            if self.clickedActor in self.getActors(): acts=[self.clickedActor]
            else: acts = self.getActors()
            for ia in acts:
                try:
                    ia.GetProperty().SetRepresentationToSurface()
                    ls = ia.GetProperty().GetLineWidth()
                    if ls==1: 
                        ia.GetProperty().EdgeVisibilityOff() 
                        ia.GetProperty().SetLineWidth(0)
                    else: ia.GetProperty().SetLineWidth(ls-1)
                except AttributeError: pass
        elif key == "L":
            if self.clickedActor in self.getActors(): acts=[self.clickedActor]
            else: acts = self.getActors()
            for ia in acts:
                try:
                    ia.GetProperty().EdgeVisibilityOn()
                    c = ia.GetProperty().GetColor()
                    ia.GetProperty().SetEdgeColor(c)
                    ls = ia.GetProperty().GetLineWidth()
                    ia.GetProperty().SetLineWidth(ls+1)
                except AttributeError: pass
#        elif key == "N":
#            if self.clickedActor in self.getActors(): acts=[self.clickedActor]
#            else: acts = self.getActors()
#            for ia in acts:
#                try:
#                    rs = vtk.vtkReverseSense()
#                    rs.ReverseNormalsOn()
#                    setInput(rs, self.getPD(ia))
#                    rs.Update()
#                    ns = rs.GetOutput().GetPointData().GetNormals()
#                    rna = vtk.vtkFloatArray.SafeDownCast(ns)
#                    ia.GetMapper().GetInput().GetPointData().SetNormals(rna)
#                    del rs
#                except: 
#                    print ("Cannot flip normals.")
        elif key == "n": # show normals to an actor
            if self.clickedActor in self.getActors(): acts=[self.clickedActor]
            else: acts = self.getActors()
            for ia in acts:
                alpha = ia.GetProperty().GetOpacity()
                c = ia.GetProperty().GetColor()
                a = self.normals(ia, ratio=1, c=c, alpha=alpha)
                self.actors.pop() #remove from list
                try:
                    i = self.actors.index(ia)
                    self.actors[i] = a
                    self.renderer.RemoveActor(ia)
                    self.interactor.Render()
                except ValueError: pass
            ii = bool(self.interactive)
            self.show(at=self.clickedr, interactive=0, axes=0)
            self.interactive = ii # restore it
        elif key == "x":
            if isinstance(self.clickedActor, vtk.vtkAssembly):
                props = vtk.vtkPropCollection()
                self.clickedActor.GetActors(props)
                actr = props.GetLastProp()
                al = np.sqrt(actr.GetProperty().GetOpacity())
                for op in np.linspace(al,0, 8): #fade away
                    actr.GetProperty().SetOpacity(op)
                    self.interactor.Render()
                self.clickedActor.RemovePart(actr)                    
            elif self.clickedActor in self.getActors():
                actr = self.clickedActor
                al = np.sqrt(actr.GetProperty().GetOpacity())
                for op in np.linspace(al,0, 8): #fade away
                    actr.GetProperty().SetOpacity(op)
                    self.interactor.Render()
                self.renderer.RemoveActor(actr)
            else: 
                if self.verbose:
                    print ('Click an actor and press x to remove it.')
                return
            if self.verbose and hasattr(actr, 'legend'):
                print ('   ...removing actor:', actr.legend)
            self._draw_legend()
            
        self.interactor.Render()


    def interact(self):
        if hasattr(self, 'interactor'):
            if self.interactor:
                self.interactor.Render()
                self.interactor.Start()

    def lastActor(self): return self.actors[-1]


    ###################################################################### Video
    def openVideo(self, name='movie.avi', fps=12, duration=None, format="XVID"):
        try:
            import cv2 #just check existence
            cv2.__version__
        except:
            print ("openVideo: cv2 not installed? Skip.")
            return
        self.videoname = name
        self.videoformat = format
        self.videoduration = duration
        self.fps = float(fps) # if duration is given, will be recalculated
        self.frames = []
        if not os.path.exists('/tmp/v'): os.mkdir('/tmp/v')
        for fl in glob("/tmp/v/*.png"): os.remove(fl)
        print ("Video", name, "is open. Press q to continue.")
        itr = bool(self.interactive)
        self.show(interactive=True)
        self.interactive = itr
        
    def addFrameVideo(self):
        if not self.videoname: return
        fr = '/tmp/v/'+str(len(self.frames))+'.png'
        screenshot(fr)
        self.frames.append(fr)

    def pauseVideo(self, pause):
        '''insert a pause, in seconds'''
        if not self.videoname: return
        fr = self.frames[-1]
        n = int(self.fps*pause)
        for i in range(n): 
            fr2='/tmp/v/'+str(len(self.frames))+'.png'
            self.frames.append(fr2)
            os.system("cp -f %s %s" % (fr, fr2))
            
    def releaseGif(self):
        if not self.videoname: return
        try: import imageio
        except: 
            print ("release_gif: imageio not installed? Skip.")
            return
        images = []
        for fl in self.frames:
            images.append(imageio.imread(fl))
        imageio.mimsave('animation.gif', images)

    def releaseVideo(self):        
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
        self.videoname = None
 
    
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
    r = np.exp(-((pc-0.0)/.2)**2/2.)
    g = np.exp(-((pc-0.5)/.2)**2/2.)
    b = np.exp(-((pc-1.0)/.2)**2/2.)
    colors3.append((r,g,b))
colors3 = colors3 * 100


#########################################################
# Useful Functions
######################################################### 
def getColor(c):
    """Convert a color to (r,g,b) format from many input formats"""
    if isinstance(c,list) or isinstance(c,tuple) : #RGB
        if c[0]<=1 and c[1]<=1 and c[2]<=1:
            return c
        else: return list(np.array(c)/255.)
    elif isinstance(c, str):
        if '#' in c: #hex to rgb
            h = c.lstrip('#')
            rgb255 = list(int(h[i:i+2], 16) for i in (0, 2 ,4))
            rgb = np.array(rgb255)/255.
            if np.sum(rgb)>1: return [0,0,0]
            return list(rgb)
        if len(c)==1: cc = color_nicks # single letter color
        else: cc = color_names         # full name color
        try: 
            ic = cc.index(c.lower())
            return colors[ic]        
        except ValueError:
            # ToDo: add vtk6 defs for colors
            print ("Unknow color name", c, 'is not in:\n', cc)
            if len(c)==1: 
                print ("Available colors:\n", color_names)
                print ("Available abbreviations:\n", color_nicks)
            return [0,0,0]
    elif isinstance(c, int): 
        return colors1[c]
    return [0,0,0]
    

def getColorName(c):
    """Convert any rgb color or numeric code to closest name color"""
    c = np.array(getColor(c)) #reformat
    mdist = 99.
    iclosest = 0
    for i in range(len(colors)):
        ci = np.array(colors[i])
        d = np.linalg.norm(c-ci)
        if d<mdist: 
            mdist = d
            iclosest = i
    return color_names[iclosest] 


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


def isInside(poly, point):
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
def write(poly, fileoutput):
    wt = vtk.vtkPolyDataWriter()
    setInput(wt, poly)
    wt.SetFileName(fileoutput)
    print ("Writing vtk file:", fileoutput)
    wt.Write()
    
vtkMV = vtk.vtkVersion().GetVTKMajorVersion() > 5
def setInput(vtkobj, p):
        if vtkMV: vtkobj.SetInputData(p)
        else: vtkobj.SetInput(p)

   
###########################################################################
if __name__ == '__main__':
###########################################################################
    '''
    Usage: 
    plotter files*.vtk  
    # valid formats [vtk,vtu,vts,vtp, ply,obj,stl,xml,pcd,xyz,txt,byu] 
    '''
###########################################################################
    import sys
    fs = sys.argv[1:]
    if len(fs) == 1 : 
        leg = False
        alpha = 1
    else: 
        leg = True
        alpha = 1./len(fs)  
        print ('Loading',len(fs),'files:', fs)
    vp = vtkPlotter(bg2=(.94,.94,1), balloon=False)
    for f in fs:
        vp.load(f, alpha=alpha)
    vp.show(legend=leg)
###########################################################################









