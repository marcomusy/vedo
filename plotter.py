#!/usr/bin/env python
#
# A helper tool for visualizing vtk objects
#
from __future__ import division, print_function
__author__  = "Marco Musy"
__license__ = "MIT"
__version__ = "6.5"
__maintainer__ = "M. Musy, G. Dalmasso"
__email__   = "marco.musy@embl.es"
__status__  = "dev"
__website__ = "https://github.com/marcomusy/vtkPlotter"


########################################################################
import time, vtk
import numpy as np

from vtkcolors import getColor
import vtkevents
import vtkvideo
import vtkloader

import vtkutils
from vtkutils import printc, makeActor, setInput, vtkMV
from vtkutils import makeAssembly, assignTexture
from vtkutils import getPolyData, getCoordinates

# to expose these methods in plotter namespace (not used in this file):
# they are also passed to the class at line ~140
from vtkutils import closestPoint, isInside, insidePoints, getMaxOfBounds
from vtkutils import normalize, clone, decimate, rotate, shrink, boolActors
from vtkutils import getCM, getVolume, getArea, write, cutterWidget
from vtkutils import ProgressBar, makePolyData


#########################################################################
class vtkPlotter:

    def help(self):
        printc("""
        A python helper class to easily draw VTK tridimensional objects.
        Please follow instructions at:
        https://github.com/marcomusy/vtkPlotter\n""", 1)
        print ("vtkPlotter version:", __version__)
        print ("VTK version:", vtk.vtkVersion().GetVTKVersion())
        try:
            import platform
            print ("Python version:", platform.python_version())
        except: pass
        print('Useful commands on graphic window:')
        self._tips()
        print( '''
        Command line usage:
            > plotter files*.vtk
            # valid file formats:
            # [vtk,vtu,vts,vtp,ply,obj,stl,xml,pcd,xyz,txt,byu,g]
        ''')
    def _tips(self):
        msg  = 'vtkPlotter '+__version__
        msg += ', vtk '+vtk.vtkVersion().GetVTKVersion()
        msg += " ----------------------- Press: \n"
        msg += "\tm   to minimise opacity of selected actor\n"
        msg += "\t.,  to increase/reduce opacity\n"
        msg += "\t/   to maximize opacity of selected actor\n"
        msg += "\tw/s to toggle wireframe/solid style\n"
        msg += "\tpP  to change point size of vertices\n"
        msg += "\tlL  to change edge line width\n"
        msg += "\tn   to show normals for selected actor\n"
        msg += "\tx   to toggle selected actor\n"
        msg += "\tX   to open a cutter widget for sel. actor\n"
        msg += "\t1-4 to change color scheme\n"
        msg += "\tV   to toggle verbose mode\n"
        msg += "\tC   to print current camera info\n"
        msg += "\tS   to save a screenshot\n"
        msg += "\tq   to continue\n"
        msg += "\te   to close current window\n"
        msg += "\tEsc to abort and exit\n"
        msg += "\tCtrl-mouse   to rotate scene\n"
        msg += "\tShift-mouse  to shift scene\n"
        msg += "\tRight-mouse  click to zoom in/out\n"
        msg += "---------------------------------------------------------"
        printc(msg, c='blue')


    def __init__(self, shape=(1,1), size='auto', N=None, screensize=(1100,1800), title='vtkPlotter',
                bg='w', bg2=None, axes=True, verbose=True, interactive=True):
        """
        size = size of the rendering window. If 'auto', guess it based on screensize.
        N    = number of desired renderers arranged in a grid automatically.
        shape= shape of the grid of renderers in format (rows, columns).
               Ignored if N is specified.
        bg   = background color
        bg2  = background color of a gradient towards the top
        interactive = if True will stop after show() to allow interaction w/ window
        """
        self.verbose    = verbose
        self.actors     = []    # list of actors to be shown
        self.clickedActor = None# holds the actor that has been clicked
        self.renderer   = None  # current renderer
        self.renderers  = []    # list of renderers
        self.interactive= interactive # allows to interact with renderer
        self.axes       = axes  # show or hide axes
        self.xtitle     = 'x'   # x axis label and units
        self.ytitle     = 'y'   # y axis label and units
        self.ztitle     = 'z'   # z axis label and units
        self.camera     = None  # current vtkCamera
        self.commoncam  = True  # share the same camera in renderers
        self.projection = 0     # value of the ParallelProjection instance variable
        self.flat       = True  # sets interpolation style to 'flat'
        self.phong      = False # sets interpolation style to 'phong'
        self.gouraud    = False # sets interpolation style to 'gouraud'
        self.bculling   = False # back face culling
        self.fculling   = False # front face culling
        self.legend     = []    # list of legend entries for actors
        self.legendSize = 0.2   # size of legend
        self.legendBG   = (.96,.96,.9) # legend background color
        self.legendPos  = 2     # 1=topright, 2=top-right, 3=bottom-left
        self.result     = dict()# stores extra output information
        self.picked3d   = None  # 3d coords of a clicked point on an actor 

        # mostly internal stuff:
        self.clickedr   = 0     # clicked renderer number
        self.camThickness = 2000
        self.locator    = None
        self.justremoved= None 
        self.caxes_exist = []
        self.icol1      = 0
        self.icol2      = 0
        self.icol3      = 0
        self.clock      = 0
        self._clockt0   = time.time()
        self.initializedPlotter= False
        self.initializedIren = False

        self._videoname   = None
        self._videoformat = None
        self._videoduration = None
        self._fps         = None
        self._frames      = None
        self.writer = None
        self.camera = vtk.vtkCamera()
        
        # share the methods in vtkutils in vtkPlotter class
        self.printc = vtkutils.printc
        self.makeActor = vtkutils.makeActor
        self.setInput = vtkutils.setInput
        self.makeAssembly = vtkutils.makeAssembly
        self.getPolyData = vtkutils.getPolyData
        self.getCoordinates = vtkutils.getCoordinates
        self.boolActors = vtkutils.boolActors
        self.closestPoint = vtkutils.closestPoint
        self.isInside = vtkutils.isInside
        self.insidePoints = vtkutils.insidePoints
        self.getMaxOfBounds = vtkutils.getMaxOfBounds
        self.normalize = vtkutils.normalize
        self.clone = vtkutils.clone
        self.decimate = vtkutils.decimate
        self.rotate = vtkutils.rotate
        self.shrink = vtkutils.shrink
        self.getCM = vtkutils.getCM
        self.getVolume = vtkutils.getVolume
        self.getArea = vtkutils.getArea
        self.write = vtkutils.write
        self.cutterWidget = vtkutils.cutterWidget
        self.ProgressBar = vtkutils.ProgressBar
        self.makePolyData = vtkutils.makePolyData
  
        if N:                # N = number of renderers. Find out the best
            if shape!=(1,1): # arrangement based on minimum nr. of empty renderers
                printc('Warning: having set N, #renderers, shape is ignored.)', c=1)
            x = float(screensize[0])
            y = float(screensize[1])
            nx= int(np.sqrt(int(N*x/y)+1))
            ny= int(np.sqrt(int(N*y/x)+1))
            lm = [(nx,ny), (nx,ny+1), (nx-1,ny), (nx+1,ny), (nx,ny-1)]
            lm+= [(nx-1,ny+1), (nx+1,ny-1), (nx+1,ny+1), (nx-1,ny-1)]
            minl=100
            ind = 0
            for i,m in enumerate(lm):
                l = m[0]*m[1]
                if N <= l < minl:
                  ind = i
                  minl = l
            shape = lm[ind]
            self.size = screensize
        elif size=='auto':        # figure out reasonable window size
            maxs = screensize
            xs = maxs[0]/2.*shape[0]
            ys = maxs[0]/2.*shape[1]
            if xs>maxs[0]:  # shrink
                xs = maxs[0]
                ys = maxs[0]/shape[0]*shape[1]
            if ys>maxs[1]:
                ys = maxs[1]
                xs = maxs[1]/shape[1]*shape[0]
            self.size = (xs,ys)
            if shape==(1,1):
                self.size = (maxs[1]/2,maxs[1]/2)

            if self.verbose and shape!=(1,1):
                print ('Window size =', self.size, 'shape =',shape)

        ############################
        # build the renderers scene:
        for i in reversed(range(shape[0])):
            for j in range(shape[1]):
                arenderer = vtk.vtkRenderer()
                arenderer.SetBackground(getColor(bg))
                if bg2:
                    arenderer.GradientBackgroundOn()
                    arenderer.SetBackground2(getColor(bg2))
                x0 = i/shape[0]
                y0 = j/shape[1]
                x1 = (i+1)/shape[0]
                y1 = (j+1)/shape[1]
                arenderer.SetViewport(y0,x0, y1,x1)
                self.renderers.append(arenderer)
                self.caxes_exist.append(False)
        self.renderWin = vtk.vtkRenderWindow()
        self.renderWin.PolygonSmoothingOn()
        self.renderWin.LineSmoothingOn()
        self.renderWin.PointSmoothingOn()
        self.renderWin.SetSize(int(self.size[1]), int(self.size[0]))
        if title: self.renderWin.SetWindowName(title)
        for r in self.renderers: self.renderWin.AddRenderer(r)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renderWin)
        vsty = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(vsty)


    ####################################### LOADER
    def load(self, inputobj, c='gold', alpha=0.2,
              wire=False, bc=None, edges=False, legend=True, texture=None):
        '''Returns a vtkActor from reading a file, directory or vtkPolyData.
           Optional args:
           c,     color in RGB format, hex, symbol or name
           alpha, transparency (0=invisible)
           wire,  show surface as wireframe
           bc,    backface color of internal surface
           legend, text to show on legend, if True picks filename.
           texture any jpg file can be used as texture
        '''
        if isinstance(inputobj, vtk.vtkPolyData):
            a = makeActor(inputobj, c, alpha, wire, bc, edges, legend, texture)
            self.actors.append(a)
            return a

        acts = vtkloader.load(inputobj, c, alpha, wire, bc, edges, legend, texture)

        if not isinstance(acts, list): acts=[acts]
        for actor in acts:
            if isinstance(actor, vtk.vtkActor):
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
                if self.bculling: actor.GetProperty().BackfaceCullingOn()
                else:             actor.GetProperty().BackfaceCullingOff()
                if self.fculling: actor.GetProperty().FrontfaceCullingOn()
                else:             actor.GetProperty().FrontfaceCullingOff()

        self.actors += acts
        if len(acts) == 1: return acts[0]
        else: return acts


    def getActors(self, obj=None):
        '''
        Return an actors list.
        If None, return actors of current renderer.
        If obj is a int, return actors of renderer #obj.
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
                printc(("Error in getActors: non existing renderer",obj), c=1)
                return []
            else:
                acs = self.renderers[obj].GetActors()
            actors=[]
            acs.InitTraversal()
            for i in range(acs.GetNumberOfItems()):
                a = acs.GetNextItem()
                if isinstance(a, vtk.vtkCubeAxesActor): continue
                if isinstance(a, vtk.vtkLightActor): continue
                actors.append(a)
            return actors

        elif isinstance(obj, str): # search the actor by the legend name
            actors=[]
            for a in self.actors:
                if hasattr(a, 'legend') and obj in a.legend:
                    actors.append(a)
                return actors

        elif isinstance(obj, vtk.vtkActor):
            return [obj]

        if self.verbose:
            printc(('Warning in getActors: unexpected input type',obj), 1)
        return []


    def moveCamera(self, camstart, camstop, fraction):
        '''
        Takes as input two vtkCamera objects and returns
        a new vtkCamera that is at intermediate position:
        fraction=0 -> camstart,  fraction=1 -> camstop.
        Press c key in interactive mode to dump a vtkCamera
        parameter for the current camera view.
        '''
        if isinstance(fraction, int) and self.verbose:
            printc("Warning in moveCamera(): fraction is integer.", 1)
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


    ##################################################################
    def light(self, pos=[1,1,1], fp=[0,0,0], deg=25,
              diffuse='y', ambient='r', specular='b', showsource=False):
        """
        Generate a source of light placed at pos, directed to focal point fp.
        If fp is a vtkActor use its position.
        deg = aperture angle of the light source
        showsource = True, will show the vtk representation of the source.
        """
        if isinstance(fp, vtk.vtkActor): fp = fp.GetPosition()
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(pos)
        light.SetPositional(1)
        light.SetConeAngle(deg)
        light.SetFocalPoint(fp)
        light.SetDiffuseColor(getColor(diffuse))
        light.SetAmbientColor(getColor(ambient))
        light.SetSpecularColor(getColor(specular))
        self.render()
        if showsource:
            lightActor = vtk.vtkLightActor()
            lightActor.SetLight(light)
            self.renderer.AddViewProp(lightActor)
            self.renderer.AddLight(light)
        return light

    
    def points(self, plist=[[1,0,0],[0,1,0],[0,0,1]],
               c='b', tags=[], r=5., alpha=1., legend=None):
        '''
        Return a vtkActor for a list of points.
        Input cols is a list of RGB colors of same length as plist
        If tags is specified the list of string is displayed along 
        with the points.
        If tags='ids' points are labeled with an integer number
        '''
        if isinstance(c, list) or isinstance(c, tuple) and len(c):
            if isinstance(c[0], list) or isinstance(c[0], tuple):
                return self._colorPoints(plist, c, r, alpha, legend)

        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(plist))
        src.Update()
        pd = src.GetOutput()
        for i,p in enumerate(plist): pd.GetPoints().SetPoint(i, p)
        actor = makeActor(pd, c, alpha)
        actor.GetProperty().SetPointSize(r)
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend)

        if tags and 0 < len(tags) <= len(plist):
            tagmap = vtk.vtkLabeledDataMapper()
            setInput(tagmap, pd)
            if tags is 'ids': 
                tagmap.SetLabelModeToLabelIds()
            else:
                vsa = vtk.vtkStringArray()
                vsa.SetName('tags')
                for t in tags: 
                    vsa.InsertNextValue(str(t))
                pd.GetPointData().AddArray(vsa)
                tagmap.SetLabelModeToLabelFieldData()
            tagmap.SetFieldDataName('tags')
            tagprop = tagmap.GetLabelTextProperty()
            tagprop.BoldOn()
            tagprop.ItalicOff()
            tagprop.ShadowOff()
            tagprop.SetColor(0,0,.0)
            tagprop.SetFontSize(12)
            tagactor = vtk.vtkActor2D()
            tagactor.SetMapper(tagmap)
            self.actors.append(tagactor)
        return actor

    def point(self, pos=[0,0,0], c='b', r=10., alpha=1., legend=None):
        return self.points([pos], c, [], r, alpha, legend)

    def _colorPoints(self, plist, cols, r, alpha, legend):
        if len(plist) != len(cols):
            printc(("Mismatch in colorPoints()", len(plist), len(cols)), 1)
            exit()
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(plist))
        src.Update()
        vertexFilter = vtk.vtkVertexGlyphFilter()
        setInput(vertexFilter, src.GetOutput())
        vertexFilter.Update()
        pd = vertexFilter.GetOutput()
        ucols = vtk.vtkUnsignedCharArray()
        ucols.SetNumberOfComponents(3)
        ucols.SetName("RGB")
        for i,p in enumerate(plist):
            pd.GetPoints().SetPoint(i, p)
            c = np.array(getColor(cols[i]))*255
            if vtkMV:
                ucols.InsertNextTuple3(c[0],c[1],c[2])
            else:
                ucols.InsertNextTupleValue(c)
        pd.GetPointData().SetScalars(ucols)
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


    def line(self, p0, p1=None, lw=1, dotted=False,
             c='r', alpha=1., legend=None):
        '''Returns the line segment between points p0 and p1
           if p0 is a list of points returns the line connecting them.
        '''
        #detect if user is passing a list of points:
        if isinstance(p0[0], list) or  isinstance(p0[0], tuple):
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
        actor = makeActor(poly, c, alpha, legend=legend)
        actor.GetProperty().SetLineWidth(lw)
        if dotted:
            actor.GetProperty().SetLineStipplePattern(0xf0f0)
            actor.GetProperty().SetLineStippleRepeatFactor(1)
        self.actors.append(actor)
        return actor


    def sphere(self, pos=[0,0,0], r=1,
               c='r', alpha=1, legend=None, texture=None, res=24):

        ss = vtk.vtkSphereSource()
        ss.SetCenter(pos)
        ss.SetRadius(r)
        ss.SetThetaResolution(res)
        ss.SetPhiResolution(res)
        ss.Update()
        pd = ss.GetOutput()

        actor = makeActor(pd, c=c, alpha=alpha, legend=legend, texture=texture)
        actor.GetProperty().SetInterpolationToPhong()
        self.actors.append(actor)
        return actor


    def box(self, pos=[0,0,0], length=1, width=2, height=3, normal=(0,0,1),
            c='g', alpha=1, legend=None, texture=None):
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
        t.Translate(pos)

        tf = vtk.vtkTransformPolyDataFilter()
        setInput(tf, poly)
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()

        actor = makeActor(pd, c=c, alpha=alpha, legend=legend, texture=texture)
        self.actors.append(actor)
        return actor


    def cube(self, pos=[0,0,0], length=1, normal=(0,0,1),
             c='g', alpha=1., legend=None, texture=None):
        return self.box(pos, length, 1, 1, normal, c, alpha, legend, texture)


    def octahedron(self, pos=[0,0,0], s=1, axis=(0,0,1),
                   c='g', alpha=1, wire=False, legend=None, texture=None):
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(6)
        pts.SetPoint(0, -s, 0, 0)
        pts.SetPoint(1, s, 0, 0)
        pts.SetPoint(2, 0, -s, 0)
        pts.SetPoint(3, 0, s, 0)
        pts.SetPoint(4, 0, 0, -s)
        pts.SetPoint(5, 0, 0, s) # axis z points to this
        t = vtk.vtkCellArray()
        t.InsertNextCell(3)
        t.InsertCellPoint(2); t.InsertCellPoint(0); t.InsertCellPoint(4)
        t.InsertNextCell(3)
        t.InsertCellPoint(1); t.InsertCellPoint(2); t.InsertCellPoint(4)
        t.InsertNextCell(3)
        t.InsertCellPoint(3); t.InsertCellPoint(1); t.InsertCellPoint(4)
        t.InsertNextCell(3)
        t.InsertCellPoint(0); t.InsertCellPoint(3); t.InsertCellPoint(4)
        t.InsertNextCell(3)
        t.InsertCellPoint(0); t.InsertCellPoint(2); t.InsertCellPoint(5)
        t.InsertNextCell(3)
        t.InsertCellPoint(2); t.InsertCellPoint(1); t.InsertCellPoint(5)
        t.InsertNextCell(3)
        t.InsertCellPoint(1); t.InsertCellPoint(3); t.InsertCellPoint(5)
        t.InsertNextCell(3)
        t.InsertCellPoint(3); t.InsertCellPoint(0); t.InsertCellPoint(5)
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(t)

        axis  = np.array(axis)/np.linalg.norm(axis)
        theta = np.arccos(axis[2])
        phi   = np.arctan2(axis[1], axis[0])
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.RotateY(theta*57.3)
        t.RotateZ(phi*57.3)
        t.Translate(pos)
        tf = vtk.vtkTransformPolyDataFilter()
        setInput(tf, pd)
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()

        actor = makeActor(pd, c=c, alpha=alpha, wire=wire,
                          legend=legend, texture=texture)
        actor.GetProperty().SetInterpolationToPhong()
        self.actors.append(actor)
        return actor


    def plane(self, pos=[0,0,0], normal=[0,0,1], s=1, c='g', bc='darkgreen',
              lw=1, alpha=1, wire=False, legend=None, texture=None):
        p = self.grid(pos, normal, s, c, bc, lw, alpha, wire, legend, texture,1)
        if not texture: p.GetProperty().SetEdgeVisibility(1)
        return p


    def grid(self, pos=[0,0,0], normal=[0,0,1], s=10, c='g', bc='darkgreen',
             lw=1, alpha=1, wire=True, legend=None, texture=None, res=10):
        '''Return a grid plane'''
        ps = vtk.vtkPlaneSource()
        ps.SetResolution(res, res)
        ps.SetCenter(np.array(pos)/s)
        ps.SetNormal(normal)
        ps.Update()
        actor = makeActor(ps.GetOutput(), 
                          c=c, bc=bc, alpha=alpha, legend=legend, texture=texture)
        actor.SetScale(s,s,s)
        if wire: actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(lw)
        actor.PickableOff()
        self.actors.append(actor)
        return actor


    def polygon(self, pos=[0,0,0], normal=[0,0,1], nsides=6, r=1,
                c='coral', bc='darkgreen', lw=1, alpha=1,
                legend=None, texture=None, followcam=False):
        ps = vtk.vtkRegularPolygonSource()
        ps.SetNumberOfSides(nsides)
        ps.SetCenter(pos)
        ps.SetRadius(r)
        ps.SetNormal(-np.array(normal))
        ps.Update()

        tf = vtk.vtkTriangleFilter()
        setInput(tf, ps.GetOutputPort())
        tf.Update()

        mapper = vtk.vtkPolyDataMapper()
        setInput(mapper, tf.GetOutputPort())
        if followcam: #follow cam
            actor = vtk.vtkFollower()
            actor.SetCamera(self.camera)
            if not self.camera:
                printc('Warning: vtkCamera does not yet exist for polygon',5)
        else:
            actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(getColor(c))
        actor.GetProperty().SetOpacity(alpha)
        actor.GetProperty().SetLineWidth(lw)
        actor.GetProperty().SetInterpolationToFlat()
        if bc: # defines a specific color for the backface
            backProp = vtk.vtkProperty()
            backProp.SetDiffuseColor(getColor(bc))
            backProp.SetOpacity(alpha)
            actor.SetBackfaceProperty(backProp)
        if texture: assignTexture(actor, texture)
        vtkutils.assignPhysicsMethods(actor)
        vtkutils.assignConvenienceMethods(actor, legend)
        self.actors.append(actor)
        return actor


    def disc(self, pos=[0,0,0], normal=[0,0,1], r1=0.5, r2=1,
             c='coral', bc='darkgreen', lw=1, alpha=1, 
             legend=None, texture=None, res=12):
        ps = vtk.vtkDiskSource()
        ps.SetInnerRadius(r1)
        ps.SetOuterRadius(r2)
        ps.SetRadialResolution(res)
        ps.SetCircumferentialResolution(res*4)
        ps.Update()
        tr = vtk.vtkTriangleFilter()
        setInput(tr, ps.GetOutputPort())
        tr.Update()

        axis  = np.array(normal)/np.linalg.norm(normal)
        theta = np.arccos(axis[2])
        phi   = np.arctan2(axis[1], axis[0])
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.RotateY(theta*57.3)
        t.RotateZ(phi*57.3)
        t.Translate(pos)
        tf = vtk.vtkTransformPolyDataFilter()
        setInput(tf, tr.GetOutput())
        tf.SetTransform(t)
        tf.Update()

        pd = tf.GetOutput()
        mapper = vtk.vtkPolyDataMapper()
        setInput(mapper, pd)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(getColor(c))
        actor.GetProperty().SetOpacity(alpha)
        actor.GetProperty().SetLineWidth(lw)
        actor.GetProperty().SetInterpolationToFlat()
        if bc: # defines a specific color for the backface
            backProp = vtk.vtkProperty()
            backProp.SetDiffuseColor(getColor(bc))
            backProp.SetOpacity(alpha)
            actor.SetBackfaceProperty(backProp)
        if texture: assignTexture(actor, texture)
        vtkutils.assignPhysicsMethods(actor)
        vtkutils.assignConvenienceMethods(actor, legend)
        self.actors.append(actor)
        return actor
        

    def arrow(self, start=[0,0,0], end=[1,1,1], axis=None,
              c='r', alpha=1, legend=None, texture=None, res=12):
        if axis:
            end = start+np.array(axis)
        axis = np.array(end) - np.array(start)
        length = np.linalg.norm(axis)
        if not length: return None
        axis = axis/length
        theta = np.arccos(axis[2])
        phi   = np.arctan2(axis[1], axis[0])
        arr = vtk.vtkArrowSource()
        arr.SetShaftResolution(res)
        arr.SetTipResolution(res)
        arr.SetTipRadius(0.06)
        arr.Update()
        actor = makeActor(arr.GetOutput(),
                          c=c, alpha=alpha, legend=legend, texture=texture)
        actor.GetProperty().SetInterpolationToPhong()
        actor.SetPosition(start)
        actor.RotateZ(phi*57.3)
        actor.RotateY(theta*57.3)
        actor.SetScale(length,length,length)
        actor.RotateY(-90) #put it along Z
        actor.DragableOff()
        actor.PickableOff()
        self.actors.append(actor)
        return actor


    def cylinder(self, pos=[0,0,0], radius=1, height=1, axis=[0,0,1],
                 c='teal', alpha=1, legend=None, texture=None, res=24):
        cyl = vtk.vtkCylinderSource()
        cyl.SetResolution(res)
        cyl.SetRadius(radius)
        cyl.SetHeight(height)
        cyl.Update()

        axis  = np.array(axis)/np.linalg.norm(axis)
        theta = np.arccos(axis[2])
        phi   = np.arctan2(axis[1], axis[0])
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.RotateX(90) #put it along Z
        t.RotateY(theta*57.3)
        t.RotateZ(phi*57.3)
        t.Translate(pos)
        tf = vtk.vtkTransformPolyDataFilter()
        setInput(tf, cyl.GetOutput())
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()

        actor = makeActor(pd, c=c, alpha=alpha, legend=legend, texture=texture)
        actor.GetProperty().SetInterpolationToPhong()
        self.actors.append(actor)
        return actor


    def paraboloid(self, pos=[0,0,0], radius=1, height=1, axis=[0,0,1],
                   c='cyan', alpha=1, legend=None, texture=None, res=50):
        quadric = vtk.vtkQuadric()
        quadric.SetCoefficients(1, 1, 0, 0, 0, 0, 0, 0, 0.25/height, 0)
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
        t.Scale(radius,radius,radius)
        t.Translate(pos)
        tf = vtk.vtkTransformPolyDataFilter()
        setInput(tf, contours.GetOutput())
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()

        actor = makeActor(pd, c=c, alpha=alpha, legend=legend, texture=texture)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetMapper().ScalarVisibilityOff()
        self.actors.append(actor)
        return actor


    def hyperboloid(self, pos=[0,0,0], a2=1, value=0.5, height=1, axis=[0,0,1],
                    c='magenta', alpha=1, legend=None, texture=None, res=50):
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
        t.Translate(pos)
        tf = vtk.vtkTransformPolyDataFilter()
        setInput(tf, contours.GetOutput())
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()

        actor = makeActor(pd, c=c, alpha=alpha, legend=legend, texture=texture)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetMapper().ScalarVisibilityOff()
        self.actors.append(actor)
        return actor


    def cone(self, pos=[0,0,0], radius=1, height=1, axis=[0,0,1],
             c='dg', alpha=1, legend=None, texture=None, res=48):
        con = vtk.vtkConeSource()
        con.SetResolution(res)
        con.SetRadius(radius)
        con.SetHeight(height)
        con.SetDirection(axis)
        con.Update()
        actor = makeActor(con.GetOutput(), c=c, alpha=alpha,
                          legend=legend, texture=texture)
        actor.GetProperty().SetInterpolationToPhong()
        actor.SetPosition(pos)
        actor.DragableOff()
        actor.PickableOff()
        self.actors.append(actor)
        return actor


    def pyramid(self, pos=[0,0,0], s=1, height=1, axis=[0,0,1],
                c='dg', alpha=1, legend=None, texture=None):
        a = self.cone(pos, s, height, axis, c, alpha, legend, texture, 4)
        return a


    def ring(self, pos=[0,0,0], radius=1, thickness=0.1, axis=[1,1,1],
             c='khaki', alpha=1, legend=None, texture=None, res=30):
        rs = vtk.vtkParametricTorus()
        rs.SetRingRadius(radius)
        rs.SetCrossSectionRadius(thickness)
        pfs = vtk.vtkParametricFunctionSource()
        pfs.SetParametricFunction(rs)
        pfs.SetUResolution(res*3)
        pfs.SetVResolution(res)
        pfs.Update()

        axis  = np.array(axis)/np.linalg.norm(axis)
        theta = np.arccos(axis[2])
        phi   = np.arctan2(axis[1], axis[0])
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.RotateY(theta*57.3)
        t.RotateZ(phi*57.3)
        t.Translate(pos)
        tf = vtk.vtkTransformPolyDataFilter()
        setInput(tf, pfs.GetOutput())
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()

        actor = makeActor(pd, c=c, alpha=alpha, legend=legend, texture=texture)
        actor.GetProperty().SetInterpolationToPhong()
        self.actors.append(actor)
        return actor


    def ellipsoid(self, pos=[0,0,0], axis1=[1,0,0], axis2=[0,2,0], axis3=[0,0,3],
                  c='c', alpha=1, legend=None, texture=None, res=24):
        """axis1 and axis2 are only used to define sizes and one azimuth angle"""
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
        t.Translate(pos)
        tf = vtk.vtkTransformPolyDataFilter()
        setInput(tf, elliSource.GetOutput())
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()

        actor= makeActor(pd, c=c, alpha=alpha, legend=legend, texture=texture)
        actor.GetProperty().BackfaceCullingOn()
        actor.GetProperty().SetInterpolationToPhong()
        self.actors.append(actor)
        return self.lastActor()


    def helix(self, pos=[0,0,0], length=2, n=6, radius=1, axis=[0,0,1],
              lw=5, c='grey', alpha=1, legend=None):
        thickness = max(length,radius)*2*lw
        trange = np.linspace(-length/2., length/2., num=10*n)
        pts = [ [np.cos(2*n*t),np.sin(2*n*t),t] for t in trange ]
        actor = self.spline(pts, 0, 2, thickness, c, alpha, False, legend)
        axis  = np.array(axis)/np.linalg.norm(axis)
        theta = np.arccos(axis[2])
        phi   = np.arctan2(axis[1], axis[0])
        actor.GetProperty().SetInterpolationToPhong()
        actor.SetPosition(pos)
        actor.RotateZ(phi*57.3)
        actor.RotateY(theta*57.3)
        self.actors.append(actor)
        if legend: setattr(actor, 'legend', legend)
        return self.lastActor()


    def _vtkspline(self, points, s, c, alpha, nodes, legend, res):
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
        actline = makeActor(profileData, c=c, alpha=alpha, legend=legend)
        actline.GetProperty().SetLineWidth(s)
        actline.GetProperty().SetInterpolationToPhong()
        if nodes:
            pts = getCoordinates(inputData)
            actnodes = self.points(pts, r=s*1.5, c=c, alpha=alpha)
            self.actors.pop()
            ass = makeAssembly([actline, actnodes], legend=legend)
            self.actors.append(ass)
            return ass
        else:
            self.actors.append(actline)
            return actline


    def spline(self, points, smooth=0.5, degree=2, 
               s=2, c='b', alpha=1., nodes=False, legend=None, res=20):
        '''
        Return a vtkActor for a spline that doesnt necessarly 
               pass exactly throught all points.
        smooth = smoothing factor, 0=interpolate points exactly, 1=average point positions
        degree = degree of the spline (1<degree<5)
        nodes  = True shows the points and therefore returns a vtkAssembly
        '''
        try:
            from scipy.interpolate import splprep, splev
        except ImportError:
            printc('Warning: ..scipy not installed, using vtkCardinalSpline instead.',5)
            return self._vtkspline(points, s, c, alpha, nodes, legend, res)

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
        acttube = makeActor(profileData, c=c, alpha=alpha, legend=legend)
        acttube.GetProperty().SetLineWidth(s)
        acttube.GetProperty().SetInterpolationToPhong()
        if nodes:
            actnodes = self.points(points, r=s*1.5, c=c, alpha=alpha)
            self.actors.pop()
            ass = makeAssembly([acttube, actnodes], legend=legend)
            self.actors.append(ass)
            return ass
        else:
            self.actors.append(acttube)
            return acttube


    def text(self, txt, pos=(0,0,0), s=1,
             c='k', alpha=1, bc=None, followcam=True, texture=None):
        '''
        Returns a vtkActor that shows a text 3D
        if cam is True the text will auto-orient to it
        '''
        tt = vtk.vtkVectorText()
        tt.SetText(txt)
        ttmapper = vtk.vtkPolyDataMapper()
        ttmapper.SetInputConnection(tt.GetOutputPort())
        if followcam: #follow cam
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
        if texture: assignTexture(ttactor, texture)
        self.actors.append(ttactor)
        return ttactor


    def xyplot(self, points=[[0,0],[1,0],[2,1],[3,2],[4,1]],
               title='', c='r', corner=1, lines=False):
        """
        Return a vtkActor that is a plot of 2D points in x and y.
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
        tprop = plot.GetAxisLabelTextProperty()
        tprop.SetColor(0,0,0)
        tprop.SetOpacity(0.7)
        tprop.SetFontFamily(0)
        tprop.BoldOff()
        tprop.ItalicOff()
        tprop.ShadowOff()
        tprop.SetFontSize(6) #not working
        plot.SetAxisTitleTextProperty(tprop)
        plot.SetAxisLabelTextProperty(tprop)
        plot.SetTitleTextProperty(tprop)
        if corner==1: plot.GetPositionCoordinate().SetValue(.0, .8, 0)
        if corner==2: plot.GetPositionCoordinate().SetValue(.7, .8, 0)
        if corner==3: plot.GetPositionCoordinate().SetValue(.0, .0, 0)
        if corner==4: plot.GetPositionCoordinate().SetValue(.7, .0, 0)
        plot.GetPosition2Coordinate().SetValue(.3, .2, 0)
        self.actors.append(plot)
        return plot


    def fxy(self, z, x=[0,3], y=[0,3],
            zlimits=[-1e+30, 1e+30], showNan=True, zlevels=10,
            c='b', bc='aqua', alpha=1, legend=True, texture=None, res=100):
        '''
        Return a surface representing the 3D function specified as a string
        or as a reference to an external function.
        Red points indicate where the function does not exist (showNan).
        zlevels will draw the specified number of z-levels contour lines.
        Examples:
            vp = plotter.vtkPlotter()
            vp.function('sin(3*x)*log(x-y)/3')
            or
            def z(x,y): return np.sin(x*y)
            vp.function(z) # or equivalently:
            vp.function(lambda x,y: np.sin(x*y))
        '''
        expr = False
        if isinstance(z, str):
            try:
                import sympy
                sx,sy = sympy.symbols('x y')
                z = z.replace('np.','')
                z = z.replace('math.','')
                expr = sympy.sympify(z)
                if legend is True: legend = z
                res = int(res/2) # sympy is slower
                if self.verbose:
                    sympy.init_printing()
                    printc('Function properties of:','c')
                    sympy.pprint(expr)
                    printc('\t simplified form: '+str(sympy.simplify(z)))
                    printc('\t expanded form: '+str(sympy.expand(z)))
                    printc('\t x derivative: '+str(sympy.diff(z, sx)))
                    printc('\t y derivative: '+str(sympy.diff(z, sy)))
            except:
                printc('Error in vtkPlotter.function: sympy not installed?',1)
                printc('or syntax error in function: '+z,1)
                printc('Use instead e.g.: def z(x,y): return math.sin(x*y)',5)
                printc('vp.function(z)', 5)
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
            setInput(tf, poly)
            tf.Update()
            poly = tf.GetOutput()

        for i in range(poly.GetNumberOfPoints()):
            px,py,_ = poly.GetPoint(i)
            xv = (px+.5)*dx+x[0]
            yv = (py+.5)*dy+y[0]
            try:
                if expr: zv = expr.subs([(sx, xv), (sy, yv)])
                else:    zv = z(xv, yv)
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
            setInput(cl, poly)
            cl.Update()
            poly = cl.GetOutput()

        a = self.cutActor(poly, (0,0,zlimits[0]), (0,0,1), False)
        poly = getPolyData(a)
        a = self.cutActor(poly, (0,0,zlimits[1]), (0,0,-1), False)
        poly = getPolyData(a)

        if c is None:
            elev = vtk.vtkElevationFilter()
            setInput(elev,poly)
            elev.Update()
            poly = elev.GetOutput()

        actor = makeActor(poly, c=c, bc=bc, alpha=alpha,
                          legend=legend, texture=texture)
        acts=[actor]

        if zlevels:
           elevation = vtk.vtkElevationFilter()
           setInput(elevation, poly)
           bounds = poly.GetBounds()
           elevation.SetLowPoint( 0,0,bounds[4])
           elevation.SetHighPoint(0,0,bounds[5])
           elevation.Update()
           bcf = vtk.vtkBandedPolyDataContourFilter()
           setInput(bcf, elevation.GetOutput())
           bcf.SetScalarModeToValue()
           bcf.GenerateContourEdgesOn()
           bcf.GenerateValues(zlevels, elevation.GetScalarRange())
           bcf.Update()
           zpoly = bcf.GetContourEdgesOutput()
           zbandsact = makeActor(zpoly, c='k', alpha=alpha)
           zbandsact.GetProperty().SetLineWidth(1.5)
           acts.append(zbandsact)

        if showNan and len(todel):
            bb = actor.GetBounds()
            zm = (bb[4]+bb[5])/2
            nans = np.array(nans)+[0,0,zm]
            nansact = self.points(nans, c='red', alpha=alpha)
            self.actors.pop()
            acts.append(nansact)

        if len(acts)>1:
            asse = makeAssembly(acts, legend)
            self.actors.append(asse)
            return asse
        else:
            self.actors.append(actor)
            return actor

    
    def addScalarBar(self, actor=None, c='k', horizontal=False):
        """
        Add a scalar bar for the specified actor.
        If actor is None will add it to the last actor in self.actors
        """
        
        if actor is None: actor=self.lastActor()
        if not isinstance(actor, vtk.vtkActor) or not hasattr(actor, 'GetMapper'): 
            printc('Error in addScalarBar: input is not a vtkActor.',1)
            return None
        lut = actor.GetMapper().GetLookupTable()
        if not lut: return None
        
        c = getColor(c)
        sb = vtk.vtkScalarBarActor()
        sb.SetLookupTable(lut)
        sb.UnconstrainedFontSizeOn()
        sb.FixedAnnotationLeaderLineColorOff()
        sb.DrawAnnotationsOn()
        sb.DrawTickLabelsOn()
        sb.SetMaximumNumberOfColors(256)

        if horizontal:
            sb.SetOrientationToHorizontal ()
            sb.SetNumberOfLabels(4)
            sb.SetTextPositionToSucceedScalarBar ()
            sb.SetPosition(0.1,.05)
            sb.SetMaximumWidthInPixels(1000)
            sb.SetMaximumHeightInPixels(70)
        else:
            sb.SetNumberOfLabels(10)
            sb.SetTextPositionToPrecedeScalarBar()            
            sb.SetPosition(.87,.05)
            sb.SetMaximumWidthInPixels(80)
            sb.SetMaximumHeightInPixels(500)

        sctxt = sb.GetLabelTextProperty()
        sctxt.SetColor(c)
        sctxt.SetShadow(0)
        sctxt.SetFontFamily(0)
        sctxt.SetItalic(0)
        sctxt.SetBold(0)
        sctxt.SetFontSize(12)
        if not self.renderer: self.render()
        self.renderer.AddActor(sb)
        self.render()
        return sb


    def normals(self, actor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8, legend=None):
        '''
        Returns a vtkActor that contains the normals at vertices shown as arrows
        '''
        maskPts = vtk.vtkMaskPoints()
        maskPts.SetOnRatio(ratio)
        maskPts.RandomModeOff()
        src = getPolyData(actor)
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
        aactor = makeAssembly([actor,glyphActor], legend=legend)
        self.actors.append(aactor)
        return aactor


    def curvature(self, actor, method=1, r=1, alpha=1, lut=None, legend=None):
        '''
        Returns a vtkActor that contains the color coded surface
        curvature following four different ways to calculate it:
        method =  0-gaussian, 1-mean, 2-max, 3-min
        '''
        poly = getPolyData(actor)
        cleaner = vtk.vtkCleanPolyData()
        setInput(cleaner, poly)
        curve = vtk.vtkCurvatures()
        curve.SetInputConnection(cleaner.GetOutputPort())
        curve.SetCurvatureType(method)
        curve.InvertMeanCurvatureOn()
        curve.Update()
        if self.verbose: print('CurvatureType set to:', method)
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
        self.actors.append(cactor)
        if legend: setattr(cactor, 'legend', legend)
        return cactor


    def boundaries(self, actor, c='p', lw=5, legend=None):
        '''Returns a vtkActor that shows the boundary lines of a surface.'''
        fe = vtk.vtkFeatureEdges()
        setInput(fe, getPolyData(actor))
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOn()
        fe.ManifoldEdgesOn()
        fe.NonManifoldEdgesOn()
        fe.ColoringOff()
        fe.Update()
        bactor = makeActor(fe.GetOutput(), c=c, alpha=1, legend=legend)
        bactor.GetProperty().SetLineWidth(lw)
        self.actors.append(bactor)
        return bactor


    ################# working with point clouds
    def fitLine(self, points, c='orange', lw=1, alpha=0.6, legend=None):
        '''
        Fits a line through points.
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
        self.result['slope'] = vv
        self.result['center'] = datamean
        self.result['variances'] = dd
        if self.verbose:
            printc("Extra info saved in vp.results['slope','center','variances']",5)
        return l


    def fitPlane(self, points, c='g', bc='darkgreen', legend=None):
        '''
        Fits a plane to a set of points.
        Extra info is stored in vp.results['normal','center','variance']
        '''
        data = np.array(points)
        datamean = data.mean(axis=0)
        uu, dd, vv = np.linalg.svd(data - datamean)
        xyz_min = points.min(axis=0)
        xyz_max = points.max(axis=0)
        s= np.linalg.norm(xyz_max - xyz_min)
        n = np.cross(vv[0],vv[1])
        pla = self.plane(datamean, n, c=c, bc=bc, s=s, lw=2, alpha=0.8, legend=legend)
        self.result['normal']  = n
        self.result['center']  = datamean
        self.result['variance']= dd[2]
        if self.verbose:
            printc("Extra info saved in vp.results['normal','center','variance']",5)
        return pla


    def pca(self, points=[[1,0,0],[0,1,0],[0,0,1],[.5,0,1],[0,.2,.3]],
            pvalue=.95, c='c', alpha=0.5, pcaAxes=False, legend=None):
        '''
        Show the oriented PCA ellipsoid that contains 95% of points.
        axes = True, show the 3 PCA semi axes
        Extra info is stored in vp.results['sphericity','a','b','c']
        sphericity = 1 for a perfect sphere
        '''
        try:
            from scipy.stats import f
        except:
            printc("Error in ellipsoid(): scipy not installed. Skip.",1)
            return None
        P = np.array(points, ndmin=2, dtype=float)
        cov = np.cov(P, rowvar=0)    # covariance matrix
        U, s, R = np.linalg.svd(cov) # singular value decomposition
        p, n = s.size, P.shape[0]
        fppf = f.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p) # f % point function
        va,vb,vc = np.sqrt(s*fppf)*2   # semi-axes (largest first)
        center = np.mean(P, axis=0)  # centroid of the hyperellipsoid
        self.result['sphericity'] = (((va-vb)/(va+vb))**2
                                   + ((va-vc)/(va+vc))**2
                                   + ((vb-vc)/(vb+vc))**2 )/3. *4.
        self.result['a'] = va
        self.result['b'] = vb
        self.result['c'] = vc
        if self.verbose:
            printc("Extra info saved in vp.results['sphericity','a','b','c']",5)
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
        actor_elli = makeActor(ftra.GetOutput(), c, alpha, legend=legend)
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
                setInput(t, l.GetOutput())
                t.Update()
                axs.append(makeActor(t.GetOutput(), c, alpha))
            asse = makeAssembly([actor_elli]+axs, legend=legend)
            self.actors.append( asse )
        else : 
            self.actors.append(actor_elli)
        return self.lastActor()


    def align(self, source, target, rigid=False, iters=100, legend=None):
        '''
        Return a vtkActor which is the same as source but
        aligned to target though IterativeClosestPoint method
        rigid = True, then no scaling is allowed.
        '''
        sprop = source.GetProperty()
        source = getPolyData(source)
        target = getPolyData(target)
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
        actor = makeActor(poly, legend=legend)
        actor.SetProperty(sprop)
        self.actors.append(actor)
        return actor


    def cutActor(self, actor, origin=(0,0,0), normal=(1,0,0),
                 showcut=True, showline=False):
        '''
        Takes actor and cuts it with the plane defined by a point
        and a normal. Substitutes it to the original actor.
        showcut  = shows the cut away part as thin wireframe
        showline = marks with a thick line the cut
        '''
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        poly = getPolyData(actor)
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
            alpha=1
            c='gold'
            bf=None
        leg = None
        if hasattr(actor, 'legend'): leg = actor.legend
        clipActor = makeActor(clipper.GetOutput(),c=c,alpha=alpha, legend=leg)
        clipActor.SetBackfaceProperty(bf)

        acts = []
        if showcut:
            cpoly = clipper.GetClippedOutput()
            restActor = makeActor(cpoly, c=c, alpha=0.05, wire=1)
            acts.append(restActor)

        cutEdges = vtk.vtkCutter()
        setInput(cutEdges, poly)
        cutEdges.SetCutFunction(plane)
        cutEdges.SetValue(0, 0.)
        cutEdges.Update()
        cutStrips = vtk.vtkStripper()
        cutStrips.SetInputConnection(cutEdges.GetOutputPort())
        cutStrips.Update()
        if showline:
            cutPoly = vtk.vtkPolyData()
            cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
            cutPoly.SetPolys(cutStrips.GetOutput().GetLines())
            cutline = makeActor(cutPoly, c=c, alpha=np.sqrt(alpha))
            cutline.GetProperty().SetRepresentationToWireframe()
            cutline.GetProperty().SetLineWidth(4)
            acts.append(cutline)

        if len(acts)>1:
            finact = makeAssembly(acts, legend=leg)
        else:
            finact = clipActor
        try:
            i = self.actors.index(actor)
            self.actors[i] = finact # substitute original actor with cut one
        except ValueError: pass
        return finact


    def subDivideMesh(self, actor, N=1, method=0, legend=None):
        '''Increases the number of points in actor'''        
        triangles = vtk.vtkTriangleFilter()
        setInput(triangles, getPolyData(actor))
        triangles.Update()
        originalMesh = triangles.GetOutput()
        if   method==0: sdf = vtk.vtkLoopSubdivisionFilter()
        elif method==1: sdf = vtk.vtkLinearSubdivisionFilter()
        elif method==2: sdf = vtk.vtkAdaptiveSubdivisionFilter()
        elif method==3: sdf = vtk.vtkButterflySubdivisionFilter()
        else:
            printc('Error in subDivideMesh: unknown method.', 'r')
            exit(1)
        if method != 2: sdf.SetNumberOfSubdivisions(N)
        setInput(sdf, originalMesh)
        sdf.Update()
        out = sdf.GetOutput()
        sactor = makeActor(out, legend=legend)
        sactor.GetProperty().SetOpacity(actor.GetProperty().GetOpacity())
        sactor.GetProperty().SetColor(actor.GetProperty().GetColor())
        try:
            i = self.actors.index(actor)
            self.actors[i] = sactor # substitute original actor
        except ValueError: pass
        return sactor


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
        ca.ZAxisLabelVisibilityOn()
        ca.SetXTitle(self.xtitle)
        ca.SetYTitle(self.ytitle)
        ca.SetZTitle(self.ztitle)
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
        if not isinstance(self.legend, list): return

        # remove old legend if present on current renderer:
        acs = self.renderer.GetActors2D()
        acs.InitTraversal()
        for i in range(acs.GetNumberOfItems()):
            a = acs.GetNextItem()
            if isinstance(a, vtk.vtkLegendBoxActor):
                self.renderer.RemoveActor(a)

        actors = self.getActors()
        acts, texts = [], []
        for i in range(len(actors)):
            a = actors[i]
            if i<len(self.legend) and self.legend[i]!='':
                if isinstance(self.legend[i], str):
                    texts.append(self.legend[i])
                    acts.append(a)
            elif hasattr(a, 'legend') and a.legend:
                if isinstance(a.legend, str):
                    texts.append(a.legend)
                    acts.append(a)

        NT = len(texts)
        if NT>25: NT=25
        vtklegend = vtk.vtkLegendBoxActor()
        vtklegend.SetNumberOfEntries(NT)
        for i in range(NT):
            ti = texts[i]
            a  = acts[i]
            c = a.GetProperty().GetColor()
            if c==(1,1,1): c=(0.2,0.2,0.2)
            vtklegend.SetEntry(i, getPolyData(a), "  "+ti, c)
        pos = self.legendPos
        width = self.legendSize
        vtklegend.SetWidth(width)
        vtklegend.SetHeight(width/5.*NT)
        sx, sy = 1-width, 1-width/5.*NT
        if   pos==1: vtklegend.GetPositionCoordinate().SetValue(  0, sy)
        elif pos==2: vtklegend.GetPositionCoordinate().SetValue( sx, sy) #default
        elif pos==3: vtklegend.GetPositionCoordinate().SetValue(  0,  0)
        elif pos==4: vtklegend.GetPositionCoordinate().SetValue( sx,  0)
        vtklegend.UseBackgroundOn()
        vtklegend.SetBackgroundColor(self.legendBG)
        vtklegend.SetBackgroundOpacity(0.6)
        vtklegend.LockBorderOn()
        self.renderer.AddActor(vtklegend)


    #################################################################################
    def show(self, actors=None, at=None,
             legend=None, axes=None, ruler=False,
             c='gold', alpha=0.2, wire=False, bc=None, edges=False,
             resetcam=True, interactive=None, outputimage=None, q=False):
        '''
        actors = a mixed list of vtkActors, vtkAssembly, vtkPolydata or filename strings
        at     = number of the renderer to plot to, if more than one exists
        legend = a string or list of string for each actor, if False will not show it
        axes   = show xyz axes
        ruler  = draws a simple ruler at the bottom
        c      = surface color, in rgb, hex or name formats
        bc     = background color, set a color for the back surface face
        wire   = show in wireframe representation
        edges  = show the edges on top of surface
        resetcam = if true re-adjust camera position to fit objects
        interactive = pause and interact w/ window or continue execution
        outputimage = filename to dump a screenshot without asking
        q      = force program exit after show() command
        '''

        # override what was stored internally with passed input
        if actors:
            if not isinstance(actors, list): self.actors = [actors]
            else: self.actors = actors
        if legend:
            if   isinstance(legend, list): self.legend = list(legend)
            elif isinstance(legend,  str): self.legend = [str(legend)]
            else:
                printc('Error in show(): legend must be list or string.', 1)
                exit()
        if not axes        is None: self.axes = axes
        if not interactive is None: self.interactive = interactive

        if self.verbose:
            print ('Drawing', len(self.actors),'actors ', end='')
            if len(self.renderers)>1 :
                print ('on window', at,'- Interactive mode: ', end='')
            else:
                print ('- Interactive mode: ', end='')
            if self.interactive: printc('On', 'green', bold=1)
            else: printc('Off', 'red', bold=0)

        if at is None and len(self.renderers)>1:
            #in case of multiple renderers a call to show w/o specifing
            # at which renderer will just render the whole thing and return
            if self.interactor:
                self.interactor.Render()
                if outputimage: vtkvideo.screenshot(outputimage)
                if self.interactive: self.interactor.Start()
                return
        if at is None: at=0

        if at < len(self.renderers):
            self.renderer = self.renderers[at]
        else:
            printc(("Error in show(): wrong renderer index", at), c=1)
            return

        if not self.camera:
            self.camera = self.renderer.GetActiveCamera()
            self.camera.SetParallelProjection(self.projection)
            self.camera.SetThickness(self.camThickness)
        if self.commoncam:
            for r in self.renderers: r.SetActiveCamera(self.camera)

        for i in range(len(self.actors)): # scan for filepaths
            a = self.actors[i]
            if isinstance(a, str): #assume a filepath was given
                out = self.load(a, c, alpha, wire, bc, edges)
                if isinstance(out, str):
                    printc(('File not found:', out), 1)
                    self.actors.pop() #something went wrong, purge list
                else:
                    self.actors[i] = self.actors.pop() #put it in right position

        for i in range(len(self.actors)): # scan for polydata
            a = self.actors[i]
            if isinstance(a, vtk.vtkPolyData):
                act = makeActor(a, c=c, alpha=alpha, wire=wire, bc=bc, edges=edges)
                self.actors[i] = act #put it in right position
                if a.GetNumberOfPoints()==0:
                    printc('Warning: actor #'+str(i)+' has zero points.',5)

        acts = self.getActors(at)
        for ia in self.actors: # add the actors that are not already in scene
            if not ia in acts:
                if ia: self.renderer.AddActor(ia)
                else:  printc('Warning: Invalid actor in actors list, skip.', 5)
        for ia in acts:        # remove the ones that are not in vp.actors
            if not ia in self.actors: self.renderer.RemoveActor(ia)

        if ruler: self._draw_ruler()
        if self.axes: self._draw_cubeaxes()
        self._draw_legend()

        if resetcam:
            self.renderer.ResetCamera()
            #self.camera.Zoom(1.05)

        if not self.initializedIren:
            self.initializedIren = True
            self.interactor.Initialize()
            def mouseleft(obj, e): vtkevents._mouseleft(self, obj, e)
            def keypress(obj, e):  vtkevents._keypress(self, obj, e)
            def stopren(obj, e):   vtkevents._stopren(self, obj, e)
            self.interactor.RemoveObservers('CharEvent')
            self.interactor.AddObserver("LeftButtonPressEvent", mouseleft)
            self.interactor.AddObserver("KeyPressEvent", keypress)
#            self.interactor.AddObserver('TimerEvent', stopren)
#            self.interactor.CreateRepeatingTimer(10)
#            self.interactor.SetTimerDuration(10) #millisec
            if self.verbose and self.interactive: self._tips()

        self.interactor.Render()

        if outputimage: vtkvideo.screenshot(outputimage)

        if self.interactive: self.interactor.Start()

        self.initializedPlotter = True
        if q : # gracefully exit
            if self.verbose: print ('q flag set to True. Exit.')
            exit(0)


    def render(self, addactor=None, resetcam=False, rate=10000):
        if addactor:
            self.addActor(addactor)
        if not self.initializedPlotter:
            before = bool(self.interactive)
            self.show(interactive=0)
            self.interactive = before
            return
        if resetcam: self.renderer.ResetCamera()
        self.interactor.Render()
#        self.interactor.Start()

        if self.clock is None: # set clock and limit rate
            self._clockt0 = time.time()
            self.clock = 0.
        else:
            t = time.time() - self._clockt0
            elapsed = t - self.clock
            mint = 1./rate
            if elapsed < mint:
                time.sleep(mint-elapsed)
            self.clock = time.time() - self._clockt0


    def lastActor(self): return self.actors[-1]


    def addActor(self, a):
        if not self.initializedPlotter:
            before = bool(self.interactive)
            self.show(interactive=0)
            self.interactive = before
            return
        self.actors.append(a)
        self.renderer.AddActor(a)


    def removeActor(self, a):
        try:
            if not self.initializedPlotter:
                self.show()
                return
            if self.renderer: self.renderer.RemoveActor(a)
            i = self.actors.index(a)
            del self.actors[i]
        except: pass


    def clear(self, actors=[]):
        """Delete specified actors, by default delete all."""
        if len(actors):
            for i,a in enumerate(actors): self.removeActor(a)
        else:
            for a in self.getActors(): self.renderer.RemoveActor(a)
            self.actors = []

    def openVideo(self, name='movie.avi', fps=12, duration=None, format="XVID"):
        return vtkvideo.openVideo(self, name, fps, duration, format)
    def addFrameVideo(self):     return vtkvideo.addFrameVideo(self)
    def pauseVideo(self, pause): return vtkvideo.pauseVideo(self, pause)
    def releaseVideo(self):      return vtkvideo.releaseVideo(self)
    def screenshot(self):        return vtkvideo.screenshot(self)



## experimental
## https://www.vtk.org/gitweb?p=VTK.git;a=blob;f=IO/Movie/Testing/Cxx/TestOggTheoraWriter.cxx
#    def openVideo2(self, name='movie.ogv', fps=12, quality=1):
#        w2if = vtk.vtkWindowToImageFilter()
#        w2if.SetInput(self.renderWin)
#        self.writer = vtk.vtkOggTheoraWriter()  #vtkAVIWriter
#        self.writer.SetInputConnection(w2if.GetOutputPort())
#        self.writer.SetFileName(name)
#        self.writer.SetRate(fps)
##        self.SetQuality(quality)
#        self.writer.Start()
#        return     
#    def addFrameVideo2(self):  
##        self.writer.Update()
#        self.writer.Write()
#        return         
#    def releaseVideo2(self):   
#        self.writer.End()
#        return 


###########################################################################
if __name__ == '__main__':
###########################################################################
    '''Usage:
    plotter files*.vtk
    # valid formats [vtk,vtu,vts,vtp,ply,obj,stl,xml,pcd,xyz,txt,byu,g]
    '''
    import sys
    fs = sys.argv[1:]
    alpha = 1
    if len(fs) == 1 :
        leg = False
    else:
        leg = None
        if len(fs): alpha = 1./len(fs)
        print ('Loading',len(fs),'files:', fs)
    vp = vtkPlotter(bg2=(.94,.94,1))
    for f in fs:
        vp.load(f, alpha=alpha)
    if len(fs):
        vp.show(legend=leg)
    else:
        help()
###########################################################################
