"""
Defines main class Plotter() to manage actors and 3D rendering.
"""

from __future__ import division, print_function
import time, sys, vtk, numpy, types

import vtkplotter.vtkio as vtkio
import vtkplotter.utils as utils
import vtkplotter.colors as colors
import vtkplotter.shapes as shapes
import vtkplotter.analysis as analysis 
from vtkplotter import __version__
from vtkplotter.utils import add_actor  


########################################################################
class Plotter:
#########################################################################           

    def _tips(self):       
        vvers = ' vtkplotter '+__version__+', vtk '+vtk.vtkVersion().GetVTKVersion()
        vvers+= ', python ' + str(sys.version_info[0])+'.'+str(sys.version_info[1])+' '
        n = len(vvers)
        if not colors._terminal_has_colors: n=0
        colors.printc(' '*n+'_'*(59-n), c='blue')        
        colors.printc(vvers, invert=1, dim=1, c='blue', end='')        
        msg = ' '*(59-n)+'|\n' + '|'+' '*58+'|\n|Press:'
        msg += "\ti     to print info about selected object          |\n"
        msg += "|\tm     to minimise opacity of selected mesh         |\n"
        msg += "|\t.,    to reduce/increase opacity                   |\n"
        msg += "|\t/     to maximize opacity                          |\n"
        msg += "|\tw/s   to toggle wireframe/solid style              |\n"
        msg += "|\tp/P   to change point size of vertices             |\n"
        msg += "|\tl/L   to change edge line width                    |\n"
        msg += "|\tx     to toggle mesh visibility                    |\n"
        msg += "|\tX     to pop up a cutter widget tool               |\n"
        msg += "|\t1-3   to change mesh color                         |\n"
        msg += "|\tk/K   to show point/cell scalars as color          |\n"
        msg += "|\tn     to show surface mesh normals                 |\n"
        msg += "|\tC     to print current camera info                 |\n"
        msg += "|\tS     to save a screenshot                         |\n"
        msg += "|\tq/e   to continue/close the rendering window       |\n"
        msg += "|\tEsc   to exit program                              |\n"
        msg += '|'+'_'*58+'|\n'
        colors.printc(msg, c='blue')


    def __init__(self, shape=(1,1), N=None, pos=(0,0),
                 size='auto', screensize='auto', title='', 
                 bg='w', bg2=None, axes=1, infinity=False,
                 sharecam=True, verbose=True, interactive=None):
        """Main class to manage actors.
        
           Optional args:

               shape= shape of the grid of renderers in format (rows, columns). Ignored if N is specified.
                
                N = number of desired renderers arranged in a grid automatically.
            
                pos, (x,y) position in pixels of top-left corneer of the rendering window on the screen
                
                size = size of the rendering window. If 'auto', guess it based on screensize.
        
                screensize = physical size of the monitor screen
                
                bg = background color or specify jpg image file name with path
                
                bg2 = background color of a gradient towards the top
                
                axes: 
                      0 = no axes, 
                      1 = vtkCubeAxes, 
                      2 = cartesian, 
                      3 = positive cartesian, 
                      4 = triad, 
                      5 = cube.
                
                projection,  if True fugue point is set at infinity (no perspective effects)
                
                sharecam,    if False each renderer will have an independent vtkCamera
                
                interactive, if True will stop after show() to allow interaction w/ window
        """
        
        if interactive is None:
            if N or shape != (1,1): 
                interactive=False
            else: 
                interactive=True
        if not interactive: verbose=False
        
        self.verbose    = verbose
        self.actors     = []    # list of actors to be shown
        self.clickedActor    = None  # holds the actor that has been clicked
        self.clickedRenderer = 0     # clicked renderer number
        self.renderer   = None  # current renderer
        self.renderers  = []    # list of renderers
        self.shape      = shape
        self.pos        = pos
        self.size       = [size[1],size[0]] # size of the rendering window
        self.interactive= interactive # allows to interact with renderer
        self.axes       = axes  # show or hide axes
        self.xtitle     = 'x'   # x axis label and units
        self.ytitle     = 'y'   # y axis label and units
        self.ztitle     = 'z'   # z axis label and units
        self.camera     = None  # current vtkCamera
        self.sharecam   = sharecam  # share the same camera if multiple renderers
        self.infinity   = infinity  # ParallelProjection On or Off
        self.flat       = True  # sets interpolation style to 'flat'
        self.phong      = False # sets interpolation style to 'phong'
        self.gouraud    = False # sets interpolation style to 'gouraud'
        self.bculling   = False # back face culling
        self.fculling   = False # front face culling
        self.legend     = []    # list of legend entries for actors
        self.legendSize = 0.15  # size of legend
        self.legendBG   = (.96,.96,.9) # legend background color
        self.legendPos  = 2     # 1=topright, 2=top-right, 3=bottom-left
        self.picked3d   = None  # 3d coords of a clicked point on an actor
        self.backgrcol  = bg

        # mostly internal stuff:
        self.camThickness = 2000
        self.justremoved  = None 
        self.caxes_exist  = []
        self.icol         = 0
        self.clock        = 0
        self._clockt0     = time.time()
        self.initializedPlotter= False
        self.initializedIren = False
        self.camera = vtk.vtkCamera()
        self.keyPressFunction = None
        self.sliders = []
        self.buttons = []
        self.widgets = []
        self.cutterWidget = None
        self.backgroundRenderer = None
        self.mouseLeftClickFunction = None
        self.mouseMiddleClickFunction = None
        self.mouseRightClickFunction = None

        # share some methods in utils in Plotter class for convenience
        self.closestPoint = utils.closestPoint
        self.insidePoints = utils.insidePoints
        self.write = vtkio.write
        
        # sort out screen size
        self.renderWin = vtk.vtkRenderWindow()
        self.renderWin.PointSmoothingOn()
        if screensize == 'auto':
            aus = self.renderWin.GetScreenSize()
            if aus and len(aus)==2 and aus[0]>100 and aus[1]>100: #seems ok
                if aus[0]/aus[1] > 2: # looks like there are 2 or more screens
                    screensize = (int(aus[0]/2), aus[1])
                else:
                    screensize = aus
            else: # it went wrong, use a default 1.5 ratio
                screensize = (2160, 1440)
            
        x, y = screensize
        if N:                # N = number of renderers. Find out the best
            if shape!=(1,1): # arrangement based on minimum nr. of empty renderers
                colors.printc('Warning: having set N, shape is ignored.', c=1)
            nx= int(numpy.sqrt(int(N*y/x)+1))
            ny= int(numpy.sqrt(int(N*x/y)+1))
            lm = [(nx,ny), (nx,ny+1), (nx-1,ny), (nx+1,ny), (nx,ny-1),
                  (nx-1,ny+1), (nx+1,ny-1), (nx+1,ny+1), (nx-1,ny-1)]
            ind, minl = 0, 1000
            for i,m in enumerate(lm):
                l = m[0]*m[1]
                if N <= l < minl:
                  ind = i
                  minl = l
            shape = lm[ind]
        if size=='auto':  # figure out a reasonable window size
            f=1.5
            xs = y/f*shape[1] # because y<x
            ys = y/f*shape[0]
            if xs>x/f:  # shrink
                xs = x/f
                ys = xs/shape[1]*shape[0]
            if ys>y/f:
                ys = y/f
                xs = ys/shape[0]*shape[1]
            self.size = (int(xs),int(ys))
            if shape==(1,1):
                self.size = (int(y/f), int(y/f))# because y<x
            if self.verbose and shape!=(1,1):
                print ('Window size =', self.size, 'shape =', shape)            

        ############################
        # build the renderers scene:
        self.shape = shape
        for i in reversed(range(shape[0])):
            for j in range(shape[1]):
                arenderer = vtk.vtkRenderer()
                if 'jpg' in str(bg).lower()  or 'jpeg' in str(bg).lower():
                    if i==0 :
                        jpeg_reader = vtk.vtkJPEGReader()
                        if not jpeg_reader.CanReadFile(bg):
                            colors.printc("Error reading background image file", bg, c=1)
                            sys.exit()
                        jpeg_reader.SetFileName(bg)
                        jpeg_reader.Update()
                        image_data = jpeg_reader.GetOutput()
                        image_actor = vtk.vtkImageActor()
                        image_actor.InterpolateOn()
                        utils.setInput(image_actor, image_data)
                        self.backgroundRenderer = vtk.vtkRenderer()
                        self.backgroundRenderer.SetLayer(0)
                        self.backgroundRenderer.InteractiveOff()
                        if bg2: 
                            self.backgroundRenderer.SetBackground(colors.getColor(bg2))
                        else:
                            self.backgroundRenderer.SetBackground(1,1,1)
                        arenderer.SetLayer(1)
                        self.renderWin.SetNumberOfLayers(2)
                        self.renderWin.AddRenderer(self.backgroundRenderer)
                        self.backgroundRenderer.AddActor(image_actor)
                else:
                    arenderer.SetBackground(colors.getColor(bg))
                    if bg2:
                        arenderer.GradientBackgroundOn()
                        arenderer.SetBackground2(colors.getColor(bg2))
                x0 = i/shape[0]
                y0 = j/shape[1]
                x1 = (i+1)/shape[0]
                y1 = (j+1)/shape[1]
                arenderer.SetViewport(y0,x0, y1,x1)
                self.renderers.append(arenderer)
                self.caxes_exist.append(None)
        
        if 'full' in size: # full screen
            self.renderWin.SetFullScreen(True)
            self.renderWin.BordersOn()
        else:
            self.renderWin.SetSize(int(self.size[0]), int(self.size[1]))
        
        self.renderWin.SetPosition(pos)
        if not title:
            title = ' vtkplotter '+__version__+', vtk '+vtk.vtkVersion().GetVTKVersion()
            title+= ', python ' + str(sys.version_info[0])+'.'+str(sys.version_info[1])
        self.renderWin.SetWindowName(title)
        for r in self.renderers: self.renderWin.AddRenderer(r)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renderWin)
        vsty = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(vsty)
       

    ############################################# LOADER
    def load(self, inputobj, c='gold', alpha=1,
             wire=False, bc=None, edges=False, legend=True, texture=None,
             smoothing=None, threshold=None, connectivity=False, scaling=None):
        ''' Returns a vtkActor from reading a file, directory or vtkPolyData.
           
            Optional args:
                
                c,       color in RGB format, hex, symbol or name
                
                alpha,   transparency (0=invisible)
                
                wire,    show surface as wireframe      
                
                bc,      backface color of internal surface      
                
                legend,  text to show on legend, True picks filename
                
                texture, any png/jpg file can be used as texture
           
            For volumetric data (tiff, slc files):
                
                smoothing,    gaussian filter to smooth vtkImageData
                
                threshold,    value to draw the isosurface
                
                connectivity, if True only keeps the largest portion of the polydata
                
                scaling,      scaling factors for x y an z coordinates 
        '''
        import os
        if isinstance(inputobj, vtk.vtkPolyData):
            a = utils.makeActor(inputobj, c, alpha, wire, bc, edges, legend, texture)
            self.actors.append(a)
            if inputobj and inputobj.GetNumberOfPoints()==0:
                colors.printc('Warning: actor has zero points.', c=5)
            return a

        acts = []
        if isinstance(legend, int): legend = bool(legend)
        if isinstance(inputobj, list):
            flist = inputobj
        else:
            import glob
            flist = sorted(glob.glob(inputobj))
        for fod in flist:
            if os.path.isfile(fod): 
                a = vtkio.loadFile(fod, c, alpha, wire, bc, edges, legend, texture,
                                   smoothing, threshold, connectivity, scaling)
                acts.append(a)
            elif os.path.isdir(fod):
                acts = vtkio.loadDir(fod, c, alpha, wire, bc, edges, legend, texture,
                                     smoothing, threshold, connectivity, scaling)
        if not len(acts):
            colors.printc('Error in load(): cannot find', inputobj, c=1)
            return None

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
        if len(acts) == 1: 
            return acts[0]
        else: 
            return acts


    def getActors(self, obj=None):
        '''Return an actors list.
        
           Optional args:

               If None, return actors of current renderer  
              
               If obj is a int, return actors of renderer #obj
                
               If obj is a vtkAssembly return the contained actors
                
               If obj is a string, return actors matching legend name
        '''
        if not self.renderer: return []
        
        if obj is None or isinstance(obj, int):
            if obj is None:
                acs = self.renderer.GetActors()
            elif obj>=len(self.renderers):
                colors.printc("Error in getActors: non existing renderer", obj, c=1)
                return []
            else:
                acs = self.renderers[obj].GetActors()
            actors=[]
            acs.InitTraversal()
            for i in range(acs.GetNumberOfItems()):
                a = acs.GetNextItem()
                if isinstance(a, vtk.vtkCubeAxesActor): continue
                r = self.renderers.index(self.renderer)
                if a == self.caxes_exist[r]: continue
                if isinstance(a, vtk.vtkLightActor): continue
                actors.append(a)
            return actors

        elif isinstance(obj , vtk.vtkAssembly):
            cl = vtk.vtkPropCollection()
            obj.GetActors(cl)
            actors=[]
            cl.InitTraversal()
            for i in range(obj.GetNumberOfPaths()):
                act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
                if isinstance(act, vtk.vtkCubeAxesActor): continue
                actors.append(act)
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
            colors.printc('Warning in getActors: unexpected input type',obj, c=1)
        return []


    def moveCamera(self, camstart, camstop, fraction):
        '''Takes as input two vtkCamera objects and returns a 
        new vtkCamera that is at intermediate position:
            
            fraction=0 -> camstart,  fraction=1 -> camstop.
            
            Press shift-C key in interactive mode to dump a vtkCamera
            parameter for the current camera view.
        '''
        if isinstance(fraction, int):
            colors.printc("Warning in moveCamera(): fraction should not be an integer", c=1)
        if fraction>1:
            colors.printc("Warning in moveCamera(): fraction is > 1", c=1)
        cam = vtk.vtkCamera()
        cam.DeepCopy(camstart)
        p1 = numpy.array(camstart.GetPosition())
        f1 = numpy.array(camstart.GetFocalPoint())
        v1 = numpy.array(camstart.GetViewUp())
        c1 = numpy.array(camstart.GetClippingRange())
        s1 = camstart.GetDistance()
        
        p2 = numpy.array(camstop.GetPosition())
        f2 = numpy.array(camstop.GetFocalPoint())
        v2 = numpy.array(camstop.GetViewUp())
        c2 = numpy.array(camstop.GetClippingRange())
        s2 = camstop.GetDistance()
        cam.SetPosition(     p2*fraction+p1*(1-fraction))
        cam.SetFocalPoint(   f2*fraction+f1*(1-fraction))
        cam.SetViewUp(       v2*fraction+v1*(1-fraction))
        cam.SetDistance(     s2*fraction+s1*(1-fraction))
        cam.SetClippingRange(c2*fraction+c1*(1-fraction))
        self.camera = cam
        self.show(resetcam=0)

    @add_actor 
    def makeActor(self, poly, c='gold', alpha=0.5, 
                  wire=False, bc=None, edges=False, legend=None, texture=None):
        '''
        Return a vtkActor from an input vtkPolyData.
        
        Options:
            
            c,       color in RGB format, hex, symbol or name
            
            alpha,   opacity value 
            
            wire,    show surface as wireframe
            
            bc,      backface color of internal surface
            
            edges,   show edges as line on top of surface
            
            legend   optional string
            
            texture  jpg file name of surface texture
        '''
        return utils.makeActor(poly, c, alpha, wire, bc, edges, legend, texture)

    @add_actor 
    def makeAssembly(self, actorlist):
        '''Group many actors as a single new actor.
        
        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/icon.py)    
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope1.py)    
        '''
        for a in actorlist:
            while a in self.actors: #update internal list
                self.actors.remove(a)
        return utils.makeAssembly(actorlist)

        
    def light(self, pos=[1,1,1], fp=[0,0,0], deg=25,
              diffuse='y', ambient='r', specular='b', showsource=False):
        """
        Generate a source of light placed at pos, directed to focal point fp.
        If fp is a vtkActor use its position.
        
        Options:
        
            deg = aperture angle of the light source
        
            showsource, if True, will show a vtk representation 
            of the source of light as an extra actor

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/lights.py)    
        """
        if isinstance(fp, vtk.vtkActor): fp = fp.GetPosition()
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(pos)
        light.SetPositional(1)
        light.SetConeAngle(deg)
        light.SetFocalPoint(fp)
        light.SetDiffuseColor(colors.getColor(diffuse))
        light.SetAmbientColor(colors.getColor(ambient))
        light.SetSpecularColor(colors.getColor(specular))
        self.render()
        if showsource:
            lightActor = vtk.vtkLightActor()
            lightActor.SetLight(light)
            self.renderer.AddViewProp(lightActor)
            self.renderer.AddLight(light)
        return light


    ####################################################### manage basic shapes
    @add_actor 
    def point(self, pos=[0,0,0], c='b', r=10, alpha=1, legend=None):
        '''Create a simple point actor.'''
        return shapes.points([pos], c, [], r, alpha, legend)
    
    @add_actor
    def points(self, plist=[[1,0,0],[0,1,0],[0,0,1]],
                c='b', tags=[], r=5, alpha=1, legend=None):
        '''
        Build a vtkActor for a list of points.
    
        c can be a list of [R,G,B] colors of same length as plist
        
        If tags (a list of strings) is specified, they will be displayed  
        along with the points.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/lorenz.py)    

        ![lorenz](https://user-images.githubusercontent.com/32848391/46818115-be7a6380-cd80-11e8-8ffb-60af2631bf71.png)
        '''
        return shapes.points(plist, c, tags, r, alpha, legend)

    @add_actor 
    def sphere(self, pos=[0,0,0], r=1,
               c='r', alpha=1, wire=False, legend=None, texture=None, res=24):
        '''Build a sphere at position pos of radius r.'''
        return shapes.sphere(pos, r, c, alpha, wire, legend, texture, res)

    @add_actor 
    def spheres(self, centers, r=1,
               c='r', alpha=1, wire=False, legend=None, texture=None, res=8):
        '''
        Build a (possibly large) set of spheres at centers of radius r.
        
        Either c or r can be a list of RGB colors or radii.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/manyspheres.py)    

        ![manysph](https://user-images.githubusercontent.com/32848391/46818673-1f566b80-cd82-11e8-9a61-be6a56160f1c.png)
        '''
        return shapes.spheres(centers, r, c, alpha, wire, legend, texture, res)
    
    @add_actor 
    def earth(self, pos=[0,0,0], r=1, lw=1):
        '''Build a textured actor representing the Earth.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/earth.py)    
        '''
        return shapes.earth(pos, r, lw)

    @add_actor 
    def line(self, p0, p1=None, lw=1, tube=False, dotted=False,
             c='r', alpha=1., legend=None):
        '''Build the line segment between points p0 and p1.
            
           If p0 is a list of points returns the line connecting them.
            
           If tube=True, lines are rendered as tubes of radius lw
        '''
        return shapes.line(p0, p1, lw, tube, dotted, c, alpha, legend)
    
    @add_actor
    def lines(self, plist0, plist1=None, lw=1, dotted=False,
              c='r', alpha=1, legend=None):
        '''Build the line segments between two lists of points plist0 and plist1.
           plist0 can be also passed in the format [[point1, point2], ...]

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitspheres2.py)    
        '''      
        return shapes.lines(plist0, plist1, lw, dotted, c, alpha, legend)
    
    @add_actor
    def arrow(self, startPoint, endPoint,
              c='r', s=None, alpha=1, legend=None, texture=None, res=12):
        '''Build a 3D arrow from startPoint to endPoint of section size s,
        expressed as the fraction of the window size.
        If s=None the arrow is scaled proportionally to its length.'''
        return shapes.arrow(startPoint, endPoint, c, s, alpha, 
                            legend, texture, res, self.renderWin.GetSize())
    @add_actor
    def arrows(self, startPoints, endPoints=None,
            c='r', s=None, alpha=1, legend=None, res=8):
        '''Build arrows between two lists of points startPoints and endPoints.
           startPoints can be also passed in the form [[point1, point2], ...]
        '''        
        rwSize = self.renderWin.GetSize()
        return shapes.arrows(startPoints, endPoints, c, s, alpha, legend, res, rwSize)

    @add_actor
    def grid(self, pos=[0,0,0], normal=[0,0,1], sx=1, sy=1, c='g', bc='darkgreen',
             lw=1, alpha=1, legend=None, resx=10, resy=10):
        '''Draw a grid of size sx and sy oriented perpendicular to vector normal
        so that it passes through point pos.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/brownian2D.py)    
        '''
        return shapes.grid(pos, normal, sx, sy, c, bc, lw, alpha, legend, resx, resy)

    @add_actor
    def plane(self, pos=[0,0,0], normal=[0,0,1], sx=1, sy=None, c='g', bc='darkgreen',
              alpha=1, legend=None, texture=None):
        '''
        Draw a plane of size sx and sy oriented perpendicular to vector normal  
        and so that it passes through point pos.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/carcrash.py)    
        '''
        return shapes.plane(pos, normal, sx, sy, c, bc, alpha, legend, texture)
    
    @add_actor
    def polygon(self, pos=[0,0,0], normal=[0,0,1], nsides=6, r=1,
                c='coral', bc='darkgreen', lw=1, alpha=1,
                legend=None, texture=None, followcam=False):
        '''Build a 2D polygon of nsides of radius r oriented as normal
        
        If followcam=True the polygon will always reorient itself to current camera.
        '''
        return shapes.polygon(pos, normal, nsides, r, c, bc, lw, alpha, legend,
                              texture, followcam, camera=self.camera)

    @add_actor
    def disc(self, pos=[0,0,0], normal=[0,0,1], r1=0.5, r2=1, c='coral', bc='darkgreen',
             lw=1, alpha=1, legend=None, texture=None, res=12):
        '''Build a 2D disc of internal radius r1 and outer radius r2,
        oriented perpendicular to normal.'''
        return shapes.disc(pos, normal, r1, r2, c, bc, lw, alpha, legend, texture, res)

    @add_actor
    def box(self, pos=[0,0,0], length=1, width=2, height=3, normal=(0,0,1),
            c='g', alpha=1, wire=False, legend=None, texture=None):
        '''Build a box of dimensions x=length, y=width and z=height oriented along vector normal.
        
        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/spring.py)    
        '''
        return shapes.box(pos, length, width, height, normal, c, alpha, wire, legend, texture)

    def cube(self, pos=[0,0,0], length=1, normal=(0,0,1),
             c='g', alpha=1., wire=False, legend=None, texture=None):
        '''Build a cube of dimensions length oriented along vector normal.
        
        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/texturecubes.py)    
        '''
        return self.box(pos, length, length, length, normal, c, alpha, wire, legend, texture)
        
    @add_actor
    def helix(self, startPoint=[0,0,0], endPoint=[1,1,1], coils=20, r=None,
              thickness=None, c='grey', alpha=1, legend=None, texture=None):
        '''
        Build a spring actor of specified nr of coils between startPoint and endPoint.

        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/spring.py)    
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope1.py)    
        [**Example3**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/multiple_pendulum.py)    
        '''
        return shapes.helix(startPoint, endPoint, coils, r, thickness, c, alpha, legend, texture)        

    @add_actor
    def cylinder(self, pos=[0,0,0], r=1, height=1, axis=[0,0,1],
                 c='teal', wire=0, alpha=1, edges=False, 
                 legend=None, texture=None, res=24):
        '''
        Build a cylinder of specified height and radius r, centered at pos.
        
        If pos is a list of 2 points, e.g. pos=[v1,v2], build a cylinder with base
        centered at v1 and top at v2.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/turing.py)    
        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope1.py)    
        '''
        return shapes.cylinder(pos, r, height, axis, c, wire, alpha, edges, legend, texture, res)

    @add_actor
    def paraboloid(self, pos=[0,0,0], r=1, height=1, axis=[0,0,1],
                   c='cyan', alpha=1, legend=None, texture=None, res=50):
        '''
        Build a paraboloid of specified height and radius r, centered at pos.
        '''
        return shapes.paraboloid(pos, r, height, axis, c, alpha, legend, texture, res)

    @add_actor
    def hyperboloid(self, pos=[0,0,0], a2=1, value=0.5, height=1, axis=[0,0,1],
                    c='magenta', alpha=1, legend=None, texture=None, res=50):
        '''
        Build a hyperboloid of specified aperture a2 and height, centered at pos.
        '''
        return shapes.hyperboloid(pos, a2, value, height, axis, c, alpha, legend, texture, res)

    @add_actor
    def cone(self, pos=[0,0,0], r=1, height=1, axis=[0,0,1],
             c='dg', alpha=1, legend=None, texture=None, res=48):
        '''
        Build a cone of specified radius r and height, centered at pos.
        '''
        return shapes.cone(pos, r, height, axis, c, alpha, legend, texture, res)

    def pyramid(self, pos=[0,0,0], s=1, height=1, axis=[0,0,1],
                c='dg', alpha=1, legend=None, texture=None):
        '''
        Build a pyramid of specified base size s and height, centered at pos.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope2.py)    
        '''
        return self.cone(pos, s, height, axis, c, alpha, legend, texture, 4)

    @add_actor
    def ring(self, pos=[0,0,0], r=1, thickness=0.1, axis=[0,0,1],
             c='khaki', alpha=1, wire=False, legend=None, texture=None, res=30):
        '''
        Build a torus of specified outer radius r internal radius thickness, centered at pos.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gas.py)    
        
        ![gas](https://user-images.githubusercontent.com/32848391/39139206-90d644ca-4721-11e8-95b9-8aceeb3ac742.gif)
        '''
        return shapes.ring(pos, r, thickness, axis, c, alpha, wire, legend, texture, res)

    @add_actor
    def ellipsoid(self, pos=[0,0,0], axis1=[1,0,0], axis2=[0,2,0], axis3=[0,0,3],
                  c='c', alpha=1, legend=None, texture=None, res=24):
        """Build a 3D ellipsoid centered at position pos.
        
        Axis1 and axis2 are only used to define sizes and one azimuth angle
        """
        return shapes.ellipsoid(pos, axis1, axis2, axis3, c, alpha, legend, texture, res)
        
    @add_actor
    def spline(self, points, smooth=0.5, degree=2, 
               s=2, c='b', alpha=1, nodes=False, legend=None, res=20):
        '''
        Return a vtkActor for a spline that doesnt necessarly pass exactly throught all points.

        Options:
            
            smooth, smoothing factor:
                    0 = interpolate points exactly, 
                    1 = average point positions

            degree = degree of the spline (1<degree<5)
            
            nodes = True, show also the input points 

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py)

        ![rspline](https://user-images.githubusercontent.com/32848391/35976041-15781de8-0cdf-11e8-997f-aeb725bc33cc.png)
        '''
        return analysis.spline(points, smooth, degree, s, c, alpha, nodes, legend, res)

    @add_actor
    def text(self, txt='Hello', pos=(0,0,0), normal=(0,0,1), s=1, depth=0.1,
             c='k', alpha=1, bc=None, texture=None, followcam=False):
        '''
        Returns a vtkActor that shows a text in 3D.
        
        Options:
            
            pos = position in 3D space
            if an integer is passed [1 -> 8], places text in one of the corners
            
            s = size of text 
            
            depth = text thickness
            
            followcam = False, if True the text will auto-orient itself to it.

        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/colorcubes.py)    
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_coloring.py)    
        '''
        return shapes.text(txt, pos, normal, s, depth, c, alpha, bc,
                           texture, followcam, cam=self.camera)


    ################# 
    @add_actor
    def xyplot(self, points=[[0,0],[1,0],[2,1],[3,2],[4,1]],
               title='', c='b', corner=1, lines=False):
        """
        Return a vtkActor that is a plot of 2D points in x and y.

        Use corner to assign its position:
            1 -> topleft, 
            2 -> topright, 
            3 -> bottomleft, 
            4 -> bottomright.         

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py)
        """
        return analysis.xyplot(points, title, c, corner, lines)
    
    def histogram(self, values, bins=10, vrange=None, 
                  title='', c='g', corner=1, lines=True):
        '''
        Build a 2D histogram from a list of values in n bins.

        Use *vrange* to restrict the range of the histogram.

        Use *corner* to assign its position:
            1 -> topleft, 
            2 -> topright, 
            3 -> bottomleft, 
            4 -> bottomright.         

        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitplanes.py)
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitspheres1.py)
        '''
        fs, edges = numpy.histogram(values, bins=bins, range=vrange)
        pts=[]
        for i in range(len(fs)): 
            pts.append( [ (edges[i]+edges[i+1])/2, fs[i] ])
        return self.xyplot(pts, title, c, corner, lines)

    @add_actor
    def fxy(self, z='sin(3*x)*log(x-y)/3', x=[0,3], y=[0,3],
            zlimits=[None,None], showNan=True, zlevels=10, wire=False,
            c='aqua', bc='aqua', alpha=1, legend=True, texture='paper', res=100):
        '''
        Build a surface representing the 3D function specified as a string
        or as a reference to an external function.
        Red points indicate where the function does not exist (showNan).
    
        zlevels will draw the specified number of z-levels contour lines.
        
        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/fxy.py)    

        ![fxy](https://user-images.githubusercontent.com/32848391/36611824-fd524fac-18d4-11e8-8c76-d3d1b1bb3954.png)
        '''
        return analysis.fxy(z, x, y, zlimits, showNan, zlevels, 
                            wire, c, bc, alpha, legend, texture, res)
    
    @add_actor
    def fitLine(self, points, c='orange', lw=1, alpha=0.6, legend=None):
        '''
        Fits a line through points.

        Extra info is stored in actor.slope, actor.center, actor.variances

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/fitline.py)
        '''
        if self.verbose:
            colors.printc("fitLine info saved in actor.slope, actor.center, actor.variances",c=5)
        return analysis.fitLine(points, c, lw, alpha, legend)

    @add_actor
    def fitPlane(self, points, c='g', bc='darkgreen', alpha=0.8, legend=None):
        '''
        Fits a plane to a set of points.

        Extra info is stored in actor.normal, actor.center, actor.variance

        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/fitline.py)    
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitplanes.py)
        '''
        if self.verbose:
            colors.printc("fitPlane info saved in actor.normal, actor.center, actor.variance",c=5)
        return analysis.fitPlane(points, c, bc, alpha, legend)

    @add_actor
    def fitSphere(self, coords, c='r', alpha=1, wire=1, legend=None):
        '''
        Fits a sphere to a set of points.
        
        Extra info is stored in actor.radius, actor.center, actor.residue

        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitspheres1.py)
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitspheres2.py)
        '''
        if self.verbose:
            colors.printc("fitSphere info saved in actor.radius, actor.center, actor.residue",c=5)
        return analysis.fitSphere(coords, c, alpha, wire, legend)

    @add_actor
    def pca(self, points=[[1,0,0],[0,1,0],[0,0,1],[.5,0,1],[0,.2,.3]],
            pvalue=.95, c='c', alpha=0.5, pcaAxes=False, legend=None):
        '''
        Show the oriented PCA ellipsoid that contains fraction pvalue of points.
        
        axes = True, show the 3 PCA semi axes
        
        Extra info is stored in actor.sphericity, actor.va, actor.vb, actor.vc
        (sphericity = 1 for a perfect sphere)
        
        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py)
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/cell_main.py)
        '''
        if self.verbose:
            colors.printc("PCA info saved in actor.sphericity, actor.va, actor.vb, actor.vc",c=5)
        return analysis.pca(points, pvalue, c, alpha, pcaAxes, legend)    
    
    @add_actor
    def align(self, source, target, iters=100, rigid=False, legend=None):
        '''
        Return a copy of source actor which is aligned to
        target actor through vtkIterativeClosestPointTransform() method.
    
        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/align1.py)
        
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/align2.py)
        '''
        return analysis.align(source, target, iters, rigid, legend)
        
    @add_actor
    def delaunay2D(self, plist, tol=None, c='gold', alpha=0.5, wire=False, bc=None, 
                   edges=False, legend=None, texture=None):
        '''Create a mesh from points in the XY plane.
        
        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/delaunay2d.py)
        '''
        return analysis.delaunay2D(plist, tol, c, alpha, wire, bc, edges, legend, texture)
        
    @add_actor
    def recoSurface(self, points, bins=256,
                    c='gold', alpha=1, wire=False, bc='t', edges=False, legend=None):
        '''
        Surface reconstruction from sparse points.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/recosurface.py)    
        
        ![reco](https://user-images.githubusercontent.com/32848391/46817107-b3263880-cd7e-11e8-985d-f5d158992f0c.png)
        '''
        return analysis.recoSurface(points, bins, c, alpha, wire, bc, edges, legend)
    
    @add_actor
    def cluster(self, points, radius, legend=None):
        '''
        Clustering of points in space.
        
        radius, is the radius of local search.
        Individual subsets can be accessed through actor.clusters

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/clustering.py)    

        ![cluster](https://user-images.githubusercontent.com/32848391/46817286-2039ce00-cd7f-11e8-8b29-42925e03c974.png)
        '''
        return analysis.cluster(points, radius, legend)
        
    def removeOutliers(self, points, radius, c='k', alpha=1, legend=None):
        '''
        Remove outliers from a cloud of points within radius search.
        If points is a list of [x,y,z] return a reduced list of points
        If input is a vtkActor return a vtkActor.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/clustering.py)    
        '''
        a = analysis.removeOutliers(points, radius, c, alpha, legend)
        if not utils.isSequence(a): self.actors.append(a)
        return a   

    @add_actor
    def normals(self, actor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8, legend=None):
        '''
        Build a vtkActor made of the normals at vertices shown as arrows.
    
        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py)    
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fatlimb.py)    
        '''
        return analysis.normals(actor, ratio, c, alpha, legend)

    @add_actor
    def curvature(self, actor, method=1, r=1, alpha=1, lut=None, legend=None):
        '''
        Build a copy of vtkActor that contains the color coded surface
        curvature following four different ways to calculate it:
            method =  0-gaussian, 1-mean, 2-max, 3-min

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py)
        '''
        return analysis.curvature(actor, method, r, alpha, lut, legend)

    @add_actor
    def boundaries(self, actor, c='p', lw=5, legend=None):
        '''Build a copy of actor that shows the boundary lines of its surface.
        
        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py)    
        '''
        return analysis.boundaries(actor, c, lw, legend)


    def cutPlane(self, actor, origin=(0,0,0), normal=(1,0,0), showcut=False):
        '''
        Takes actor and cuts it with the plane defined by a point and a normal. 
            showcut  = shows the cut away part as thin wireframe

        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py)    
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/trail.py)            
        '''
        cactor = utils.cutPlane(actor, origin, normal, showcut)
        try:
            i = self.actors.index(actor)
            self.actors[i] = cactor # substitute original actor with cut one
        except ValueError: 
            self.actors.append(cactor)
        return cactor #NB: original actor is modified


    def smoothMLS1D(self, actor, f=0.2, showNLines=0):
        '''
        Smooth actor or points with a Moving Least Squares variant.
        The list actor.variances contain the residue calculated for each point.
        Input actor's polydata is modified.
        
        Options:

            f, smoothing factor - typical range s [0,2] 
            
            showNLines, build an actor showing the fitting line for N random points            
    
        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares1D.py)    
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/skeletonize.py)    

        ![skel](https://user-images.githubusercontent.com/32848391/46820954-c5f13b00-cd87-11e8-87aa-286528a09de8.png)
        '''        
        actor = analysis.smoothMLS1D(actor, f, showNLines)
        return actor #NB: original actor is modified

    def smoothMLS2D(self, actor, f=0.2, decimate=1, recursive=0, showNPlanes=0):
        '''
        Smooth actor or points with a Moving Least Squares variant.
        The list actor.variances contain the residue calculated for each point.
        Input actor's polydata is modified.
        
        Options:
        
            f, smoothing factor - typical range s [0,2]
            
            decimate, decimation factor (an integer number) 
            
            recursive, move points while algorithm proceedes
            
            showNPlanes, build an actor showing the fitting plane for N random points            
        
        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/mesh_smoothers.py)    
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares2D.py)    
        [**Example3**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/recosurface.py.py)    
        '''        
        actor = analysis.smoothMLS2D(actor, f, decimate, recursive, showNPlanes)
        return actor #NB: original actor is modified

    
    def addScalarBar(self, actor=None, c='k', horizontal=False):
        """
        Add a 2D scalar bar for the specified actor.

        If actor is None will add it to the last actor in self.actors

        [**Example1**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/boolean.py)    
        [**Example2**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_coloring.py)    
        [**Example3**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitspheres2.py)    
        """
        
        if actor is None: actor=self.lastActor()
        if not isinstance(actor, vtk.vtkActor) or not hasattr(actor, 'GetMapper'): 
            colors.printc('Error in addScalarBar: input is not a vtkActor.',c=1)
            return None
        lut = actor.GetMapper().GetLookupTable()
        if not lut: return None
      
        c = colors.getColor(c)
        sb = vtk.vtkScalarBarActor()
        sb.SetLookupTable(lut)
        
        if vtk.vtkVersion().GetVTKMajorVersion() > 7: 
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
            sb.SetMaximumHeightInPixels(50)
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


    def addScalarBar3D(self, obj=None, pos=[0,0,0], normal=[0,0,1], sx=.1, sy=2, 
                       nlabels=9, ncols=256, cmap='jet', c='k', alpha=1):
        '''
        Draw a 3D scalar bar.
        
        obj input can be:
            
            a list of numbers,
            
            a list of two numbers in the form (min, max)
            
            a vtkActor containing a set of scalars associated to vertices or cells,
            if None the last actor in the list of actors will be used.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_coloring.py)    
        '''
        gap = 0.4
        if obj is None: obj = self.lastActor()
        if isinstance(obj, vtk.vtkActor):
            poly = utils.polydata(obj)
            vtkscalars = poly.GetPointData().GetScalars()
            if vtkscalars is None:
                vtkscalars = poly.GetCellData().GetScalars()
            if vtkscalars is None:
                print('Error in scalarBar3D: actor has no scalar array.')
                sys.exit()
            npscalars = utils.vtk_to_numpy(vtkscalars)
            vmin, vmax = numpy.min(npscalars), numpy.max(npscalars)
        elif utils.isSequence(obj):
            vmin, vmax = numpy.min(obj), numpy.max(obj)
        else:
            print('Error in scalarBar3D: input must be vtkActor or list.', type(obj))
            sys.exit()
        # build the color scale part
        scale = shapes.grid([-sx*gap,0,0], c='w', alpha=alpha, sx=sx, sy=sy, resx=1, resy=ncols)
        scale.GetProperty().SetRepresentationToSurface()
        cscals = utils.cellCenters(scale)[:,1] 
        utils.cellColors(scale, cscals, cmap)
        # build text
        nlabels = numpy.min([nlabels, ncols])
        tlabs = numpy.linspace(vmin, vmax, num=nlabels, endpoint=True)
        tacts = []
        prec = (vmax-vmin)/abs(vmax+vmin)*2
        prec = int(abs(numpy.log10(prec))+2.5)
        for i,t in enumerate(tlabs):
            tx = str(utils.to_precision(t, prec))
            y = -sy/1.98+sy*i/(nlabels-1)
            a = shapes.text(tx, pos=[sx*gap,y,0], s=sy/50, c=c, alpha=alpha, depth=0)
            tacts.append( a )
        sact = utils.makeAssembly([scale]+tacts)
        nax = numpy.linalg.norm(normal)
        if nax: normal = numpy.array(normal)/nax
        theta = numpy.arccos(normal[2])
        phi   = numpy.arctan2(normal[1], normal[0])
        sact.RotateZ(phi*57.3)
        sact.RotateY(theta*57.3)
        sact.SetPosition(pos)
        utils.assignConvenienceMethods(sact, None)
        utils.assignPhysicsMethods(sact)
        if not self.renderer: self.render()
        self.renderer.AddActor(sact)
        self.render()
        return sact


    def addSlider(self, sliderfunc, xmin=0, xmax=1, value=None, pos=4, s=.04,
                  title='', c='k', showValue=True):
        '''
        Add a slider widget with external custom function.
        
        Options:
        
            sliderfunc, external function to be called by the widget 
                    
            xmin, lower value
            
            xmax, upper value
            
            value, current value
            
            pos, position corner number: horizontal [1-4] or vertical [11-14]
                 it can also be specified by corners coordinates [(x1,y1), (x2,y2)]

            title, title label

            c, color string or number
            
            showValue, if true current value is shown                    

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/sliders.py)    
        '''        
        if numpy.sum(colors.getColor(self.backgrcol)) < 3 and c=='k': c='w'
        c = colors.getColor(c)
            
        sliderRep = vtk.vtkSliderRepresentation2D()
        sliderRep.SetMinimumValue(xmin)
        sliderRep.SetMaximumValue(xmax)
        if value is None: value=xmin
        sliderRep.SetValue(value)
        sliderRep.SetSliderLength(0.015)
        sliderRep.SetSliderWidth(0.025)
        sliderRep.SetEndCapLength(0.0015)
        sliderRep.SetEndCapWidth(0.0125)
        sliderRep.SetTubeWidth(.0075)
        sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        if utils.isSequence(pos):
            sliderRep.GetPoint1Coordinate().SetValue(pos[0][0],pos[0][1])
            sliderRep.GetPoint2Coordinate().SetValue(pos[1][0],pos[1][1])
        elif pos==1: #top-left horizontal
            sliderRep.GetPoint1Coordinate().SetValue(.04 ,.96)
            sliderRep.GetPoint2Coordinate().SetValue(.45, .96)
        elif pos==2:
            sliderRep.GetPoint1Coordinate().SetValue(.55 ,.96)
            sliderRep.GetPoint2Coordinate().SetValue(.96, .96)
        elif pos==3:
            sliderRep.GetPoint1Coordinate().SetValue(.04 ,.04)
            sliderRep.GetPoint2Coordinate().SetValue(.45, .04)
        elif pos==4: #bottom-right
            sliderRep.GetPoint1Coordinate().SetValue(.55 ,.04)
            sliderRep.GetPoint2Coordinate().SetValue(.96, .04)
        elif pos==5: #bottom margin horizontal
            sliderRep.GetPoint1Coordinate().SetValue(.04 ,.04)
            sliderRep.GetPoint2Coordinate().SetValue(.96, .04)
        elif pos==11:#top-left vertical
            sliderRep.GetPoint1Coordinate().SetValue(.04, .54)
            sliderRep.GetPoint2Coordinate().SetValue(.04 ,.9)
        elif pos==12:
            sliderRep.GetPoint1Coordinate().SetValue(.96,.54)
            sliderRep.GetPoint2Coordinate().SetValue(.96,.9)
        elif pos==13:
            sliderRep.GetPoint1Coordinate().SetValue(.04 ,.1)
            sliderRep.GetPoint2Coordinate().SetValue(.04, .54)
        elif pos==14:#bottom-right vertical
            sliderRep.GetPoint1Coordinate().SetValue(.96 ,.1)
            sliderRep.GetPoint2Coordinate().SetValue(.96, .54)
        elif pos==15:#right margin vertical
            sliderRep.GetPoint1Coordinate().SetValue(.96 ,.1)
            sliderRep.GetPoint2Coordinate().SetValue(.96, .9)

        if showValue:
            if isinstance(xmin, int) and isinstance(xmax, int):
                frm = '%0.0f'
            else:
                frm = '%0.1f'
            sliderRep.SetLabelFormat(frm) # default is '%0.3g'    
            sliderRep.GetLabelProperty().SetShadow(0)
            sliderRep.GetLabelProperty().SetBold(0)
            sliderRep.GetLabelProperty().SetOpacity(0.6)
            sliderRep.GetLabelProperty().SetColor(c)
            if pos>10:
                sliderRep.GetLabelProperty().SetOrientation(90)
        else:
            sliderRep.ShowSliderLabelOff()
        sliderRep.GetTubeProperty().SetColor(c)
        sliderRep.GetTubeProperty().SetOpacity(0.6)
        sliderRep.GetSliderProperty().SetColor(c)
        sliderRep.GetSelectedProperty().SetColor(.8, 0, 0)
        sliderRep.GetCapProperty().SetColor(c)
        
        if title:
            sliderRep.SetTitleText(title)
            sliderRep.SetTitleHeight(.015)
            sliderRep.GetTitleProperty().SetShadow(0)
            sliderRep.GetTitleProperty().SetColor(c)
            sliderRep.GetTitleProperty().SetOpacity(.6)
            sliderRep.GetTitleProperty().SetBold(0)
            if not utils.isSequence(pos) :
                if pos>10:
                    sliderRep.GetTitleProperty().SetOrientation(90)
            else:
                if abs(pos[0][0]-pos[1][0])<.1:
                    sliderRep.GetTitleProperty().SetOrientation(90)
       
        sliderWidget = vtk.vtkSliderWidget()
        sliderWidget.SetInteractor(self.interactor)
        sliderWidget.SetRepresentation(sliderRep)
        sliderWidget.AddObserver("InteractionEvent", sliderfunc)
        sliderWidget.EnabledOn()      
        self.sliders.append([sliderWidget, sliderfunc])
        return sliderWidget
        
    
    def addButton(self, fnc, states=['On', 'Off'], c=['w','w'], bc=['dg','dr'],
                  pos=[20,40], size=24, font='arial', bold=False, italic=False, 
                  alpha=1, angle=0):
        '''Add a button to the renderer window.

        Options:
            
            states, a list of possible states ['On', 'Off']
            
            c,      color for each state
            
            bc,     background color for each state 
            
            pos,    2d position in pixels from left-bottom corner
            
            size,   size of button font
            
            font,   font type (arial, courier, times)
            
            bold,   bold face
            
            italic, italic face
            
            alpha,  opacity level
            
            angle,  anticlockwise rotation in degrees           


        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/buttons.py)    
        '''
        if not self.renderer:
            colors.printc('Error: Use addButton() after rendering the scene.', c=1)
            return
        bu = vtkio.Button(fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle)
        self.renderer.AddActor2D(bu.actor)
        self.renderWin.Render()
        self.buttons.append(bu)
        return bu
    
    
    def addCutterTool(self, actor):  
        '''Create handles to cut away parts of a mesh.
        
        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/cutter.py)    
        '''
        if not isinstance(actor, vtk.vtkActor):
            return None
        
        if not self.renderer:
            self.render()
        
        self.clickedActor = actor
        apd = utils.polydata(actor)
        
        planes = vtk.vtkPlanes()
        planes.SetBounds(apd.GetBounds())
    
        clipper = vtk.vtkClipPolyData()
        utils.setInput(clipper, apd)
        clipper.SetClipFunction(planes)
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOn()
    
        act0Mapper = vtk.vtkPolyDataMapper() # the part which stays
        act0Mapper.SetInputConnection(clipper.GetOutputPort())
        act0 = vtk.vtkActor()
        act0.SetMapper(act0Mapper)
        act0.GetProperty().SetColor(actor.GetProperty().GetColor())
        act0.GetProperty().SetOpacity(1)
        
        act1Mapper = vtk.vtkPolyDataMapper() # the part which is cut away
        act1Mapper.SetInputConnection(clipper.GetClippedOutputPort())
        act1 = vtk.vtkActor()
        act1.SetMapper(act1Mapper)
        act1.GetProperty().SetOpacity(.05)
        act1.GetProperty().SetRepresentationToWireframe()
        act1.VisibilityOn()

        self.renderer.AddActor(act0)
        self.renderer.AddActor(act1)
        self.renderer.RemoveActor(actor)
        
        def SelectPolygons(vobj, event): 
            vobj.GetPlanes(planes)
        
        boxWidget = vtk.vtkBoxWidget()
        boxWidget.OutlineCursorWiresOn()
        boxWidget.GetSelectedOutlineProperty().SetColor(1,0,1)
        boxWidget.GetOutlineProperty().SetColor(0.1,0.1,0.1)
        boxWidget.GetOutlineProperty().SetOpacity(0.8)
        boxWidget.SetPlaceFactor(1.05)
        boxWidget.SetInteractor(self.interactor)
        utils.setInput(boxWidget, apd)
        boxWidget.PlaceWidget()    
        boxWidget.AddObserver("InteractionEvent", SelectPolygons)
        boxWidget.On()
        
        self.cutterWidget = boxWidget
        self.clickedActor = act0
        ia = self.actors.index(actor)
        self.actors[ia] = act0
        
        colors.printc('Mesh Cutter Tool:', c='m', invert=1)
        colors.printc('  Move gray handles to cut away parts of the mesh', c='m')
        colors.printc("  Press X to save file to: clipped.vtk", c='m')
        
        self.interactor.Start()
        boxWidget.Off()

        self.widgets.append(boxWidget)
        return act0
    
    
    def addIcon(self, iconActor, pos=3, size=0.08):
        '''
        Add an inset icon mesh into the same renderer.
        
        Options:
            
            pos, can be in the range [1-4] indicating one of the 4 corners,
                 or can be a tuple (x,y) in fraction of the renderer size.
                
            size, size of the square inset.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/icon.py)    
        '''              
        if not self.renderer:
            colors.printc('Warning: Use addIcon() after rendering the scene.', c=3)
            self.render()
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker( iconActor )
        widget.SetInteractor( self.interactor )
        if utils.isSequence(pos):
            widget.SetViewport(pos[0]-size, pos[1]-size, pos[0]+size, pos[1]+size)
        else:
            if   pos <2: widget.SetViewport(        0, 1-2*size, size*2,      1 )
            elif pos==2: widget.SetViewport( 1-2*size, 1-2*size,      1,      1 )
            elif pos==3: widget.SetViewport(        0,        0, size*2, size*2 )
            elif pos==4: widget.SetViewport( 1-2*size,        0,      1, size*2 )
        widget.EnabledOn()
        widget.InteractiveOff()
        self.widgets.append(widget)
        if iconActor in self.actors:
            self.actors.remove(iconActor)
        return widget


    def addTrail(self, actor=None, maxlength=None, n=25, c=None, alpha=None, lw=1):
        '''
        Add a trailing line to an existing actor.
        
        Options:

            maxlength, length of trailing line in absolute units
            
            n, number of segments, controls precision

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/trail.py)    
        '''
        if not actor:
            actor = self.actors[-1]
        if maxlength is None:
            maxlength = utils.diagonalSize(actor)*20
        if not hasattr(actor, 'trailPoints'):
            pos = actor.GetPosition()
            setattr(actor,'trailPoints', [None]*n)
            setattr(actor,'trailSegmentSize', maxlength/n)

            ppoints = vtk.vtkPoints() # Generate the polyline
            poly = vtk.vtkPolyData()
            for i in range(n): 
                x,y,z = pos
                ppoints.InsertPoint(i, x,y,z)
            lines = vtk.vtkCellArray() 
            lines.InsertNextCell(n)
            for i in range(n): lines.InsertCellPoint(i)
            poly.SetPoints(ppoints)
            poly.SetLines(lines)
            mapper = vtk.vtkPolyDataMapper()
            if c is None:
                col = actor.GetProperty().GetColor()
            else:
                col = colors.getColor(c)
            al = colors.getAlpha(c)
            if al: alpha = al
            if alpha is None:
                alpha = actor.GetProperty().GetOpacity()
            utils.setInput(mapper, poly)
            tline = vtk.vtkActor()
            tline.SetMapper(mapper)
            tline.GetProperty().SetColor(col)
            tline.GetProperty().SetOpacity(alpha)
            tline.GetProperty().SetLineWidth(lw)
            setattr(actor,'trail', tline) # holds the vtkActor
            self.actors.append(tline)

            def _fupdateTrail(self):
                currentpos = numpy.array(self.GetPosition())
                lastpos = self.trailPoints[-1]
                if lastpos is None: # reset list
                    self.trailPoints = [currentpos]*len(self.trailPoints)
                    return
                if numpy.linalg.norm(currentpos-lastpos) < self.trailSegmentSize: 
                    return 
                
                self.trailPoints.append(currentpos) #cycle
                self.trailPoints.pop(0)    

                poly = utils.polydata(self.trail)
                vtkpts = poly.GetPoints()
                for i in range(poly.GetNumberOfPoints()):
                    x,y,z = self.trailPoints[i]
                    vtkpts.SetPoint(i, x,y,z)
                vtkpts.Modified() # needed by vtk<7
                return self
            actor.updateTrail = types.MethodType(_fupdateTrail, actor )
            return tline


    def _draw_axes(self, c=None):
        if c is None: #automatic black or white
            c = (1, 1, 1)
            if numpy.sum(self.renderer.GetBackground()) > 1.5:
                c = (0.1, 0.1, 0.1)
        r = self.renderers.index(self.renderer)
        if self.caxes_exist[r] or not self.axes: return
        if not self.renderer: return
        vbb = self.renderer.ComputeVisiblePropBounds()

        if self.axes == 1 or self.axes == True:
            ca = vtk.vtkCubeAxesActor()
            ca.SetBounds(vbb)
            if self.camera: ca.SetCamera(self.camera)
            else: ca.SetCamera(self.renderer.GetActiveCamera())
            if utils.vtkMV:
                ca.GetXAxesLinesProperty().SetColor(c)
                ca.GetYAxesLinesProperty().SetColor(c)
                ca.GetZAxesLinesProperty().SetColor(c)
                for i in range(3):
                    ca.GetLabelTextProperty(i).SetColor(c)
                    ca.GetTitleTextProperty(i).SetColor(c)
                ca.SetTitleOffset(8)
            else:
                ca.GetProperty().SetColor(c)
            ca.SetFlyMode(3)
            ca.SetLabelScaling(False, 1,1,1)
            ca.SetXTitle(self.xtitle)
            ca.SetYTitle(self.ytitle)
            ca.SetZTitle(self.ztitle)
            if self.xtitle=='': 
                ca.SetXAxisVisibility(0)
                ca.XAxisLabelVisibilityOff()
            if self.ytitle=='': 
                ca.SetYAxisVisibility(0)
                ca.YAxisLabelVisibilityOff()
            if self.ztitle=='': 
                ca.SetZAxisVisibility(0)
                ca.ZAxisLabelVisibilityOff()
            ca.XAxisMinorTickVisibilityOff()
            ca.YAxisMinorTickVisibilityOff()
            ca.ZAxisMinorTickVisibilityOff()
            self.caxes_exist[r] = ca
            self.renderer.AddActor(ca)

        elif 1 < self.axes < 4:
            xcol, ycol, zcol = 'db', 'dg', 'dr' # dark blue, green red
            s = 1
            alpha = 1
            centered = False
            x0, x1, y0, y1, z0, z1 = vbb
            dx, dy, dz = x1-x0, y1-y0, z1-z0
            aves = numpy.sqrt(dx*dx+dy*dy+dz*dz)/2
            x0, x1 = min(x0, 0), max(x1, 0)
            y0, y1 = min(y0, 0), max(y1, 0)
            z0, z1 = min(z0, 0), max(z1, 0)
            if self.axes==3: 
                if x1>0: x0=0
                if y1>0: y0=0
                if z1>0: z0=0

            dx, dy, dz = x1-x0, y1-y0, z1-z0
            acts=[]
            if (x0*x1<=0 or y0*z1<=0 or z0*z1<=0): # some ranges contain origin
                zero = self.sphere(r=aves/120*s, c='k', alpha=alpha, res=10)
                acts += [zero]
                self.actors.pop()

            if len(self.xtitle) and dx>aves/100:
                xl = shapes.cylinder([[x0, 0, 0], [x1, 0, 0]], r=aves/250*s, c=xcol, alpha=alpha)
                xc = shapes.cone(pos=[x1, 0, 0], c=xcol, alpha=alpha,
                                    r=aves/100*s, height=aves/25*s, axis=[1, 0, 0], res=10)
                wpos = [x1-(len(self.xtitle)+1)*aves/40*s, -aves/25*s, 0] # aligned to arrow tip
                if centered: wpos = [(x0+x1)/2-len(self.xtitle)/2*aves/40*s, -aves/25*s, 0] 
                xt = shapes.text(self.xtitle, pos=wpos, normal=(0,0,1) , s=aves/40*s, c=xcol)
                acts += [xl,xc,xt]

            if len(self.ytitle) and dy>aves/100:
                yl = shapes.cylinder([[0, y0, 0], [0, y1, 0]], r=aves/250*s, c=ycol, alpha=alpha)
                yc = shapes.cone(pos=[0, y1, 0], c=ycol, alpha=alpha,
                                    r=aves/100*s, height=aves/25*s, axis=[0, 1, 0], res=10)
                wpos = [-aves/40*s, y1-(len(self.ytitle)+1)*aves/40*s, 0]
                if centered: wpos = [ -aves/40*s, (y0+y1)/2-len(self.ytitle)/2*aves/40*s, 0] 
                yt = shapes.text(self.ytitle, normal=(0,0,1) , s=aves/40*s, c=ycol)
                yt.rotate(90, [0,0,1]).pos(wpos)
                acts += [yl,yc,yt]

            if len(self.ztitle) and dz>aves/100:
                zl = shapes.cylinder([[0, 0, z0], [0, 0, z1]], r=aves/250*s, c=zcol, alpha=alpha)
                zc = shapes.cone(pos=[0, 0, z1], c=zcol, alpha=alpha,
                                    r=aves/100*s, height=aves/25*s, axis=[0, 0, 1], res=10)
                wpos = [-aves/50*s, -aves/50*s, z1-(len(self.ztitle)+1)*aves/40*s]
                if centered: wpos = [ -aves/50*s,  -aves/50*s, (z0+z1)/2-len(self.ztitle)/2*aves/40*s]
                zt = shapes.text(self.ztitle, normal=(1, -1,0) , s=aves/40*s, c=zcol)
                zt.rotate(180, (1, -1, 0)).pos(wpos)
                acts += [zl,zc,zt]
            ass = utils.makeAssembly(acts)
            self.caxes_exist[r] = ass
            self.renderer.AddActor(ass)

        elif self.axes == 4:
            axact = vtk.vtkAxesActor()
            axact.SetShaftTypeToCylinder()
            axact.SetCylinderRadius(.03)
            axact.SetXAxisLabelText(self.xtitle)
            axact.SetYAxisLabelText(self.ytitle)
            axact.SetZAxisLabelText(self.ztitle)
            axact.GetXAxisShaftProperty().SetColor(0,0,1)
            axact.GetZAxisShaftProperty().SetColor(1,0,0)
            axact.GetXAxisTipProperty().SetColor(0,0,1)
            axact.GetZAxisTipProperty().SetColor(1,0,0)
            bc = numpy.array(self.renderer.GetBackground())
            if numpy.sum(bc)<1.5: lc=(1,1,1)
            else: lc=(0,0,0)
            axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
            axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
            axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
            axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
            axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
            axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
            axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
            axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
            axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
            axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
            axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
            axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
            self.addIcon(axact, size=0.1)

        elif self.axes == 5:
            axact = vtk.vtkAnnotatedCubeActor()
            axact.GetCubeProperty().SetColor(.75,.75,.75)
            axact.SetTextEdgesVisibility(0)
            axact.SetFaceTextScale(.4)
            axact.GetXPlusFaceProperty().SetColor(colors.getColor('b'))
            axact.GetXMinusFaceProperty().SetColor(colors.getColor('db'))
            axact.GetYPlusFaceProperty().SetColor(colors.getColor('g'))
            axact.GetYMinusFaceProperty().SetColor(colors.getColor('dg'))
            axact.GetZPlusFaceProperty().SetColor(colors.getColor('r'))
            axact.GetZMinusFaceProperty().SetColor(colors.getColor('dr'))
            self.addIcon(axact, size=.06)
        
        else:
            colors.printc('axes type keyword must be in range [1-5]', c=1)


    def _draw_ruler(self):
        #draws a simple ruler at the bottom of the window
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
        if not utils.isSequence(self.legend): return

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
            vtklegend.SetEntry(i, utils.polydata(a), "  "+ti, c)
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
    def show(self, actors=None, at=None, legend=None, axes=None, ruler=False,
             c=None, alpha=None , wire=False, bc=None, 
             resetcam=True, zoom=False, interactive=None, execute=None, 
             azimuth=0, elevation=0,
             q=False):
        '''Render a list of actors.
        
        Options:
            
            actors = a mixed list of vtkActor, vtkAssembly, vtkPolydata, vtkVolume or filename strings
            
            at     = number of the renderer to plot to, if more than one exists
            
            legend = a string or list of string for each actor, if False will not show it
            
            axes   = set typ of axes to be shown:
                
                     0, no axes,
                
                     1, show vtk default axes,
                     
                     2, show negative axes starting from origin
                     
                     3, show positive axes starting from origin
                     
                     4, show axis widget as a triad on the bottom left corner
                     
                     5, show annotated cube widget on the bottom left corner
            
            ruler  = draws a simple ruler at the bottom
            
            c      = surface color, in rgb, hex or name formats
            
            bc     = set a color for the internal surface face
            
            wire   = show actor in wireframe representation
            
            resetcam = re-adjust camera position to fit objects
            
            interactive = pause and interact with window (True) or continue execution (False)
            
            execute = holds an external function to be called, allowing interaction with scene
            
            q      = force program to quit after show() command returns
        '''
        
        def scan(wannabeacts):
            scannedacts=[]
            if not utils.isSequence(wannabeacts): wannabeacts = [wannabeacts]
            for a in wannabeacts: # scan content of list
                if isinstance(a, vtk.vtkActor):      
                    if c is not None: a.GetProperty().SetColor(colors.getColor(c))
            
                    if alpha is not None: a.GetProperty().SetOpacity(alpha)
                    
                    if wire: a.GetProperty().SetRepresentationToWireframe()
            
                    if bc: # defines a specific color for the backface
                        backProp = vtk.vtkProperty()
                        backProp.SetDiffuseColor(colors.getColor(bc))
                        if alpha: backProp.SetOpacity(alpha)
                        a.SetBackfaceProperty(backProp)
                    scannedacts.append(a)
                elif isinstance(a, vtk.vtkAssembly):   scannedacts.append(a)
                elif isinstance(a, vtk.vtkActor2D):    scannedacts.append(a)
                elif isinstance(a, vtk.vtkImageActor): scannedacts.append(a)
                elif isinstance(a, vtk.vtkVolume):     scannedacts.append(a)
                elif isinstance(a, vtk.vtkPolyData):
                    out = self.load(a, c, alpha, wire, bc, False)
                    self.actors.pop()
                    scannedacts.append(out) 
                elif isinstance(a, str): # assume a filepath was given
                    out = self.load(a, c, alpha, wire, bc, False)
                    self.actors.pop()
                    if isinstance(out, str):
                        colors.printc('File not found:', out, c=1)
                        scannedacts.append(None)
                    else:
                        scannedacts.append(out) 
                elif a is None: 
                    pass
                else: 
                    colors.printc('Cannot understand input in show():', type(a), c=1)
                    scannedacts.append(None)
            return scannedacts

        if actors:
            actors2show = scan(actors)
            for a in actors2show:
                if a not in self.actors: self.actors.append(a)
        else:
            actors2show = scan(self.actors)
            self.actors = list(actors2show)
        
        if legend:
            if   utils.isSequence(legend): self.legend = list(legend)
            elif isinstance(legend,  str): self.legend = [str(legend)]
            else:
                colors.printc('Error in show(): legend must be list or string.', c=1)
                sys.exit()
        if not (axes is None): self.axes = axes
  
        if not (interactive is None): self.interactive = interactive
  
        if at is None and len(self.renderers)>1:
            #in case of multiple renderers a call to show w/o specifing
            # at which renderer will just render the whole thing and return
            if self.interactor:
                if zoom: self.camera.Zoom(zoom)
                self.interactor.Render()
                if self.interactive: self.interactor.Start()
                return
            
        if at is None: at=0

        if at < len(self.renderers):
            self.renderer = self.renderers[at]
        else:
            colors.printc("Error in show(): wrong renderer index", at, c=1)
            return

        if not self.camera:
            self.camera = self.renderer.GetActiveCamera()
        self.camera.SetParallelProjection(self.infinity)
        self.camera.SetThickness(self.camThickness)
            
        if self.sharecam:
            for r in self.renderers: r.SetActiveCamera(self.camera)
        
        if len(self.renderers)==1: 
            self.renderer.SetActiveCamera(self.camera)

        
        ############################### rendering
        for ia in actors2show:        # add the actors that are not already in scene            
            if ia: 
                if isinstance(ia, vtk.vtkVolume):
                    self.renderer.AddVolume(ia)
                else:
                    self.renderer.AddActor(ia)
            else:  
                colors.printc('Warning: Invalid actor in actors list, skip.', c=5)
        for ia in self.getActors(at): # remove the ones that are not in actors2show
            if ia not in actors2show: 
                self.renderer.RemoveActor(ia)

        if ruler: self._draw_ruler()
        if self.axes: self._draw_axes()
        self._draw_legend()

        if resetcam: self.renderer.ResetCamera()

        if not self.initializedIren:
            self.initializedIren = True
            self.interactor.Initialize()
            self.interactor.RemoveObservers('CharEvent')
            def mouseleft(obj, e): vtkio._mouseleft(self, obj, e)
            self.interactor.AddObserver("LeftButtonPressEvent", mouseleft)
            def mouseright(obj, e): vtkio._mouseright(self, obj, e)
            self.interactor.AddObserver("RightButtonPressEvent", mouseright)
            def mousemiddle(obj, e): vtkio._mousemiddle(self, obj, e)
            self.interactor.AddObserver("MiddleButtonPressEvent", mousemiddle)
            def keypress( obj, e): vtkio._keypress( self, obj, e)
            self.interactor.AddObserver("KeyPressEvent", keypress)
            if execute:
                self.interactor.AddObserver('TimerEvent', execute) #vtkTimerCallback()
                self.interactor.CreateRepeatingTimer(1) # calls execute every millisecond
            if self.verbose and self.interactive: self._tips()

        self.initializedPlotter = True

        if zoom: self.camera.Zoom(zoom)
        if elevation: self.camera.Elevation(elevation)
        if azimuth: self.camera.Azimuth(azimuth)
        self.interactor.Render()

        if self.interactive:                
            self.interactor.Start()

        if q : # gracefully exit
            if self.verbose: print ('q flag set to True. Exit.')
            sys.exit(0)


    def render(self, addActor=None, at=None, axes=None, resetcam=False, zoom=False, rate=None):
        '''Render current window.'''
        if addActor:
            if utils.isSequence(addActor): 
                for a in addActor: self.addActor(a)
            else: self.addActor(addActor)

        if not self.initializedPlotter:
            before = bool(self.interactive)
            self.verbose = False
            self.show(interactive=0, at=at, axes=axes, zoom=zoom)
            self.interactive = before
            return
        if resetcam: self.renderer.ResetCamera()
        self.interactor.Render()

        if rate:
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
        '''Add a vtkActor to current renderer.'''
        if not self.initializedPlotter:
            before = bool(self.interactive)
            self.show(interactive=0)
            self.interactive = before
            return
        self.actors.append(a)
        self.renderer.AddActor(a)

    def removeActor(self, a):
        '''Remove vtkActor or actor index from current renderer.'''
        try:
            if not self.initializedPlotter:
                self.show()
                return
            if self.renderer: self.renderer.RemoveActor(a)
            i = self.actors.index(a)
            del self.actors[i]
        except: pass

    def clear(self, actors=[]):
        """Delete specified list of actors, by default delete all."""
        if len(actors):
            for i,a in enumerate(actors): self.removeActor(a)
        else:
            for a in self.getActors(): self.renderer.RemoveActor(a)
            self.actors = []
    
    def openVideo(self, name='movie.avi', fps=12, duration=None):
        '''Open a video file.
        
        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/makeVideo.py)    
        '''
        return vtkio.Video(self.renderWin, name, fps, duration)

    def screenshot(self, filename='screenshot.png'):
        '''Save a screenshot of the current rendering window.'''
        vtkio.screenshot(self.renderWin, filename)




