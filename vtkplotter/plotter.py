from __future__ import division, print_function
import time
import sys
import vtk
import numpy

from vtkplotter import __version__
import vtkplotter.vtkio as vtkio
import vtkplotter.utils as utils
import vtkplotter.colors as colors
import vtkplotter.shapes as shapes
from vtkplotter.actors import Assembly, Actor
import vtkplotter.docs as docs
import vtkplotter.settings as settings

__doc__ = """
Defines main class ``Plotter`` to manage actors and 3D rendering.
"""+docs._defs

__all__ = [
    'show',
    'Plotter',
]


########################################################################
def show(actors=None,
         at=None, shape=(1, 1), N=None,
         pos=(0, 0), size='auto', screensize='auto', title='',
         bg='blackboard', bg2=None, axes=1, infinity=False,
         verbose=True, interactive=None, offscreen=False,
         resetcam=True, zoom=None, viewup='', azimuth=0, elevation=0, roll=0,
         interactorStyle=0, newPlotter=False, depthpeeling=False, q=False):
    '''
    Create on the fly an instance of class ``Plotter`` and show the object(s) provided.

    Allowed input objects are: ``filename``, ``vtkPolyData``, ``vtkActor``, 
    ``vtkActor2D``, ``vtkImageActor``, ``vtkAssembly`` or ``vtkVolume``.

    If filename is given, its type is guessed based on its extension.
    Supported formats are: 
    `vtu, vts, vtp, ply, obj, stl, 3ds, xml, neutral, gmsh, pcd, xyz, txt, byu,
    tif, slc, vti, mhd, png, jpg`.

    :param bool newPlotter: if set to `True`, a call to ``show`` will instantiate
        a new ``Plotter`` object (a new window) instead of reusing the first created.
        See e.g.: |readVolumeAsIsoSurface.py|_
    :return: the current ``Plotter`` class instance.

    .. note:: With multiple renderers, keyword ``at`` can become a `list`, e.g.:

        >>> from vtkplotter import *
        >>> s = Sphere()
        >>> c = Cube()
        >>> p = Paraboloid()
        >>> show([s, c], at=[0, 1], shape=(3,1))
        >>> show(p, at=2)
        >>> #
        >>> # is equivalent to:
        >>> vp = Plotter(shape=(3,1))
        >>> s = Sphere()
        >>> c = Cube()
        >>> p = Paraboloid()
        >>> vp.show(s, at=0)
        >>> vp.show(p, at=1)
        >>> vp.show(c, at=2, interactive=True)
    '''
    if settings.plotter_instance and newPlotter == False:
        vp = settings.plotter_instance
    else:
        if utils.isSequence(at):
            if not utils.isSequence(actors):
                colors.printc("show() Error: input must be a list.", c=1)
                exit()
            if len(at) != len(actors):
                colors.printc("show() Error: lists 'input' and 'at', must have equal lengths.", c=1)
                exit()
            if len(at) > 1 and shape == (1, 1) and N == None:
                N = len(at)
        elif at is None and (N or shape != (1, 1)):
            if not utils.isSequence(actors):
                colors.printc('show() Error: N or shape is set, but actor is not a sequence.', c=1)
                exit()
            at = range(len(actors))

        vp = Plotter(shape=shape, N=N,
                     pos=pos, size=size, screensize=screensize, title=title,
                     bg=bg, bg2=bg2, axes=axes, infinity=infinity, depthpeeling=depthpeeling,
                     verbose=verbose, interactive=interactive, offscreen=offscreen)
        settings.plotter_instance = vp
        if not(vp in settings.plotter_instances):
            settings.plotter_instances.append(vp)

    if utils.isSequence(at):
        no = len(actors)-1
        for i, a in enumerate(actors):
            if i == no and interactive is None:
                interactive = True
            vp.show(a, at=i, zoom=zoom, resetcam=resetcam,
                    viewup=viewup, azimuth=azimuth, elevation=elevation, roll=roll,
                    interactive=interactive, interactorStyle=interactorStyle, q=q)
    else:
        vp.show(actors, at=at, zoom=zoom,
                viewup=viewup, azimuth=azimuth, elevation=elevation, roll=roll,
                interactive=interactive, interactorStyle=interactorStyle, q=q)

    return vp


########################################################################
class Plotter:
    """
    Main class to manage actors.

    :param list shape: shape of the grid of renderers in format (rows, columns). 
        Ignored if N is specified.
    :param int N: number of desired renderers arranged in a grid automatically.
    :param list pos: (x,y) position in pixels of top-left corneer of the rendering window 
        on the screen
    :param size: size of the rendering window. If 'auto', guess it based on screensize.
    :param screensize: physical size of the monitor screen 
    :param bg: background color or specify jpg image file name with path
    :param bg2: background color of a gradient towards the top
    :param int axes:  

      - 0,  no axes,
      - 1,  draw three gray grid walls
      - 2,  show cartesian axes from (0,0,0)
      - 3,  show positive range of cartesian axes from (0,0,0)
      - 4,  show a triad at bottom left
      - 5,  show a cube at bottom left
      - 6,  mark the corners of the bounding box
      - 7,  draw a simple ruler at the bottom of the window
      - 8,  show the ``vtkCubeAxesActor`` object,
      - 9,  show the bounding box outLine,

    :param bool infinity: if True fugue point is set at infinity (no perspective effects)
    :param bool sharecam: if False each renderer will have an independent vtkCamera
    :param bool interactive: if True will stop after show() to allow interaction w/ window
    :param bool offscreen: if True will not show the rendering window    
    :param bool depthpeeling: depth-peel volumes along with the translucent geometry

    |multiwindows|
    """

    def __init__(self, shape=(1, 1), N=None, pos=(0, 0),
                 size='auto', screensize='auto', title='',
                 bg='blackboard', bg2=None, axes=1, infinity=False,
                 sharecam=True, verbose=True, interactive=None, offscreen=False,
                 depthpeeling=False):

        global plotter_instance
        plotter_instance = self

        if interactive is None:
            if N or shape != (1, 1):
                interactive = False
            else:
                interactive = True

        if not interactive:
            verbose = False

        self.verbose = verbose
        self.actors = []      # list of actors to be shown
        self.clickedActor = None  # holds the actor that has been clicked
        self.clickedRenderer = 0  # clicked renderer number
        self.renderer = None  # current renderer
        self.renderers = []   # list of renderers
        self.shape = shape
        self.pos = pos
        self.size = [size[1], size[0]]  # size of the rendering window
        self.interactive = interactive  # allows to interact with renderer
        self.axes = axes        # show axes type nr.
        self.title = title      # window title
        self.xtitle = 'x'       # x axis label and units
        self.ytitle = 'y'       # y axis label and units
        self.ztitle = 'z'       # z axis label and units
        self.sharecam = sharecam  # share the same camera if multiple renderers
        self.infinity = infinity  # ParallelProjection On or Off
        self.flat = True       # sets interpolation style to 'flat'
        self.phong = False     # sets interpolation style to 'phong'
        self.gouraud = False   # sets interpolation style to 'gouraud'
        self.bculling = False  # back face culling
        self.fculling = False  # front face culling
        self._legend = []       # list of legend entries for actors
        self.legendSize = 0.15  # size of legend
        self.legendBC = (.96, .96, .9)  # legend background color
        self.legendPos = 2      # 1=topright, 2=top-right, 3=bottom-left
        self.picked3d = None    # 3d coords of a clicked point on an actor
        self.backgrcol = bg
        self.offscreen = offscreen

        # mostly internal stuff:
        self.camThickness = 2000
        self.justremoved = None
        self.axes_exist = []
        self.icol = 0
        self.clock = 0
        self._clockt0 = time.time()
        self.initializedPlotter = False
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

        self.write = vtkio.write

        # sort out screen size
        self.renderWin = vtk.vtkRenderWindow()
        self.renderWin.PointSmoothingOn()
        if screensize == 'auto':
            aus = self.renderWin.GetScreenSize()
            if aus and len(aus) == 2 and aus[0] > 100 and aus[1] > 100:  # seems ok
                if aus[0]/aus[1] > 2:  # looks like there are 2 or more screens
                    screensize = (int(aus[0]/2), aus[1])
                else:
                    screensize = aus
            else:  # it went wrong, use a default 1.5 ratio
                screensize = (2160, 1440)

        x, y = screensize
        if N:                    # N = number of renderers. Find out the best
            if shape != (1, 1):  # arrangement based on minimum nr. of empty renderers
                colors.printc('Warning: having set N, shape is ignored.', c=1)
            nx = int(numpy.sqrt(int(N*y/x)+1))
            ny = int(numpy.sqrt(int(N*x/y)+1))
            lm = [(nx, ny), (nx, ny+1), (nx-1, ny), (nx+1, ny), (nx, ny-1),
                  (nx-1, ny+1), (nx+1, ny-1), (nx+1, ny+1), (nx-1, ny-1)]
            ind, minl = 0, 1000
            for i, m in enumerate(lm):
                l = m[0]*m[1]
                if N <= l < minl:
                    ind = i
                    minl = l
            shape = lm[ind]
        if size == 'auto':  # figure out a reasonable window size
            f = 1.5
            xs = y/f*shape[1]  # because y<x
            ys = y/f*shape[0]
            if xs > x/f:  # shrink
                xs = x/f
                ys = xs/shape[1]*shape[0]
            if ys > y/f:
                ys = y/f
                xs = ys/shape[0]*shape[1]
            self.size = (int(xs), int(ys))
            if shape == (1, 1):
                self.size = (int(y/f), int(y/f))  # because y<x

        ############################
        # build the renderers scene:
        self.shape = shape
        for i in reversed(range(shape[0])):
            for j in range(shape[1]):
                arenderer = vtk.vtkRenderer()
                arenderer.SetUseDepthPeeling(depthpeeling)
                if 'jpg' in str(bg).lower() or 'jpeg' in str(bg).lower():
                    if i == 0:
                        jpeg_reader = vtk.vtkJPEGReader()
                        if not jpeg_reader.CanReadFile(bg):
                            colors.printc("Error reading background image file", bg, c=1)
                            sys.exit()
                        jpeg_reader.SetFileName(bg)
                        jpeg_reader.Update()
                        image_data = jpeg_reader.GetOutput()
                        image_actor = vtk.vtkImageActor()
                        image_actor.InterpolateOn()
                        image_actor.SetInputData(image_data)
                        self.backgroundRenderer = vtk.vtkRenderer()
                        self.backgroundRenderer.SetLayer(0)
                        self.backgroundRenderer.InteractiveOff()
                        if bg2:
                            self.backgroundRenderer.SetBackground(
                                colors.getColor(bg2))
                        else:
                            self.backgroundRenderer.SetBackground(1, 1, 1)
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
                arenderer.SetViewport(y0, x0, y1, x1)
                self.renderers.append(arenderer)
                self.axes_exist.append(None)

        if 'full' in size and not offscreen:  # full screen
            self.renderWin.SetFullScreen(True)
            self.renderWin.BordersOn()
        else:
            self.renderWin.SetSize(int(self.size[0]), int(self.size[1]))

        self.renderWin.SetPosition(pos)

        if not title:
            title = ' vtkplotter '+__version__+', vtk '+vtk.vtkVersion().GetVTKVersion()
            title += ', python ' + str(sys.version_info[0])+'.'+str(sys.version_info[1])
        
        self.renderWin.SetWindowName(title)

        if not settings.usingQt:
            for r in self.renderers:
                self.renderWin.AddRenderer(r)

        if offscreen:
            self.renderWin.SetOffScreenRendering(True)
            self.interactive = False
            self.interactor = None
            ######
            return
            ######

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renderWin)
        vsty = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(vsty)

        def mouseleft(obj, e):
            vtkio._mouseleft(self, obj, e)
        self.interactor.AddObserver("LeftButtonPressEvent", mouseleft)

        def mouseright(obj, e):
            vtkio._mouseright(self, obj, e)
        self.interactor.AddObserver("RightButtonPressEvent", mouseright)

        def mousemiddle(obj, e):
            vtkio._mousemiddle(self, obj, e)
        self.interactor.AddObserver("MiddleButtonPressEvent", mousemiddle)

        def keypress(obj, e):
            vtkio._keypress(self, obj, e)
        self.interactor.AddObserver("KeyPressEvent", keypress)

        if settings.allowInteraction:
            self._update_observer = None
            self._update_win_clock = time.time()

            def win_interact(iren, event):
                if event == 'TimerEvent':
                    iren.TerminateApp()
            self.interactor.AddObserver('TimerEvent', win_interact)

            def _allowInteraction():
                timenow = time.time()
                if timenow - self._update_win_clock > 0.1:
                    self._update_win_clock = timenow
                    self._update_observer = self.interactor.CreateRepeatingTimer(1)
                    self.interactor.Start()
                    self.interactor.DestroyTimer(self._update_observer)
            self.allowInteraction = _allowInteraction


    #################################################### LOADER
    def load(self, inputobj, c='gold', alpha=1,
             wire=False, bc=None, legend=True, texture=None,
             smoothing=None, threshold=None, connectivity=False):
        ''' 
        Returns a ``vtkActor`` from reading a file, directory or ``vtkPolyData``.

        :param c: color in RGB format, hex, symbol or name
        :param alpha:   transparency (0=invisible)
        :param wire:    show surface as wireframe
        :param bc:      backface color of internal surface
        :param legend:  text to show on legend, True picks filename
        :param texture: any png/jpg file can be used as texture

        For volumetric data (tiff, slc, vti files):

        :param smoothing:    gaussian filter to smooth vtkImageData
        :param threshold:    value to draw the isosurface
        :param bool connectivity: if True only keeps the largest portion of the polydata
        '''
        import os
        if isinstance(inputobj, vtk.vtkPolyData):
            a = Actor(inputobj, c, alpha, wire, bc, legend, texture)
            self.actors.append(a)
            if inputobj and inputobj.GetNumberOfPoints() == 0:
                colors.printc('Warning: actor has zero points.', c=5)
            return a

        acts = []
        if isinstance(legend, int):
            legend = bool(legend)
        if isinstance(inputobj, list):
            flist = inputobj
        else:
            import glob
            flist = sorted(glob.glob(inputobj))
        for fod in flist:
            if os.path.isfile(fod):
                a = vtkio._loadFile(fod, c, alpha, wire, bc, legend, texture,
                                    smoothing, threshold, connectivity)
                acts.append(a)
            elif os.path.isdir(fod):
                acts = vtkio._loadDir(fod, c, alpha, wire, bc, legend, texture,
                                      smoothing, threshold, connectivity)
        if not len(acts):
            colors.printc('Error in load(): cannot find', inputobj, c=1)
            return None

        for actor in acts:
            if isinstance(actor, vtk.vtkActor):
                if self.flat:
                    actor.GetProperty().SetInterpolationToFlat()
                    self.phong = False
                    self.gouraud = False
                    actor.GetProperty().SetSpecular(0)
                if self.phong:
                    actor.GetProperty().SetInterpolationToPhong()
                    self.flat = False
                    self.gouraud = False
                if self.gouraud:
                    actor.GetProperty().SetInterpolationToGouraud()
                    self.flat = False
                    self.phong = False
                if self.bculling:
                    actor.GetProperty().BackfaceCullingOn()
                else:
                    actor.GetProperty().BackfaceCullingOff()
                if self.fculling:
                    actor.GetProperty().FrontfaceCullingOn()
                else:
                    actor.GetProperty().FrontfaceCullingOff()

        self.actors += acts
        if len(acts) == 1:
            return acts[0]
        else:
            return acts

    def getActors(self, obj=None):
        '''
        Return an actors list.

        If ``obj`` is:
            ``None``, return actors of current renderer

            ``int``, return actors in given renderer number 

            ``vtkAssembly`` return the contained actors

            ``string``, return actors matching legend name
        '''
        if not self.renderer:
            return []

        if obj is None or isinstance(obj, int):
            if obj is None:
                acs = self.renderer.GetActors()
            elif obj >= len(self.renderers):
                colors.printc("Error in getActors: non existing renderer", obj, c=1)
                return []
            else:
                acs = self.renderers[obj].GetActors()
            actors = []
            acs.InitTraversal()
            for i in range(acs.GetNumberOfItems()):
                a = acs.GetNextItem()
                if a.GetPickable():
                    r = self.renderers.index(self.renderer)
                    if a == self.axes_exist[r]:
                        continue
                    actors.append(a)
            return actors

        elif isinstance(obj, vtk.vtkAssembly):
            cl = vtk.vtkPropCollection()
            obj.GetActors(cl)
            actors = []
            cl.InitTraversal()
            for i in range(obj.GetNumberOfPaths()):
                act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
                if act.GetPickable():
                    actors.append(act)
            return actors

        elif isinstance(obj, str):  # search the actor by the legend name
            actors = []
            for a in self.actors:
                if hasattr(a, '_legend') and obj in a._legend and a.GetPickable():
                    actors.append(a)
            return actors

        elif isinstance(obj, vtk.vtkActor):
            return [obj]

        if self.verbose:
            colors.printc(
                'Warning in getActors: unexpected input type', obj, c=1)
        return []


    def add(self, actors):
        '''Append input object to the internal list of actors to be shown.

        :return: returns input actor for possible concatenation.
        '''
        if utils.isSequence(actors):
            for a in actors:
                if a not in self.actors:
                    self.actors.append(a)
            return None
        else:
            self.actors.append(actors)
            return actors

    def moveCamera(self, camstart, camstop, fraction):
        '''
        Takes as input two ``vtkCamera`` objects and returns a
        new ``vtkCamera`` that is at an intermediate position:

        fraction=0 -> camstart,  fraction=1 -> camstop.

        Press ``shift-C`` key in interactive mode to dump a python snipplet
        of parameters for the current camera view.
        '''
        if isinstance(fraction, int):
            colors.printc(
                "Warning in moveCamera(): fraction should not be an integer", c=1)
        if fraction > 1:
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
        cam.SetPosition(p2*fraction+p1*(1-fraction))
        cam.SetFocalPoint(f2*fraction+f1*(1-fraction))
        cam.SetViewUp(v2*fraction+v1*(1-fraction))
        cam.SetDistance(s2*fraction+s1*(1-fraction))
        cam.SetClippingRange(c2*fraction+c1*(1-fraction))
        self.camera = cam
        save_int = self.interactive
        self.show(resetcam=0, interactive=0)
        self.interactive = save_int

    def Actor(self, poly=None, c='gold', alpha=0.5,
              wire=False, bc=None, legend=None, texture=None):
        '''
        Return a ``vtkActor`` from an input ``vtkPolyData``.

        :param c: color name, number, or list of [R,G,B] colors
        :type c: int, str, list
        :param float alpha: transparency in range [0,1].
        :param bool wire: show surface as wireframe
        :param bc: backface color of the internal surface
        :type c: int, str, list
        :param str legend: legend text
        :param str texture: jpg file name of surface texture
        '''
        a = Actor(poly, c, alpha, wire, bc, legend, texture)
        self.actors.append(a)
        return a


    def Assembly(self, actorlist):
        '''Group many actors as a single new actor which is a ``vtkAssembly`` derived object.

        .. hint:: |icon| |icon.py|_

        '''
        for a in actorlist:
            while a in self.actors:  # update internal list
                self.actors.remove(a)
        a = Assembly(actorlist)
        self.actors.append(a)
        return a

    def light(self, pos=(1, 1, 1), fp=(0, 0, 0), deg=25,
              diffuse='y', ambient='r', specular='b', showsource=False):
        """
        Generate a source of light placed at pos, directed to focal point fp.

        :param fp: focal Point, if this is a ``vtkActor`` use its position.
        :type fp: vtkActor, list
        :param deg: aperture angle of the light source
        :param showsource: if `True`, will show a vtk representation
                            of the source of light as an extra actor

        .. hint:: |lights.py|_
        """
        if isinstance(fp, vtk.vtkActor):
            fp = fp.GetPosition()
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(pos)
        light.SetPositional(1)
        light.SetConeAngle(deg)
        light.SetFocalPoint(fp)
        light.SetDiffuseColor(colors.getColor(diffuse))
        light.SetAmbientColor(colors.getColor(ambient))
        light.SetSpecularColor(colors.getColor(specular))
        save_int = self.interactive
        self.show(interactive=0)
        self.interactive = save_int
        if showsource:
            lightActor = vtk.vtkLightActor()
            lightActor.SetLight(light)
            self.renderer.AddViewProp(lightActor)
            self.renderer.AddLight(light)
        return light


    ################################################################## 
    def addScalarBar(self, actor=None, c=None, title='', horizontal=False):
        """
        Add a 2D scalar bar for the specified actor.

        If `actor` is ``None`` will add it to the last actor in ``self.actors``.

        .. hint:: |mesh_bands| |mesh_bands.py|_
        """
        if actor is None:
            actor = self.lastActor()
        if not hasattr(actor, 'mapper'):
            colors.printc('Error in addScalarBar: input is not a Actor.', c=1)
            return None

        lut = actor.mapper.GetLookupTable()
        if not lut:
            return None
        vtkscalars = actor.poly.GetPointData().GetScalars()
        if vtkscalars is None:
            vtkscalars = actor.poly.GetCellData().GetScalars()
        if not vtkscalars:
            return None
        actor.mapper.SetScalarRange(vtkscalars.GetRange())

        if c is None:
            if self.renderer:  # automatic black or white
                c = (0.9, 0.9, 0.9)
                if numpy.sum(self.renderer.GetBackground()) > 1.5:
                    c = (0.1, 0.1, 0.1)
            else:
                c = 'k'

        c = colors.getColor(c)
        sb = vtk.vtkScalarBarActor()
        sb.SetLookupTable(lut)
        if title:
            titprop = vtk.vtkTextProperty()
            titprop.BoldOn()
            titprop.ItalicOff()
            titprop.ShadowOff()
            titprop.SetColor(c)
            titprop.SetVerticalJustificationToTop()
            sb.SetTitle(title)
            sb.SetVerticalTitleSeparation(15)
            sb.SetTitleTextProperty(titprop)

        if vtk.vtkVersion().GetVTKMajorVersion() > 7:
            sb.UnconstrainedFontSizeOn()
            sb.FixedAnnotationLeaderLineColorOff()
            sb.DrawAnnotationsOn()
            sb.DrawTickLabelsOn()
        sb.SetMaximumNumberOfColors(512)

        if horizontal:
            sb.SetOrientationToHorizontal()
            sb.SetNumberOfLabels(4)
            sb.SetTextPositionToSucceedScalarBar()
            sb.SetPosition(0.1, .05)
            sb.SetMaximumWidthInPixels(1000)
            sb.SetMaximumHeightInPixels(50)
        else:
            sb.SetNumberOfLabels(10)
            sb.SetTextPositionToPrecedeScalarBar()
            sb.SetPosition(.87, .05)
            sb.SetMaximumWidthInPixels(80)
            sb.SetMaximumHeightInPixels(500)

        sctxt = sb.GetLabelTextProperty()
        sctxt.SetColor(c)
        sctxt.SetShadow(0)
        sctxt.SetFontFamily(0)
        sctxt.SetItalic(0)
        sctxt.SetBold(0)
        sctxt.SetFontSize(12)
        if not self.renderer:
            save_int = self.interactive
            self.show(interactive=0)
            self.interactive = save_int
        sb.PickableOff()
        self.renderer.AddActor(sb)
        self.renderer.Render()
        return sb


    def addScalarBar3D(self, obj=None, at=0, pos=(0, 0, 0), normal=(0, 0, 1), sx=.1, sy=2,
                       nlabels=9, ncols=256, cmap=None, c='k', alpha=1):
        '''
        Draw a 3D scalar bar.

        ``obj`` input can be:
            - a list of numbers,
            - a list of two numbers in the form `(min, max)`,
            - a ``vtkActor`` already containing a set of scalars associated to vertices or cells,
            - if ``None`` the last actor in the list of actors will be used.

        .. hint:: |mesh_coloring| |mesh_coloring.py|_
        '''
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

        gap = 0.4  # space btw nrs and scale
        vtkscalars_name = ''
        if obj is None:
            obj = self.lastActor()
        if isinstance(obj, vtk.vtkActor):
            poly = obj.GetMapper().GetInput()
            vtkscalars = poly.GetPointData().GetScalars()
            if vtkscalars is None:
                vtkscalars = poly.GetCellData().GetScalars()
            if vtkscalars is None:
                print('Error in addScalarBar3D: actor has no scalar array.', [obj])
                sys.exit()
            npscalars = vtk_to_numpy(vtkscalars)
            vmin, vmax = numpy.min(npscalars), numpy.max(npscalars)
            vtkscalars_name = vtkscalars.GetName().split('_')[-1]
        elif utils.isSequence(obj):
            vmin, vmax = numpy.min(obj), numpy.max(obj)
            vtkscalars_name = 'jet'
        else:
            print('Error in addScalarBar3D(): input must be vtkActor or list.', type(obj))
            sys.exit()

        if cmap is None:
            cmap = vtkscalars_name

        # build the color scale part
        scale = shapes.Grid([-sx*gap, 0, 0], c='w',
                            alpha=alpha, sx=sx, sy=sy, resx=1, resy=ncols)
        scale.GetProperty().SetRepresentationToSurface()
        cscals = scale.cellCenters()[:, 1]

        def _cellColors(scale, scalars, cmap, alpha):
            mapper = scale.GetMapper()
            cpoly = mapper.GetInput()
            n = len(scalars)
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(n)
            lut.Build()
            for i in range(n):
                r, g, b = colors.colorMap(i, cmap, 0, n)
                lut.SetTableValue(i, r, g, b, alpha)
            arr = numpy_to_vtk(numpy.ascontiguousarray(scalars), deep=True)
            vmin, vmax = numpy.min(scalars), numpy.max(scalars)
            mapper.SetScalarRange(vmin, vmax)
            mapper.SetLookupTable(lut)
            mapper.ScalarVisibilityOn()
            cpoly.GetCellData().SetScalars(arr)
        _cellColors(scale, cscals, cmap, alpha)

        # build text
        nlabels = numpy.min([nlabels, ncols])
        tlabs = numpy.linspace(vmin, vmax, num=nlabels, endpoint=True)
        tacts = []
        prec = (vmax-vmin)/abs(vmax+vmin)*2
        prec = int(3+abs(numpy.log10(prec+1)))
        for i, t in enumerate(tlabs):
            tx = utils.precision(t, prec)
            y = -sy/1.98+sy*i/(nlabels-1)
            a = shapes.Text(tx, pos=[sx*gap, y, 0],
                            s=sy/50, c=c, alpha=alpha, depth=0)
            a.PickableOff()
            tacts.append(a)
        sact = Assembly([scale]+tacts)
        nax = numpy.linalg.norm(normal)
        if nax:
            normal = numpy.array(normal)/nax
        theta = numpy.arccos(normal[2])
        phi = numpy.arctan2(normal[1], normal[0])
        sact.RotateZ(phi*57.3)
        sact.RotateY(theta*57.3)
        sact.SetPosition(pos)
        if not self.renderers[at]:
            save_int = self.interactive
            self.show(interactive=0)
            self.interactive = save_int
        self.renderers[at].AddActor(sact)
        self.renderers[at].Render()
        sact.PickableOff()
        return sact

    def addSlider2D(self, sliderfunc, xmin, xmax, value=None, pos=4, s=.04,
                    title='', c=None, showValue=True):
        '''
        Add a slider widget which can call an external custom function.

        :param sliderfunc: external function to be called by the widget
        :param float xmin:  lower value
        :param float xmax:  upper value
        :param float value: current value
        :param list pos:  position corner number: horizontal [1-4] or vertical [11-14]
                            it can also be specified by corners coordinates [(x1,y1), (x2,y2)]
        :param str title: title text
        :param bool showValue:  if true current value is shown

        .. hint:: |sliders| |sliders.py|_
        '''
        if c is None:  # automatic black or white
            c = (0.8, 0.8, 0.8)
            if numpy.sum(colors.getColor(self.backgrcol)) > 1.5:
                c = (0.2, 0.2, 0.2)
        c = colors.getColor(c)

        if value is None or value < xmin:
            value = xmin

        sliderRep = vtk.vtkSliderRepresentation2D()
        sliderRep.SetMinimumValue(xmin)
        sliderRep.SetMaximumValue(xmax)
        sliderRep.SetValue(value)
        sliderRep.SetSliderLength(0.015)
        sliderRep.SetSliderWidth(0.025)
        sliderRep.SetEndCapLength(0.0015)
        sliderRep.SetEndCapWidth(0.0125)
        sliderRep.SetTubeWidth(.0075)
        sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        if utils.isSequence(pos):
            sliderRep.GetPoint1Coordinate().SetValue(pos[0][0], pos[0][1])
            sliderRep.GetPoint2Coordinate().SetValue(pos[1][0], pos[1][1])
        elif pos == 1:  # top-left horizontal
            sliderRep.GetPoint1Coordinate().SetValue(.04, .96)
            sliderRep.GetPoint2Coordinate().SetValue(.45, .96)
        elif pos == 2:
            sliderRep.GetPoint1Coordinate().SetValue(.55, .96)
            sliderRep.GetPoint2Coordinate().SetValue(.96, .96)
        elif pos == 3:
            sliderRep.GetPoint1Coordinate().SetValue(.04, .04)
            sliderRep.GetPoint2Coordinate().SetValue(.45, .04)
        elif pos == 4:  # bottom-right
            sliderRep.GetPoint1Coordinate().SetValue(.55, .04)
            sliderRep.GetPoint2Coordinate().SetValue(.96, .04)
        elif pos == 5:  # bottom margin horizontal
            sliderRep.GetPoint1Coordinate().SetValue(.04, .04)
            sliderRep.GetPoint2Coordinate().SetValue(.96, .04)
        elif pos == 11:  # top-left vertical
            sliderRep.GetPoint1Coordinate().SetValue(.04, .54)
            sliderRep.GetPoint2Coordinate().SetValue(.04, .9)
        elif pos == 12:
            sliderRep.GetPoint1Coordinate().SetValue(.96, .54)
            sliderRep.GetPoint2Coordinate().SetValue(.96, .9)
        elif pos == 13:
            sliderRep.GetPoint1Coordinate().SetValue(.04, .1)
            sliderRep.GetPoint2Coordinate().SetValue(.04, .54)
        elif pos == 14:  # bottom-right vertical
            sliderRep.GetPoint1Coordinate().SetValue(.96, .1)
            sliderRep.GetPoint2Coordinate().SetValue(.96, .54)
        elif pos == 15:  # right margin vertical
            sliderRep.GetPoint1Coordinate().SetValue(.96, .1)
            sliderRep.GetPoint2Coordinate().SetValue(.96, .9)

        if showValue:
            if isinstance(xmin, int) and isinstance(xmax, int):
                frm = '%0.0f'
            else:
                frm = '%0.1f'
            sliderRep.SetLabelFormat(frm)  # default is '%0.3g'
            sliderRep.GetLabelProperty().SetShadow(0)
            sliderRep.GetLabelProperty().SetBold(0)
            sliderRep.GetLabelProperty().SetOpacity(0.6)
            sliderRep.GetLabelProperty().SetColor(c)
            if isinstance(pos, int) and pos > 10:
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
            sliderRep.SetTitleHeight(.012)
            sliderRep.GetTitleProperty().SetShadow(0)
            sliderRep.GetTitleProperty().SetColor(c)
            sliderRep.GetTitleProperty().SetOpacity(.6)
            sliderRep.GetTitleProperty().SetBold(0)
            if not utils.isSequence(pos):
                if isinstance(pos, int) and pos > 10:
                    sliderRep.GetTitleProperty().SetOrientation(90)
            else:
                if abs(pos[0][0]-pos[1][0]) < 0.1:
                    sliderRep.GetTitleProperty().SetOrientation(90)

        sliderWidget = vtk.vtkSliderWidget()
        sliderWidget.SetInteractor(self.interactor)
        sliderWidget.SetAnimationModeToJump()
        sliderWidget.SetRepresentation(sliderRep)
        sliderWidget.AddObserver("InteractionEvent", sliderfunc)
        sliderWidget.EnabledOn()
        self.sliders.append([sliderWidget, sliderfunc])
        return sliderWidget

    def addSlider3D(self, sliderfunc, pos1, pos2, xmin, xmax, value=None,
                    s=0.03, title='', rotation=0, c=None, showValue=True):
        '''
        Add a 3D slider widget which can call an external custom function.

        :param sliderfunc: external function to be called by the widget
        :param list pos1: first position coordinates
        :param list pos2: second position coordinates
        :param float xmin:  lower value
        :param float xmax:  upper value
        :param float value: initial value
        :param float s: label scaling factor
        :param str title: title text
        :param c: slider color
        :param float rotation: title rotation around slider axis
        :param bool showValue: if True current value is shown

        .. hint:: |sliders3d| |sliders3d.py|_
        '''
        if c is None:  # automatic black or white
            c = (0.8, 0.8, 0.8)
            if numpy.sum(colors.getColor(self.backgrcol)) > 1.5:
                c = (0.2, 0.2, 0.2)
        else:
            c = colors.getColor(c)

        if value is None or value < xmin:
            value = xmin

        t = 1.5/numpy.sqrt(utils.mag(numpy.array(pos2)-pos1))  # better norm

        sliderRep = vtk.vtkSliderRepresentation3D()
        sliderRep.SetMinimumValue(xmin)
        sliderRep.SetValue(value)
        sliderRep.SetMaximumValue(xmax)

        sliderRep.GetPoint1Coordinate().SetCoordinateSystemToWorld()
        sliderRep.GetPoint2Coordinate().SetCoordinateSystemToWorld()
        sliderRep.GetPoint1Coordinate().SetValue(pos2)
        sliderRep.GetPoint2Coordinate().SetValue(pos1)

        sliderRep.SetSliderWidth(0.03*t)
        sliderRep.SetTubeWidth(0.01*t)
        sliderRep.SetSliderLength(0.04*t)
        sliderRep.SetSliderShapeToCylinder()
        sliderRep.GetSelectedProperty().SetColor(1, 0, 0)
        sliderRep.GetSliderProperty().SetColor(numpy.array(c)/2)
        sliderRep.GetCapProperty().SetOpacity(0)

        sliderRep.SetRotation(rotation)

        if not showValue:
            sliderRep.ShowSliderLabelOff()

        sliderRep.SetTitleText(title)
        sliderRep.SetTitleHeight(s*t)
        sliderRep.SetLabelHeight(s*t*0.85)

        sliderRep.GetTubeProperty()
        sliderRep.GetTubeProperty().SetColor(c)

        sliderWidget = vtk.vtkSliderWidget()
        sliderWidget.SetInteractor(self.interactor)
        sliderWidget.SetRepresentation(sliderRep)
        sliderWidget.SetAnimationModeToJump()
        sliderWidget.AddObserver("InteractionEvent", sliderfunc)
        sliderWidget.EnabledOn()
        self.sliders.append([sliderWidget, sliderfunc])
        return sliderWidget


    def addButton(self, fnc, states=('On', 'Off'), c=('w', 'w'), bc=('dg', 'dr'),
                  pos=[20, 40], size=24, font='arial', bold=False, italic=False,
                  alpha=1, angle=0):
        '''Add a button to the renderer window.

        :param list states: a list of possible states ['On', 'Off']
        :param c:      a list of colors for each state
        :param bc:     a list of background colors for each state
        :param pos:    2D position in pixels from left-bottom corner
        :param size:   size of button font
        :param str font:   font type (arial, courier, times)
        :param bool bold:   bold face (False)
        :param bool italic: italic face (False)
        :param float alpha:  opacity level
        :param float angle:  anticlockwise rotation in degrees

        .. hint:: |buttons| |buttons.py|_
        '''
        if not self.renderer:
            colors.printc('Error: Use addButton() after rendering the scene.', c=1)
            return
        bu = vtkio.Button(fnc, states, c, bc, pos, size,
                          font, bold, italic, alpha, angle)
        self.renderer.AddActor2D(bu.actor)
        self.renderWin.Render()
        self.buttons.append(bu)
        return bu

    def addCutterTool(self, actor):
        '''Create handles to cut away parts of a mesh.

        .. hint:: |cutter| |cutter.py|_
        '''
        if not isinstance(actor, vtk.vtkActor):
            return None

        if not self.renderer:
            save_int = self.interactive
            self.show(interactive=0)
            self.interactive = save_int

        self.clickedActor = actor
        if hasattr(actor, 'polydata'):
            apd = actor.polydata()
        else:
            apd = actor.GetMapper().GetInput()


        planes = vtk.vtkPlanes()
        planes.SetBounds(apd.GetBounds())

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(apd)
        clipper.SetClipFunction(planes)
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOn()

        act0Mapper = vtk.vtkPolyDataMapper()  # the part which stays
        act0Mapper.SetInputConnection(clipper.GetOutputPort())
        act0 = Actor()
        act0.SetMapper(act0Mapper)
        act0.GetProperty().SetColor(actor.GetProperty().GetColor())
        act0.GetProperty().SetOpacity(1)

        act1Mapper = vtk.vtkPolyDataMapper()  # the part which is cut away
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
        boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
        boxWidget.GetOutlineProperty().SetColor(0.1, 0.1, 0.1)
        boxWidget.GetOutlineProperty().SetOpacity(0.8)
        boxWidget.SetPlaceFactor(1.05)
        boxWidget.SetInteractor(self.interactor)
        boxWidget.SetInputData(apd)
        boxWidget.PlaceWidget()
        boxWidget.AddObserver("InteractionEvent", SelectPolygons)
        boxWidget.On()

        self.cutterWidget = boxWidget
        self.clickedActor = act0
        ia = self.actors.index(actor)
        self.actors[ia] = act0

        colors.printc('Mesh Cutter Tool:', c='m', invert=1)
        colors.printc(
            '  Move gray handles to cut away parts of the mesh', c='m')
        colors.printc("  Press X to save file to: clipped.vtk", c='m')

        self.interactor.Start()
        boxWidget.Off()
        self.widgets.append(boxWidget)

        self.interactor.Start()  # allow extra interaction

        return act0
    

    def addIcon(self, iconActor, pos=3, size=0.08):
        '''
        Add an inset icon mesh into the same renderer.

        :param pos: icon position in the range [1-4] indicating one of the 4 corners,
                    or it can be a tuple (x,y) as a fraction of the renderer size.
        :param float size: size of the square inset.

        .. hint:: |icon| |icon.py|_
        '''
        if not self.renderer:
            colors.printc('Warning: Use addIcon() after first rendering the scene.', c=3)
            save_int = self.interactive
            self.show(interactive=0)
            self.interactive = save_int
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(iconActor)
        widget.SetInteractor(self.interactor)
        if utils.isSequence(pos):
            widget.SetViewport(pos[0]-size, pos[1]-size,
                               pos[0]+size, pos[1]+size)
        else:
            if pos < 2:
                widget.SetViewport(0, 1-2*size, size*2, 1)
            elif pos == 2:
                widget.SetViewport(1-2*size, 1-2*size, 1, 1)
            elif pos == 3:
                widget.SetViewport(0, 0, size*2, size*2)
            elif pos == 4:
                widget.SetViewport(1-2*size, 0, 1, size*2)
        widget.EnabledOn()
        widget.InteractiveOff()
        self.widgets.append(widget)
        if iconActor in self.actors:
            self.actors.remove(iconActor)
        return widget


    def drawAxes(self, axtype=None, c=None):
        '''
        Draw axes on scene. Available axes types:

        :param int axtype: 

              - 0,  no axes,
              - 1,  draw three gray grid walls
              - 2,  show cartesian axes from (0,0,0)
              - 3,  show positive range of cartesian axes from (0,0,0)
              - 4,  show a triad at bottom left
              - 5,  show a cube at bottom left
              - 6,  mark the corners of the bounding box
              - 7,  draw a simple ruler at the bottom of the window
              - 8,  show the ``vtkCubeAxesActor`` object,
              - 9,  show the bounding box outLine,
        '''
        if axtype is not None:
            self.axes = axtype  # overrride

        if not self.axes:
            return

        if c is None:  # automatic black or white
            c = (0.9, 0.9, 0.9)
            if numpy.sum(self.renderer.GetBackground()) > 1.5:
                c = (0.1, 0.1, 0.1)

        if not self.renderer:
            return

        r = self.renderers.index(self.renderer)
        if self.axes_exist[r]:
            return

        # calculate max actors bounds
        bns = []
        for a in self.actors:
            if a and a.GetPickable():
                b = a.GetBounds()
                if b:
                    bns.append(b)
        if len(bns):
            max_bns = numpy.max(bns, axis=0)
            min_bns = numpy.min(bns, axis=0)
            vbb = (min_bns[0], max_bns[1], min_bns[2],
                   max_bns[3], min_bns[4], max_bns[5])
        else:
            vbb = self.renderer.ComputeVisiblePropBounds()
            max_bns = vbb
            min_bns = vbb
        sizes = (max_bns[1]-min_bns[0],
                 max_bns[3]-min_bns[2],
                 max_bns[5]-min_bns[4])

        ############################################################
        if self.axes == 1 or self.axes == True:  # gray grid walls
            nd = 4     # number of divisions in the smallest axis
            off = -0.04  # label offset
            step = numpy.min(sizes)/nd
            if not step:
                # bad proportions, use vtkCubeAxesActor
                self.drawAxes(axtype=8, c=c)
                self.axes = 1
                return

            rx, ry, rz = numpy.rint(sizes/step).astype(int)
            if max([rx/ry, ry/rx, rx/rz, rz/rx, ry/rz, rz/ry]) > 15:
                # bad proportions, use vtkCubeAxesActor
                self.drawAxes(axtype=8, c=c)
                self.axes = 1
                return

            gxy = shapes.Grid(pos=(.5,.5,0), normal=[0,0,1], bc=None, resx=rx, resy=ry)
            gxz = shapes.Grid(pos=(.5,0,.5), normal=[0,1,0], bc=None, resx=rz, resy=rx)
            gyz = shapes.Grid(pos=(0,.5,.5), normal=[1,0,0], bc=None, resx=rz, resy=ry)
            gxy.alpha(0.06).wire(False).color(c).lineWidth(1)
            gxz.alpha(0.04).wire(False).color(c).lineWidth(1)
            gyz.alpha(0.04).wire(False).color(c).lineWidth(1)

            xa = shapes.Line([0, 0, 0], [1, 0, 0], c=c, lw=1)
            ya = shapes.Line([0, 0, 0], [0, 1, 0], c=c, lw=1)
            za = shapes.Line([0, 0, 0], [0, 0, 1], c=c, lw=1)

            xt, yt, zt, ox, oy, oz = [None]*6
            if self.xtitle:
                if min_bns[0]<=0 and max_bns[1]>0: # mark x origin
                    ox = shapes.Cube([-min_bns[0]/sizes[0],0,0], side=.008, c=c)
                if len(self.xtitle) == 1: # add axis length info
                    self.xtitle += ' /' + utils.precision(sizes[0], 4)
                wpos = [1-(len(self.xtitle)+1)/40, off, 0]
                xt = shapes.Text(self.xtitle, pos=wpos, normal=(0,0,1), s=.025, c=c)

            if self.ytitle:
                if min_bns[2]<=0 and max_bns[3]>0: # mark y origin
                    oy = shapes.Cube([0,-min_bns[2]/sizes[1],0], side=.008, c=c)
                yt = shapes.Text(self.ytitle, pos=(0,0,0), normal=(0,0,1), s=.025, c=c)
                if len(self.ytitle) == 1:
                    wpos = [off, 1-(len(self.ytitle)+1)/40,  0]
                    yt.pos(wpos)
                else:
                    wpos = [off*.7, 1-(len(self.ytitle)+1)/40,  0]
                    yt.rotateZ(90).pos(wpos)

            if self.ztitle:
                if min_bns[4]<=0 and max_bns[5]>0: # mark z origin
                    oz = shapes.Cube([0,0,-min_bns[4]/sizes[2]], side=.008, c=c)
                zt = shapes.Text(self.ztitle, pos=(0,0,0), normal=(1,-1,0), s=.025, c=c)
                if len(self.ztitle) == 1:
                    wpos = [off*.6, off*.6, 1-(len(self.ztitle)+1)/40]
                    zt.rotate(90, (1,-1,0)).pos(wpos)
                else:
                    wpos = [off*.3, off*.3, 1-(len(self.ztitle)+1)/40]
                    zt.rotate(180, (1,-1,0)).pos(wpos)

            acts = [gxy, gxz, gyz, xa, ya, za, xt, yt, zt, ox, oy, oz]
            for a in acts:
                if a:
                    a.PickableOff()
            aa = Assembly(acts)
            aa.pos(min_bns[0], min_bns[2], min_bns[4])
            aa.SetScale(sizes)
            aa.PickableOff()
            self.renderer.AddActor(aa)

        elif self.axes == 2 or self.axes == 3:
            vbb = self.renderer.ComputeVisiblePropBounds()  # to be double checked
            xcol, ycol, zcol = 'db', 'dg', 'dr'
            s = 1
            alpha = 1
            centered = False
            x0, x1, y0, y1, z0, z1 = vbb
            dx, dy, dz = x1-x0, y1-y0, z1-z0
            aves = numpy.sqrt(dx*dx+dy*dy+dz*dz)/2
            x0, x1 = min(x0, 0), max(x1, 0)
            y0, y1 = min(y0, 0), max(y1, 0)
            z0, z1 = min(z0, 0), max(z1, 0)

            if self.axes == 3:
                if x1 > 0:
                    x0 = 0
                if y1 > 0:
                    y0 = 0
                if z1 > 0:
                    z0 = 0

            dx, dy, dz = x1-x0, y1-y0, z1-z0
            acts = []
            if (x0*x1 <= 0 or y0*z1 <= 0 or z0*z1 <= 0):  # some ranges contain origin
                zero = shapes.Sphere(r=aves/120*s, c='k', alpha=alpha, res=10)
                acts += [zero]

            if len(self.xtitle) and dx > aves/100:
                xl = shapes.Cylinder([[x0, 0, 0], [x1, 0, 0]], r=aves/250*s, c=xcol, alpha=alpha)
                xc = shapes.Cone(pos=[x1, 0, 0], c=xcol, alpha=alpha,
                                 r=aves/100*s, height=aves/25*s, axis=[1, 0, 0], res=10)
                wpos = [x1-(len(self.xtitle)+1)*aves/40*s, -aves/25*s, 0]  # aligned to arrow tip
                if centered:
                    wpos = [(x0+x1)/2-len(self.xtitle) / 2*aves/40*s, -aves/25*s, 0]
                xt = shapes.Text(self.xtitle, pos=wpos, normal=(0, 0, 1), s=aves/40*s, c=xcol)
                acts += [xl, xc, xt]

            if len(self.ytitle) and dy > aves/100:
                yl = shapes.Cylinder([[0, y0, 0], [0, y1, 0]], r=aves/250*s, c=ycol, alpha=alpha)
                yc = shapes.Cone(pos=[0, y1, 0], c=ycol, alpha=alpha,
                                 r=aves/100*s, height=aves/25*s, axis=[0, 1, 0], res=10)
                wpos = [-aves/40*s, y1-(len(self.ytitle)+1)*aves/40*s, 0]
                if centered:
                    wpos = [-aves/40*s, (y0+y1)/2 - len(self.ytitle)/2*aves/40*s, 0]
                yt = shapes.Text(self.ytitle, pos=(0,0,0), normal=(0, 0, 1), s=aves/40*s, c=ycol)
                yt.rotate(90, [0, 0, 1]).pos(wpos)
                acts += [yl, yc, yt]

            if len(self.ztitle) and dz > aves/100:
                zl = shapes.Cylinder([[0, 0, z0], [0, 0, z1]], r=aves/250*s, c=zcol, alpha=alpha)
                zc = shapes.Cone(pos=[0, 0, z1], c=zcol, alpha=alpha,
                                 r=aves/100*s, height=aves/25*s, axis=[0, 0, 1], res=10)
                wpos = [-aves/50*s, -aves/50*s, z1 - (len(self.ztitle)+1)*aves/40*s]
                if centered:
                    wpos = [-aves/50*s, -aves/50*s, (z0+z1)/2-len(self.ztitle)/2*aves/40*s]
                zt = shapes.Text(self.ztitle, pos=(0,0,0), normal=(1, -1, 0), s=aves/40*s, c=zcol)
                zt.rotate(180, (1, -1, 0)).pos(wpos)
                acts += [zl, zc, zt]
            for a in acts:
                a.PickableOff()
            ass = Assembly(acts)
            ass.PickableOff()
            self.renderer.AddActor(ass)

        elif self.axes == 4:
            axact = vtk.vtkAxesActor()
            axact.SetShaftTypeToCylinder()
            axact.SetCylinderRadius(.03)
            axact.SetXAxisLabelText(self.xtitle)
            axact.SetYAxisLabelText(self.ytitle)
            axact.SetZAxisLabelText(self.ztitle)
            axact.GetXAxisShaftProperty().SetColor(0, 0, 1)
            axact.GetZAxisShaftProperty().SetColor(1, 0, 0)
            axact.GetXAxisTipProperty().SetColor(0, 0, 1)
            axact.GetZAxisTipProperty().SetColor(1, 0, 0)
            bc = numpy.array(self.renderer.GetBackground())
            if numpy.sum(bc) < 1.5:
                lc = (1, 1, 1)
            else:
                lc = (0, 0, 0)
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
            axact.PickableOff()
            self.addIcon(axact, size=0.1)

        elif self.axes == 5:
            axact = vtk.vtkAnnotatedCubeActor()
            axact.GetCubeProperty().SetColor(.75, .75, .75)
            axact.SetTextEdgesVisibility(0)
            axact.SetFaceTextScale(.4)
            axact.GetXPlusFaceProperty().SetColor(colors.getColor('b'))
            axact.GetXMinusFaceProperty().SetColor(colors.getColor('db'))
            axact.GetYPlusFaceProperty().SetColor(colors.getColor('g'))
            axact.GetYMinusFaceProperty().SetColor(colors.getColor('dg'))
            axact.GetZPlusFaceProperty().SetColor(colors.getColor('r'))
            axact.GetZMinusFaceProperty().SetColor(colors.getColor('dr'))
            axact.PickableOff()
            self.addIcon(axact, size=.06)

        elif self.axes == 6:
            ocf = vtk.vtkOutlineCornerFilter()
            ocf.SetCornerFactor(0.1)
            largestact, sz = None, -1
            for a in self.actors:
                if a.GetPickable():
                    d = a.diagonalSize()
                    if sz < d:
                        largestact = a
                        sz = d
            if isinstance(largestact, Assembly):
                ocf.SetInputData(largestact.getActor(0).polydata())
            else:
                ocf.SetInputData(largestact.polydata())
            ocf.Update()
            ocMapper = vtk.vtkHierarchicalPolyDataMapper()
            ocMapper.SetInputConnection(0, ocf.GetOutputPort(0))
            ocActor = vtk.vtkActor()
            ocActor.SetMapper(ocMapper)
            bc = numpy.array(self.renderer.GetBackground())
            if numpy.sum(bc) < 1.5:
                lc = (1, 1, 1)
            else:
                lc = (0, 0, 0)
            ocActor.GetProperty().SetColor(lc)
            ocActor.PickableOff()
            self.renderer.AddActor(ocActor)

        elif self.axes == 7:
            # draws a simple ruler at the bottom of the window
            ls = vtk.vtkLegendScaleActor()
            ls.RightAxisVisibilityOff()
            ls.TopAxisVisibilityOff()
            ls.LegendVisibilityOff()
            ls.LeftAxisVisibilityOff()
            ls.GetBottomAxis().SetNumberOfMinorTicks(1)
            ls.GetBottomAxis().GetProperty().SetColor(0, 0, 0)
            ls.GetBottomAxis().GetLabelTextProperty().SetColor(0, 0, 0)
            ls.GetBottomAxis().GetLabelTextProperty().BoldOff()
            ls.GetBottomAxis().GetLabelTextProperty().ItalicOff()
            ls.GetBottomAxis().GetLabelTextProperty().ShadowOff()
            ls.PickableOff()
            self.renderer.AddActor(ls)

        elif self.axes == 8:
            ca = vtk.vtkCubeAxesActor()
            ca.SetBounds(vbb)
            if self.camera:
                ca.SetCamera(self.camera)
            else:
                ca.SetCamera(self.renderer.GetActiveCamera())
            ca.GetXAxesLinesProperty().SetColor(c)
            ca.GetYAxesLinesProperty().SetColor(c)
            ca.GetZAxesLinesProperty().SetColor(c)
            for i in range(3):
                ca.GetLabelTextProperty(i).SetColor(c)
                ca.GetTitleTextProperty(i).SetColor(c)
            ca.SetTitleOffset(5)
            ca.SetFlyMode(3)
            ca.SetXTitle(self.xtitle)
            ca.SetYTitle(self.ytitle)
            ca.SetZTitle(self.ztitle)
            if self.xtitle == '':
                ca.SetXAxisVisibility(0)
                ca.XAxisLabelVisibilityOff()
            if self.ytitle == '':
                ca.SetYAxisVisibility(0)
                ca.YAxisLabelVisibilityOff()
            if self.ztitle == '':
                ca.SetZAxisVisibility(0)
                ca.ZAxisLabelVisibilityOff()
            ca.PickableOff()
            self.renderer.AddActor(ca)
            self.axes_exist[r] = ca
            return

        elif self.axes == 9:
            src = vtk.vtkCubeSource()
            src.SetXLength(vbb[1]-vbb[0])
            src.SetYLength(vbb[3]-vbb[2])
            src.SetZLength(vbb[5]-vbb[4])
            src.Update()
            ca = Actor(src.GetOutput(), c=c, alpha=0.5, wire=1)
            ca.pos((vbb[0]+vbb[1])/2,
                   (vbb[3]+vbb[2])/2,
                   (vbb[5]+vbb[4])/2)
            ca.PickableOff()
            self.renderer.AddActor(ca)

        else:
            colors.printc('Keyword axes must be in range [0-9].', c=1)
            colors.printc('''Available axes types:
      0 = no axes,
      1 = draw three gray grid walls
      2 = show cartesian axes from (0,0,0)
      3 = show positive range of cartesian axes from (0,0,0)
      4 = show a triad at bottom left
      5 = show a cube at bottom left
      6 = mark the corners of the bounding box
      7 = draw a simple ruler at the bottom of the window
      8 = show the vtkCubeAxesActor object
      9 = show the bounding box outline''', c=1, bold=0)

        self.axes_exist[r] = True
        return


    def _draw_legend(self):
        if not utils.isSequence(self._legend):
            return
        if len(self.renderers) > 4:
            return

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
            if i < len(self._legend) and self._legend[i] != '':
                if isinstance(self._legend[i], str):
                    texts.append(self._legend[i])
                    acts.append(a)
            elif hasattr(a, '_legend') and a._legend:
                if isinstance(a._legend, str):
                    texts.append(a._legend)
                    acts.append(a)

        NT = len(texts)
        if NT > 25:
            NT = 25
        vtklegend = vtk.vtkLegendBoxActor()
        vtklegend.SetNumberOfEntries(NT)
        for i in range(NT):
            ti = texts[i]
            a = acts[i]
            c = a.GetProperty().GetColor()
            if c == (1, 1, 1):
                c = (0.2, 0.2, 0.2)
            vtklegend.SetEntry(i, a.polydata(), "  "+ti, c)
        pos = self.legendPos
        width = self.legendSize
        vtklegend.SetWidth(width)
        vtklegend.SetHeight(width/5.*NT)
        sx, sy = 1-width, 1-width/5.*NT
        if pos == 1:
            vtklegend.GetPositionCoordinate().SetValue(0, sy)
        elif pos == 2:
            vtklegend.GetPositionCoordinate().SetValue(sx, sy)  # default
        elif pos == 3:
            vtklegend.GetPositionCoordinate().SetValue(0,  0)
        elif pos == 4:
            vtklegend.GetPositionCoordinate().SetValue(sx,  0)
        vtklegend.UseBackgroundOn()
        vtklegend.SetBackgroundColor(self.legendBC)
        vtklegend.SetBackgroundOpacity(0.6)
        vtklegend.LockBorderOn()
        self.renderer.AddActor(vtklegend)
        

    #################################################################################
    def show(self, actors=None, at=None, legend=None, axes=None,
             c=None, alpha=None, wire=False, bc=None,
             resetcam=True, zoom=False, interactive=None, rate=None,
             viewup='', azimuth=0, elevation=0, roll=0,
             interactorStyle=0, q=False):
        '''
        Render a list of actors.

        Allowed input objects are: ``filename``, ``vtkPolyData``, ``vtkActor``, 
        ``vtkActor2D``, ``vtkImageActor``, ``vtkAssembly`` or ``vtkVolume``.

        If filename is given, its type is guessed based on its extension.
        Supported formats are: 
        `vtu, vts, vtp, ply, obj, stl, 3ds, xml, neutral, gmsh, pcd, xyz, txt, byu,
        tif, slc, vti, mhd, png, jpg`.

        :param int at: number of the renderer to plot to, if more than one exists
        :param legend: a string or list of string for each actor, if False will not show it.
        :param int axes: set the type of axes to be shown

              - 0,  no axes,
              - 1,  draw three gray grid walls
              - 2,  show cartesian axes from (0,0,0)
              - 3,  show positive range of cartesian axes from (0,0,0)
              - 4,  show a triad at bottom left
              - 5,  show a cube at bottom left
              - 6,  mark the corners of the bounding box
              - 7,  draw a simple ruler at the bottom of the window
              - 8,  show the ``vtkCubeAxesActor`` object,
              - 9,  show the bounding box outLine,

        :param c:     surface color, in rgb, hex or name formats
        :param bc:    set a color for the internal surface face
        :param bool wire:  show actor in wireframe representation
        :param float azimuth/elevation/roll:  move camera accordingly
        :param str viewup:  either ['x', 'y', 'z'] or a vector to set vertical direction
        :param bool resetcam:  re-adjust camera position to fit objects
        :param bool interactive:  pause and interact with window (True) 
            or continue execution (False)
        :param float rate:  maximum rate of `show()` in Hertz
        :param int interactorStyle: set the type of interaction

            - 0, TrackballCamera
            - 1, TrackballActor
            - 2, JoystickCamera
            - 3, Unicam
            - 4, Flight
            - 5, RubberBand3D
            - 6, RubberBandZoom

        :param bool q:  force program to quit after `show()` command returns.
        '''

        if self.offscreen:
            interactive = False
            self.interactive = False

        def scan(wannabeacts):
            scannedacts = []
            if not utils.isSequence(wannabeacts):
                wannabeacts = [wannabeacts]
            for a in wannabeacts:  # scan content of list
                if isinstance(a, vtk.vtkActor):
                    if c is not None:
                        a.GetProperty().SetColor(colors.getColor(c))
                    if alpha is not None:
                        a.GetProperty().SetOpacity(alpha)
                    if wire:
                        a.GetProperty().SetRepresentationToWireframe()
                    if bc:  # defines a specific color for the backface
                        backProp = vtk.vtkProperty()
                        backProp.SetDiffuseColor(colors.getColor(bc))
                        if alpha is not None:
                            backProp.SetOpacity(alpha)
                        a.SetBackfaceProperty(backProp)
                    scannedacts.append(a)
                    if hasattr(a, 'trail') and a.trail and not a.trail in self.actors:
                        scannedacts.append(a.trail)
                elif isinstance(a, vtk.vtkAssembly):
                    scannedacts.append(a)
                    if a.trail and not a.trail in self.actors:
                        scannedacts.append(a.trail)
                elif isinstance(a, vtk.vtkActor2D):
                    scannedacts.append(a)
                elif isinstance(a, vtk.vtkImageActor):
                    scannedacts.append(a)
                elif isinstance(a, vtk.vtkVolume):
                    scannedacts.append(a)
                elif isinstance(a, vtk.vtkPolyData):
                    out = self.load(a, c, alpha, wire, bc, False)
                    self.actors.pop()
                    scannedacts.append(out)
                elif isinstance(a, str):  # assume a filepath was given
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

        if actors is not None:
            self.actors = []
            actors2show = scan(actors)
            for a in actors2show:
                if a not in self.actors:
                    self.actors.append(a)
        else:
            actors2show = scan(self.actors)
            self.actors = list(actors2show)

        if legend:
            if utils.isSequence(legend):
                self._legend = list(legend)
            elif isinstance(legend,  str):
                self._legend = [str(legend)]
            else:
                colors.printc(
                    'Error in show(): legend must be list or string.', c=1)
                sys.exit()

        if not (axes is None):
            self.axes = axes

        if not (interactive is None):
            self.interactive = interactive

        if at is None and len(self.renderers) > 1:
            # in case of multiple renderers a call to show w/o specifing
            # at which renderer will just render the whole thing and return
            if self.interactor:
                if zoom:
                    self.camera.Zoom(zoom)
                self.interactor.Render()
                if self.interactive:
                    self.interactor.Start()
                return

        if at is None:
            at = 0

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
            for r in self.renderers:
                r.SetActiveCamera(self.camera)

        if len(self.renderers) == 1:
            self.renderer.SetActiveCamera(self.camera)

        # rendering
        for ia in actors2show:        # add the actors that are not already in scene
            if ia:
                if isinstance(ia, vtk.vtkVolume):
                    self.renderer.AddVolume(ia)
                else:
                    self.renderer.AddActor(ia)
            else:
                colors.printc(
                    'Warning: Invalid actor in actors list, skip.', c=5)
        # remove the ones that are not in actors2show
        for ia in self.getActors(at):
            if ia not in actors2show:
                self.renderer.RemoveActor(ia)

        if self.axes:
            self.drawAxes()
        self._draw_legend()

        if resetcam or self.initializedIren == False:
            self.renderer.ResetCamera()

        if not self.initializedIren and self.interactor:
            self.initializedIren = True
            self.interactor.Initialize()
            self.interactor.RemoveObservers('CharEvent')

            if self.verbose and self.interactive:
                docs.onelinetip()

        self.initializedPlotter = True

        if zoom:
            self.camera.Zoom(zoom)
        if azimuth:
            self.camera.Azimuth(azimuth)
        if elevation:
            self.camera.Elevation(elevation)
        if roll:
            self.camera.Roll(roll)

        if len(viewup):
            if viewup == 'x':
                viewup = [1, 0, 0]
            elif viewup == 'y':
                viewup = [0, 1, 0]
            elif viewup == 'z':
                viewup = [0, 0, 1]
                self.camera.Azimuth(60)
                self.camera.Elevation(30)
            self.camera.Azimuth(0.01)  # otherwise camera gets stuck
            self.camera.SetViewUp(viewup)

        self.renderer.ResetCameraClippingRange()

        self.renderWin.Render()

        scbflag = False
        for a in self.actors:
            if hasattr(a, 'scalarbar') and a.scalarbar is not None and utils.isSequence(a.scalarbar):
                if len(a.scalarbar) == 3:  # addScalarBar
                    s1, s2, s3 = a.scalarbar
                    sb = self.addScalarBar(a, s1, s2, s3)
                    scbflag = True
                    a.scalarbar = sb  # save scalarbar actor
                elif len(a.scalarbar) == 10:  # addScalarBar3D
                    s0, s1, s2, s3, s4, s5, s6, s7, s8 = a.scalarbar
                    sb = self.addScalarBar3D(a, at, s0, s1, s2, s3, s4, s5, s6, s7, s8)
                    scbflag = True
                    a.scalarbar = sb  # save scalarbar actor
        if scbflag:
            self.renderWin.Render()

        if settings.allowInteraction and not self.offscreen:
            self.allowInteraction()

        if settings.interactorStyle is not None:
            interactorStyle = settings.interactorStyle

        if interactorStyle == 0 or interactorStyle == 'TrackballCamera':
            pass
        elif interactorStyle == 1 or interactorStyle == 'TrackballActor':
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
        elif interactorStyle == 2 or interactorStyle == 'JoystickCamera':
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleJoystickCamera())
        elif interactorStyle == 3 or interactorStyle == 'Unicam':
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleUnicam())
        elif interactorStyle == 4 or interactorStyle == 'Flight':
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleFlight())
        elif interactorStyle == 5 or interactorStyle =='RubberBand3D':
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBand3D())
        elif interactorStyle == 6 or interactorStyle == 'RubberBandZoom':
            self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandZoom())

        if self.interactor and self.interactive:
            self.interactor.Start()

        if rate:
            if self.clock is None:  # set clock and limit rate
                self._clockt0 = time.time()
                self.clock = 0.
            else:
                t = time.time() - self._clockt0
                elapsed = t - self.clock
                mint = 1./rate
                if elapsed < mint:
                    time.sleep(mint-elapsed)
                self.clock = time.time() - self._clockt0

        if q:  # gracefully exit
            if self.verbose:
                print('q flag set to True. Exit.')
            sys.exit(0)


    def lastActor(self):
        '''Return last added ``Actor``.'''
        return self.actors[-1]

    def removeActor(self, a):
        '''Remove ``vtkActor`` or actor index from current renderer.'''
        try:
            if not self.initializedPlotter:
                save_int = self.interactive
                self.show(interactive=0)
                self.interactive = save_int
                return
            if self.renderer:
                self.renderer.RemoveActor(a)
            i = self.actors.index(a)
            del self.actors[i]
        except:
            pass

    def clear(self, actors=()):
        """Delete specified list of actors, by default delete all."""
        if len(actors):
            for a in actors:
                self.removeActor(a)
        else:
            for a in self.getActors():
                self.renderer.RemoveActor(a)
            self.actors = []

    def Video(self, name='movie.avi', fps=12, duration=None):
        '''Open a video file.

        :param int fps: set the number of frames per second.
        :param flaot duration: set the total `duration` of the video and 
            recalculates `fps` accordingly.

        .. hint:: |makeVideo| |makeVideo.py|_
        '''
        return vtkio.Video(self.renderWin, name, fps, duration)

    def screenshot(self, filename='screenshot.png'):
        '''Save a screenshot of the current rendering window.'''
        vtkio.screenshot(self.renderWin, filename)
