from __future__ import division, print_function
import vtk
from vedo.addons import addScalarBar
from vedo.plotter import Plotter
from vedo.pyplot import cornerHistogram
from vedo.utils import mag, precision, linInterpolate, isSequence
from vedo.colors import printc, colorMap, getColor
from vedo.shapes import Text2D
from vedo import settings
import numpy as np

__all__ = ["Slicer", "Slicer2d", 'RayCaster',
           'IsosurfaceBrowser', 'Browser']

# globals
_cmap_slicer='gist_ncar_r'
_alphaslider0, _alphaslider1, _alphaslider2 = 0.33, 0.66, 1  # defaults
_kact=0

##########################################################################
def Slicer(volume,
           alpha=1,
           cmaps=('gist_ncar_r', "hot_r", "bone_r", "jet", "Spectral_r"),
           map2cells=False,  # buggy
           clamp=True,
           useSlider3D=False,
           size=(850,700),
           screensize="auto",
           title="",
           bg="white",
           bg2="lightblue",
           axes=1,
           showHisto=True,
           showIcon=True,
           draggable=False,
           verbose=True,
           ):
    """
    Generate a ``Plotter`` window with slicing planes for the input Volume.
    Returns the ``Plotter`` object.

    :param float alpha: transparency of the slicing planes
    :param list cmaps: list of color maps names to cycle when clicking button
    :param bool map2cells: scalars are mapped to cells, not intepolated.
    :param bool clamp: clamp scalar to reduce the effect of tails in color mapping
    :param bool useSlider3D: show sliders attached along the axes
    :param list size: rendering window size in pixels
    :param list screensize: size of the screen can be specified
    :param str title: window title
    :param bg: background color
    :param bg2: background gradient color
    :param int axes: axis type number
    :param bool showHisto: show histogram on bottom left
    :param bool showIcon: show a small 3D rendering icon of the volume
    :param bool draggable: make the icon draggable
    """
    global _cmap_slicer

    if verbose: printc("Slicer tool", invert=1, c="m")
    ################################
    vp = Plotter(bg=bg, bg2=bg2,
                 size=size,
                 screensize=screensize,
                 title=title,
                 interactive=False,
                 verbose=verbose)

    ################################
    box = volume.box().wireframe().alpha(0)

    vp.show(box, viewup="z", axes=axes)
    if showIcon:
        vp.showInset(volume, pos=(.85,.85), size=0.15, c='w', draggable=draggable)

    # inits
    la, ld = 0.7, 0.3 #ambient, diffuse
    dims = volume.dimensions()
    data = volume.getPointArray()
    rmin, rmax = volume.imagedata().GetScalarRange()
    if clamp:
        hdata, edg = np.histogram(data, bins=50)
        logdata = np.log(hdata+1)
        # mean  of the logscale plot
        meanlog = np.sum(np.multiply(edg[:-1], logdata))/np.sum(logdata)
        rmax = min(rmax, meanlog+(meanlog-rmin)*0.9)
        rmin = max(rmin, meanlog-(rmax-meanlog)*0.9)
        if verbose:
            printc('scalar range clamped to: (' +
                   precision(rmin, 3) +', '+  precision(rmax, 3)+')', c='m', bold=0)
    _cmap_slicer = cmaps[0]
    visibles = [None, None, None]
    msh = volume.zSlice(int(dims[2]/2))
    msh.alpha(alpha).lighting('', la, ld, 0)
    msh.cmap(_cmap_slicer, vmin=rmin, vmax=rmax)
    if map2cells: msh.mapPointsToCells()
    vp.renderer.AddActor(msh)
    visibles[2] = msh
    addScalarBar(msh, pos=(0.04,0.0), horizontal=True, titleFontSize=0)

    def sliderfunc_x(widget, event):
        i = int(widget.GetRepresentation().GetValue())
        msh = volume.xSlice(i).alpha(alpha).lighting('', la, ld, 0)
        msh.cmap(_cmap_slicer, vmin=rmin, vmax=rmax)
        if map2cells: msh.mapPointsToCells()
        vp.renderer.RemoveActor(visibles[0])
        if i and i<dims[0]: vp.renderer.AddActor(msh)
        visibles[0] = msh

    def sliderfunc_y(widget, event):
        i = int(widget.GetRepresentation().GetValue())
        msh = volume.ySlice(i).alpha(alpha).lighting('', la, ld, 0)
        msh.cmap(_cmap_slicer, vmin=rmin, vmax=rmax)
        if map2cells: msh.mapPointsToCells()
        vp.renderer.RemoveActor(visibles[1])
        if i and i<dims[1]: vp.renderer.AddActor(msh)
        visibles[1] = msh

    def sliderfunc_z(widget, event):
        i = int(widget.GetRepresentation().GetValue())
        msh = volume.zSlice(i).alpha(alpha).lighting('', la, ld, 0)
        msh.cmap(_cmap_slicer, vmin=rmin, vmax=rmax)
        if map2cells: msh.mapPointsToCells()
        vp.renderer.RemoveActor(visibles[2])
        if i and i<dims[2]: vp.renderer.AddActor(msh)
        visibles[2] = msh

    cx, cy, cz, ch = 'dr', 'dg', 'db', (0.3,0.3,0.3)
    if np.sum(vp.renderer.GetBackground()) < 1.5:
        cx, cy, cz = 'lr', 'lg', 'lb'
        ch = (0.8,0.8,0.8)

    if not useSlider3D:
        vp.addSlider2D(sliderfunc_x, 0, dims[0], title='X', titleSize=0.5,
                       pos=[(0.8,0.12), (0.95,0.12)], showValue=False, c=cx)
        vp.addSlider2D(sliderfunc_y, 0, dims[1], title='Y', titleSize=0.5,
                       pos=[(0.8,0.08), (0.95,0.08)], showValue=False, c=cy)
        vp.addSlider2D(sliderfunc_z, 0, dims[2], title='Z', titleSize=0.6,
                       value=int(dims[2]/2),
                       pos=[(0.8,0.04), (0.95,0.04)], showValue=False, c=cz)
    else: # 3d sliders attached to the axes bounds
        bs = box.bounds()
        vp.addSlider3D(sliderfunc_x,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[1], bs[2], bs[4]),
            xmin=0, xmax=dims[0],
            t=box.diagonalSize()/mag(box.xbounds())*0.6,
            c=cx,
            showValue=False,
        )
        vp.addSlider3D(sliderfunc_y,
            pos1=(bs[1], bs[2], bs[4]),
            pos2=(bs[1], bs[3], bs[4]),
            xmin=0, xmax=dims[1],
            t=box.diagonalSize()/mag(box.ybounds())*0.6,
            c=cy,
            showValue=False,
        )
        vp.addSlider3D(sliderfunc_z,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[0], bs[2], bs[5]),
            xmin=0, xmax=dims[2],
            value=int(dims[2]/2),
            t=box.diagonalSize()/mag(box.zbounds())*0.6,
            c=cz,
            showValue=False,
        )


    #################
    def buttonfunc():
        global _cmap_slicer
        bu.switch()
        _cmap_slicer = bu.status()
        for mesh in visibles:
            if mesh:
                mesh.cmap(_cmap_slicer, vmin=rmin, vmax=rmax)
                if map2cells:
                    mesh.mapPointsToCells()
        vp.renderer.RemoveActor(mesh.scalarbar)
        mesh.scalarbar = addScalarBar(mesh,
                                      pos=(0.04,0.0),
                                      horizontal=True,
                                      titleFontSize=0)
        vp.renderer.AddActor(mesh.scalarbar)

    bu = vp.addButton(buttonfunc,
        pos=(0.27, 0.005),
        states=cmaps,
        c=["db"]*len(cmaps),
        bc=["lb"]*len(cmaps),  # colors of states
        size=14,
        bold=True,
    )

    #################
    hist = None
    if showHisto:
        hist = cornerHistogram(data, s=0.2,
                               bins=25, logscale=1, pos=(0.02, 0.02),
                               c=ch, bg=ch, alpha=0.7)

    comment = None
    if verbose:
        comment = Text2D("Use sliders to slice volume\nClick button to change colormap",
                         font='', s=0.8)

    vp.show(msh, hist, comment, interactive=False)
    vp.interactive = True
    if verbose:
        printc("Press button to cycle through color maps,", c="m")
        printc("Use sliders to select the slicing planes.", c="m")
    return vp


########################################################################################
def Slicer2d(volume, size=(900,900), bg=(0.6,0.6,0.7), zoom=1.3):
    """Create a 2D window with a single balck a nd white
    slice of a Volume, wich can be oriented arbitrarily in space.
    """
    img = volume.imagedata()

    ren1 = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    im = vtk.vtkImageResliceMapper()
    im.SetInputData(img)
    im.SliceFacesCameraOn()
    im.SliceAtFocalPointOn()
    im.BorderOn()

    ip = vtk.vtkImageProperty()
    ip.SetInterpolationTypeToLinear()

    ia = vtk.vtkImageSlice()
    ia.SetMapper(im)
    ia.SetProperty(ip)

    ren1.AddViewProp(ia)
    ren1.SetBackground(bg)
    renWin.SetSize(size)

    iren = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleImage()
    style.SetInteractionModeToImage3D()
    iren.SetInteractorStyle(style)
    renWin.SetInteractor(iren)

    renWin.Render()
    cam1 = ren1.GetActiveCamera()
    cam1.ParallelProjectionOn()
    ren1.ResetCameraClippingRange()
    cam1.Zoom(zoom)
    renWin.Render()

    printc("Slicer2D tool", invert=1, c="m")
    printc(
        """Press  SHIFT+Left mouse    to rotate the camera for oblique slicing
       SHIFT+Middle mouse  to slice perpendicularly through the image
       Left mouse and Drag to modify luminosity and contrast
       X                   to Reset to sagittal view
       Y                   to Reset to coronal view
       Z                   to Reset to axial view
       R                   to Reset the Window/Levels
       Q                   to Quit.""",
        c="m",
    )

    iren.Start()
    return iren



########################################################################
def RayCaster(volume):
    """
    Generate a ``Plotter`` window for Volume rendering using ray casting.
    Returns the ``Plotter`` object.
    """
    vp = settings.plotter_instance
    if not vp:
        vp = Plotter(axes=4, bg='bb')

    volumeProperty = volume.GetProperty()
    img = volume.imagedata()

    if volume.dimensions()[2]<3:
        print("Error in raycaster: not enough depth", volume.dimensions())
        return vp
    printc("GPU Ray-casting tool", c="b", invert=1)

    smin, smax = img.GetScalarRange()

    x0alpha = smin + (smax - smin) * 0.25
    x1alpha = smin + (smax - smin) * 0.5
    x2alpha = smin + (smax - smin) * 1.0

    ############################## color map slider
    # Create transfer mapping scalar value to color
    cmaps = ["jet",
            "viridis",
            "bone",
            "hot",
            "plasma",
            "winter",
            "cool",
            "gist_earth",
            "coolwarm",
            "tab10",
            ]
    cols_cmaps = []
    for cm in cmaps:
        cols = colorMap(range(0, 21), cm, 0, 20)  # sample 20 colors
        cols_cmaps.append(cols)
    Ncols = len(cmaps)
    csl = (0.9, 0.9, 0.9)
    if sum(getColor(vp.renderer.GetBackground())) > 1.5:
        csl = (0.1, 0.1, 0.1)

    def sliderColorMap(widget, event):
        sliderRep = widget.GetRepresentation()
        k = int(sliderRep.GetValue())
        sliderRep.SetTitleText(cmaps[k])
        volume.color(cmaps[k])

    w1 = vp.addSlider2D(
        sliderColorMap,
        0,
        Ncols - 1,
        value=0,
        showValue=0,
        title=cmaps[0],
        c=csl,
        pos=[(0.8, 0.05), (0.965, 0.05)],
    )
    w1.GetRepresentation().SetTitleHeight(0.018)

    ############################## alpha sliders
    # Create transfer mapping scalar value to opacity
    opacityTransferFunction = volumeProperty.GetScalarOpacity()

    def setOTF():
        opacityTransferFunction.RemoveAllPoints()
        opacityTransferFunction.AddPoint(smin, 0.0)
        opacityTransferFunction.AddPoint(smin + (smax - smin) * 0.1, 0.0)
        opacityTransferFunction.AddPoint(x0alpha, _alphaslider0)
        opacityTransferFunction.AddPoint(x1alpha, _alphaslider1)
        opacityTransferFunction.AddPoint(x2alpha, _alphaslider2)

    setOTF()

    def sliderA0(widget, event):
        global _alphaslider0
        _alphaslider0 = widget.GetRepresentation().GetValue()
        setOTF()

    vp.addSlider2D(sliderA0, 0, 1,
                    value=_alphaslider0,
                    pos=[(0.84, 0.1), (0.84, 0.26)],
                    c=csl, showValue=0)

    def sliderA1(widget, event):
        global _alphaslider1
        _alphaslider1 = widget.GetRepresentation().GetValue()
        setOTF()

    vp.addSlider2D(sliderA1, 0, 1,
                    value=_alphaslider1,
                    pos=[(0.89, 0.1), (0.89, 0.26)],
                    c=csl, showValue=0)

    def sliderA2(widget, event):
        global _alphaslider2
        _alphaslider2 = widget.GetRepresentation().GetValue()
        setOTF()

    w2 = vp.addSlider2D(sliderA2, 0, 1,
                        value=_alphaslider2,
                        pos=[(0.96, 0.1), (0.96, 0.26)],
                        c=csl, showValue=0,
                        title="Opacity levels")
    w2.GetRepresentation().SetTitleHeight(0.016)

    # add a button
    def buttonfuncMode():
        s = volume.mode()
        snew = (s + 1) % 2
        volume.mode(snew)
        bum.switch()

    bum = vp.addButton(
        buttonfuncMode,
        pos=(0.7, 0.035),
        states=["composite", "max proj."],
        c=["bb", "gray"],
        bc=["gray", "bb"],  # colors of states
        font="arial",
        size=16,
        bold=0,
        italic=False,
    )
    bum.status(volume.mode())

    def CheckAbort(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    vp.window.AddObserver("AbortCheckEvent", CheckAbort)

    # add histogram of scalar
    from vedo.pyplot import cornerHistogram

    plot = cornerHistogram(volume.getPointArray(),
        bins=25, logscale=1, c="gray", bg="gray", pos=(0.78, 0.065)
    )
    plot.GetPosition2Coordinate().SetValue(0.197, 0.20, 0)
    plot.GetXAxisActor2D().SetFontFactor(0.7)
    plot.GetProperty().SetOpacity(0.5)

    vp.add(plot)
    vp.add(volume)

    return vp


def IsosurfaceBrowser(volume, c=None, alpha=1, lego=False, cmap='hot', pos=None):
    """
    Generate a ``Plotter`` window for Volume isosurfacing using a slider.
    Returns the ``Plotter`` object.
    """
    vp = settings.plotter_instance
    if not vp:
        vp = Plotter(axes=4, bg='w', title="Isosurface Browser")

    scrange = volume.scalarRange()
    threshold = (scrange[1] - scrange[0]) / 3.0 + scrange[0]

    if lego:
        sliderpos = ((0.79, 0.035), (0.975, 0.035))
        slidertitle = ""
        showval = False
        mesh = volume.legosurface(vmin=threshold, cmap=cmap).alpha(alpha)
        mesh.addScalarBar(horizontal=True)
    else:
        sliderpos = 4
        slidertitle = "threshold"
        showval = True
        mesh = volume.isosurface(threshold)
        mesh.color(c).alpha(alpha)

    if pos is not None:
        sliderpos = pos

    vp.actors = [mesh] + vp.actors

    ############################## threshold slider
    bacts = dict()
    def sliderThres(widget, event):

        prevact = vp.actors[0]
        wval =  widget.GetRepresentation().GetValue()
        wval_2 = precision(wval, 2)
        if wval_2 in bacts.keys():  # reusing the already available mesh
            mesh = bacts[wval_2]
        else:                       # else generate it
            if lego:
                mesh = volume.legosurface(vmin=wval, cmap=cmap)
            else:
                mesh = volume.isosurface(threshold=wval).color(c).alpha(alpha)
            bacts.update({wval_2: mesh}) # store it

        vp.renderer.RemoveActor(prevact)
        vp.renderer.AddActor(mesh)
        vp.actors[0] = mesh

    dr = scrange[1] - scrange[0]
    vp.addSlider2D( sliderThres,
                    scrange[0] + 0.02 * dr,
                    scrange[1] - 0.02 * dr,
                    value=threshold,
                    pos=sliderpos,
                    title=slidertitle,
                    showValue=showval)
    return vp


##############################################################################
def Browser(meshes, sliderpos=((0.55, 0.07),(0.96, 0.07)), c=None):
    """
    Generate a ``Plotter`` window to browse a list of objects using a slider.
    Returns the ``Plotter`` object.
    """

    vp = settings.plotter_instance
    if not vp:
        vp = Plotter(axes=1, bg='white', title="Browser")
    vp.actors = meshes

    # define the slider
    def sliderfunc(widget, event=None):
        k = int(widget.GetRepresentation().GetValue())
        ak = vp.actors[k]
        for a in vp.actors:
            if a == ak:
                a.on()
            else:
                a.off()
        tx = str(k)
        if ak.filename:
            tx = ak.filename.split("/")[-1]
            tx = tx.split("\\")[-1] # windows os
        elif ak.name:
            tx = ak.name
        widget.GetRepresentation().SetTitleText(tx)
        #printc("Browser Mode", c="y", invert=1, end="")
        #if tx:
        #    printc(": showing #", k, tx, " "*abs(40-len(tx))+"\r",
        #           c="y", bold=0, end="")

    printc("Browser Mode", c="y", invert=1, end="")
    printc(" loaded", len(meshes), "objects", c="y", bold=False)

    wid = vp.addSlider2D(sliderfunc, 0.5, len(meshes)-0.5,
                         pos=sliderpos, font='courier', c=c, showValue=False)
    wid.GetRepresentation().SetTitleHeight(0.020)
    sliderfunc(wid) # init call

    return vp


########################################################################
class Animation(Plotter):
    """
    Animate simultaneously various objects
    by specifying event times and durations of different visual effects.

    See examples
    `here <https://github.com/marcomusy/vedo/blob/master/vedo/examples/other>`_.

    |animation1| |animation2|

    N.B.: this is still an experimental feature at the moment.

    A ``Plotter`` derived class that allows to animate simultaneously various objects
    by specifying event times and durations of different visual effects.

    :param float totalDuration: expand or shrink the total duration of video to this value
    :param float timeResolution: in seconds, save a frame at this rate
    :param bool showProgressBar: show the progressbar
    :param str videoFileName: output file name of the video
    :param int videoFPS: desired value of the nr of frames per second.
    """

    def __init__(self, totalDuration=None, timeResolution=0.02, showProgressBar=True,
                 videoFileName='animation.mp4', videoFPS=12):
        Plotter.__init__(self)
        self.verbose = False
        self.resetcam = True

        self.events = []
        self.timeResolution = timeResolution
        self.totalDuration = totalDuration
        self.showProgressBar = showProgressBar
        self.videoFileName = videoFileName
        self.videoFPS = videoFPS
        self.bookingMode = True
        self._inputvalues = []
        self._performers = []
        self._lastT = None
        self._lastDuration = None
        self._lastActs = None
        self.eps = 0.00001


    def _parse(self, objs, t, duration):
        if t is None:
            if self._lastT:
                t = self._lastT
            else:
                t = 0.0
        if duration is None:
            if self._lastDuration:
                duration = self._lastDuration
            else:
                duration = 0.0
        if objs is None:
            if self._lastActs:
                objs = self._lastActs
            else:
                printc('Need to specify actors!', c='r')
                raise RuntimeError

        objs2 = objs

        if isSequence(objs):
            objs2 = objs
        else:
            objs2 = [objs]

        #quantize time steps and duration
        t = int(t/self.timeResolution+0.5)*self.timeResolution
        nsteps =   int(duration/self.timeResolution+0.5)
        duration = nsteps*self.timeResolution

        rng = np.linspace(t, t+duration, nsteps+1)

        self._lastT = t
        self._lastDuration = duration
        self._lastActs = objs2

        for a in objs2:
            if a not in self.actors:
                self.actors.append(a)

        return objs2, t, duration, rng


    def switchOn(self, acts=None, t=None, duration=None):
        """Switch on the input list of meshes."""
        return self.fadeIn(acts, t, 0)

    def switchOff(self, acts=None, t=None, duration=None):
        """Switch off the input list of meshes."""
        return self.fadeOut(acts, t, 0)


    def fadeIn(self, acts=None, t=None, duration=None):
        """Gradually switch on the input list of meshes by increasing opacity."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = linInterpolate(tt, [t,t+duration], [0,1])
                self.events.append((tt, self.fadeIn, acts, alpha))
        else:
            for a in self._performers:
                if a.alpha() >= self._inputvalues:
                    continue
                a.alpha(self._inputvalues)
        return self

    def fadeOut(self, acts=None, t=None, duration=None):
        """Gradually switch off the input list of meshes by increasing transparency."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = linInterpolate(tt, [t,t+duration], [1,0])
                self.events.append((tt, self.fadeOut, acts, alpha))
        else:
            for a in self._performers:
                if a.alpha() <= self._inputvalues:
                    continue
                a.alpha(self._inputvalues)
        return self


    def changeAlphaBetween(self, alpha1, alpha2, acts=None, t=None, duration=None):
        """Gradually change transparency for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = linInterpolate(tt, [t,t+duration], [alpha1, alpha2])
                self.events.append((tt, self.fadeOut, acts, alpha))
        else:
            for a in self._performers:
                a.alpha(self._inputvalues)
        return self


    def changeColor(self,  c, acts=None, t=None, duration=None):
        """Gradually change color for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            col2 = getColor(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    col1 = a.color()
                    r = linInterpolate(tt, [t,t+duration], [col1[0], col2[0]])
                    g = linInterpolate(tt, [t,t+duration], [col1[1], col2[1]])
                    b = linInterpolate(tt, [t,t+duration], [col1[2], col2[2]])
                    inputvalues.append((r,g,b))
                self.events.append((tt, self.changeColor, acts, inputvalues))
        else:
            for i,a in enumerate(self._performers):
                a.color(self._inputvalues[i])
        return self


    def changeBackColor(self, c, acts=None, t=None, duration=None):
        """Gradually change backface color for the input list of meshes.
        An initial backface color should be set in advance."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            col2 = getColor(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    if a.GetBackfaceProperty():
                        col1 = a.backColor()
                        r = linInterpolate(tt, [t,t+duration], [col1[0], col2[0]])
                        g = linInterpolate(tt, [t,t+duration], [col1[1], col2[1]])
                        b = linInterpolate(tt, [t,t+duration], [col1[2], col2[2]])
                        inputvalues.append((r,g,b))
                    else:
                        inputvalues.append(None)
                self.events.append((tt, self.changeBackColor, acts, inputvalues))
        else:
            for i,a in enumerate(self._performers):
                a.backColor(self._inputvalues[i])
        return self


    def changeToWireframe(self, acts=None, t=None):
        """Switch representation to wireframe for the input list of meshes at time `t`."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, None)
            self.events.append((t, self.changeToWireframe, acts, True))
        else:
            for a in self._performers:
                a.wireframe(self._inputvalues)
        return self

    def changeToSurface(self, acts=None, t=None):
        """Switch representation to surface for the input list of meshes at time `t`."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, None)
            self.events.append((t, self.changeToSurface, acts, False))
        else:
            for a in self._performers:
                a.wireframe(self._inputvalues)
        return self


    def changeLineWidth(self, lw, acts=None, t=None, duration=None):
        """Gradually change line width of the mesh edges for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    newlw = linInterpolate(tt, [t,t+duration], [a.lw(), lw])
                    inputvalues.append(newlw)
                self.events.append((tt, self.changeLineWidth, acts, inputvalues))
        else:
            for i,a in enumerate(self._performers):
                a.lw(self._inputvalues[i])
        return self


    def changeLineColor(self, c, acts=None, t=None, duration=None):
        """Gradually change line color of the mesh edges for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            col2 = getColor(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    col1 = a.lineColor()
                    r = linInterpolate(tt, [t,t+duration], [col1[0], col2[0]])
                    g = linInterpolate(tt, [t,t+duration], [col1[1], col2[1]])
                    b = linInterpolate(tt, [t,t+duration], [col1[2], col2[2]])
                    inputvalues.append((r,g,b))
                self.events.append((tt, self.changeLineColor, acts, inputvalues))
        else:
            for i,a in enumerate(self._performers):
                a.lineColor(self._inputvalues[i])
        return self


    def changeLighting(self, style, acts=None, t=None, duration=None):
        """Gradually change the lighting style for the input list of meshes.

        Allowed styles are: [metallic, plastic, shiny, glossy, default].
        """
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            c = (1,1,0.99)
            if   style=='metallic': pars = [0.1, 0.3, 1.0, 10, c]
            elif style=='plastic' : pars = [0.3, 0.4, 0.3,  5, c]
            elif style=='shiny'   : pars = [0.2, 0.6, 0.8, 50, c]
            elif style=='glossy'  : pars = [0.1, 0.7, 0.9, 90, c]
            elif style=='default' : pars = [0.1, 1.0, 0.05, 5, c]
            else:
                printc('Unknown lighting style:', [style], c='r')

            for tt in rng:
                inputvalues = []
                for a in acts:
                    pr = a.GetProperty()
                    aa = pr.GetAmbient()
                    ad = pr.GetDiffuse()
                    asp = pr.GetSpecular()
                    aspp = pr.GetSpecularPower()
                    naa  = linInterpolate(tt, [t,t+duration], [aa,  pars[0]])
                    nad  = linInterpolate(tt, [t,t+duration], [ad,  pars[1]])
                    nasp = linInterpolate(tt, [t,t+duration], [asp, pars[2]])
                    naspp= linInterpolate(tt, [t,t+duration], [aspp,pars[3]])
                    inputvalues.append((naa, nad, nasp, naspp))
                self.events.append((tt, self.changeLighting, acts, inputvalues))
        else:
            for i,a in enumerate(self._performers):
                pr = a.GetProperty()
                vals = self._inputvalues[i]
                pr.SetAmbient(vals[0])
                pr.SetDiffuse(vals[1])
                pr.SetSpecular(vals[2])
                pr.SetSpecularPower(vals[3])
        return self


    def move(self, act=None, pt=(0,0,0), t=None, duration=None, style='linear'):
        """Smoothly change the position of a specific object to a new point in space."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                printc('Error in move(), can move only one object.', c='r')
            cpos = acts[0].pos()
            pt = np.array(pt)
            dv = (pt - cpos)/len(rng)
            for j,tt in enumerate(rng):
                i = j+1
                if 'quad' in style:
                    x = i/len(rng)
                    y = x*x
                    #print(x,y)
                    self.events.append((tt, self.move, acts, cpos+dv*i*y))
                else:
                    self.events.append((tt, self.move, acts, cpos+dv*i))
        else:
            self._performers[0].pos(self._inputvalues)
        return self


    def rotate(self, act=None, axis=(1,0,0), angle=0, t=None, duration=None):
        """Smoothly rotate a specific object by a specified angle and axis."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                printc('Error in rotate(), can move only one object.', c='r')
            for tt in rng:
                ang = angle/len(rng)
                self.events.append((tt, self.rotate, acts, (axis, ang)))
        else:
            ax = self._inputvalues[0]
            if   ax == 'x':
                self._performers[0].rotateX(self._inputvalues[1])
            elif ax == 'y':
                self._performers[0].rotateY(self._inputvalues[1])
            elif ax == 'z':
                self._performers[0].rotateZ(self._inputvalues[1])
        return self


    def scale(self, acts=None, factor=1, t=None, duration=None):
        """Smoothly scale a specific object to a specified scale factor."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                fac = linInterpolate(tt, [t,t+duration], [1, factor])
                self.events.append((tt, self.scale, acts, fac))
        else:
            for a in self._performers:
                a.scale(self._inputvalues)
        return self


    def meshErode(self, act=None, corner=6, t=None, duration=None):
        """Erode a mesh by removing cells that are close to one of the 8 corners
        of the bounding box.
        """
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                printc('Error in meshErode(), can erode only one object.', c='r')
            diag = acts[0].diagonalSize()
            x0,x1, y0,y1, z0,z1 = acts[0].GetBounds()
            corners = [ (x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0),
                        (x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1) ]
            pcl = acts[0].closestPoint(corners[corner])
            dmin = np.linalg.norm(pcl - corners[corner])
            for tt in rng:
                d = linInterpolate(tt, [t,t+duration], [dmin, diag*1.01])
                if d>0:
                    ids = acts[0].closestPoint(corners[corner],
                                               radius=d, returnIds=True)
                    if len(ids) <= acts[0].N():
                        self.events.append((tt, self.meshErode, acts, ids))
        else:
            self._performers[0].deletePoints(self._inputvalues)
        return self


    def moveCamera(self, camstart=None, camstop=None, t=None, duration=None):
        """
        Smoothly move camera between two ``vtkCamera`` objects.
        """
        if self.bookingMode:
            if camstart is None:
                if not self.camera:
                    printc("Error in moveCamera(), no camera exists.")
                    return
                camstart = self.camera
            acts, t, duration, rng = self._parse(None, t, duration)
            p1 = np.array(camstart.GetPosition())
            f1 = np.array(camstart.GetFocalPoint())
            v1 = np.array(camstart.GetViewUp())
            c1 = np.array(camstart.GetClippingRange())
            s1 = camstart.GetDistance()

            p2 = np.array(camstop.GetPosition())
            f2 = np.array(camstop.GetFocalPoint())
            v2 = np.array(camstop.GetViewUp())
            c2 = np.array(camstop.GetClippingRange())
            s2 = camstop.GetDistance()
            for tt in rng:
                np1 = linInterpolate(tt, [t,t+duration], [p1,p2])
                nf1 = linInterpolate(tt, [t,t+duration], [f1,f2])
                nv1 = linInterpolate(tt, [t,t+duration], [v1,v2])
                nc1 = linInterpolate(tt, [t,t+duration], [c1,c2])
                ns1 = linInterpolate(tt, [t,t+duration], [s1,s2])
                inps = (np1, nf1, nv1, nc1, ns1)
                self.events.append((tt, self.moveCamera, acts, inps))
        else:
            if not self.camera:
                return
            np1, nf1, nv1, nc1, ns1 = self._inputvalues
            self.camera.SetPosition(np1)
            self.camera.SetFocalPoint(nf1)
            self.camera.SetViewUp(nv1)
            self.camera.SetClippingRange(nc1)
            self.camera.SetDistance(ns1)


    def play(self):
        """Play the internal list of events and save a video."""

        from vedo import Video
        self.events = sorted(self.events, key=lambda x: x[0])
        self.bookingMode = False

        for a in self.actors: a.alpha(0)

        #if self.showProgressBar:
        #    pb = ProgressBar(0, len(self.events), c='g')

        if self.totalDuration is None:
            self.totalDuration = self.events[-1][0] - self.events[0][0]
        vd = Video(self.videoFileName, fps=self.videoFPS, duration=self.totalDuration)

        ttlast=0
        for e in self.events:

            tt, action, self._performers, self._inputvalues = e
            action(0,0)

            dt = tt-ttlast
            if dt > self.eps:
                self.show(interactive=False, resetcam=self.resetcam)
                vd.addFrame()

                if dt > self.timeResolution+self.eps:
                    vd.pause(dt)

            ttlast = tt

            #if self.showProgressBar:
            #    pb.print('t='+str(int(tt*100)/100)+'s,  '+action.__name__)

        self.show(interactive=False, resetcam=self.resetcam)
        vd.addFrame()

        vd.close()
        self.show(interactive=True, resetcam=self.resetcam)
        self.bookingMode = True




