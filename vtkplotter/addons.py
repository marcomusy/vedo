"""
Additional objects like axes, legends etc..
"""
from __future__ import division, print_function
import vtkplotter.colors as colors
import vtkplotter.shapes as shapes
from vtkplotter.actors import Assembly, Actor
import vtkplotter.utils as utils
import vtkplotter.settings as settings
import vtkplotter.docs as docs
import numpy
import vtk

__doc__ = (
    """
Additional objects like axes, legends etc..
"""
    + docs._defs
)

__all__ = [
        "addLight",
        "addScalarBar",
        "addScalarBar3D",
        "addSlider2D",
        "addSlider3D",
        "addButton",
        "addCutterTool",
        "addIcon",
        "addAxes",
        "addLegend",
        ]

def addLight(
    pos=(1, 1, 1),
    focalPoint=(0, 0, 0),
    deg=90,
    ambient=None,
    diffuse=None,
    specular=None,
    removeOthers=False,
    showsource=False,
):
    """
    Generate a source of light placed at pos, directed to focal point.

    :param focalPoint: focal point, if this is a ``vtkActor`` use its position.
    :type fp: vtkActor, list
    :param deg: aperture angle of the light source
    :param showsource: if `True`, will show a vtk representation
                        of the source of light as an extra actor

    .. hint:: |lights.py|_
    """
    if isinstance(focalPoint, vtk.vtkActor):
        focalPoint = focalPoint.GetPosition()
    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(pos)
    light.SetPositional(1)
    light.SetConeAngle(deg)
    light.SetFocalPoint(focalPoint)
    if diffuse  is not None: light.SetDiffuseColor(colors.getColor(diffuse))
    if ambient  is not None: light.SetAmbientColor(colors.getColor(ambient))
    if specular is not None: light.SetSpecularColor(colors.getColor(specular))
    if showsource:
        lightActor = vtk.vtkLightActor()
        lightActor.SetLight(light)
        settings.plotter_instance.renderer.AddViewProp(lightActor)
    if removeOthers:
        settings.plotter_instance.renderer.RemoveAllLights()
    settings.plotter_instance.renderer.AddLight(light)
    return light


def addScalarBar(actor,
                 pos=(0.8,0.05),
                 title="",
                 titleXOffset=0,
                 titleYOffset=15,
                 titleFontSize=12,
                 nlabels=10,
                 c=None,
                 horizontal=False,
                 vmin=None, vmax=None,
                 ):
    """Add a 2D scalar bar for the specified actor.

    .. hint:: |mesh_coloring| |mesh_coloring.py|_ |scalarbars.py|_
    """
    vp = settings.plotter_instance

    if not hasattr(actor, "mapper"):
        colors.printc("~times addScalarBar(): input is invalid,", type(actor), c=1)
        return None

    if vp and vp.renderer:
        c = (0.9, 0.9, 0.9)
        if numpy.sum(vp.renderer.GetBackground()) > 1.5:
            c = (0.1, 0.1, 0.1)
        if isinstance(actor.scalarbar, vtk.vtkActor):
            vp.renderer.RemoveActor(actor.scalarbar)
        elif isinstance(actor.scalarbar, Assembly):
            for a in actor.scalarbar.getActors():
                vp.renderer.RemoveActor(a)
    if c is None: c = 'gray'

    if isinstance(actor, Actor):
        lut = actor.mapper.GetLookupTable()
        if not lut:
            return None
        vtkscalars = actor.poly.GetPointData().GetScalars()
        if vtkscalars is None:
            vtkscalars = actor.poly.GetCellData().GetScalars()
        if not vtkscalars:
            return None

        rng = list(vtkscalars.GetRange())
        if vmin is not None: rng[0] = vmin
        if vmax is not None: rng[1] = vmax
        actor.mapper.SetScalarRange(rng)

    elif isinstance(actor, 'Volume'):
        # to be implemented
        pass

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
        sb.SetVerticalTitleSeparation(titleYOffset)
#        sb.SetHorizontalTitleSeparation(titleXOffset)
        sb.SetTitleTextProperty(titprop)

    if vtk.vtkVersion().GetVTKMajorVersion() > 7:
        sb.UnconstrainedFontSizeOn()
        sb.FixedAnnotationLeaderLineColorOff()
        sb.DrawAnnotationsOn()
        sb.DrawTickLabelsOn()
    sb.SetMaximumNumberOfColors(256)

    if horizontal:
        sb.SetOrientationToHorizontal()
        sb.SetNumberOfLabels(int((nlabels-1)/2.))
        sb.SetTextPositionToSucceedScalarBar()
        sb.SetPosition(pos)
        sb.SetMaximumWidthInPixels(1000)
        sb.SetMaximumHeightInPixels(50)
    else:
        sb.SetNumberOfLabels(nlabels)
        sb.SetTextPositionToPrecedeScalarBar()
        sb.SetPosition(pos[0]+0.07, pos[1])
        sb.SetMaximumWidthInPixels(80)
        sb.SetMaximumHeightInPixels(500)

    sctxt = sb.GetLabelTextProperty()
    sctxt.SetColor(c)
    sctxt.SetShadow(0)
    sctxt.SetFontFamily(0)
    sctxt.SetItalic(0)
    sctxt.SetBold(0)
    sctxt.SetFontSize(titleFontSize)
    sb.PickableOff()
    actor.scalarbar = sb
    return sb


def addScalarBar3D(
    obj,
    pos=(0, 0, 0),
    normal=(0, 0, 1),
    sx=0.1,
    sy=2,
    title='',
    titleXOffset = -1.5,
    titleYOffset = 0.0,
    titleSize =  1.5,
    titleRotation = 0.0,
    nlabels=9,
    precision=3,
    labelOffset = 0.4,
    c=None,
    alpha=1,
    cmap=None,
):
    """Draw a 3D scalar bar.

    ``obj`` input can be:
        - a list of numbers,
        - a list of two numbers in the form `(min, max)`,
        - a ``vtkActor`` already containing a set of scalars associated to vertices or cells,
        - if ``None`` the last actor in the list of actors will be used.

    :param float sx: thickness of scalarbar
    :param float sy: length of scalarbar
    :param str title: scalar bar title
    :param float titleXOffset: horizontal space btw title and color scalarbar
    :param float titleYOffset: vertical space offset
    :param float titleSize: size of title wrt numeric labels
    :param float titleRotation: title rotation in degrees
    :param int nlabels: number of numeric labels
    :param int precision: number of significant digits
    :param float labelOffset: space btw numeric labels and scale
    :param str cmap: specify cmap to be used

    .. hint:: |scalarbars| |scalarbars.py|_
    """
    vp = settings.plotter_instance
    if vp and c is None:  # automatic black or white
        c = (0.8, 0.8, 0.8)
        if numpy.sum(colors.getColor(vp.backgrcol)) > 1.5:
            c = (0.2, 0.2, 0.2)
    if c is None: c = 'gray'
    c = colors.getColor(c)

    if isinstance(obj, Actor):
        if cmap is None:
            lut = obj.mapper.GetLookupTable()
            if not lut:
                print("Error in ScalarBar3D: actor has no active scalar array.", [obj])
                return None
        else:
            lut = cmap
        vmin,vmax = obj.mapper.GetScalarRange()

    elif utils.isSequence(obj):
        vmin, vmax = numpy.min(obj), numpy.max(obj)
    else:
        print("Error in ScalarBar3D(): input must be Actor or list.", type(obj))
        raise RuntimeError()

    # build the color scale part
    scale = shapes.Grid([-sx *labelOffset, 0, 0], c=c, alpha=alpha,
                        sx=sx, sy=sy, resx=1, resy=256)
    scale.lw(0).wireframe(False)
    cscals = scale.cellCenters()[:, 1]
    scale.cellColors(cscals, lut, alpha)
    scale.lighting(ambient=1, diffuse=0, specular=0, specularPower=0)

    # build text
    tlabs = numpy.linspace(vmin, vmax, num=nlabels, endpoint=True)
    tacts = []
    for i, t in enumerate(tlabs):
        tx = utils.precision(t, precision, vrange=vmax-vmin)
        y = -sy / 1.98 + sy * i / (nlabels - 1)
        a = shapes.Text(tx, pos=[sx*labelOffset, y, 0], s=sy/50, c=c, alpha=alpha, depth=0)
        a.lighting(ambient=1, diffuse=0, specular=0, specularPower=0)
        a.PickableOff()
        tacts.append(a)

    # build title
    if title:
        t = shapes.Text(title, (0,0,0), s=sy/50*titleSize, c=c, alpha=alpha, depth=0,
                        justify='centered')
        t.rotateZ(90+titleRotation).pos(sx*titleXOffset,titleYOffset,0)
        t.lighting(ambient=1, diffuse=0, specular=0, specularPower=0)
        t.PickableOff()
        tacts.append(t)

    sact = Assembly([scale] + tacts)
    normal = numpy.array(normal) / numpy.linalg.norm(normal)
    theta = numpy.arccos(normal[2])
    phi = numpy.arctan2(normal[1], normal[0])
    sact.RotateZ(numpy.rad2deg(phi))
    sact.RotateY(numpy.rad2deg(theta))
    sact.SetPosition(pos)
    sact.PickableOff()
    if isinstance(obj, Actor):
        obj.scalarbar = sact
    return sact


def addSlider2D(sliderfunc, xmin, xmax, value=None, pos=4,
                title='', c=None, showValue=True):
    """Add a slider widget which can call an external custom function.

    :param sliderfunc: external function to be called by the widget
    :param float xmin:  lower value
    :param float xmax:  upper value
    :param float value: current value
    :param list pos:  position corner number: horizontal [1-4] or vertical [11-14]
                        it can also be specified by corners coordinates [(x1,y1), (x2,y2)]
    :param str title: title text
    :param bool showValue:  if true current value is shown

    |sliders| |sliders.py|_
    """
    vp = settings.plotter_instance
    if c is None:  # automatic black or white
        c = (0.8, 0.8, 0.8)
        if numpy.sum(colors.getColor(vp.backgrcol)) > 1.5:
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
    sliderRep.SetTubeWidth(0.0075)
    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    if utils.isSequence(pos):
        sliderRep.GetPoint1Coordinate().SetValue(pos[0][0], pos[0][1])
        sliderRep.GetPoint2Coordinate().SetValue(pos[1][0], pos[1][1])
    elif pos == 1:  # top-left horizontal
        sliderRep.GetPoint1Coordinate().SetValue(0.04, 0.96)
        sliderRep.GetPoint2Coordinate().SetValue(0.45, 0.96)
    elif pos == 2:
        sliderRep.GetPoint1Coordinate().SetValue(0.55, 0.96)
        sliderRep.GetPoint2Coordinate().SetValue(0.96, 0.96)
    elif pos == 3:
        sliderRep.GetPoint1Coordinate().SetValue(0.04, 0.04)
        sliderRep.GetPoint2Coordinate().SetValue(0.45, 0.04)
    elif pos == 4:  # bottom-right
        sliderRep.GetPoint1Coordinate().SetValue(0.55, 0.04)
        sliderRep.GetPoint2Coordinate().SetValue(0.96, 0.04)
    elif pos == 5:  # bottom margin horizontal
        sliderRep.GetPoint1Coordinate().SetValue(0.04, 0.04)
        sliderRep.GetPoint2Coordinate().SetValue(0.96, 0.04)
    elif pos == 11:  # top-left vertical
        sliderRep.GetPoint1Coordinate().SetValue(0.04, 0.54)
        sliderRep.GetPoint2Coordinate().SetValue(0.04, 0.9)
    elif pos == 12:
        sliderRep.GetPoint1Coordinate().SetValue(0.96, 0.54)
        sliderRep.GetPoint2Coordinate().SetValue(0.96, 0.9)
    elif pos == 13:
        sliderRep.GetPoint1Coordinate().SetValue(0.04, 0.1)
        sliderRep.GetPoint2Coordinate().SetValue(0.04, 0.54)
    elif pos == 14:  # bottom-right vertical
        sliderRep.GetPoint1Coordinate().SetValue(0.96, 0.1)
        sliderRep.GetPoint2Coordinate().SetValue(0.96, 0.54)
    elif pos == 15:  # right margin vertical
        sliderRep.GetPoint1Coordinate().SetValue(0.96, 0.1)
        sliderRep.GetPoint2Coordinate().SetValue(0.96, 0.9)

    if showValue:
        if isinstance(xmin, int) and isinstance(xmax, int):
            frm = "%0.0f"
        else:
            frm = "%0.1f"
        sliderRep.SetLabelFormat(frm)  # default is '%0.3g'
        sliderRep.GetLabelProperty().SetShadow(0)
        sliderRep.GetLabelProperty().SetBold(0)
        sliderRep.GetLabelProperty().SetOpacity(1)
        sliderRep.GetLabelProperty().SetColor(c)
        if isinstance(pos, int) and pos > 10:
            sliderRep.GetLabelProperty().SetOrientation(90)
    else:
        sliderRep.ShowSliderLabelOff()
    sliderRep.GetTubeProperty().SetColor(c)
    sliderRep.GetTubeProperty().SetOpacity(0.6)
    sliderRep.GetSliderProperty().SetColor(c)
    sliderRep.GetSelectedProperty().SetColor(0.8, 0, 0)
    sliderRep.GetCapProperty().SetColor(c)

    if title:
        sliderRep.SetTitleText(title)
        sliderRep.SetTitleHeight(0.012)
        sliderRep.GetTitleProperty().SetShadow(0)
        sliderRep.GetTitleProperty().SetColor(c)
        sliderRep.GetTitleProperty().SetOpacity(1)
        sliderRep.GetTitleProperty().SetBold(0)
        if not utils.isSequence(pos):
            if isinstance(pos, int) and pos > 10:
                sliderRep.GetTitleProperty().SetOrientation(90)
        else:
            if abs(pos[0][0] - pos[1][0]) < 0.1:
                sliderRep.GetTitleProperty().SetOrientation(90)

    sliderWidget = vtk.vtkSliderWidget()
    sliderWidget.SetInteractor(vp.interactor)
    sliderWidget.SetAnimationModeToJump()
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.AddObserver("InteractionEvent", sliderfunc)
    sliderWidget.EnabledOn()
    vp.sliders.append([sliderWidget, sliderfunc])
    return sliderWidget


def addSlider3D(
    sliderfunc,
    pos1,
    pos2,
    xmin,
    xmax,
    value=None,
    s=0.03,
    title="",
    rotation=0,
    c=None,
    showValue=True,
):
    """Add a 3D slider widget which can call an external custom function.

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

    |sliders3d| |sliders3d.py|_
    """
    vp = settings.plotter_instance
    if c is None:  # automatic black or white
        c = (0.8, 0.8, 0.8)
        if numpy.sum(colors.getColor(vp.backgrcol)) > 1.5:
            c = (0.2, 0.2, 0.2)
    else:
        c = colors.getColor(c)

    if value is None or value < xmin:
        value = xmin

    t = 1.5 / numpy.sqrt(utils.mag(numpy.array(pos2) - pos1))  # better norm

    sliderRep = vtk.vtkSliderRepresentation3D()
    sliderRep.SetMinimumValue(xmin)
    sliderRep.SetValue(value)
    sliderRep.SetMaximumValue(xmax)

    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToWorld()
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToWorld()
    sliderRep.GetPoint1Coordinate().SetValue(pos2)
    sliderRep.GetPoint2Coordinate().SetValue(pos1)

    sliderRep.SetSliderWidth(0.03 * t)
    sliderRep.SetTubeWidth(0.01 * t)
    sliderRep.SetSliderLength(0.04 * t)
    sliderRep.SetSliderShapeToCylinder()
    sliderRep.GetSelectedProperty().SetColor(1, 0, 0)
    sliderRep.GetSliderProperty().SetColor(numpy.array(c) / 2)
    sliderRep.GetCapProperty().SetOpacity(0)

    sliderRep.SetRotation(rotation)

    if not showValue:
        sliderRep.ShowSliderLabelOff()

    sliderRep.SetTitleText(title)
    sliderRep.SetTitleHeight(s * t)
    sliderRep.SetLabelHeight(s * t * 0.85)

    sliderRep.GetTubeProperty()
    sliderRep.GetTubeProperty().SetColor(c)

    sliderWidget = vtk.vtkSliderWidget()
    sliderWidget.SetInteractor(vp.interactor)
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.SetAnimationModeToJump()
    sliderWidget.AddObserver("InteractionEvent", sliderfunc)
    sliderWidget.EnabledOn()
    vp.sliders.append([sliderWidget, sliderfunc])
    return sliderWidget


def addButton(
    fnc,
    states=("On", "Off"),
    c=("w", "w"),
    bc=("dg", "dr"),
    pos=(20, 40),
    size=24,
    font="arial",
    bold=False,
    italic=False,
    alpha=1,
    angle=0,
):
    """Add a button to the renderer window.

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

    |buttons| |buttons.py|_
    """
    vp = settings.plotter_instance
    if not vp.renderer:
        colors.printc("~timesError: Use addButton() after rendering the scene.", c=1)
        return
    bu = Button(fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle)
    vp.renderer.AddActor2D(bu.actor)
    vp.window.Render()
    vp.buttons.append(bu)
    return bu


def addCutterTool(actor):
    """Create handles to cut away parts of a mesh.

    |cutter| |cutter.py|_
    """
    if isinstance(actor, vtk.vtkVolume):
        return _addVolumeCutterTool(actor)
    elif isinstance(actor, vtk.vtkImageData):
        from vtkplotter import Volume
        return _addVolumeCutterTool(Volume(actor))

    vp = settings.plotter_instance
    if not vp.renderer:
        save_int = vp.interactive
        vp.show(interactive=0)
        vp.interactive = save_int

    vp.clickedActor = actor
    apd = actor.polydata()

    planes = vtk.vtkPlanes()
    planes.SetBounds(apd.GetBounds())

    clipper = vtk.vtkClipPolyData()
    clipper.GenerateClipScalarsOn()
    clipper.SetInputData(apd)
    clipper.SetClipFunction(planes)
    clipper.InsideOutOn()
    clipper.GenerateClippedOutputOn()
    clipper.Update()
    cpoly = clipper.GetOutput()

    act0 = Actor(cpoly, alpha=actor.alpha()) # the main cut part
    act0.mapper.SetLookupTable(actor.mapper.GetLookupTable())
    act0.mapper.SetScalarRange(actor.mapper.GetScalarRange())

    act1 = Actor()
    act1.mapper.SetInputConnection(clipper.GetClippedOutputPort()) # needs OutputPort??
    act1.alpha(0.04).color((0.5,0.5,0.5)).wireframe()

    vp.renderer.RemoveActor(actor)
    vp.renderer.AddActor(act0)
    vp.renderer.AddActor(act1)
    vp.renderer.ResetCamera()

    def selectPolygons(vobj, event):
        vobj.GetPlanes(planes)

    boxWidget = vtk.vtkBoxWidget()
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
    boxWidget.GetOutlineProperty().SetColor(0.2, 0.2, 0.2)
    boxWidget.GetOutlineProperty().SetOpacity(0.8)
    boxWidget.SetPlaceFactor(1.05)
    boxWidget.SetInteractor(vp.interactor)
    boxWidget.SetInputData(apd)
    boxWidget.PlaceWidget()
    boxWidget.AddObserver("InteractionEvent", selectPolygons)
    boxWidget.On()

    vp.cutterWidget = boxWidget
    vp.clickedActor = act0
    if actor in vp.actors:
        ia = vp.actors.index(actor)
        vp.actors[ia] = act0

    colors.printc("Mesh Cutter Tool:", c="m", invert=1)
    colors.printc("  Move gray handles to cut away parts of the mesh", c="m")
    colors.printc("  Press X to save file to: clipped.vtk", c="m")
    vp.interactor.Start()

    boxWidget.Off()
    vp.widgets.append(boxWidget)

    vp.interactor.Start()  # allow extra interaction
    return act0

def _addVolumeCutterTool(vol):
    vp = settings.plotter_instance
    if not vp.renderer:
        save_int = vp.interactive
        vp.show(interactive=0)
        vp.interactive = save_int

    boxWidget = vtk.vtkBoxWidget()
    boxWidget.SetInteractor(vp.interactor)
    boxWidget.SetPlaceFactor(1.0)

    vp.cutterWidget = boxWidget

    planes = vtk.vtkPlanes()
    def clipVolumeRender(obj, event):
        obj.GetPlanes(planes)
        vol.mapper.SetClippingPlanes(planes)
        #vol.mapper.Modified()
        #vol.mapper.Update()

    boxWidget.SetInputData(vol.inputdata())
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
    boxWidget.GetOutlineProperty().SetColor(0.2, 0.2, 0.2)
    boxWidget.GetOutlineProperty().SetOpacity(0.7)
    boxWidget.SetPlaceFactor(1.0)
    boxWidget.PlaceWidget()
    boxWidget.InsideOutOn()
    boxWidget.AddObserver("InteractionEvent", clipVolumeRender)

    colors.printc("Volume Cutter Tool:", c="m", invert=1)
    colors.printc("  Move gray handles to cut parts of the volume", c="m")

    vp.renderer.ResetCamera()
    boxWidget.On()

    vp.interactor.Start()
    boxWidget.Off()
    vp.widgets.append(boxWidget)


def addIcon(iconActor, pos=3, size=0.08):
    """Add an inset icon mesh into the renderer.

    :param pos: icon position in the range [1-4] indicating one of the 4 corners,
                or it can be a tuple (x,y) as a fraction of the renderer size.
    :param float size: size of the icon space.

    |icon| |icon.py|_
    """
    vp = settings.plotter_instance
    if not vp.renderer:
        colors.printc("~lightningWarning: Use addIcon() after first rendering the scene.", c=3)
        save_int = vp.interactive
        vp.show(interactive=0)
        vp.interactive = save_int
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(iconActor)
    widget.SetInteractor(vp.interactor)
    if utils.isSequence(pos):
        widget.SetViewport(pos[0] - size, pos[1] - size, pos[0] + size, pos[1] + size)
    else:
        if pos < 2:
            widget.SetViewport(0, 1 - 2 * size, size * 2, 1)
        elif pos == 2:
            widget.SetViewport(1 - 2 * size, 1 - 2 * size, 1, 1)
        elif pos == 3:
            widget.SetViewport(0, 0, size * 2, size * 2)
        elif pos == 4:
            widget.SetViewport(1 - 2 * size, 0, 1, size * 2)
    widget.EnabledOn()
    widget.InteractiveOff()
    vp.widgets.append(widget)
    if iconActor in vp.actors:
        vp.actors.remove(iconActor)
    return widget

def computeVisibleBounds():
    """Calculate max actors bounds and sizes."""
    bns = []
    for a in settings.plotter_instance.actors:
        if a and a.GetPickable():
            b = a.GetBounds()
            if b:
                bns.append(b)
    if len(bns):
        max_bns = numpy.max(bns, axis=0)
        min_bns = numpy.min(bns, axis=0)
        vbb = (min_bns[0], max_bns[1], min_bns[2], max_bns[3], min_bns[4], max_bns[5])
    else:
        vbb = settings.plotter_instance.renderer.ComputeVisiblePropBounds()
        max_bns = vbb
        min_bns = vbb
    sizes = numpy.array([max_bns[1]-min_bns[0], max_bns[3]-min_bns[2], max_bns[5]-min_bns[4]])
    return vbb, sizes, min_bns, max_bns


def addAxes(axtype=None, c=None):
    """Draw axes on scene. Available axes types:

    :param int axtype:

          - 0,  no axes,
          - 1,  draw three gray grid walls
          - 2,  show cartesian axes from (0,0,0)
          - 3,  show positive range of cartesian axes from (0,0,0)
          - 4,  show a triad at bottom left
          - 5,  show a cube at bottom left
          - 6,  mark the corners of the bounding box
          - 7,  draw a simple ruler at the bottom of the window
          - 8,  show the ``vtkCubeAxesActor`` object
          - 9,  show the bounding box outLine
          - 10, show three circles representing the maximum bounding box

    Axis type-1 can be fully customized by passing a dictionary ``axes=dict()`` where:

        - `xtitle`,            ['x'], x-axis title text.
        - `ytitle`,            ['y'], y-axis title text.
        - `ztitle`,            ['z'], z-axis title text.
        - `numberOfDivisions`, [automatic], number of divisions on the shortest axis
        - `axesLineWidth`,       [1], width of the axes lines
        - `gridLineWidth`,       [1], width of the grid lines
        - `reorientShortTitle`, [True], titles shorter than 2 letter are placed horizontally
        - `originMarkerSize`, [0.01], draw a small cube on the axis where the origin is
        - `enableLastLabel`, [False], show last numeric label on axes
        - `titleDepth`,          [0], extrusion fractional depth of title text
        - `xyGrid`,           [True], show a gridded wall on plane xy
        - `yzGrid`,           [True], show a gridded wall on plane yz
        - `zxGrid`,           [True], show a gridded wall on plane zx
        - `zxGrid2`,         [False], show zx plane on opposite side of the bounding box
        - `xyGridTransparent`  [False], make grid plane completely transparent
        - `xyGrid2Transparent` [False], make grid plane completely transparent on opposite side box
        - `xyPlaneColor`,   ['gray'], color of the plane
        - `xyGridColor`,    ['gray'], grid line color
        - `xyAlpha`,          [0.15], grid plane opacity
        - `showTicks`,        [True], show major ticks
        - `xTitlePosition`,   [0.32], title fractional positions along axis
        - `xTitleOffset`,     [0.05], title fractional offset distance from axis line
        - `xTitleJustify`, ["top-right"], title justification
        - `xTitleRotation`,      [0], add a rotation of the axis title
        - `xLineColor`,  [automatic], color of the x-axis
        - `xTitleColor`, [automatic], color of the axis title
        - `xTitleBackfaceColor`, [None],  color of axis title on its backface
        - `xTitleSize`,      [0.025], size of the axis title
        - `xHighlightZero`,   [True], draw a line highlighting zero position if in range
        - `xHighlightZeroColor`, [automatic], color of the line highlighting the zero position
        - `xTickRadius`,     [0.005], radius of the major ticks
        - `xTickThickness`, [0.0025], thickness of the major ticks along their axis
        - `xTickColor`,  [automatic], color of major ticks
        - `xMinorTicks`,         [1], number of minor ticks between two major ticks
        - `tipSize`,          [0.01], size of the arrow tip
        - `xLabelPrecision`,     [2], nr. of significative digits to be shown
        - `xLabelSize`,      [0.015], size of the numeric labels along axis
        - `xLabelOffset`,    [0.025], offset of numeric labels

        :Example:

            .. code-block:: python

                from vtkplotter import Box, show
                b = Box(pos=(0,0,0), length=80, width=90, height=70).alpha(0)

                show(b, axes={ 'xtitle':'Some long variable [a.u.]',
                               'numberOfDivisions':4,
                               # ...
                             }
                )

    |customAxes| |customAxes.py|_
    """
    vp = settings.plotter_instance
    if axtype is not None:
        vp.axes = axtype  # overrride

    r = vp.renderers.index(vp.renderer)

    if not vp.axes:
        return

    if c is None:  # automatic black or white
        c = (0.9, 0.9, 0.9)
        if numpy.sum(vp.renderer.GetBackground()) > 1.5:
            c = (0.1, 0.1, 0.1)
    else:
        c = colors.getColor(c) # for speed

    if not vp.renderer:
        return

    if vp.axes_instances[r]:
        return

    ############################################################
    if vp.axes == 1 or vp.axes == True or isinstance(vp.axes, dict):  # custom grid walls

        if isinstance(vp.axes, dict):
            axes = vp.axes
            axes_copy = dict(vp.axes)
        else:
            axes = dict() # will cause popping the defaults
            axes_copy = 1

        xtitle = axes.pop('xtitle', vp.xtitle)
        ytitle = axes.pop('ytitle', vp.ytitle)
        ztitle = axes.pop('ztitle', vp.ztitle)

        limitRatio = axes.pop('limitRatio', 20)

        vbb, sizes, min_bns, max_bns = computeVisibleBounds()

        if sizes[0] and (sizes[1]/sizes[0] > limitRatio or sizes[2]/sizes[0] > limitRatio):
            sizes[0] = 0
            xtitle = ''
        if sizes[1] and (sizes[0]/sizes[1] > limitRatio or sizes[2]/sizes[1] > limitRatio):
            sizes[1] = 0
            ytitle = ''
        if sizes[2] and (sizes[0]/sizes[2] > limitRatio or sizes[1]/sizes[2] > limitRatio):
            sizes[2] = 0
            ztitle = ''
        rats = []
        if sizes[0]: rats += [sizes[1]/sizes[0], sizes[2]/sizes[0]]
        if sizes[1]: rats += [sizes[0]/sizes[1], sizes[2]/sizes[1]]
        if sizes[2]: rats += [sizes[0]/sizes[2], sizes[1]/sizes[2]]
        if not len(rats):
            return
        rats = max(rats)
        if rats == 0:
            return

        nrDiv = max(1, int(6.5/rats))

        numberOfDivisions = axes.pop('numberOfDivisions', nrDiv) # number of divisions on the shortest axis

        axesLineWidth = axes.pop('axesLineWidth', 1)
        gridLineWidth = axes.pop('gridLineWidth', 1)
        reorientShortTitle = axes.pop('reorientShortTitle', True)
        originMarkerSize = axes.pop('originMarkerSize', 0)
        enableLastLabel = axes.pop('enableLastLabel', False)
        titleDepth = axes.pop('titleDepth', 0)

        xTitlePosition = axes.pop('xTitlePosition', 0.95) # title fractional positions along axis [0->1]
        yTitlePosition = axes.pop('yTitlePosition', 0.95)
        zTitlePosition = axes.pop('zTitlePosition', 0.95)

        xTitleOffset = axes.pop('xTitleOffset', 0.05)  # title fractional offsets
        yTitleOffset = axes.pop('yTitleOffset', 0.05)  #
        zTitleOffset = axes.pop('zTitleOffset', 0.05)  #

        xTitleJustify = axes.pop('xTitleJustify', "top-right")
        yTitleJustify = axes.pop('yTitleJustify', "bottom-right")
        zTitleJustify = axes.pop('zTitleJustify', "bottom-right")

        xTitleRotation = axes.pop('xTitleRotation', 0)
        yTitleRotation = axes.pop('yTitleRotation', 90)
        zTitleRotation = axes.pop('zTitleRotation', 135)

        xTitleSize = axes.pop('xTitleSize', 0.025)
        yTitleSize = axes.pop('yTitleSize', 0.025)
        zTitleSize = axes.pop('zTitleSize', 0.025)

        xTitleColor = axes.pop('xTitleColor', c)
        yTitleColor = axes.pop('yTitleColor', c)
        zTitleColor = axes.pop('zTitleColor', c)

        xTitleBackfaceColor = axes.pop('xTitleBackfaceColor', None)
        yTitleBackfaceColor = axes.pop('yTitleBackfaceColor', None)
        zTitleBackfaceColor = axes.pop('zTitleBackfaceColor', None)

        xKeepAspectRatio = axes.pop('xKeepAspectRatio', True)
        yKeepAspectRatio = axes.pop('yKeepAspectRatio', True)
        zKeepAspectRatio = axes.pop('zKeepAspectRatio', True)

        xyGrid = axes.pop('xyGrid', True)
        yzGrid = axes.pop('yzGrid', True)
        zxGrid = axes.pop('zxGrid', False)
        xyGrid2 = axes.pop('xyGrid2', False) # opposite side grid
        yzGrid2 = axes.pop('yzGrid2', False)
        zxGrid2 = axes.pop('zxGrid2', False)
        if settings.plotter_instance and settings.plotter_instance.renderer.GetUseDepthPeeling():
            xyGrid = False
            yzGrid = False
            zxGrid = False
            xyGrid2 = False
            yzGrid2 = False
            zxGrid2 = False

        xyGridTransparent = axes.pop('xyGridTransparent', False)
        yzGridTransparent = axes.pop('yzGridTransparent', False)
        zxGridTransparent = axes.pop('zxGridTransparent', False)
        xyGrid2Transparent = axes.pop('xyGrid2Transparent', False)
        yzGrid2Transparent = axes.pop('yzGrid2Transparent', False)
        zxGrid2Transparent = axes.pop('zxGrid2Transparent', False)

        xyPlaneColor = axes.pop('xyPlaneColor', c)
        yzPlaneColor = axes.pop('yzPlaneColor', c)
        zxPlaneColor = axes.pop('zxPlaneColor', c)
        xyGridColor = axes.pop('xyGridColor', c)
        yzGridColor = axes.pop('yzGridColor', c)
        zxGridColor = axes.pop('zxGridColor', c)
        xyAlpha = axes.pop('xyAlpha', 0.05)
        yzAlpha = axes.pop('yzAlpha', 0.05)
        zxAlpha = axes.pop('zxAlpha', 0.05)

        xLineColor = axes.pop('xLineColor', c)
        yLineColor = axes.pop('yLineColor', c)
        zLineColor = axes.pop('zLineColor', c)

        xHighlightZero = axes.pop('xHighlightZero', False)
        yHighlightZero = axes.pop('yHighlightZero', False)
        zHighlightZero = axes.pop('zHighlightZero', False)
        xHighlightZeroColor = axes.pop('xHighlightZeroColor', 'red')
        yHighlightZeroColor = axes.pop('yHighlightZeroColor', 'green')
        zHighlightZeroColor = axes.pop('zHighlightZeroColor', 'blue')

        showTicks = axes.pop('showTicks', True)
        xTickRadius = axes.pop('xTickRadius', 0.005)
        yTickRadius = axes.pop('yTickRadius', 0.005)
        zTickRadius = axes.pop('zTickRadius', 0.005)

        xTickThickness = axes.pop('xTickThickness', 0.0025)
        yTickThickness = axes.pop('yTickThickness', 0.0025)
        zTickThickness = axes.pop('zTickThickness', 0.0025)

        xTickColor = axes.pop('xTickColor', xLineColor)
        yTickColor = axes.pop('yTickColor', yLineColor)
        zTickColor = axes.pop('zTickColor', zLineColor)

        xMinorTicks = axes.pop('xMinorTicks', 1)
        yMinorTicks = axes.pop('yMinorTicks', 1)
        zMinorTicks = axes.pop('zMinorTicks', 1)

        tipSize = axes.pop('tipSize', 0.01)

        xLabelPrecision = axes.pop('xLabelPrecision', 2) # nr. of significant digits
        yLabelPrecision = axes.pop('yLabelPrecision', 2)
        zLabelPrecision = axes.pop('zLabelPrecision', 2)

        xLabelSize = axes.pop('xLabelSize', 0.0175)
        yLabelSize = axes.pop('yLabelSize', 0.0175)
        zLabelSize = axes.pop('zLabelSize', 0.0175)

        xLabelOffset = axes.pop('xLabelOffset', 0.015)
        yLabelOffset = axes.pop('yLabelOffset', 0.015)
        zLabelOffset = axes.pop('zLabelOffset', 0.01)

        ########################
        step = numpy.min(sizes[numpy.nonzero(sizes)]) / numberOfDivisions
        rx, ry, rz = numpy.rint(sizes / step).astype(int)
        if rx==0: xtitle=''
        if ry==0: ytitle=''
        if rz==0: ztitle=''

        if enableLastLabel:
            enableLastLabel = 1
        else:
            enableLastLabel = 0

        ################################################ axes lines
        lines = []
        if xtitle: lines.append(shapes.Line([0, 0, 0], [1, 0, 0], c=xLineColor, lw=axesLineWidth))
        if ytitle: lines.append(shapes.Line([0, 0, 0], [0, 1, 0], c=yLineColor, lw=axesLineWidth))
        if ztitle: lines.append(shapes.Line([0, 0, 0], [0, 0, 1], c=zLineColor, lw=axesLineWidth))

        ################################################ grid planes
        grids = []
        if xyGrid and xtitle and ytitle:
            gxy = shapes.Grid(pos=(0.5, 0.5, 0), normal=[0, 0, 1], resx=rx, resy=ry)
            gxy.alpha(xyAlpha).wireframe(xyGridTransparent).c(xyPlaneColor).lw(gridLineWidth).lc(xyGridColor)
            grids.append(gxy)
        if yzGrid and ytitle and ztitle:
            gyz = shapes.Grid(pos=(0, 0.5, 0.5), normal=[1, 0, 0], resx=rz, resy=ry)
            gyz.alpha(yzAlpha).wireframe(yzGridTransparent).c(yzPlaneColor).lw(gridLineWidth).lc(yzGridColor)
            grids.append(gyz)
        if zxGrid and ztitle and xtitle:
            gzx = shapes.Grid(pos=(0.5, 0, 0.5), normal=[0, 1, 0], resx=rz, resy=rx)
            gzx.alpha(zxAlpha).wireframe(zxGridTransparent).c(zxPlaneColor).lw(gridLineWidth).lc(zxGridColor)
            grids.append(gzx)

        grids2 = []
        if xyGrid2 and xtitle and ytitle:
            gxy2 = shapes.Grid(pos=(0.5, 0.5, 1), normal=[0, 0, 1], resx=rx, resy=ry)
            gxy2.alpha(xyAlpha).wireframe(xyGrid2Transparent).c(xyPlaneColor).lw(gridLineWidth).lc(xyGridColor)
            grids2.append(gxy2)
        if yzGrid2 and ytitle and ztitle:
            gyz2 = shapes.Grid(pos=(1, 0.5, 0.5), normal=[1, 0, 0], resx=rz, resy=ry)
            gyz2.alpha(yzAlpha).wireframe(yzGrid2Transparent).c(yzPlaneColor).lw(gridLineWidth).lc(yzGridColor)
            grids2.append(gyz2)
        if zxGrid2 and ztitle and xtitle:
            gzx2 = shapes.Grid(pos=(0.5, 1, 0.5), normal=[0, 1, 0], resx=rz, resy=rx)
            gzx2.alpha(zxAlpha).wireframe(zxGrid2Transparent).c(zxPlaneColor).lw(gridLineWidth).lc(zxGridColor)
            grids2.append(gzx2)


        ################################################ zero lines highlights
        highlights = []
        if xyGrid and xtitle and ytitle:
            if xHighlightZero and min_bns[0] <= 0 and max_bns[1] > 0:
                xhl = -min_bns[0] / sizes[0]
                hxy = shapes.Line([xhl,0,0], [xhl,1,0], c=xHighlightZeroColor)
                hxy.alpha(numpy.sqrt(xyAlpha)).lw(gridLineWidth*2)
                highlights.append(hxy)
            if yHighlightZero and min_bns[2] <= 0 and max_bns[3] > 0:
                yhl = -min_bns[2] / sizes[1]
                hyx = shapes.Line([0,yhl,0], [1,yhl,0], c=yHighlightZeroColor)
                hyx.alpha(numpy.sqrt(yzAlpha)).lw(gridLineWidth*2)
                highlights.append(hyx)

        if yzGrid and ytitle and ztitle:
            if yHighlightZero and min_bns[2] <= 0 and max_bns[3] > 0:
                yhl = -min_bns[2] / sizes[1]
                hyz = shapes.Line([0,yhl,0], [0,yhl,1], c=yHighlightZeroColor)
                hyz.alpha(numpy.sqrt(yzAlpha)).lw(gridLineWidth*2)
                highlights.append(hyz)
            if zHighlightZero and min_bns[4] <= 0 and max_bns[5] > 0:
                zhl = -min_bns[4] / sizes[2]
                hzy = shapes.Line([0,0,zhl], [0,1,zhl], c=zHighlightZeroColor)
                hzy.alpha(numpy.sqrt(yzAlpha)).lw(gridLineWidth*2)
                highlights.append(hzy)

        if zxGrid and ztitle and xtitle:
            if zHighlightZero and min_bns[4] <= 0 and max_bns[5] > 0:
                zhl = -min_bns[4] / sizes[2]
                hzx = shapes.Line([0,0,zhl], [1,0,zhl], c=zHighlightZeroColor)
                hzx.alpha(numpy.sqrt(zxAlpha)).lw(gridLineWidth*2)
                highlights.append(hzx)
            if xHighlightZero and min_bns[0] <= 0 and max_bns[1] > 0:
                xhl = -min_bns[0] / sizes[0]
                hxz = shapes.Line([xhl,0,0], [xhl,0,1], c=xHighlightZeroColor)
                hxz.alpha(numpy.sqrt(zxAlpha)).lw(gridLineWidth*2)
                highlights.append(hxz)

        ################################################ aspect ratio scales
        x_aspect_ratio_scale=1
        y_aspect_ratio_scale=1
        z_aspect_ratio_scale=1
        if xtitle:
            if sizes[0] > sizes[1]:
                x_aspect_ratio_scale = (1, sizes[0]/sizes[1], 1)
            else:
                x_aspect_ratio_scale = (sizes[1]/sizes[0], 1, 1)

        if ytitle:
            if sizes[0] > sizes[1]:
                y_aspect_ratio_scale = (sizes[0]/sizes[1], 1, 1)
            else:
                y_aspect_ratio_scale = (1, sizes[1]/sizes[0], 1)

        if ztitle:
            smean = (sizes[0]+sizes[1])/2
            if smean:
                if sizes[2] > smean:
                    zarfact = smean/sizes[2]
                    z_aspect_ratio_scale = (zarfact, zarfact*sizes[2]/smean, zarfact)
                else:
                    z_aspect_ratio_scale = (smean/sizes[2], 1, 1)

        ################################################ axes titles
        titles = []
        if xtitle:
            xt = shapes.Text(xtitle, pos=(0,0,0), s=xTitleSize, bc=xTitleBackfaceColor,
                             c=xTitleColor, justify=xTitleJustify, depth=titleDepth)
            if reorientShortTitle and len(ytitle) < 3:  # title is short
                wpos = [xTitlePosition, -xTitleOffset +0.02, 0]
            else:
                wpos = [xTitlePosition, -xTitleOffset, 0]
            if xKeepAspectRatio: xt.SetScale(x_aspect_ratio_scale)
            xt.rotateX(xTitleRotation).pos(wpos)
            titles.append(xt.lighting(specular=0, diffuse=0, ambient=1))

        if ytitle:
            yt = shapes.Text(ytitle, pos=(0, 0, 0), s=yTitleSize, bc=yTitleBackfaceColor,
                             c=yTitleColor, justify=yTitleJustify, depth=titleDepth)
            if reorientShortTitle and len(ytitle) < 3:  # title is short
                wpos = [-yTitleOffset +0.03-0.01*len(ytitle), yTitlePosition, 0]
                if yKeepAspectRatio: yt.SetScale(x_aspect_ratio_scale) #x!
            else:
                wpos = [-yTitleOffset, yTitlePosition, 0]
                if yKeepAspectRatio: yt.SetScale(y_aspect_ratio_scale)
                yt.rotateZ(yTitleRotation)
            yt.pos(wpos)
            titles.append(yt.lighting(specular=0, diffuse=0, ambient=1))

        if ztitle:
            zt = shapes.Text(ztitle, pos=(0, 0, 0), s=zTitleSize, bc=zTitleBackfaceColor,
                             c=zTitleColor, justify=zTitleJustify, depth=titleDepth)
            if reorientShortTitle and len(ztitle) < 3:  # title is short
                wpos = [(-zTitleOffset+0.02-0.003*len(ztitle))/1.42,
                        (-zTitleOffset+0.02-0.003*len(ztitle))/1.42, zTitlePosition]
                if zKeepAspectRatio:
                    zr2 = (z_aspect_ratio_scale[1], z_aspect_ratio_scale[0], z_aspect_ratio_scale[2])
                    zt.SetScale(zr2)
                zt.rotateX(90).rotateY(45).pos(wpos)
            else:
                if zKeepAspectRatio: zt.SetScale(z_aspect_ratio_scale)
                wpos = [-zTitleOffset/1.42, -zTitleOffset/1.42, zTitlePosition]
                zt.rotateY(-90).rotateX(zTitleRotation).pos(wpos)
            titles.append(zt.lighting(specular=0, diffuse=0, ambient=1))

        ################################################ cube origin ticks
        originmarks = []
        if originMarkerSize:
            if xtitle:
                if min_bns[0] <= 0 and max_bns[1] > 0:  # mark x origin
                    ox = shapes.Cube([-min_bns[0] / sizes[0], 0, 0], side=originMarkerSize, c=xLineColor)
                    originmarks.append(ox.lighting(specular=0, diffuse=0, ambient=1))

            if ytitle:
                if min_bns[2] <= 0 and max_bns[3] > 0:  # mark y origin
                    oy = shapes.Cube([0, -min_bns[2] / sizes[1], 0], side=originMarkerSize, c=yLineColor)
                    originmarks.append(oy.lighting(specular=0, diffuse=0, ambient=1))

            if ztitle:
                if min_bns[4] <= 0 and max_bns[5] > 0:  # mark z origin
                    oz = shapes.Cube([0, 0, -min_bns[4] / sizes[2]], side=originMarkerSize, c=zLineColor)
                    originmarks.append(oz.lighting(specular=0, diffuse=0, ambient=1))

        ################################################ arrow cone
        cones = []
        if tipSize:
            if xtitle:
                cx = shapes.Cone((1,0,0), r=tipSize, height=tipSize*2, axis=(1,0,0), c=xLineColor, res=10)
                cones.append(cx.lighting(specular=0, diffuse=0, ambient=1))
            if ytitle:
                cy = shapes.Cone((0,1,0), r=tipSize, height=tipSize*2, axis=(0,1,0), c=yLineColor, res=10)
                cones.append(cy.lighting(specular=0, diffuse=0, ambient=1))
            if ztitle:
                cz = shapes.Cone((0,0,1), r=tipSize, height=tipSize*2, axis=(0,0,1), c=zLineColor, res=10)
                cones.append(cz.lighting(specular=0, diffuse=0, ambient=1))

        ################################################ cylindrical ticks
        ticks = []
        if showTicks:
            if xtitle:
                for coo in range(1 ,rx):
                    v = [coo/rx,0,0]
                    xds = shapes.Cylinder(v, r=xTickRadius, height=xTickThickness, axis=(1,0,0), res=10)
                    ticks.append(xds.c(xTickColor).lighting(specular=0, ambient=1))
            if ytitle:
                for coo in range(1 ,ry):
                    v = [0,coo/ry,0]
                    yds = shapes.Cylinder(v, r=yTickRadius, height=yTickThickness, axis=(0,1,0), res=10)
                    ticks.append(yds.c(yTickColor).lighting(specular=0, ambient=1))
            if ztitle:
                for coo in range(1 ,rz):
                    v = [0,0,coo/rz]
                    zds = shapes.Cylinder(v, r=zTickRadius, height=zTickThickness, axis=(0,0,1), res=10)
                    ticks.append(zds.c(zTickColor).lighting(specular=0, ambient=1))

        ################################################ MINOR cylindrical ticks
        minorticks = []
        if xMinorTicks and xtitle:
            xMinorTicks += 1
            for coo in range(1, rx*xMinorTicks):
                v = [coo/rx/xMinorTicks,0,0]
                mxds = shapes.Cylinder(v, r=xTickRadius/1.5, height=xTickThickness, axis=(1,0,0), res=6)
                minorticks.append(mxds.c(xTickColor).lighting(specular=0, ambient=1))
        if yMinorTicks and ytitle:
            yMinorTicks += 1
            for coo in range(1, ry*yMinorTicks):
                v = [0, coo/ry/yMinorTicks,0]
                myds = shapes.Cylinder(v, r=yTickRadius/1.5, height=yTickThickness, axis=(0,1,0), res=6)
                minorticks.append(myds.c(yTickColor).lighting(specular=0, ambient=1))
        if zMinorTicks and ztitle:
            zMinorTicks += 1
            for coo in range(1, rz*zMinorTicks):
                v = [0, 0, coo/rz/zMinorTicks]
                mzds = shapes.Cylinder(v, r=zTickRadius/1.5, height=zTickThickness, axis=(0,0,1), res=6)
                minorticks.append(mzds.c(zTickColor).lighting(specular=0, ambient=1))

        ################################################ axes tick NUMERIC labels
        labels = []
        if xLabelSize:
            if xtitle:
                if rx > 12: rx = int(rx/2)
                for ic in range(1, rx+enableLastLabel):
                    v = (ic/rx, -xLabelOffset, 0)
                    val = v[0]*sizes[0]+min_bns[0]
                    if abs(val)>1 and sizes[0]<1: xLabelPrecision = int(xLabelPrecision-numpy.log10(sizes[0]))
                    tval = utils.precision(val, xLabelPrecision, vrange=sizes[0])
                    xlab = shapes.Text(tval, pos=v, s=xLabelSize, justify="center-top", depth=0)
                    if xKeepAspectRatio: xlab.SetScale(x_aspect_ratio_scale)
                    labels.append(xlab.c(xTickColor).lighting(specular=0, ambient=1))
        if yLabelSize:
            if ytitle:
                if ry > 12: ry = int(ry/2)
                for ic in range(1, ry+enableLastLabel):
                    v = (-yLabelOffset, ic/ry, 0)
                    val = v[1]*sizes[1]+min_bns[2]
                    if abs(val)>1 and sizes[1]<1: yLabelPrecision = int(yLabelPrecision-numpy.log10(sizes[1]))
                    tval = utils.precision(val, yLabelPrecision, vrange=sizes[1])
                    ylab = shapes.Text(tval, pos=(0,0,0), s=yLabelSize, justify="center-bottom", depth=0)
                    if yKeepAspectRatio: ylab.SetScale(y_aspect_ratio_scale)
                    ylab.rotateZ(yTitleRotation).pos(v)
                    labels.append(ylab.c(yTickColor).lighting(specular=0, ambient=1))
        if zLabelSize:
            if ztitle:
                if rz > 12: rz = int(rz/2)
                for ic in range(1, rz+enableLastLabel):
                    v = (-zLabelOffset, -zLabelOffset, ic/rz)
                    val = v[2]*sizes[2]+min_bns[4]
                    tval = utils.precision(val, zLabelPrecision, vrange=sizes[2])
                    if abs(val)>1 and sizes[2]<1: zLabelPrecision = int(zLabelPrecision-numpy.log10(sizes[2]))
                    zlab = shapes.Text(tval, pos=(0,0,0), s=zLabelSize, justify="center-bottom", depth=0)
                    if zKeepAspectRatio: zlab.SetScale(z_aspect_ratio_scale)
                    zlab.rotateY(-90).rotateX(zTitleRotation).pos(v)
                    labels.append(zlab.c(zTickColor).lighting(specular=0, ambient=1))

        acts = grids + grids2 + lines + highlights + titles
        acts += minorticks + originmarks + ticks + cones + labels
        for a in acts:
            a.PickableOff()
        asse = Assembly(acts)
        asse.pos(min_bns[0], min_bns[2], min_bns[4])
        asse.SetScale(sizes)
        asse.PickableOff()
        vp.renderer.AddActor(asse)
        vp.axes_instances[r] = asse
        vp.axes = axes_copy #un-pop


    elif vp.axes == 2 or vp.axes == 3:
        x0, x1, y0, y1, z0, z1 = vp.renderer.ComputeVisiblePropBounds()
        xcol, ycol, zcol = "db", "dg", "dr"
        s = 1
        alpha = 1
        centered = False
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        aves = numpy.sqrt(dx * dx + dy * dy + dz * dz) / 2
        x0, x1 = min(x0, 0), max(x1, 0)
        y0, y1 = min(y0, 0), max(y1, 0)
        z0, z1 = min(z0, 0), max(z1, 0)

        if vp.axes == 3:
            if x1 > 0:
                x0 = 0
            if y1 > 0:
                y0 = 0
            if z1 > 0:
                z0 = 0

        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        acts = []
        if x0 * x1 <= 0 or y0 * z1 <= 0 or z0 * z1 <= 0:  # some ranges contain origin
            zero = shapes.Sphere(r=aves / 120 * s, c="k", alpha=alpha, res=10)
            acts += [zero]

        if len(vp.xtitle) and dx > aves/100:
            xl = shapes.Cylinder([[x0, 0, 0], [x1, 0, 0]], r=aves/250*s, c=xcol, alpha=alpha)
            xc = shapes.Cone(pos=[x1, 0, 0], c=xcol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[1, 0, 0], res=10)
            wpos = [x1-(len(vp.xtitle)+1)*aves/40*s, -aves/25*s, 0]  # aligned to arrow tip
            if centered:
                wpos = [(x0 + x1) / 2 - len(vp.xtitle) / 2 * aves / 40 * s, -aves / 25 * s, 0]
            xt = shapes.Text(vp.xtitle, pos=wpos, s=aves / 40 * s, c=xcol)
            acts += [xl, xc, xt]

        if len(vp.ytitle) and dy > aves/100:
            yl = shapes.Cylinder([[0, y0, 0], [0, y1, 0]], r=aves/250*s, c=ycol, alpha=alpha)
            yc = shapes.Cone(pos=[0, y1, 0], c=ycol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[0, 1, 0], res=10)
            wpos = [-aves/40*s, y1-(len(vp.ytitle)+1)*aves/40*s, 0]
            if centered:
                wpos = [-aves / 40 * s, (y0 + y1) / 2 - len(vp.ytitle) / 2 * aves / 40 * s, 0]
            yt = shapes.Text(vp.ytitle, pos=(0, 0, 0), s=aves / 40 * s, c=ycol)
            yt.rotate(90, [0, 0, 1]).pos(wpos)
            acts += [yl, yc, yt]

        if len(vp.ztitle) and dz > aves/100:
            zl = shapes.Cylinder([[0, 0, z0], [0, 0, z1]], r=aves/250*s, c=zcol, alpha=alpha)
            zc = shapes.Cone(pos=[0, 0, z1], c=zcol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[0, 0, 1], res=10)
            wpos = [-aves/50*s, -aves/50*s, z1 - (len(vp.ztitle)+1)*aves/40*s]
            if centered:
                wpos = [-aves/50*s, -aves/50*s, (z0+z1)/2-len(vp.ztitle)/2*aves/40*s]
            zt = shapes.Text(vp.ztitle, pos=(0,0,0), s=aves/40*s, c=zcol)
            zt.rotate(180, (1, -1, 0)).pos(wpos)
            acts += [zl, zc, zt]
        for a in acts:
            a.PickableOff()
        ass = Assembly(acts)
        ass.PickableOff()
        vp.renderer.AddActor(ass)
        vp.axes_instances[r] = ass

    elif vp.axes == 4:
        axact = vtk.vtkAxesActor()
        axact.SetShaftTypeToCylinder()
        axact.SetCylinderRadius(0.03)
        axact.SetXAxisLabelText(vp.xtitle)
        axact.SetYAxisLabelText(vp.ytitle)
        axact.SetZAxisLabelText(vp.ztitle)
        axact.GetXAxisShaftProperty().SetColor(0, 0, 1)
        axact.GetZAxisShaftProperty().SetColor(1, 0, 0)
        axact.GetXAxisTipProperty().SetColor(0, 0, 1)
        axact.GetZAxisTipProperty().SetColor(1, 0, 0)
        bc = numpy.array(vp.renderer.GetBackground())
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
        icn = addIcon(axact, size=0.1)
        vp.axes_instances[r] = icn

    elif vp.axes == 5:
        axact = vtk.vtkAnnotatedCubeActor()
        axact.GetCubeProperty().SetColor(0.75, 0.75, 0.75)
        axact.SetTextEdgesVisibility(0)
        axact.SetFaceTextScale(0.2)
        axact.SetXPlusFaceText ( "right" )
        axact.SetXMinusFaceText( "left " )
        axact.SetYPlusFaceText ( "front" )
        axact.SetYMinusFaceText( "back " )
        axact.SetZPlusFaceText ( " top " )
        axact.SetZMinusFaceText( "bttom" )
        axact.SetZFaceTextRotation(90)

        axact.GetXPlusFaceProperty().SetColor(colors.getColor("b"))
        axact.GetXMinusFaceProperty().SetColor(colors.getColor("db"))
        axact.GetYPlusFaceProperty().SetColor(colors.getColor("g"))
        axact.GetYMinusFaceProperty().SetColor(colors.getColor("dg"))
        axact.GetZPlusFaceProperty().SetColor(colors.getColor("r"))
        axact.GetZMinusFaceProperty().SetColor(colors.getColor("dr"))
        axact.PickableOff()
        icn = addIcon(axact, size=0.06)
        vp.axes_instances[r] = icn

    elif vp.axes == 6:
        ocf = vtk.vtkOutlineCornerFilter()
        ocf.SetCornerFactor(0.1)
        largestact, sz = None, -1
        for a in vp.actors:
            if a.GetPickable():
                b = a.GetBounds()
                if b is None:
                    return
                d = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
                if sz < d:
                    largestact = a
                    sz = d
        if isinstance(largestact, Assembly):
            ocf.SetInputData(largestact.getActor(0).GetMapper().GetInput())
        else:
            ocf.SetInputData(largestact.GetMapper().GetInput())
        ocf.Update()
        ocMapper = vtk.vtkHierarchicalPolyDataMapper()
        ocMapper.SetInputConnection(0, ocf.GetOutputPort(0))
        ocActor = vtk.vtkActor()
        ocActor.SetMapper(ocMapper)
        bc = numpy.array(vp.renderer.GetBackground())
        if numpy.sum(bc) < 1.5:
            lc = (1, 1, 1)
        else:
            lc = (0, 0, 0)
        ocActor.GetProperty().SetColor(lc)
        ocActor.PickableOff()
        vp.renderer.AddActor(ocActor)
        vp.axes_instances[r] = ocActor

    elif vp.axes == 7:
        # draws a simple ruler at the bottom of the window
        ls = vtk.vtkLegendScaleActor()
        ls.RightAxisVisibilityOff()
        ls.TopAxisVisibilityOff()
        ls.LegendVisibilityOff()
        ls.LeftAxisVisibilityOff()
        ls.GetBottomAxis().SetNumberOfMinorTicks(1)
        ls.GetBottomAxis().GetProperty().SetColor(c)
        ls.GetBottomAxis().GetLabelTextProperty().SetColor(c)
        ls.GetBottomAxis().GetLabelTextProperty().BoldOff()
        ls.GetBottomAxis().GetLabelTextProperty().ItalicOff()
        ls.GetBottomAxis().GetLabelTextProperty().ShadowOff()
        ls.PickableOff()
        vp.renderer.AddActor(ls)
        vp.axes_instances[r] = ls

    elif vp.axes == 8:
        vbb = computeVisibleBounds()[0]
        ca = vtk.vtkCubeAxesActor()
        ca.SetBounds(vbb)
        if vp.camera:
            ca.SetCamera(vp.camera)
        else:
            ca.SetCamera(vp.renderer.GetActiveCamera())
        ca.GetXAxesLinesProperty().SetColor(c)
        ca.GetYAxesLinesProperty().SetColor(c)
        ca.GetZAxesLinesProperty().SetColor(c)
        for i in range(3):
            ca.GetLabelTextProperty(i).SetColor(c)
            ca.GetTitleTextProperty(i).SetColor(c)
        ca.SetTitleOffset(5)
        ca.SetFlyMode(3)
        ca.SetXTitle(vp.xtitle)
        ca.SetYTitle(vp.ytitle)
        ca.SetZTitle(vp.ztitle)
        if vp.xtitle == "":
            ca.SetXAxisVisibility(0)
            ca.XAxisLabelVisibilityOff()
        if vp.ytitle == "":
            ca.SetYAxisVisibility(0)
            ca.YAxisLabelVisibilityOff()
        if vp.ztitle == "":
            ca.SetZAxisVisibility(0)
            ca.ZAxisLabelVisibilityOff()
        ca.PickableOff()
        vp.renderer.AddActor(ca)
        vp.axes_instances[r] = ca
        return

    elif vp.axes == 9:
        vbb = computeVisibleBounds()[0]
        src = vtk.vtkCubeSource()
        src.SetXLength(vbb[1] - vbb[0])
        src.SetYLength(vbb[3] - vbb[2])
        src.SetZLength(vbb[5] - vbb[4])
        src.Update()
        ca = Actor(src.GetOutput(), c, 0.5).wireframe(True)
        ca.pos((vbb[0] + vbb[1]) / 2, (vbb[3] + vbb[2]) / 2, (vbb[5] + vbb[4]) / 2)
        ca.PickableOff()
        vp.renderer.AddActor(ca)
        vp.axes_instances[r] = ca

    elif vp.axes == 10:
        vbb = computeVisibleBounds()[0]
        x0 = (vbb[0] + vbb[1]) / 2, (vbb[3] + vbb[2]) / 2, (vbb[5] + vbb[4]) / 2
        rx, ry, rz = (vbb[1]-vbb[0])/2, (vbb[3]-vbb[2])/2, (vbb[5]-vbb[4])/2
        rm = max(rx, ry, rz)
        xc = shapes.Disc(x0, r1=rm, r2=rm, c='lr', res=1, resphi=72)
        yc = shapes.Disc(x0, r1=rm, r2=rm, c='lg', res=1, resphi=72).rotateX(90)
        zc = shapes.Disc(x0, r1=rm, r2=rm, c='lb', res=1, resphi=72).rotateY(90)
        xc.clean().alpha(0.2).wireframe().lineWidth(2.5).PickableOff()
        yc.clean().alpha(0.2).wireframe().lineWidth(2.5).PickableOff()
        zc.clean().alpha(0.2).wireframe().lineWidth(2.5).PickableOff()
        ca = xc + yc + zc
        ca.PickableOff()
        vp.renderer.AddActor(ca)
        vp.axes_instances[r] = ca

    else:
        colors.printc('~bomb Keyword axes must be in range [0-10].', c=1)
        colors.printc('''
  ~target Available axes types:
  0 = no axes,
  1 = draw three customizable gray grid walls
  2 = show cartesian axes from (0,0,0)
  3 = show positive range of cartesian axes from (0,0,0)
  4 = show a triad at bottom left
  5 = show a cube at bottom left
  6 = mark the corners of the bounding box
  7 = draw a simple ruler at the bottom of the window
  8 = show the vtkCubeAxesActor object
  9 = show the bounding box outline
  10 = show three circles representing the maximum bounding box
  ''', c=1, bold=0)

    if not vp.axes_instances[r]:
        vp.axes_instances[r] = True
    return


def addRendererFrame(c=None, alpha=0.5, bg=None, lw=0.5):

    if c is None:  # automatic black or white
        c = (0.9, 0.9, 0.9)
        if numpy.sum(settings.plotter_instance.renderer.GetBackground())>1.5:
            c = (0.1, 0.1, 0.1)
    c = colors.getColor(c)

    ppoints = vtk.vtkPoints()  # Generate the polyline
    psqr = [[0,0],[0,1],[1,1],[1,0],[0,0]]
    for i, pt in enumerate(psqr):
            ppoints.InsertPoint(i, pt[0], pt[1], 0)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(psqr))
    for i in range(len(psqr)):
        lines.InsertCellPoint(i)
    pd = vtk.vtkPolyData()
    pd.SetPoints(ppoints)
    pd.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(pd)
    cs = vtk.vtkCoordinate()
    cs.SetCoordinateSystemToNormalizedViewport()
    mapper.SetTransformCoordinate(cs)

    fractor = vtk.vtkActor2D()
    fractor.GetPositionCoordinate().SetValue(0, 0)
    fractor.GetPosition2Coordinate().SetValue(1, 1)
    fractor.SetMapper(mapper)
    fractor.GetProperty().SetColor(c)
    fractor.GetProperty().SetOpacity(alpha)
    fractor.GetProperty().SetLineWidth(lw)

    settings.plotter_instance.renderer.AddActor(fractor)


def addLegend():

    vp = settings.plotter_instance
    if not utils.isSequence(vp._legend):
        return

    # remove old legend if present on current renderer:
    acs = vp.renderer.GetActors2D()
    acs.InitTraversal()
    for i in range(acs.GetNumberOfItems()):
        a = acs.GetNextItem()
        if isinstance(a, vtk.vtkLegendBoxActor):
            vp.renderer.RemoveActor(a)

    actors = vp.getActors()
    acts, texts = [], []
    for i in range(len(actors)):
        a = actors[i]
        if i < len(vp._legend) and vp._legend[i] != "":
            if isinstance(vp._legend[i], str):
                texts.append(vp._legend[i])
                acts.append(a)
        elif hasattr(a, "_legend") and a._legend:
            if isinstance(a._legend, str):
                texts.append(a._legend)
                acts.append(a)

    NT = len(texts)
    if NT > 20:
        NT = 20
    vtklegend = vtk.vtkLegendBoxActor()
    vtklegend.SetNumberOfEntries(NT)
    vtklegend.ScalarVisibilityOff()
    for i in range(NT):
        ti = texts[i]
        if not ti:
            continue
        a = acts[i]
        c = a.GetProperty().GetColor()
        if c == (1, 1, 1):
            c = (0.2, 0.2, 0.2)
        vtklegend.SetEntry(i, a.polydata(), "  " + ti, c)
    pos = vp.legendPos
    width = vp.legendSize
    vtklegend.SetWidth(width)
    vtklegend.SetHeight(width / 5.0 * NT)
    sx, sy = 1 - width, 1 - width / 5.0 * NT
    if pos == 1:
        vtklegend.GetPositionCoordinate().SetValue(0, sy)
    elif pos == 2:
        vtklegend.GetPositionCoordinate().SetValue(sx, sy)  # default
    elif pos == 3:
        vtklegend.GetPositionCoordinate().SetValue(0, 0)
    elif pos == 4:
        vtklegend.GetPositionCoordinate().SetValue(sx, 0)
    vtklegend.UseBackgroundOn()
    vtklegend.SetBackgroundColor(vp.legendBC)
    vtklegend.SetBackgroundOpacity(0.6)
    vtklegend.LockBorderOn()
    vp.renderer.AddActor(vtklegend)


###########################################################################################
class Button:
    """
    Build a Button object to be shown in the rendering window.

    |buttons| |buttons.py|_
    """

    def __init__(self, fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle):
        """
        Build a Button object to be shown in the rendering window.
        """
        self._status = 0
        self.states = states
        self.colors = c
        self.bcolors = bc
        self.function = fnc
        self.actor = vtk.vtkTextActor()

        self.actor.GetActualPositionCoordinate().SetCoordinateSystemToNormalizedViewport()

        self.actor.SetPosition(pos[0], pos[1])
        #self.actor.SetDisplayPosition(pos[0], pos[1]) # pixel coords
        self.framewidth = 2
        self.offset = 5
        self.spacer = " "

        self.textproperty = self.actor.GetTextProperty()
        self.textproperty.SetJustificationToCentered()
        if font.lower() == "courier":
            self.textproperty.SetFontFamilyToCourier()
        elif font.lower() == "times":
            self.textproperty.SetFontFamilyToTimes()
        else:
            self.textproperty.SetFontFamilyToArial()
        self.textproperty.SetFontSize(size)
        self.textproperty.SetBackgroundOpacity(alpha)
        self.textproperty.BoldOff()
        if bold:
            self.textproperty.BoldOn()
        self.textproperty.ItalicOff()
        if italic:
            self.textproperty.ItalicOn()
        self.textproperty.ShadowOff()
        self.textproperty.SetOrientation(angle)
        self.showframe = hasattr(self.textproperty, "FrameOn")
        self.status(0)

    def status(self, s=None):
        """
        Set/Get the status of the button.
        """
        if s is None:
            return self.states[self._status]

        if isinstance(s, str):
            s = self.states.index(s)
        self._status = s
        self.textproperty.SetLineOffset(self.offset)
        self.actor.SetInput(self.spacer + self.states[s] + self.spacer)
        s = s % len(self.colors)  # to avoid mismatch
        self.textproperty.SetColor(colors.getColor(self.colors[s]))
        bcc = numpy.array(colors.getColor(self.bcolors[s]))
        self.textproperty.SetBackgroundColor(bcc)
        if self.showframe:
            self.textproperty.FrameOn()
            self.textproperty.SetFrameWidth(self.framewidth)
            self.textproperty.SetFrameColor(numpy.sqrt(bcc))
        return self

    def switch(self):
        """
        Change/cycle button status to the next defined status in states list.
        """
        self._status = (self._status + 1) % len(self.states)
        self.status(self._status)
        return self
