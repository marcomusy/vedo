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
        "addScalarBar",
        "addScalarBar3D",
        "addSlider2D",
        "addSlider3D",
        "addButton",
        "addCutterTool",
        "addIcon",
        "addAxes",
        "addFrame",
        "addLegend",
        ]


def addScalarBar(actor=None, c=None, title="", horizontal=False, vmin=None, vmax=None):
    """Add a 2D scalar bar for the specified actor.

    If `actor` is ``None`` will add it to the last actor in ``self.actors``.

    .. hint:: |mesh_bands| |mesh_bands.py|_
    """
    vp = settings.plotter_instance
    
    if actor is None:
        actor = vp.lastActor()
        
    if not hasattr(actor, "mapper"):
        colors.printc("~times Error in addScalarBar: input is not a Actor.", c=1)
        return None

    if vp and vp.renderer and actor.scalarbar_actor:
        vp.renderer.RemoveActor(actor.scalarbar)
            

    lut = actor.mapper.GetLookupTable()
    if not lut:
        return None
    vtkscalars = actor.poly.GetPointData().GetScalars()
    if vtkscalars is None:
        vtkscalars = actor.poly.GetCellData().GetScalars()
    if not vtkscalars:
        return None
    
    rng = list(vtkscalars.GetRange())
    if vmin:
        rng[0] = vmin
    if vmin:
        rng[1] = vmax
    actor.mapper.SetScalarRange(rng)

    if c is None:
        if vp.renderer:  # automatic black or white
            c = (0.9, 0.9, 0.9)
            if numpy.sum(vp.renderer.GetBackground()) > 1.5:
                c = (0.1, 0.1, 0.1)
        else:
            c = "k"

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
        sb.SetPosition(0.8, 0.05)
        sb.SetMaximumWidthInPixels(1000)
        sb.SetMaximumHeightInPixels(50)
    else:
        sb.SetNumberOfLabels(10)
        sb.SetTextPositionToPrecedeScalarBar()
        sb.SetPosition(0.87, 0.05)
        sb.SetMaximumWidthInPixels(80)
        sb.SetMaximumHeightInPixels(500)

    sctxt = sb.GetLabelTextProperty()
    sctxt.SetColor(c)
    sctxt.SetShadow(0)
    sctxt.SetFontFamily(0)
    sctxt.SetItalic(0)
    sctxt.SetBold(0)
    sctxt.SetFontSize(12)
    if not vp.renderer:
        save_int = vp.interactive
        vp.show(interactive=0)
        vp.interactive = save_int
    sb.PickableOff()
    vp.renderer.AddActor(sb)
    vp.scalarbars.append(sb)
    actor.scalarbar_actor = sb
    vp.renderer.Render()
    return sb


def addScalarBar3D(
    obj=None,
    at=0,
    pos=(0, 0, 0),
    normal=(0, 0, 1),
    sx=0.1,
    sy=2,
    nlabels=9,
    ncols=256,
    cmap=None,
    c=None,
    alpha=1,
):
    """Draw a 3D scalar bar.

    ``obj`` input can be:
        - a list of numbers,
        - a list of two numbers in the form `(min, max)`,
        - a ``vtkActor`` already containing a set of scalars associated to vertices or cells,
        - if ``None`` the last actor in the list of actors will be used.

    .. hint:: |scalbar| |mesh_coloring.py|_
    """
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    vp = settings.plotter_instance
    if c is None:  # automatic black or white
        c = (0.8, 0.8, 0.8)
        if numpy.sum(colors.getColor(vp.backgrcol)) > 1.5:
            c = (0.2, 0.2, 0.2)
    c = colors.getColor(c)
    
    gap = 0.4  # space btw nrs and scale
    vtkscalars_name = ""
    if obj is None:
        obj = vp.lastActor()
    if isinstance(obj, vtk.vtkActor):
        poly = obj.GetMapper().GetInput()
        vtkscalars = poly.GetPointData().GetScalars()
        if vtkscalars is None:
            vtkscalars = poly.GetCellData().GetScalars()
        if vtkscalars is None:
            print("Error in addScalarBar3D: actor has no scalar array.", [obj])
            exit()
        npscalars = vtk_to_numpy(vtkscalars)
        vmin, vmax = numpy.min(npscalars), numpy.max(npscalars)
        vtkscalars_name = vtkscalars.GetName().split("_")[-1]
    elif utils.isSequence(obj):
        vmin, vmax = numpy.min(obj), numpy.max(obj)
        vtkscalars_name = "jet"
    else:
        print("Error in addScalarBar3D(): input must be vtkActor or list.", type(obj))
        exit()

    if cmap is None:
        cmap = vtkscalars_name

    # build the color scale part
    scale = shapes.Grid([-sx * gap, 0, 0], c=c, alpha=alpha, sx=sx, sy=sy, resx=1, resy=ncols)
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
    prec = (vmax - vmin) / abs(vmax + vmin) * 2
    prec = int(3 + abs(numpy.log10(prec + 1)))
    for i, t in enumerate(tlabs):
        tx = utils.precision(t, prec)
        y = -sy / 1.98 + sy * i / (nlabels - 1)
        a = shapes.Text(tx, pos=[sx * gap, y, 0], s=sy / 50, c=c, alpha=alpha, depth=0)
        a.PickableOff()
        tacts.append(a)
    sact = Assembly([scale] + tacts)
    nax = numpy.linalg.norm(normal)
    if nax:
        normal = numpy.array(normal) / nax
    theta = numpy.arccos(normal[2])
    phi = numpy.arctan2(normal[1], normal[0])
    sact.RotateZ(phi * 57.3)
    sact.RotateY(theta * 57.3)
    sact.SetPosition(pos)
    if not vp.renderers[at]:
        save_int = vp.interactive
        vp.show(interactive=0)
        vp.interactive = save_int
    vp.renderers[at].AddActor(sact)
    vp.renderers[at].Render()
    sact.PickableOff()
    vp.scalarbars.append(sact)
    if isinstance(obj, Actor):
        obj.scalarbar_actor = sact
    return sact


def addSlider2D(sliderfunc, xmin, xmax, value=None, pos=4, s=.04,
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

    .. hint:: |sliders| |sliders.py|_
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
        sliderRep.GetLabelProperty().SetOpacity(0.6)
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
        sliderRep.GetTitleProperty().SetOpacity(0.6)
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

    .. hint:: |sliders3d| |sliders3d.py|_
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

    .. hint:: |buttons| |buttons.py|_
    """
    vp = settings.plotter_instance
    if not vp.renderer:
        colors.printc("~timesError: Use addButton() after rendering the scene.", c=1)
        return
    import vtkplotter.vtkio as vtkio
    bu = vtkio.Button(fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle)
    vp.renderer.AddActor2D(bu.actor)
    vp.window.Render()
    vp.buttons.append(bu)
    return bu


def addCutterTool(actor):
    """Create handles to cut away parts of a mesh.

    .. hint:: |cutter| |cutter.py|_
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
    if hasattr(actor, "polydata"):
        apd = actor.polydata()
    else:
        apd = actor.GetMapper().GetInput()

    planes = vtk.vtkPlanes()
    planes.SetBounds(apd.GetBounds())

    clipper = vtk.vtkClipPolyData()
    clipper.GenerateClipScalarsOff()
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
    act1.GetProperty().SetOpacity(0.02)
    act1.GetProperty().SetRepresentationToWireframe()
    act1.VisibilityOn()

    vp.renderer.AddActor(act0)
    vp.renderer.AddActor(act1)
    vp.renderer.RemoveActor(actor)

    def SelectPolygons(vobj, event):
        vobj.GetPlanes(planes)

    boxWidget = vtk.vtkBoxWidget()
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
    boxWidget.GetOutlineProperty().SetColor(0.1, 0.1, 0.1)
    boxWidget.GetOutlineProperty().SetOpacity(0.8)
    boxWidget.SetPlaceFactor(1.05)
    boxWidget.SetInteractor(vp.interactor)
    boxWidget.SetInputData(apd)
    boxWidget.PlaceWidget()
    boxWidget.AddObserver("InteractionEvent", SelectPolygons)
    boxWidget.On()

    vp.cutterWidget = boxWidget
    vp.clickedActor = act0
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
    def ClipVolumeRender(obj, event):
        obj.GetPlanes(planes)
        vol.mapper.SetClippingPlanes(planes)

    boxWidget.SetInputData(vol.image)
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
    boxWidget.GetOutlineProperty().SetColor(0.1, 0.1, 0.1)
    boxWidget.GetOutlineProperty().SetOpacity(0.7)
    boxWidget.SetPlaceFactor(1.05)
    boxWidget.PlaceWidget()
    boxWidget.InsideOutOn()
    boxWidget.AddObserver("InteractionEvent", ClipVolumeRender)

    colors.printc("Mesh Cutter Tool:", c="m", invert=1)
    colors.printc("  Move gray handles to cut away parts of the mesh", c="m")
    colors.printc("  Press X to save file to: clipped.vtk", c="m")
    
    vp.renderer.ResetCamera()
    boxWidget.On()

    vp.interactor.Start()
    boxWidget.Off()
    vp.widgets.append(boxWidget)


def addIcon(iconActor, pos=3, size=0.08):
    """Add an inset icon mesh into the same renderer.

    :param pos: icon position in the range [1-4] indicating one of the 4 corners,
                or it can be a tuple (x,y) as a fraction of the renderer size.
    :param float size: size of the square inset.

    .. hint:: |icon| |icon.py|_
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

    if not vp.renderer:
        return

    if vp.axes_exist[r]:
        return

    # calculate max actors bounds
    bns = []
    for a in vp.actors:
        if a and a.GetPickable():
            b = a.GetBounds()
            if b:
                bns.append(b)
    if len(bns):
        max_bns = numpy.max(bns, axis=0)
        min_bns = numpy.min(bns, axis=0)
        vbb = (min_bns[0], max_bns[1], min_bns[2], max_bns[3], min_bns[4], max_bns[5])
    else:
        vbb = vp.renderer.ComputeVisiblePropBounds()
        max_bns = vbb
        min_bns = vbb
    sizes = (max_bns[1] - min_bns[0], max_bns[3] - min_bns[2], max_bns[5] - min_bns[4])


    ############################################################
    if vp.axes == 1 or vp.axes == True:  # gray grid walls
        nd = 4  # number of divisions in the smallest axis
        off = -0.04  # label offset
        step = numpy.min(sizes) / nd
        if not step:
            # bad proportions, use vtkCubeAxesActor
            vp.addAxes(axtype=8, c=c)
            vp.axes = 1
            return

        rx, ry, rz = numpy.rint(sizes / step).astype(int)
        if max([rx / ry, ry / rx, rx / rz, rz / rx, ry / rz, rz / ry]) > 15:
            # bad proportions, use vtkCubeAxesActor
            vp.addAxes(axtype=8, c=c)
            vp.axes = 1
            return

        gxy = shapes.Grid(pos=(0.5, 0.5, 0), normal=[0, 0, 1], bc=None, resx=rx, resy=ry)
        gxz = shapes.Grid(pos=(0.5, 0, 0.5), normal=[0, 1, 0], bc=None, resx=rz, resy=rx)
        gyz = shapes.Grid(pos=(0, 0.5, 0.5), normal=[1, 0, 0], bc=None, resx=rz, resy=ry)
        gxy.alpha(0.06).wire(False).color(c).lineWidth(1)
        gxz.alpha(0.04).wire(False).color(c).lineWidth(1)
        gyz.alpha(0.04).wire(False).color(c).lineWidth(1)

        xa = shapes.Line([0, 0, 0], [1, 0, 0], c=c, lw=1)
        ya = shapes.Line([0, 0, 0], [0, 1, 0], c=c, lw=1)
        za = shapes.Line([0, 0, 0], [0, 0, 1], c=c, lw=1)

        xt, yt, zt, ox, oy, oz = [None] * 6
        if vp.xtitle:
            xtitle = vp.xtitle
            if min_bns[0] <= 0 and max_bns[1] > 0:  # mark x origin
                ox = shapes.Cube([-min_bns[0] / sizes[0], 0, 0], side=0.008, c=c)
            if len(vp.xtitle) == 1:  # add axis length info
                xtitle = vp.xtitle + " /" + utils.precision(sizes[0], 4)
            wpos = [1 - (len(vp.xtitle) + 1) / 40, off, 0]
            xt = shapes.Text(xtitle, pos=wpos, normal=(0, 0, 1),
                             s=0.025, c=c, justify="bottom-right")

        if vp.ytitle:
            if min_bns[2] <= 0 and max_bns[3] > 0:  # mark y origin
                oy = shapes.Cube([0, -min_bns[2] / sizes[1], 0], side=0.008, c=c)
            yt = shapes.Text(vp.ytitle, pos=(0, 0, 0), normal=(0, 0, 1),
                             s=0.025, c=c, justify="bottom-right")
            if len(vp.ytitle) == 1:
                wpos = [off, 1 - (len(vp.ytitle) + 1) / 40, 0]
                yt.pos(wpos)
            else:
                wpos = [off * 0.7, 1 - (len(vp.ytitle) + 1) / 40, 0]
                yt.rotateZ(90).pos(wpos)

        if vp.ztitle:
            if min_bns[4] <= 0 and max_bns[5] > 0:  # mark z origin
                oz = shapes.Cube([0, 0, -min_bns[4] / sizes[2]], side=0.008, c=c)
            zt = shapes.Text(vp.ztitle, pos=(0, 0, 0), normal=(1, -1, 0),
                             s=0.025, c=c, justify="bottom-right")
            if len(vp.ztitle) == 1:
                wpos = [off * 0.6, off * 0.6, 1 - (len(vp.ztitle) + 1) / 40]
                zt.rotate(90, (1, -1, 0)).pos(wpos)
            else:
                wpos = [off * 0.3, off * 0.3, 1 - (len(vp.ztitle) + 1) / 40]
                zt.rotate(180, (1, -1, 0)).pos(wpos)

        acts = [gxy, gxz, gyz, xa, ya, za, xt, yt, zt, ox, oy, oz]
        for a in acts:
            if a:
                a.PickableOff()
        aa = Assembly(acts)
        aa.pos(min_bns[0], min_bns[2], min_bns[4])
        aa.SetScale(sizes)
        aa.PickableOff()
        vp.renderer.AddActor(aa)
        vp.axes_exist[r] = aa

    elif vp.axes == 2 or vp.axes == 3:
        vbb = vp.renderer.ComputeVisiblePropBounds()  # to be double checked
        xcol, ycol, zcol = "db", "dg", "dr"
        s = 1
        alpha = 1
        centered = False
        x0, x1, y0, y1, z0, z1 = vbb
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
            xt = shapes.Text(vp.xtitle, pos=wpos, normal=(0, 0, 1), s=aves / 40 * s, c=xcol)
            acts += [xl, xc, xt]

        if len(vp.ytitle) and dy > aves/100:
            yl = shapes.Cylinder([[0, y0, 0], [0, y1, 0]], r=aves/250*s, c=ycol, alpha=alpha)
            yc = shapes.Cone(pos=[0, y1, 0], c=ycol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[0, 1, 0], res=10)
            wpos = [-aves/40*s, y1-(len(vp.ytitle)+1)*aves/40*s, 0]
            if centered:
                wpos = [-aves / 40 * s, (y0 + y1) / 2 - len(vp.ytitle) / 2 * aves / 40 * s, 0]
            yt = shapes.Text(vp.ytitle, pos=(0, 0, 0), normal=(0, 0, 1), s=aves / 40 * s, c=ycol)
            yt.rotate(90, [0, 0, 1]).pos(wpos)
            acts += [yl, yc, yt]

        if len(vp.ztitle) and dz > aves/100:
            zl = shapes.Cylinder([[0, 0, z0], [0, 0, z1]], r=aves/250*s, c=zcol, alpha=alpha)
            zc = shapes.Cone(pos=[0, 0, z1], c=zcol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[0, 0, 1], res=10)
            wpos = [-aves/50*s, -aves/50*s, z1 - (len(vp.ztitle)+1)*aves/40*s]
            if centered:
                wpos = [-aves/50*s, -aves/50*s, (z0+z1)/2-len(vp.ztitle)/2*aves/40*s]
            zt = shapes.Text(vp.ztitle, pos=(0,0,0), normal=(1, -1, 0), s=aves/40*s, c=zcol)
            zt.rotate(180, (1, -1, 0)).pos(wpos)
            acts += [zl, zc, zt]
        for a in acts:
            a.PickableOff()
        ass = Assembly(acts)
        ass.PickableOff()
        vp.renderer.AddActor(ass)
        vp.axes_exist[r] = ass

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
        vp.axes_exist[r] = icn

    elif vp.axes == 5:
        axact = vtk.vtkAnnotatedCubeActor()
        axact.GetCubeProperty().SetColor(0.75, 0.75, 0.75)
        axact.SetTextEdgesVisibility(0)
        axact.SetFaceTextScale(0.4)
        axact.GetXPlusFaceProperty().SetColor(colors.getColor("b"))
        axact.GetXMinusFaceProperty().SetColor(colors.getColor("db"))
        axact.GetYPlusFaceProperty().SetColor(colors.getColor("g"))
        axact.GetYMinusFaceProperty().SetColor(colors.getColor("dg"))
        axact.GetZPlusFaceProperty().SetColor(colors.getColor("r"))
        axact.GetZMinusFaceProperty().SetColor(colors.getColor("dr"))
        axact.PickableOff()
        icn = addIcon(axact, size=0.06)
        vp.axes_exist[r] = icn

    elif vp.axes == 6:
        ocf = vtk.vtkOutlineCornerFilter()
        ocf.SetCornerFactor(0.1)
        largestact, sz = None, -1
        for a in vp.actors:
            if a.GetPickable():
                b = a.GetBounds()
                d = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
                if sz < d:
                    largestact = a
                    sz = d
        if isinstance(largestact, Assembly):
            ocf.SetInputData(largestact.getActor(0).GetMapper().GetInput())
        else:
            ocf.SetInputData(largestact.polydata())
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
        vp.axes_exist[r] = ocActor

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
        vp.axes_exist[r] = ls

    elif vp.axes == 8:
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
        vp.axes_exist[r] = ca
        return

    elif vp.axes == 9:
        src = vtk.vtkCubeSource()
        src.SetXLength(vbb[1] - vbb[0])
        src.SetYLength(vbb[3] - vbb[2])
        src.SetZLength(vbb[5] - vbb[4])
        src.Update()
        ca = Actor(src.GetOutput(), c=c, alpha=0.5, wire=1)
        ca.pos((vbb[0] + vbb[1]) / 2, (vbb[3] + vbb[2]) / 2, (vbb[5] + vbb[4]) / 2)
        ca.PickableOff()
        vp.renderer.AddActor(ca)
        vp.axes_exist[r] = ca

    elif vp.axes == 10:
        x0 = (vbb[0] + vbb[1]) / 2, (vbb[3] + vbb[2]) / 2, (vbb[5] + vbb[4]) / 2
        rx, ry, rz = (vbb[1]-vbb[0])/2, (vbb[3]-vbb[2])/2, (vbb[5]-vbb[4])/2
        rm = max(rx, ry, rz)
        xc = shapes.Disc(x0, (0,0,1), r1=rm, r2=rm, c='lr', bc=None, res=1, resphi=72)
        yc = shapes.Disc(x0, (0,1,0), r1=rm, r2=rm, c='lg', bc=None, res=1, resphi=72)
        zc = shapes.Disc(x0, (1,0,0), r1=rm, r2=rm, c='lb', bc=None, res=1, resphi=72)
        xc.clean().alpha(0.2).wire().lineWidth(2.5).PickableOff()
        yc.clean().alpha(0.2).wire().lineWidth(2.5).PickableOff()
        zc.clean().alpha(0.2).wire().lineWidth(2.5).PickableOff()
        ca = xc + yc + zc        
        ca.PickableOff()
        vp.renderer.AddActor(ca)
        vp.axes_exist[r] = ca

    else:
        colors.printc('~bomb Keyword axes must be in range [0-10].', c=1)
        colors.printc('''
  ~target Available axes types:
  0 = no axes,
  1 = draw three gray grid walls
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
    
    if not vp.axes_exist[r]:
        vp.axes_exist[r] = True
    return


def addFrame(c=None, alpha=0.5, bg=None, lw=0.5):

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
