"""
Additional objects like axes, legends etc..
"""
from __future__ import division, print_function
from vtkplotter.colors import printc, getColor
from vtkplotter.assembly import Assembly
from vtkplotter.mesh import Mesh, merge
from vtkplotter.utils import mag, isSequence, make_ticks
import vtkplotter.shapes as shapes
import vtkplotter.settings as settings
import vtkplotter.docs as docs
import numpy as np
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
        "addLegend",
        ]


#####################################################################
def addLight(
    pos,
    focalPoint=(0, 0, 0),
    deg=180,
    c='white',
    intensity=0.4,
    removeOthers=False,
    showsource=False,
):
    """
    Generate a source of light placed at pos, directed to focal point.
    Returns a ``vtkLight`` object.

    :param focalPoint: focal point, if this is a ``vtkActor`` use its position.
    :type fp: vtkActor, list
    :param deg: aperture angle of the light source
    :param c: set light color
    :param float intensity: intensity between 0 and 1.
    :param bool removeOthers: remove all other lights in the scene
    :param bool showsource: if `True`, will show a representation
                            of the source of light as an extra Mesh

    .. hint:: |lights.py|_
    """
    if isinstance(focalPoint, vtk.vtkActor):
        focalPoint = focalPoint.GetPosition()
    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(pos)
    light.SetPositional(True)
    light.SetConeAngle(deg)
    light.SetFocalPoint(focalPoint)
    light.SetIntensity(intensity)
    light.SetColor(getColor(c))
    if showsource:
        lightActor = vtk.vtkLightActor()
        lightActor.SetLight(light)
        settings.plotter_instance.renderer.AddViewProp(lightActor)
    if removeOthers:
        settings.plotter_instance.renderer.RemoveAllLights()
    settings.plotter_instance.renderer.AddLight(light)
    return light


#####################################################################
def addScalarBar(mesh,
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
    """Add a 2D scalar bar for the specified mesh.

    .. hint:: |mesh_coloring| |mesh_coloring.py|_ |scalarbars.py|_
    """
    vp = settings.plotter_instance

    if not hasattr(mesh, "mapper"):
        printc("~times addScalarBar(): input is invalid,", type(mesh), c=1)
        return None

    if vp and vp.renderer:
        c = (0.9, 0.9, 0.9)
        if np.sum(vp.renderer.GetBackground()) > 1.5:
            c = (0.1, 0.1, 0.1)
        if isinstance(mesh.scalarbar, vtk.vtkActor):
            vp.renderer.RemoveActor(mesh.scalarbar)
        elif isinstance(mesh.scalarbar, Assembly):
            for a in mesh.scalarbar.getMeshes():
                vp.renderer.RemoveActor(a)
    if c is None: c = 'gray'

    if isinstance(mesh, Mesh):
        lut = mesh.mapper().GetLookupTable()
        if not lut:
            return None
        vtkscalars = mesh._polydata.GetPointData().GetScalars()
        if vtkscalars is None:
            vtkscalars = mesh._polydata.GetCellData().GetScalars()
        if not vtkscalars:
            return None

        rng = list(vtkscalars.GetRange())
        if vmin is not None: rng[0] = vmin
        if vmax is not None: rng[1] = vmax
        mesh.mapper().SetScalarRange(rng)

    elif isinstance(mesh, 'Volume'):
        # to be implemented
        pass

    c = getColor(c)
    sb = vtk.vtkScalarBarActor()
    sb.SetLookupTable(lut)
    if title:
        titprop = vtk.vtkTextProperty()
        titprop.BoldOn()
        titprop.ItalicOff()
        titprop.ShadowOff()
        titprop.SetColor(c)
        titprop.SetVerticalJustificationToTop()
        titprop.SetFontSize(titleFontSize)
        sb.SetTitle(title)
        sb.SetVerticalTitleSeparation(titleYOffset)
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
    mesh.scalarbar = sb
    return sb


#####################################################################
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
    prec=2,
    labelOffset = 0.4,
    c=None,
    alpha=1,
    cmap=None,
):
    """Draw a 3D scalar bar.

    ``obj`` input can be:
        - a list of numbers,
        - a list of two numbers in the form `(min, max)`,
        - a ``Mesh`` already containing a set of scalars associated to vertices or cells,
        - if ``None`` the last mesh in the list of meshs will be used.

    :param float sx: thickness of scalarbar
    :param float sy: length of scalarbar
    :param str title: scalar bar title
    :param float titleXOffset: horizontal space btw title and color scalarbar
    :param float titleYOffset: vertical space offset
    :param float titleSize: size of title wrt numeric labels
    :param float titleRotation: title rotation in degrees
    :param int nlabels: number of numeric labels
    :param int prec: number of significant digits
    :param float labelOffset: space btw numeric labels and scale
    :param str cmap: specify cmap to be used

    .. hint:: |scalarbars| |scalarbars.py|_
    """
    vp = settings.plotter_instance
    if vp and c is None:  # automatic black or white
        c = (0.8, 0.8, 0.8)
        if np.sum(getColor(vp.backgrcol)) > 1.5:
            c = (0.2, 0.2, 0.2)
    if c is None: c = 'gray'
    c = getColor(c)

    if isinstance(obj, Mesh):
        if cmap is None:
            lut = obj.mapper().GetLookupTable()
            if not lut:
                print("Error in ScalarBar3D: mesh has no active scalar array.", [obj])
                return None
        else:
            lut = cmap
        vmin, vmax = obj.mapper().GetScalarRange()

    elif isSequence(obj):
        vmin, vmax = np.min(obj), np.max(obj)
    else:
        print("Error in ScalarBar3D(): input must be Mesh or list.", type(obj))
        raise RuntimeError()

    # build the color scale part
    scale = shapes.Grid([-sx *labelOffset, 0, 0], c=c, alpha=alpha,
                        sx=sx, sy=sy, resx=1, resy=256)
    scale.lw(0).wireframe(False)
    cscals = scale.cellCenters()[:, 1]
    scale.cellColors(cscals, lut, alpha)
    scale.lighting(ambient=1, diffuse=0, specular=0, specularPower=0)

    # build text
    tacts = []

    ticks_pos, ticks_txt = make_ticks(vmin, vmax, nlabels)
    nlabels = len(ticks_pos)-1
    for i, p in enumerate(ticks_pos):
        tx = ticks_txt[i]
        y = -sy / 1.98 + sy * i / nlabels
        a = shapes.Text(tx, pos=[sx*labelOffset, y, 0], s=sy/50, c=c, alpha=alpha)
        a.lighting(ambient=1, diffuse=0, specular=0, specularPower=0)
        a.PickableOff()
        tacts.append(a)

    # build title
    if title:
        t = shapes.Text(title, (0,0,0), s=sy/50*titleSize, c=c, alpha=alpha, justify='centered')
        t.RotateZ(90+titleRotation)
        t.pos(sx*titleXOffset,titleYOffset,0)
        t.lighting(ambient=1, diffuse=0, specular=0, specularPower=0)
        t.PickableOff()
        tacts.append(t)

    sact = Assembly([scale] + tacts)
    normal = np.array(normal) / np.linalg.norm(normal)
    theta = np.arccos(normal[2])
    phi = np.arctan2(normal[1], normal[0])
    sact.RotateZ(np.rad2deg(phi))
    sact.RotateY(np.rad2deg(theta))
    sact.SetPosition(pos)
    sact.PickableOff()
    if isinstance(obj, Mesh):
        obj.scalarbar = sact
    return sact

#####################################################################
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
        if np.sum(getColor(vp.backgrcol)) > 1.5:
            c = (0.2, 0.2, 0.2)
    c = getColor(c)

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
    if isSequence(pos):
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
    sliderRep.GetSelectedProperty().SetColor(np.sqrt(np.array(c)))
    sliderRep.GetCapProperty().SetColor(c)

    if title:
        sliderRep.SetTitleText(title)
        sliderRep.SetTitleHeight(0.012)
        sliderRep.GetTitleProperty().SetShadow(0)
        sliderRep.GetTitleProperty().SetColor(c)
        sliderRep.GetTitleProperty().SetOpacity(1)
        sliderRep.GetTitleProperty().SetBold(0)
        if not isSequence(pos):
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

#####################################################################
def addSlider3D(
    sliderfunc,
    pos1,
    pos2,
    xmin,
    xmax,
    value=None,
    s=0.03,
    t=1,
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
    :param float t: tube scaling factor
    :param str title: title text
    :param c: slider color
    :param float rotation: title rotation around slider axis
    :param bool showValue: if True current value is shown

    |sliders3d| |sliders3d.py|_
    """
    vp = settings.plotter_instance
    if c is None:  # automatic black or white
        c = (0.8, 0.8, 0.8)
        if np.sum(getColor(vp.backgrcol)) > 1.5:
            c = (0.2, 0.2, 0.2)
    else:
        c = getColor(c)

    if value is None or value < xmin:
        value = xmin

    sliderRep = vtk.vtkSliderRepresentation3D()
    sliderRep.SetMinimumValue(xmin)
    sliderRep.SetMaximumValue(xmax)
    sliderRep.SetValue(value)

    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToWorld()
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToWorld()
    sliderRep.GetPoint1Coordinate().SetValue(pos2)
    sliderRep.GetPoint2Coordinate().SetValue(pos1)

    sliderRep.SetSliderWidth(0.03 * t)
    sliderRep.SetTubeWidth(0.01 * t)
    sliderRep.SetSliderLength(0.04 * t)
    sliderRep.SetSliderShapeToCylinder()
    sliderRep.GetSelectedProperty().SetColor(np.sqrt(np.array(c)))
    sliderRep.GetSliderProperty().SetColor(np.array(c) / 1.5)
    sliderRep.GetCapProperty().SetOpacity(0)
    sliderRep.SetRotation(rotation)

    if not showValue:
        sliderRep.ShowSliderLabelOff()

    sliderRep.SetTitleText(title)
    sliderRep.SetTitleHeight(s * t)
    sliderRep.SetLabelHeight(s * t * 0.85)

    sliderRep.GetTubeProperty().SetColor(c)

    sliderWidget = vtk.vtkSliderWidget()
    sliderWidget.SetInteractor(vp.interactor)
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.SetAnimationModeToJump()
    sliderWidget.AddObserver("InteractionEvent", sliderfunc)
    sliderWidget.EnabledOn()
    vp.sliders.append([sliderWidget, sliderfunc])
    return sliderWidget

#####################################################################
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
        printc("~timesError: Use addButton() after rendering the scene.", c=1)
        return
    bu = Button(fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle)
    vp.renderer.AddActor2D(bu.actor)
    vp.window.Render()
    vp.buttons.append(bu)
    return bu


def addCutterTool(mesh):
    """Create handles to cut away parts of a mesh.

    |cutter| |cutter.py|_
    """
    if isinstance(mesh, vtk.vtkVolume):
        return _addVolumeCutterTool(mesh)
    elif isinstance(mesh, vtk.vtkImageData):
        from vtkplotter import Volume
        return _addVolumeCutterTool(Volume(mesh))

    vp = settings.plotter_instance
    if not vp.renderer:
        save_int = vp.interactive
        vp.show(interactive=0)
        vp.interactive = save_int

    vp.clickedActor = mesh
    apd = mesh.polydata()

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

    act0 = Mesh(cpoly, alpha=mesh.alpha()) # the main cut part
    act0.mapper().SetLookupTable(mesh.mapper().GetLookupTable())
    act0.mapper().SetScalarRange(mesh.mapper().GetScalarRange())

    act1 = Mesh()
    act1.mapper().SetInputConnection(clipper.GetClippedOutputPort()) # needs OutputPort??
    act1.alpha(0.04).color((0.5,0.5,0.5)).wireframe()

    vp.renderer.RemoveActor(mesh)
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
    if mesh in vp.actors:
        ia = vp.actors.index(mesh)
        vp.actors[ia] = act0

    printc("Mesh Cutter Tool:", c="m", invert=1)
    printc("  Move gray handles to cut away parts of the mesh", c="m")
    printc("  Press X to save file to: clipped.vtk", c="m")
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

    vp.renderer.AddVolume(vol)

    planes = vtk.vtkPlanes()
    def clipVolumeRender(obj, event):
        obj.GetPlanes(planes)
        vol.mapper().SetClippingPlanes(planes)

    boxWidget.SetInputData(vol.inputdata())
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
    boxWidget.GetOutlineProperty().SetColor(0.2, 0.2, 0.2)
    boxWidget.GetOutlineProperty().SetOpacity(0.7)
    boxWidget.SetPlaceFactor(1.0)
    boxWidget.PlaceWidget()
    boxWidget.InsideOutOn()
    boxWidget.AddObserver("InteractionEvent", clipVolumeRender)

    printc("Volume Cutter Tool:", c="m", invert=1)
    printc("  Move gray handles to cut parts of the volume", c="m")

    vp.interactor.Render()
    boxWidget.On()

    vp.interactor.Start()
    boxWidget.Off()
    vp.widgets.append(boxWidget)

#####################################################################
def addIcon(mesh, pos=3, size=0.08):
    """Add an inset icon mesh into the renderer.

    :param pos: icon position in the range [1-4] indicating one of the 4 corners,
                or it can be a tuple (x,y) as a fraction of the renderer size.
    :param float size: size of the icon space.

    |icon| |icon.py|_
    """
    vp = settings.plotter_instance
    if not vp.renderer:
        printc("~lightningWarning: Use addIcon() after first rendering the scene.", c=3)
        save_int = vp.interactive
        vp.show(interactive=0)
        vp.interactive = save_int
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(mesh)
    widget.SetInteractor(vp.interactor)
    if isSequence(pos):
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
    if mesh in vp.actors:
        vp.actors.remove(mesh)
    return widget

#####################################################################
def computeVisibleBounds():
    """Calculate max meshs bounds and sizes."""
    bns = []
    for a in settings.plotter_instance.actors:
        if a and a.GetPickable():
            b = a.GetBounds()
            if b:
                bns.append(b)
    if len(bns):
        max_bns = np.max(bns, axis=0)
        min_bns = np.min(bns, axis=0)
        vbb = (min_bns[0], max_bns[1], min_bns[2], max_bns[3], min_bns[4], max_bns[5])
    else:
        vbb = settings.plotter_instance.renderer.ComputeVisiblePropBounds()
        max_bns = vbb
        min_bns = vbb
    sizes = np.array([max_bns[1]-min_bns[0], max_bns[3]-min_bns[2], max_bns[5]-min_bns[4]])
    return [vbb, sizes, min_bns, max_bns]


#####################################################################
def buildAxes(obj=None,
              xtitle=None, ytitle=None, ztitle=None,
              xrange=None, yrange=None, zrange=None,
              c=None,
              numberOfDivisions=None,
              limitRatio=0.04,
              axesLineWidth=1,
              gridLineWidth=1,
              reorientShortTitle=True,
              titleDepth=0,
              xTitlePosition=0.95, yTitlePosition=0.95, zTitlePosition=0.95,
              xTitleOffset=0.05, yTitleOffset=0.05, zTitleOffset=0.05,
              xTitleJustify="top-right", yTitleJustify="bottom-right", zTitleJustify="bottom-right",
              xTitleRotation=0, yTitleRotation=90, zTitleRotation=135,
              xTitleSize=0.025, yTitleSize=0.025, zTitleSize=0.025,
              xTitleColor=None, yTitleColor=None, zTitleColor=None,
              xTitleBackfaceColor=None, yTitleBackfaceColor=None, zTitleBackfaceColor=None,
              xKeepAspectRatio=True, yKeepAspectRatio=True, zKeepAspectRatio=True,
              xyGrid=True, yzGrid=True, zxGrid=False,
              xyGrid2=False, yzGrid2=False, zxGrid2=False,
              xyGridTransparent=False, yzGridTransparent=False, zxGridTransparent=False,
              xyGrid2Transparent=False, yzGrid2Transparent=False, zxGrid2Transparent=False,
              xyPlaneColor=None, yzPlaneColor=None, zxPlaneColor=None,
              xyGridColor=None, yzGridColor=None, zxGridColor=None,
              xyAlpha=0.05, yzAlpha=0.05, zxAlpha=0.05,
              xyFrameLine=None, yzFrameLine=None, zxFrameLine=None,
              xLineColor=None, yLineColor=None, zLineColor=None,
              xOriginMarkerSize=0, yOriginMarkerSize=0, zOriginMarkerSize=0,
              xHighlightZero=False, yHighlightZero=False, zHighlightZero=False,
              xHighlightZeroColor=(1,0,0), yHighlightZeroColor=(0,1,0), zHighlightZeroColor=(0,0,1),
              showTicks=True,
              xTickLength=0.015, yTickLength=0.015, zTickLength=0.015,
              xTickThickness=0.0025, yTickThickness=0.0025, zTickThickness=0.0025,
              xTickColor=None, yTickColor=None, zTickColor=None,
              xMinorTicks=1, yMinorTicks=1, zMinorTicks=1,
              tipSize=None,
              xLabelSize=0.0175, yLabelSize=0.0175, zLabelSize=0.0175,
              xLabelOffset=0.015, yLabelOffset=0.015, zLabelOffset=0.01,
              xPositionsAndLabels=None, yPositionsAndLabels=None, zPositionsAndLabels=None,
              useGlobal=False,
              tol=0.0001,
              ):
    """Draw axes on a ``Mesh`` or ``Volume``.
    Returns a ``Assembly`` object.

    - `xtitle`,                ['x'], x-axis title text
    - `xrange`,               [None], x-axis range in format (xmin, ymin), default is automatic.
    - `numberOfDivisions`,    [None], approximate number of divisions on the longest axis
    - `axesLineWidth`,           [1], width of the axes lines
    - `gridLineWidth`,           [1], width of the grid lines
    - `reorientShortTitle`,   [True], titles shorter than 2 letter are placed horizontally
    - `originMarkerSize`,     [0.01], draw a small cube on the axis where the origin is
    - `titleDepth`,              [0], extrusion fractional depth of title text
    - `xyGrid`,               [True], show a gridded wall on plane xy
    - `yzGrid`,               [True], show a gridded wall on plane yz
    - `zxGrid`,               [True], show a gridded wall on plane zx
    - `zxGrid2`,             [False], show zx plane on opposite side of the bounding box
    - `xyGridTransparent`    [False], make grid plane completely transparent
    - `xyGrid2Transparent`   [False], make grid plane completely transparent on opposite side box
    - `xyPlaneColor`,       ['gray'], color of the plane
    - `xyGridColor`,        ['gray'], grid line color
    - `xyAlpha`,              [0.15], grid plane opacity
    - `xyFrameLine`,          [None], add a frame for the plane
    - `showTicks`,            [True], show major ticks
    - `xTitlePosition`,       [0.32], title fractional positions along axis
    - `xTitleOffset`,         [0.05], title fractional offset distance from axis line
    - `xTitleJustify`, ["top-right"], title justification
    - `xTitleRotation`,          [0], add a rotation of the axis title
    - `xLineColor`,      [automatic], color of the x-axis
    - `xTitleColor`,     [automatic], color of the axis title
    - `xTitleBackfaceColor`,  [None],  color of axis title on its backface
    - `xTitleSize`,          [0.025], size of the axis title
    - `xHighlightZero`,       [True], draw a line highlighting zero position if in range
    - `xHighlightZeroColor`, [autom], color of the line highlighting the zero position
    - `xTickLength`,         [0.005], radius of the major ticks
    - `xTickThickness`,     [0.0025], thickness of the major ticks along their axis
    - `xTickColor`,      [automatic], color of major ticks
    - `xMinorTicks`,             [1], number of minor ticks between two major ticks
    - `tipSize`,              [0.01], size of the arrow tip
    - `xPositionsAndLabels`       [], assign custom tick positions and labels [(pos1, label1), ...]
    - `xLabelPrecision`,         [2], nr. of significative digits to be shown
    - `xLabelSize`,          [0.015], size of the numeric labels along axis
    - `xLabelOffset`,        [0.025], offset of numeric labels
    - `limitRatio`,           [0.04], below this ratio don't plot small axis

    :Example:

        .. code-block:: python

            from vtkplotter import Box, show

            b = Box(pos=(1,2,3), length=8, width=9, height=7).alpha(0)
            bax = buildAxes(b, c='white')  # returns Assembly object

            show(b, bax)

    |customAxes| |customAxes.py|_
    """
    ncolls = len(settings.collectable_actors)
    if c is None:  # automatic black or white
        c = (0.9, 0.9, 0.9)
        bgcol = (0,0,0)
        vp = settings.plotter_instance
        if vp and vp.renderer:
            bgcol = vp.renderer.GetBackground()
        else:
            if isinstance(obj, Mesh):
                bgcol = obj.color()
        if np.sum(bgcol) > 1.5:
            c = (0.1, 0.1, 0.1)
    else:
        c = getColor(c)

    if useGlobal:
        vbb, ss, min_bns, max_bns = computeVisibleBounds()
    else:
        if obj:
            vbb = list(obj.GetBounds())
            ss = np.array([vbb[1]-vbb[0], vbb[3]-vbb[2], vbb[5]-vbb[4]])
            min_bns = vbb
            max_bns = vbb
        else:
            vbb = np.zeros(6)
            ss = np.zeros(3)
            if xrange is None or yrange is None or zrange is None:
                printc("ERROR in buildAxes(): no mesh given, so must specify ranges.", c='r')
                raise RuntimeError()

    if xrange is not None:
        vbb[0], vbb[1] = xrange
        ss[0] = xrange[1] - xrange[0]
        min_bns = vbb
        max_bns = vbb
    if yrange is not None:
        vbb[2], vbb[3] = yrange
        ss[1] = yrange[1] - yrange[0]
        min_bns = vbb
        max_bns = vbb
    if zrange is not None:
        vbb[4], vbb[5] = zrange
        ss[2] = zrange[1] - zrange[0]
        min_bns = vbb
        max_bns = vbb

    if xtitle is None: xtitle = settings.xtitle
    if ytitle is None: ytitle = settings.ytitle
    if ztitle is None: ztitle = settings.ztitle

    ssmax = max(ss)
    if not ssmax:
        return
    if ss[0]/ssmax < limitRatio:
        ss[0] = 0
        xtitle = ''
    if ss[1]/ssmax < limitRatio:
        ss[1] = 0
        ytitle = ''
    if ss[2]/ssmax < limitRatio:
        ss[2] = 0
        ztitle = ''

    if not xTitleColor:  xTitleColor = c
    if not yTitleColor:  yTitleColor = c
    if not zTitleColor:  zTitleColor = c
    if not xyPlaneColor: xyPlaneColor = c
    if not yzPlaneColor: yzPlaneColor = c
    if not zxPlaneColor: zxPlaneColor = c
    if not xyGridColor: xyGridColor = c
    if not yzGridColor: yzGridColor = c
    if not zxGridColor: zxGridColor = c
    if not xLineColor:  xLineColor = c
    if not yLineColor:  yLineColor = c
    if not zLineColor:  zLineColor = c
    if not xTickColor:  xTickColor = xLineColor
    if not yTickColor:  yTickColor = yLineColor
    if not zTickColor:  zTickColor = zLineColor
    if settings.useDepthPeeling:
        xyGrid = False
        yzGrid = False
        zxGrid = False
        xyGrid2 = False
        yzGrid2 = False
        zxGrid2 = False

    ndiv = 4
    if not ztitle or not ytitle or not xtitle: # make more default ticks if 2D
        ndiv = 6
        if not ztitle:
            if xyFrameLine is None:
                xyFrameLine = True
            if tipSize is None:
                tipSize = False

    if tipSize is None:
        tipSize = 0.006

    if not numberOfDivisions: numberOfDivisions = ndiv

    rx, ry, rz = np.round(ss/max(ss)*numberOfDivisions+1).astype(int)
    #printc('numberOfDivisions', numberOfDivisions, '\t r=', rx, ry, rz)

    if xtitle: xticks_float, xticks_str = make_ticks(vbb[0], vbb[1], rx, xPositionsAndLabels)
    if ytitle: yticks_float, yticks_str = make_ticks(vbb[2], vbb[3], ry, yPositionsAndLabels)
    if ztitle: zticks_float, zticks_str = make_ticks(vbb[4], vbb[5], rz, zPositionsAndLabels)


    ################################## calculate aspect ratio scales
    x_aspect_ratio_scale=1
    y_aspect_ratio_scale=1
    z_aspect_ratio_scale=1
    if xtitle:
        if ss[0] > ss[1]:
            x_aspect_ratio_scale = (1, ss[0]/ss[1], 1)
        else:
            x_aspect_ratio_scale = (ss[1]/ss[0], 1, 1)
    if ytitle:
        if ss[0] > ss[1]:
            y_aspect_ratio_scale = (ss[0]/ss[1], 1, 1)
        else:
            y_aspect_ratio_scale = (1, ss[1]/ss[0], 1)
    if ztitle:
        smean = (ss[0]+ss[1])/2
        if smean:
            if ss[2] > smean:
                zarfact = smean/ss[2]
                z_aspect_ratio_scale = (zarfact, zarfact*ss[2]/smean, zarfact)
            else:
                z_aspect_ratio_scale = (smean/ss[2], 1, 1)


    ################################################ axes lines
    lines = []
    if xtitle:
        axlinex = shapes.Line([0, 0, 0], [1, 0, 0], c=xLineColor, lw=axesLineWidth)
        axlinex.name = 'xAxis'
        lines.append(axlinex)
    if ytitle:
        axliney = shapes.Line([0, 0, 0], [0, 1, 0], c=yLineColor, lw=axesLineWidth)
        axliney.name = 'yAxis'
        lines.append(axliney)
    if ztitle:
        axlinez = shapes.Line([0, 0, 0], [0, 0, 1], c=zLineColor, lw=axesLineWidth)
        axlinez.name = 'zAxis'
        lines.append(axlinez)

    ################################################ grid planes
    # all shapes have a name to keep track of them in the Assembly if user wants to unpack it
    grids = []
    if xyGrid and xtitle and ytitle:
        gxy = shapes.Grid(sx=xticks_float, sy=yticks_float)
        gxy.alpha(xyAlpha).wireframe(xyGridTransparent).c(xyPlaneColor).lw(gridLineWidth).lc(xyGridColor)
        gxy.name = "xyGrid"
        grids.append(gxy)
    if yzGrid and ytitle and ztitle:
        gyz = shapes.Grid(sx=zticks_float, sy=yticks_float)
        gyz.RotateY(-90)
        gyz.alpha(yzAlpha).wireframe(yzGridTransparent).c(yzPlaneColor).lw(gridLineWidth).lc(yzGridColor)
        gyz.name = "yzGrid"
        grids.append(gyz)
    if zxGrid and ztitle and xtitle:
        gzx = shapes.Grid(sx=xticks_float, sy=zticks_float)
        gzx.RotateX(90)
        gzx.alpha(zxAlpha).wireframe(zxGridTransparent).c(zxPlaneColor).lw(gridLineWidth).lc(zxGridColor)
        gzx.name = "zxGrid"
        grids.append(gzx)

    grids2 = []
    if xyGrid2 and xtitle and ytitle:
        gxy2 = shapes.Grid(sx=xticks_float, sy=yticks_float).z(1)
        gxy2.alpha(xyAlpha).wireframe(xyGrid2Transparent).c(xyPlaneColor).lw(gridLineWidth).lc(xyGridColor)
        gxy2.name = "xyGrid2"
        grids2.append(gxy2)
    if yzGrid2 and ytitle and ztitle:
        gyz2 = shapes.Grid(sx=zticks_float, sy=yticks_float).x(1)
        gyz2.RotateY(-90)
        gyz2.alpha(yzAlpha).wireframe(yzGrid2Transparent).c(yzPlaneColor).lw(gridLineWidth).lc(yzGridColor)
        gyz2.name = "yzGrid2"
        grids2.append(gyz2)
    if zxGrid2 and ztitle and xtitle:
        gzx2 = shapes.Grid(sx=xticks_float, sy=zticks_float).y(1)
        gzx2.RotateX(90)
        gzx2.alpha(zxAlpha).wireframe(zxGrid2Transparent).c(zxPlaneColor).lw(gridLineWidth).lc(zxGridColor)
        gzx2.name = "zxGrid2"
        grids2.append(gzx2)

    ################################################ frame lines
    framelines = []
    if xyFrameLine and xtitle and ytitle:
        frxy = shapes.Line([[0,1,0],[1,1,0],[1,0,0]], c=xLineColor, lw=axesLineWidth)
        frxy.name = 'xyFrameLine'
        framelines.append(frxy)
    if yzFrameLine and ytitle and ztitle:
        frxy = shapes.Line([[0,0,1],[0,1,1],[0,1,0]], c=yLineColor, lw=axesLineWidth)
        frxy.name = 'yzFrameLine'
        framelines.append(frxy)
    if zxFrameLine and ztitle and xtitle:
        frxy = shapes.Line([[0,0,1],[1,0,1],[1,0,0]], c=xLineColor, lw=axesLineWidth)
        frxy.name = 'zxFrameLine'
        framelines.append(frxy)

    ################################################ zero lines highlights
    highlights = []
    if xyGrid and xtitle and ytitle:
        if xHighlightZero and min_bns[0] <= 0 and max_bns[1] > 0:
            xhl = -min_bns[0] / ss[0]
            hxy = shapes.Line([xhl,0,0], [xhl,1,0], c=xHighlightZeroColor)
            hxy.alpha(np.sqrt(xyAlpha)).lw(gridLineWidth*2)
            hxy.name = "xyHighlightZero"
            highlights.append(hxy)
        if yHighlightZero and min_bns[2] <= 0 and max_bns[3] > 0:
            yhl = -min_bns[2] / ss[1]
            hyx = shapes.Line([0,yhl,0], [1,yhl,0], c=yHighlightZeroColor)
            hyx.alpha(np.sqrt(yzAlpha)).lw(gridLineWidth*2)
            hyx.name = "yxHighlightZero"
            highlights.append(hyx)

    if yzGrid and ytitle and ztitle:
        if yHighlightZero and min_bns[2] <= 0 and max_bns[3] > 0:
            yhl = -min_bns[2] / ss[1]
            hyz = shapes.Line([0,yhl,0], [0,yhl,1], c=yHighlightZeroColor)
            hyz.alpha(np.sqrt(yzAlpha)).lw(gridLineWidth*2)
            hyz.name = "yzHighlightZero"
            highlights.append(hyz)
        if zHighlightZero and min_bns[4] <= 0 and max_bns[5] > 0:
            zhl = -min_bns[4] / ss[2]
            hzy = shapes.Line([0,0,zhl], [0,1,zhl], c=zHighlightZeroColor)
            hzy.alpha(np.sqrt(yzAlpha)).lw(gridLineWidth*2)
            hzy.name = "zyHighlightZero"
            highlights.append(hzy)

    if zxGrid and ztitle and xtitle:
        if zHighlightZero and min_bns[4] <= 0 and max_bns[5] > 0:
            zhl = -min_bns[4] / ss[2]
            hzx = shapes.Line([0,0,zhl], [1,0,zhl], c=zHighlightZeroColor)
            hzx.alpha(np.sqrt(zxAlpha)).lw(gridLineWidth*2)
            hzx.name = "zxHighlightZero"
            highlights.append(hzx)
        if xHighlightZero and min_bns[0] <= 0 and max_bns[1] > 0:
            xhl = -min_bns[0] / ss[0]
            hxz = shapes.Line([xhl,0,0], [xhl,0,1], c=xHighlightZeroColor)
            hxz.alpha(np.sqrt(zxAlpha)).lw(gridLineWidth*2)
            hxz.name = "xzHighlightZero"
            highlights.append(hxz)

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
        xt.RotateX(xTitleRotation)
        xt.pos(wpos)
        xt.name = "xtitle "+str(xtitle)
        titles.append(xt)

    if ytitle:
        yt = shapes.Text(ytitle, pos=(0, 0, 0), s=yTitleSize, bc=yTitleBackfaceColor,
                         c=yTitleColor, justify=yTitleJustify, depth=titleDepth)
        if reorientShortTitle and len(ytitle) < 3:  # title is short
            wpos = [-yTitleOffset +0.03-0.01*len(ytitle), yTitlePosition, 0]
            if yKeepAspectRatio: yt.SetScale(x_aspect_ratio_scale) #x!
        else:
            wpos = [-yTitleOffset, yTitlePosition, 0]
            if yKeepAspectRatio: yt.SetScale(y_aspect_ratio_scale)
            yt.RotateZ(yTitleRotation)
        yt.pos(wpos)
        yt.name = "ytitle "+str(ytitle)
        titles.append(yt)

    if ztitle:
        zt = shapes.Text(ztitle, pos=(0, 0, 0), s=zTitleSize, bc=zTitleBackfaceColor,
                         c=zTitleColor, justify=zTitleJustify, depth=titleDepth)
        if reorientShortTitle and len(ztitle) < 3:  # title is short
            wpos = [(-zTitleOffset+0.02-0.003*len(ztitle))/1.42,
                    (-zTitleOffset+0.02-0.003*len(ztitle))/1.42, zTitlePosition]
            if zKeepAspectRatio:
                zr2 = (z_aspect_ratio_scale[1], z_aspect_ratio_scale[0], z_aspect_ratio_scale[2])
                zt.SetScale(zr2)
            zt.RotateX(90)
            zt.RotateY(45)
            zt.pos(wpos)
        else:
            if zKeepAspectRatio: zt.SetScale(z_aspect_ratio_scale)
            wpos = [-zTitleOffset/1.42, -zTitleOffset/1.42, zTitlePosition]
            zt.RotateY(-90)
            zt.RotateX(zTitleRotation)
            zt.pos(wpos)
        zt.name = "ztitle "+str(ztitle)
        titles.append(zt)

    ################################################ cube origin ticks
    originmarks = []
    if xOriginMarkerSize:
        if xtitle:
            if min_bns[0] <= 0 and max_bns[1] > 0:  # mark x origin
                ox = shapes.Cube([-min_bns[0] / ss[0], 0, 0], side=xOriginMarkerSize, c=xLineColor)
                ox.name = "xOriginMarker"
                originmarks.append(ox)

    if yOriginMarkerSize:
        if ytitle:
            if min_bns[2] <= 0 and max_bns[3] > 0:  # mark y origin
                oy = shapes.Cube([0, -min_bns[2] / ss[1], 0], side=yOriginMarkerSize, c=yLineColor)
                oy.name = "yOriginMarker"
                originmarks.append(oy)

    if zOriginMarkerSize:
        if ztitle:
            if min_bns[4] <= 0 and max_bns[5] > 0:  # mark z origin
                oz = shapes.Cube([0, 0, -min_bns[4] / ss[2]], side=zOriginMarkerSize, c=zLineColor)
                oz.name = "zOriginMarker"
                originmarks.append(oz)

    ################################################ arrow cone
    cones = []
    if tipSize:
        if xtitle:
            cx = shapes.Cone((1,0,0), r=tipSize, height=tipSize*2, axis=(1,0,0), c=xLineColor, res=10)
            cx.name = "xTipCone"
            cones.append(cx)
        if ytitle:
            cy = shapes.Cone((0,1,0), r=tipSize, height=tipSize*2, axis=(0,1,0), c=yLineColor, res=10)
            cy.name = "yTipCone"
            cones.append(cy)
        if ztitle:
            cz = shapes.Cone((0,0,1), r=tipSize, height=tipSize*2, axis=(0,0,1), c=zLineColor, res=10)
            cz.name = "zTipCone"
            cones.append(cz)

    ################################################ MAJOR ticks
    majorticks, minorticks= [], []
    xticks, yticks, zticks = [],[],[]
    if showTicks:
        if xtitle:
            for i in range(1, len(xticks_float)-1):
                v1 = (xticks_float[i]-xTickThickness/2, -xTickLength/2, 0)
                v2 = (xticks_float[i]+xTickThickness/2,  xTickLength/2, 0)
                xticks.append(shapes.Rectangle(v1, v2))
            if len(xticks)>1:
                xmajticks = merge(xticks).c(xTickColor)
                xmajticks.name = "xMajorTicks"
                majorticks.append(xmajticks)
        if ytitle:
            for i in range(1, len(yticks_float)-1):
                v1 = (-yTickLength/2, yticks_float[i]-yTickThickness/2, 0)
                v2 = ( yTickLength/2, yticks_float[i]+yTickThickness/2, 0)
                yticks.append(shapes.Rectangle(v1, v2))
            if len(yticks)>1:
                ymajticks = merge(yticks).c(yTickColor)
                ymajticks.name = "yMajorTicks"
                majorticks.append(ymajticks)
        if ztitle:
            for i in range(1, len(zticks_float)-1):
                v1 = (zticks_float[i]-zTickThickness/2, -zTickLength/2.84, 0)
                v2 = (zticks_float[i]+zTickThickness/2,  zTickLength/2.84, 0)
                zticks.append(shapes.Rectangle(v1, v2))
            if len(zticks)>1:
                zmajticks = merge(zticks).c(zTickColor)
                zmajticks.RotateZ(-45)
                zmajticks.RotateY(-90)
                zmajticks.name = "zMajorTicks"
                majorticks.append(zmajticks)

        ################################################ MINOR ticks
        if xMinorTicks and xtitle and len(xticks)>1:
            xMinorTicks += 1
            ticks = []
            for i in range(1,len(xticks)):
                t0, t1 = xticks[i-1].pos(), xticks[i].pos()
                dt = t1 - t0
                for j in range(1, xMinorTicks):
                    mt = dt*(j/xMinorTicks) + t0
                    v1 = (mt[0]-xTickThickness/4, -xTickLength/4, 0)
                    v2 = (mt[0]+xTickThickness/4,  xTickLength/4, 0)
                    ticks.append(shapes.Rectangle(v1, v2))
            if len(ticks):
                xminticks = merge(ticks).c(xTickColor)
                xminticks.name = "xMinorTicks"
                minorticks.append(xminticks)

        if yMinorTicks and ytitle and len(yticks)>1:
            yMinorTicks += 1
            ticks = []
            for i in range(1,len(yticks)):
                t0, t1 = yticks[i-1].pos(), yticks[i].pos()
                dt = t1 - t0
                for j in range(1, yMinorTicks):
                    mt = dt*(j/yMinorTicks) + t0
                    v1 = (-yTickLength/4, mt[1]-yTickThickness/4, 0)
                    v2 = ( yTickLength/4, mt[1]+yTickThickness/4, 0)
                    ticks.append(shapes.Rectangle(v1, v2))
            if len(ticks):
                yminticks = merge(ticks).c(yTickColor)
                yminticks.name = "yMinorTicks"
                minorticks.append(yminticks)

        if zMinorTicks and ztitle and len(zticks)>1:
            zMinorTicks += 1
            ticks = []
            for i in range(1,len(zticks)):
                t0, t1 = zticks[i-1].pos(), zticks[i].pos()
                dt = t1 - t0
                for j in range(1, zMinorTicks):
                    mt = dt*(j/zMinorTicks) + t0
                    v1 = (mt[0]-zTickThickness/4, -zTickLength/5., 0)
                    v2 = (mt[0]+zTickThickness/4,  zTickLength/5., 0)
                    ticks.append(shapes.Rectangle(v1, v2))
            if len(ticks):
                zminticks = merge(ticks).c(zTickColor)
                zminticks.RotateZ(-45)
                zminticks.RotateY(-90)
                zminticks.name = "zMinorTicks"
                minorticks.append(zminticks)


    ################################################ axes tick NUMERIC text labels
    labels = []
    if xLabelSize and xtitle:
        for i in range(1, len(xticks_str)):
            t = xticks_str[i]
            if not t: continue
            v = (xticks_float[i], -xLabelOffset, 0)
            xlab = shapes.Text(t, pos=v, s=xLabelSize, justify="center-top", depth=0)
            if xKeepAspectRatio: xlab.SetScale(x_aspect_ratio_scale)
            xlab.name = "xNumericLabel"+str(i)+" "+t
            labels.append(xlab.c(xTickColor))

    if yLabelSize and ytitle:
        for i in range(1,len(yticks_str)):
            t = yticks_str[i]
            if not t: continue
            v = (-yLabelOffset, yticks_float[i], 0)
            ylab = shapes.Text(t, pos=(0,0,0), s=yLabelSize, justify="center-bottom", depth=0)
            if yKeepAspectRatio: ylab.SetScale(y_aspect_ratio_scale)
            ylab.RotateZ(yTitleRotation)
            ylab.pos(v)
            ylab.name = "yNumericLabel"+str(i)+" "+t
            labels.append(ylab.c(yTickColor))

    if zLabelSize and ztitle:
        for i in range(1, len(zticks_str)):
            t = zticks_str[i]
            if not t: continue
            v = (-zLabelOffset, -zLabelOffset, zticks_float[i])
            zlab = shapes.Text(t, pos=(0,0,0), s=zLabelSize, justify="center-bottom", depth=0)
            if zKeepAspectRatio: zlab.SetScale(z_aspect_ratio_scale)
            zlab.RotateY(-90)
            zlab.RotateX(zTitleRotation)
            zlab.pos(v)
            zlab.name = "zNumericLabel"+str(i)+" "+t
            labels.append(zlab.c(zTickColor))

    acts = titles + lines + labels + grids + grids2 + highlights + framelines
    acts += majorticks + minorticks + originmarks + cones

    tol *= mag(ss)
    orig = np.array([min_bns[0], min_bns[2], min_bns[4]]) - tol
    for a in acts:
        a.PickableOff()
        a.SetPosition(a.GetPosition() + orig)
        pr = a.GetProperty()
        pr.SetAmbient(0.9)
        pr.SetDiffuse(0.1)
        pr.SetSpecular(0)
    asse = Assembly(acts)
    asse.SetOrigin(orig)
    asse.SetScale(ss)
    asse.PickableOff()
    settings.collectable_actors = settings.collectable_actors[:ncolls]
    return asse


def addGlobalAxes(axtype=None, c=None):
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
          - 11, show a large grid on the x-y plane (use with zoom=8)
          - 12, show polar axes

    Axis type-1 can be fully customized by passing a dictionary ``axes=dict()`` where:


        - `xtitle`,                ['x'], x-axis title text
        - `xrange`,               [None], x-axis range in format (xmin, ymin), default is automatic.
        - `numberOfDivisions`,    [None], approximate number of divisions on the longest axis
        - `axesLineWidth`,           [1], width of the axes lines
        - `gridLineWidth`,           [1], width of the grid lines
        - `reorientShortTitle`,   [True], titles shorter than 2 letter are placed horizontally
        - `originMarkerSize`,     [0.01], draw a small cube on the axis where the origin is
        - `titleDepth`,              [0], extrusion fractional depth of title text
        - `xyGrid`,               [True], show a gridded wall on plane xy
        - `yzGrid`,               [True], show a gridded wall on plane yz
        - `zxGrid`,               [True], show a gridded wall on plane zx
        - `zxGrid2`,             [False], show zx plane on opposite side of the bounding box
        - `xyGridTransparent`    [False], make grid plane completely transparent
        - `xyGrid2Transparent`   [False], make grid plane completely transparent on opposite side box
        - `xyPlaneColor`,       ['gray'], color of the plane
        - `xyGridColor`,        ['gray'], grid line color
        - `xyAlpha`,              [0.15], grid plane opacity
        - `xyFrameLine`,          [None], add a frame for the plane
        - `showTicks`,            [True], show major ticks
        - `xTitlePosition`,       [0.32], title fractional positions along axis
        - `xTitleOffset`,         [0.05], title fractional offset distance from axis line
        - `xTitleJustify`, ["top-right"], title justification
        - `xTitleRotation`,          [0], add a rotation of the axis title
        - `xLineColor`,      [automatic], color of the x-axis
        - `xTitleColor`,     [automatic], color of the axis title
        - `xTitleBackfaceColor`,  [None],  color of axis title on its backface
        - `xTitleSize`,          [0.025], size of the axis title
        - `xHighlightZero`,       [True], draw a line highlighting zero position if in range
        - `xHighlightZeroColor`, [autom], color of the line highlighting the zero position
        - `xTickLength`,         [0.005], radius of the major ticks
        - `xTickThickness`,     [0.0025], thickness of the major ticks along their axis
        - `xTickColor`,      [automatic], color of major ticks
        - `xMinorTicks`,             [1], number of minor ticks between two major ticks
        - `tipSize`,              [0.01], size of the arrow tip
        - `xPositionsAndLabels`       [], assign custom tick positions and labels [(pos1, label1), ...]
        - `xLabelPrecision`,         [2], nr. of significative digits to be shown
        - `xLabelSize`,          [0.015], size of the numeric labels along axis
        - `xLabelOffset`,        [0.025], offset of numeric labels
        - `limitRatio`,           [0.04], below this ratio don't plot small axis

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
        vp.axes = axtype  # override

    r = vp.renderers.index(vp.renderer)

    if not vp.axes:
        return

    if c is None:  # automatic black or white
        c = (0.9, 0.9, 0.9)
        if np.sum(vp.renderer.GetBackground()) > 1.5:
            c = (0.1, 0.1, 0.1)
    else:
        c = getColor(c) # for speed

    if not vp.renderer:
        return

    if vp.axes_instances[r]:
        return

    ############################################################
    # custom grid walls
    if vp.axes == 1 or vp.axes is True or isinstance(vp.axes, dict):

        if isinstance(vp.axes, dict):
            vp.axes.update({'useGlobal':True})
            asse = buildAxes(None, **vp.axes)
        else:
            asse = buildAxes(None, useGlobal=True)

        vp.renderer.AddActor(asse)
        vp.axes_instances[r] = asse


    elif vp.axes == 2 or vp.axes == 3:
        x0, x1, y0, y1, z0, z1 = vp.renderer.ComputeVisiblePropBounds()
        xcol, ycol, zcol = "dr", "dg", "db"
        s = 1
        alpha = 1
        centered = False
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        aves = np.sqrt(dx * dx + dy * dy + dz * dz) / 2
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
            yt.pos(wpos).RotateZ(90)
            acts += [yl, yc, yt]

        if len(vp.ztitle) and dz > aves/100:
            zl = shapes.Cylinder([[0, 0, z0], [0, 0, z1]], r=aves/250*s, c=zcol, alpha=alpha)
            zc = shapes.Cone(pos=[0, 0, z1], c=zcol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[0, 0, 1], res=10)
            wpos = [-aves/50*s, -aves/50*s, z1 - (len(vp.ztitle)+1)*aves/40*s]
            if centered:
                wpos = [-aves/50*s, -aves/50*s, (z0+z1)/2-len(vp.ztitle)/2*aves/40*s]
            zt = shapes.Text(vp.ztitle, pos=(0,0,0), s=aves/40*s, c=zcol)
            zt.pos(wpos).RotateZ(45)
            zt.RotateX(90)
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
        axact.GetXAxisShaftProperty().SetColor(1, 0, 0)
        axact.GetYAxisShaftProperty().SetColor(0, 1, 0)
        axact.GetZAxisShaftProperty().SetColor(0, 0, 1)
        axact.GetXAxisTipProperty().SetColor(1, 0, 0)
        axact.GetYAxisTipProperty().SetColor(0, 1, 0)
        axact.GetZAxisTipProperty().SetColor(0, 0, 1)
        bc = np.array(vp.renderer.GetBackground())
        if np.sum(bc) < 1.5:
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

        axact.GetXPlusFaceProperty().SetColor(getColor("r"))
        axact.GetXMinusFaceProperty().SetColor(getColor("dr"))
        axact.GetYPlusFaceProperty().SetColor(getColor("g"))
        axact.GetYMinusFaceProperty().SetColor(getColor("dg"))
        axact.GetZPlusFaceProperty().SetColor(getColor("b"))
        axact.GetZMinusFaceProperty().SetColor(getColor("db"))
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
            ocf.SetInputData(largestact.unpack(0).GetMapper().GetInput())
        else:
            ocf.SetInputData(largestact.GetMapper().GetInput())
        ocf.Update()
        ocMapper = vtk.vtkHierarchicalPolyDataMapper()
        ocMapper.SetInputConnection(0, ocf.GetOutputPort(0))
        ocActor = vtk.vtkActor()
        ocActor.SetMapper(ocMapper)
        bc = np.array(vp.renderer.GetBackground())
        if np.sum(bc) < 1.5:
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

    elif vp.axes == 9:
        vbb = computeVisibleBounds()[0]
        src = vtk.vtkCubeSource()
        src.SetXLength(vbb[1] - vbb[0])
        src.SetYLength(vbb[3] - vbb[2])
        src.SetZLength(vbb[5] - vbb[4])
        src.Update()
        ca = Mesh(src.GetOutput(), c, 0.5).wireframe(True)
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
        yc = shapes.Disc(x0, r1=rm, r2=rm, c='lg', res=1, resphi=72)
        yc.RotateX(90)
        zc = shapes.Disc(x0, r1=rm, r2=rm, c='lb', res=1, resphi=72)
        yc.RotateY(90)
        xc.clean().alpha(0.2).wireframe().lineWidth(2.5).PickableOff()
        yc.clean().alpha(0.2).wireframe().lineWidth(2.5).PickableOff()
        zc.clean().alpha(0.2).wireframe().lineWidth(2.5).PickableOff()
        ca = xc + yc + zc
        ca.PickableOff()
        vp.renderer.AddActor(ca)
        vp.axes_instances[r] = ca

    elif vp.axes == 11:
        vbb, ss = computeVisibleBounds()[0:2]
        xpos, ypos = (vbb[1] + vbb[0]) /2, (vbb[3] + vbb[2]) /2
        gr = shapes.Grid((xpos, ypos, vbb[4]),
                         sx=ss[0]*8, sy=ss[1]*8,
                         resx=7, resy=7,
                         c=c, alpha=0.2)
        gr.lighting('ambient').PickableOff()
        vp.renderer.AddActor(gr)
        vp.axes_instances[r] = gr

    elif vp.axes == 12:
        polaxes = vtk.vtkPolarAxesActor()
        vbb = computeVisibleBounds()[0]

        if vp.xtitle == 'x':
            polaxes.SetPolarAxisTitle('radial distance')
        else:
            polaxes.SetPolarAxisTitle(vp.xtitle)
        polaxes.SetPole(0,0, vbb[4])
        rd = max(abs(vbb[0]), abs(vbb[2]), abs(vbb[1]), abs(vbb[3]))
        polaxes.SetMaximumRadius(rd)
        polaxes.AutoSubdividePolarAxisOff()
        polaxes.SetNumberOfPolarAxisTicks(10)
        polaxes.SetCamera(vp.renderer.GetActiveCamera())
        polaxes.SetPolarLabelFormat("%6.1f")
        polaxes.PolarLabelVisibilityOff() # due to bad overlap of labels

        polaxes.GetPolarArcsProperty().SetColor(c)
        polaxes.GetPolarAxisProperty().SetColor(c)
        polaxes.GetPolarAxisTitleTextProperty().SetColor(c)
        polaxes.GetPolarAxisLabelTextProperty().SetColor(c)
        polaxes.GetLastRadialAxisTextProperty().SetColor(c)
        polaxes.GetSecondaryRadialAxesTextProperty().SetColor(c)
        polaxes.GetSecondaryRadialAxesProperty().SetColor(c)
        polaxes.GetSecondaryPolarArcsProperty().SetColor(c)

        polaxes.SetMinimumAngle(0.)
        polaxes.SetMaximumAngle(315.)
        polaxes.SetNumberOfPolarAxisTicks(5)
        polaxes.PickableOff()
        vp.renderer.AddActor(polaxes)
        vp.axes_instances[r] = polaxes

    else:
        printc('~bomb Keyword axes must be in range [0-10].', c=1)
        printc('''
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
  11 = show a large grid on the x-y plane (use with zoom=8)
  12 = show polar axes.
  ''', c=1, bold=0)

    if not vp.axes_instances[r]:
        vp.axes_instances[r] = True
    return


#####################################################################
def addRendererFrame(c=None, alpha=0.5, bg=None, lw=0.5):

    if c is None:  # automatic black or white
        c = (0.9, 0.9, 0.9)
        if np.sum(settings.plotter_instance.renderer.GetBackground())>1.5:
            c = (0.1, 0.1, 0.1)
    c = getColor(c)

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
    if not isSequence(vp._legend):
        return

    # remove old legend if present on current renderer:
    acs = vp.renderer.GetActors2D()
    acs.InitTraversal()
    for i in range(acs.GetNumberOfItems()):
        a = acs.GetNextItem()
        if isinstance(a, vtk.vtkLegendBoxActor):
            vp.renderer.RemoveActor(a)

    meshs = vp.getMeshes()
    acts, texts = [], []
    for i, a in enumerate(meshs):
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
        self.textproperty.SetColor(getColor(self.colors[s]))
        bcc = np.array(getColor(self.bcolors[s]))
        self.textproperty.SetBackgroundColor(bcc)
        if self.showframe:
            self.textproperty.FrameOn()
            self.textproperty.SetFrameWidth(self.framewidth)
            self.textproperty.SetFrameColor(np.sqrt(bcc))
        return self

    def switch(self):
        """
        Change/cycle button status to the next defined status in states list.
        """
        self._status = (self._status + 1) % len(self.states)
        self.status(self._status)
        return self
