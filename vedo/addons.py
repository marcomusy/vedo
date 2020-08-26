#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import vedo
from vedo.colors import printc, getColor
from vedo.assembly import Assembly
from vedo.mesh import Mesh, merge
from vedo.pointcloud import Points
from vedo.utils import mag, isSequence, make_ticks
from vedo.utils import versor, ctf2lut, vtkVersionIsAtLeast
from vedo.utils import precision as nrprecision
import vedo.shapes as shapes
import vedo.settings as settings
import vedo.docs as docs
from vedo.volume import Volume
from vedo.tetmesh import TetMesh
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
        "buildAxes",
        "buildRulerAxes",
        "Goniometer",
        "Ruler",
        ]


#####################################################################
def Ruler(
    p1, p2,
    label="",
    s=None,
    font="",
    italic=0,
    prefix="",
    units="",  #eg.'μm'
    c=(0.2, 0.1, 0.1),
    alpha=1,
    lw=1,
    precision=3,
    labelRotation=0,
    axisRotation=0,
    tickAngle=90,
):
    """
    Build a 3D ruler to indicate the distance of two points p1 and p2.

    :param str label: alternative fixed label to be shown
    :param float s: size of the label
    :param str font: font name
    :param float italic: italicness of the font [0,1]
    :param str units: string to be appended to the numeric value
    :param float lw: line width in pixel units
    :param int precision: nr of significant digits to be shown
    :param float labelRotation: initial rotation of the label around the z-axis
    :param float axisRotation: initial rotation of the line around the main axis
    :param float tickAngle: initial rotation of the line around the main axis

    |goniometer| |goniometer.py|_
    """
    ncolls = len(settings.collectable_actors)

    if isinstance(p1, Points): p1 = p1.GetPosition()
    if isinstance(p2, Points): p2 = p2.GetPosition()
    if len(p1)==2: p1=[p1[0],p1[1],0.0]
    if len(p2)==2: p2=[p2[0],p2[1],0.0]
    p1, p2 = np.array(p1), np.array(p2)
    q1, q2 = [0, 0, 0], [mag(p2 - p1), 0, 0]
    q1, q2 = np.array(q1), np.array(q2)
    v = q2 - q1
    d = mag(v)

    if s is None:
        s = d*0.02

    if not label:
        label = str(d)
        if precision:
            label = nrprecision(d, precision)
    if prefix:
        label = prefix+ '~' + label
    if units:
        label += '~'+ units

    lb = shapes.Text(label, pos=(q1 + q2) / 2, s=s,
                     font=font, italic=italic, justify="center")
    if labelRotation:
        lb.RotateZ(labelRotation)

    x0, x1 = lb.xbounds()
    gap = [(x1 - x0) / 2, 0, 0]
    pc1 = (v / 2 - gap) * 0.9 + q1
    pc2 = q2 - (v / 2 - gap) * 0.9

    lc1 = shapes.Line(q1 - v / 50, pc1)
    lc2 = shapes.Line(q2 + v / 50, pc2)

    zs = np.array([0, d / 50, 0])
    ml1 = shapes.Line(-zs, zs).pos(q1)
    ml2 = shapes.Line(-zs, zs).pos(q2)
    ml1.RotateZ(tickAngle-90)
    ml2.RotateZ(tickAngle-90)

    c1 = shapes.Circle(q1, r=d / 180, res=20)
    c2 = shapes.Circle(q2, r=d / 180, res=20)

    acts = [lb, lc1, lc2, c1, c2, ml1, ml2]
    macts = merge(acts).pos(p1).c(c).alpha(alpha)
    macts.GetProperty().LightingOff()
    macts.GetProperty().SetLineWidth(lw)
    macts.base = q1
    macts.top = q2
    macts.orientation(p2 - p1, rotation=axisRotation).bc('t').pickable(False)
    settings.collectable_actors = settings.collectable_actors[:ncolls]
    return macts


#####################################################################
def Goniometer(
        p1,p2,p3,
        font="",
        arcSize=0.4,
        fill=0.1,
        s=1,
        italic=0,
        rotation=0,
        prefix="",
        c=(0.2, 0, 0),
        alpha=1,
        lw=1,
        precision=3,
    ):
    """
    Build a graphical goniometer to measure the angle formed by 3 points in space.

    Parameters
    ----------
    p1 : list
        first point.
    p2 : list
        the vertex point.
    p3 : list
        the last point defining the angle.
    font : str, optional
        Font name to be used. The default is "".
    arcSize : float, optional
        dimension of the arc wrt the smallest axis. The default is 0.4.
    fill : bool, optional
        fill the arc area. The default is 0.1.
    s : float, optional
        size of the text. The default is 1.
    italic : float, bool, optional
        italic text. The default is 0.
    rotation : float, optional
        rotation of text in degrees. The default is 0.
    prefix : str, optional
        append this string to the numeric value of the angle. The default is "".
    c : list, optional
        color of the goniometer. The default is (0.2, 0, 0).
    alpha : float, optional
        transparency level. The default is 1.
    lw : float, optional
        line width. The default is 1.
    precision : int, optional
        number of significant digits. The default is 3.


    |goniometer| |goniometer.py|_
    """
    ncolls = len(settings.collectable_actors)

    if isinstance(p1, Points): p1 = p1.GetPosition()
    if isinstance(p2, Points): p2 = p2.GetPosition()
    if isinstance(p3, Points): p3 = p3.GetPosition()
    if len(p1)==2: p1=[p1[0], p1[1], 0.0]
    if len(p2)==2: p2=[p2[0], p2[1], 0.0]
    if len(p3)==2: p3=[p3[0], p3[1], 0.0]
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

    acts=[]
    ln = shapes.Line([p1,p2,p3], lw=lw, c=c).alpha(alpha).lighting('off')
    acts.append(ln)

    va = versor(p1-p2)
    vb = versor(p3-p2)
    r = min(mag(p3-p2), mag(p1-p2))*arcSize
    ptsarc = []
    res = 120
    imed = int(res/2)
    for i in range(res+1):
        vi = versor(vb*i/res + va*(res-i)/res)
        if i==imed: vc = np.array(vi)
        ptsarc.append(p2+vi*r)
    arc = shapes.Line(ptsarc).lw(lw).c(c).alpha(alpha).lighting('off')
    acts.append(arc)

    angle = np.arccos(np.dot(va,vb))*180/np.pi

    lb = shapes.Text(prefix+nrprecision(angle,precision)+'º', s=r/12*s,
                     font=font, italic=italic, justify="center")
    cr = np.cross(va,vb)
    lb.pos(p2+vc*r/1.75).orientation(cr*np.sign(cr[2]), rotation=rotation)
    # lb.base=np.array([0,0,0])
    # lb.top=np.array([1,0,0])
    # lb.pos(p2+vc*r/1.75).orientation(va, rotation=0)
    lb.c(c).alpha(alpha).bc('tomato').lighting('off')
    acts.append(lb)

    if fill:
        pts = [p2] + arc.points().tolist() + [p2]
        msh = Mesh([pts, [list(range(arc.N()+2))]], c=c, alpha=fill).triangulate()
        msh.addPos(0,0,r/10000) # to resolve 2d conflicts..
        acts.append(msh)

    asse = Assembly(acts)
    settings.collectable_actors = settings.collectable_actors[:ncolls]
    return asse


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

        self.framewidth = 2
        self.offset = 5
        self.spacer = " "

        self.textproperty = self.actor.GetTextProperty()
        self.textproperty.SetJustificationToCentered()
        if font.lower() == "courier":
            self.textproperty.SetFontFamilyToCourier()
        elif font.lower() == "times":
            self.textproperty.SetFontFamilyToTimes()
        elif font.lower() == "arial":
            self.textproperty.SetFontFamilyToArial()
        else:
            self.textproperty.SetFontFamily(vtk.VTK_FONT_FILE)
            self.textproperty.SetFontFile(settings.fonts_path + font +'.ttf')
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
def addScalarBar(obj,
                 pos=(0.8,0.05),
                 title="",
                 titleXOffset=0,
                 titleYOffset=15,
                 titleFontSize=12,
                 nlabels=None,
                 c=None,
                 horizontal=False,
                 useAlpha=True,
                 ):
    """Add a 2D scalar bar for the specified obj.

    .. hint:: |mesh_coloring| |mesh_coloring.py|_ |scalarbars.py|_
    """
    plt = settings.plotter_instance

    if not hasattr(obj, "mapper"):
        printc("\times addScalarBar(): input is invalid,", type(obj), c='r')
        return None

    if plt and plt.renderer:
        c = (0.9, 0.9, 0.9)
        if np.sum(plt.renderer.GetBackground()) > 1.5:
            c = (0.1, 0.1, 0.1)
        if isinstance(obj.scalarbar, vtk.vtkActor):
            plt.renderer.RemoveActor(obj.scalarbar)
        elif isinstance(obj.scalarbar, Assembly):
            for a in obj.scalarbar.getMeshes():
                plt.renderer.RemoveActor(a)
    if c is None: c = 'gray'

    if isinstance(obj, Points):
        lut = obj.mapper().GetLookupTable()
        if not lut:
            return None
        vtkscalars = obj._polydata.GetPointData().GetScalars()
        if vtkscalars is None:
            vtkscalars = obj._polydata.GetCellData().GetScalars()
        if not vtkscalars:
            return None

    elif isinstance(obj, (Volume, TetMesh)):
        lut = ctf2lut(obj)

    else:
        return obj

    c = getColor(c)
    sb = vtk.vtkScalarBarActor()
    #sb.SetLabelFormat('%-#6.3g')
    #print(sb.GetLabelFormat())
    sb.SetLookupTable(lut)
    sb.SetUseOpacity(useAlpha)
    sb.SetDrawFrame(0)
    sb.SetDrawBackground(0)

    if title:
        titprop = sb.GetTitleTextProperty()
        titprop.BoldOn()
        titprop.ItalicOff()
        titprop.ShadowOff()
        titprop.SetColor(c)
        titprop.SetVerticalJustificationToTop()
        titprop.SetFontSize(titleFontSize)
        titprop.SetFontFamily(vtk.VTK_FONT_FILE)
        titprop.SetFontFile(settings.fonts_path + settings.defaultFont +'.ttf')
        sb.SetTitle(title)
        sb.SetVerticalTitleSeparation(titleYOffset)
        sb.SetTitleTextProperty(titprop)

    #     lpr = sb.GetAnnotationTextProperty ()# GetLabelTextProperty()
    #     lpr.BoldOff()
    #     lpr.ItalicOff()
    #     lpr.ShadowOff()
    #     lpr.SetColor(c)
    #     lpr.SetFontFamily(vtk.VTK_FONT_FILE)
    #     lpr.SetFontFile(settings.fonts_path + settings.defaultFont +'.ttf')

    sb.UnconstrainedFontSizeOn()
    sb.DrawAnnotationsOn()
    sb.DrawTickLabelsOn()
    sb.SetMaximumNumberOfColors(256)

    if horizontal:
        sb.SetOrientationToHorizontal()
        sb.SetNumberOfLabels(3)
        sb.SetTextPositionToSucceedScalarBar()
        sb.SetPosition(pos)
        sb.SetMaximumWidthInPixels(1000)
        sb.SetMaximumHeightInPixels(50)
    else:
        sb.SetNumberOfLabels(7)
        sb.SetTextPositionToPrecedeScalarBar()
        sb.SetPosition(pos[0]+0.09, pos[1])
        sb.SetMaximumWidthInPixels(60)
        sb.SetMaximumHeightInPixels(250)

    if nlabels is not None:
        sb.SetNumberOfLabels(nlabels)

    sctxt = sb.GetLabelTextProperty()
    sb.SetAnnotationTextProperty(sctxt)
    sctxt.SetColor(c)
    sctxt.SetShadow(0)
    sctxt.SetFontFamily(0)
    sctxt.SetItalic(0)
    sctxt.SetBold(0)
    sctxt.SetFontSize(titleFontSize)
    sb.PickableOff()
    obj.scalarbar = sb
    return sb


#####################################################################
def addScalarBar3D(
    obj,
    pos=None,
    sx=None,
    sy=None,
    title='',
    titleFont="",
    titleXOffset=-1.5,
    titleYOffset=0.0,
    titleSize=1.5,
    titleRotation=0.0,
    nlabels=9,
    labelFont="",
    labelOffset=0.375,
    italic=0,
    c=None,
    useAlpha=True,
    drawBox=True,
):
    """
    Draw a 3D scalar bar.

    ``obj`` input can be:
        - a list of numbers,
        - a list of two numbers in the form `(min, max)`,
        - a ``Mesh`` already containing a set of scalars associated to vertices or cells,
        - if ``None`` the last object in the list of actors will be used.

    :param float sx: thickness of scalarbar
    :param float sy: length of scalarbar
    :param str title: scalar bar title
    :param float titleXOffset: horizontal space btw title and color scalarbar
    :param float titleYOffset: vertical space offset
    :param float titleSize: size of title wrt numeric labels
    :param float titleRotation: title rotation in degrees
    :param int nlabels: number of numeric labels
    :param float labelOffset: space btw numeric labels and scale
    :param bool useAlpha: render transparency of the color bar, otherwise ignore
    :param bool drawBox: draw a box around the colorbar (useful with useAlpha=True)

    .. hint:: |scalarbars| |scalarbars.py|_
    """
    plt = settings.plotter_instance
    if plt and c is None:  # automatic black or white
        c = (0.9, 0.9, 0.9)
        if np.sum(getColor(plt.backgrcol)) > 1.5:
            c = (0.1, 0.1, 0.1)
    if c is None: c = (0,0,0)
    c = getColor(c)

    bns = obj.GetBounds()
    if sy is None:
        sy = (bns[3]-bns[2])
    if sx is None:
        sx = sy/18

    if isinstance(obj, Points):
        lut = obj.mapper().GetLookupTable()
        if not lut:
            print("Error in addScalarBar3D: mesh has no lookup table.", [obj])
            return None
        vmin, vmax = obj.mapper().GetScalarRange()

    elif isinstance(obj, (Volume, TetMesh)):
        lut = ctf2lut(obj)
        vmin, vmax = lut.GetRange()

    elif isSequence(obj):
        vmin, vmax = np.min(obj), np.max(obj)

    else:
        print("Error in ScalarBar3D(): input must be Mesh or list.", type(obj))
        return obj

    # build the color scale part
    scale = shapes.Grid([-sx *labelOffset, 0, 0], c=c, alpha=1,
                        sx=sx, sy=sy, resx=1, resy=256)
    scale.lw(0).wireframe(False)
    cscals = scale.cellCenters()[:, 1]
    scale.cmap(lut, cscals, on='cells')
    scale.lighting('off')
    xbns = scale.xbounds()

    if pos is None:
        d=sx/2
        if title:
            d = np.sqrt((bns[1]-bns[0])**2+sy*sy)/20
        pos=(bns[1]-xbns[0]+d,
             (bns[2]+bns[3])/2,
             bns[4])

    tacts = []
    ticks_pos, ticks_txt = make_ticks(vmin, vmax, nlabels)
    nlabels2 = len(ticks_pos)-1
    for i, p in enumerate(ticks_pos):
        tx = ticks_txt[i]
        if i and tx:
            # build numeric text
            y = -sy /2 + sy * i / nlabels2
            a = shapes.Text(tx, pos=[sx*labelOffset, y, 0], s=sy/60,
                            justify='center-left', c=c, italic=italic, font=labelFont)
            tacts.append(a)
            # build ticks
            tic = shapes.Line([xbns[1], y, 0],
                              [xbns[1]+sx*labelOffset/4, y, 0], lw=0.1, c=c)
            tacts.append(tic)

    # build title
    if title:
        t = shapes.Text(title, (0,0,0), s=sy/50*titleSize,
                        c=c, justify='centered', italic=italic, font=titleFont)
        t.RotateZ(90+titleRotation)
        t.pos(sx*titleXOffset,titleYOffset,0)
        tacts.append(t)

    if drawBox:
        tacts.append(scale.box().lw(0.1))

    for a in tacts: a.PickableOff()

    mtacts = merge(tacts).lighting('off')
    mtacts.PickableOff()
    scale.PickableOff()

    sact = Assembly(scale, tacts)
    sact.SetPosition(pos)
    sact.PickableOff()
    sact.UseBoundsOff()
    sact.name = 'ScalarBar3D'
    return sact


#####################################################################
def addSlider2D(sliderfunc, xmin, xmax, value=None, pos=4,
                title='', font='arial', titleSize=1, c=None, showValue=True):
    """Add a slider widget which can call an external custom function.

    :param sliderfunc: external function to be called by the widget
    :param float xmin:  lower value
    :param float xmax:  upper value
    :param float value: current value
    :param list pos: position corner number: horizontal [1-5] or vertical [11-15]
          it can also be specified by corners coordinates [(x1,y1), (x2,y2)]
    :param str title: title text
    :param str font: title font [arial, courier]
    :param float titleSize: title text scale [1.0]
    :param bool showValue:  if true current value is shown

    |sliders1| |sliders1.py|_ |sliders2.py|_
    """
    plt = settings.plotter_instance
    if c is None:  # automatic black or white
        c = (0.8, 0.8, 0.8)
        if np.sum(getColor(plt.backgrcol)) > 1.5:
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
        sliderRep.GetPoint1Coordinate().SetValue(0.04, 0.045)
        sliderRep.GetPoint2Coordinate().SetValue(0.45, 0.045)
    elif pos == 4:  # bottom-right
        sliderRep.GetPoint1Coordinate().SetValue(0.55, 0.045)
        sliderRep.GetPoint2Coordinate().SetValue(0.96, 0.045)
    elif pos == 5:  # bottom margin horizontal
        sliderRep.GetPoint1Coordinate().SetValue(0.04, 0.045)
        sliderRep.GetPoint2Coordinate().SetValue(0.96, 0.045)
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

    sliderRep.SetTitleHeight(0.022*titleSize)
    sliderRep.GetTitleProperty().SetShadow(0)
    sliderRep.GetTitleProperty().SetColor(c)
    sliderRep.GetTitleProperty().SetOpacity(1)
    sliderRep.GetTitleProperty().SetBold(0)
    if font == 'courier':
        sliderRep.GetTitleProperty().SetFontFamilyToCourier()
    elif font.lower() == "times":
        sliderRep.GetTitleProperty().SetFontFamilyToTimes()
    else:
        sliderRep.GetTitleProperty().SetFontFamilyToArial()

    if title:
        sliderRep.SetTitleText(title)
        if not isSequence(pos):
            if isinstance(pos, int) and pos > 10:
                sliderRep.GetTitleProperty().SetOrientation(90)
        else:
            if abs(pos[0][0] - pos[1][0]) < 0.1:
                sliderRep.GetTitleProperty().SetOrientation(90)

    sliderWidget = vtk.vtkSliderWidget()
    sliderWidget.SetInteractor(plt.interactor)
    sliderWidget.SetAnimationModeToJump()
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.AddObserver("InteractionEvent", sliderfunc)
    if plt.renderer:
        sliderWidget.SetCurrentRenderer(plt.renderer)
    sliderWidget.EnabledOn()
    plt.sliders.append([sliderWidget, sliderfunc])
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
    plt = settings.plotter_instance
    if c is None:  # automatic black or white
        c = (0.8, 0.8, 0.8)
        if np.sum(getColor(plt.backgrcol)) > 1.5:
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

    # sliderRep.SetPoint1InWorldCoordinates(pos2[0], pos2[1], pos2[2])
    # sliderRep.SetPoint2InWorldCoordinates(pos1[0], pos1[1], pos1[2])

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
    sliderWidget.SetInteractor(plt.interactor)
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.SetAnimationModeToJump()
    sliderWidget.AddObserver("InteractionEvent", sliderfunc)
    sliderWidget.EnabledOn()
    plt.sliders.append([sliderWidget, sliderfunc])
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
    plt = settings.plotter_instance
    if not plt.renderer:
        printc("\timesError: Use addButton() after rendering the scene.", c='r')
        return
    bu = Button(fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle)
    plt.renderer.AddActor2D(bu.actor)
    plt.window.Render()
    plt.buttons.append(bu)
    return bu


def addCutterTool(mesh):
    """Create handles to cut away parts of a mesh.

    |cutter| |cutter.py|_
    """
    if isinstance(mesh, vtk.vtkVolume):
        return _addVolumeCutterTool(mesh)
    elif isinstance(mesh, vtk.vtkImageData):
        return _addVolumeCutterTool(Volume(mesh))

    plt = settings.plotter_instance
    if not plt.renderer:
        save_int = plt.interactive
        plt.show(interactive=0)
        plt.interactive = save_int

    plt.clickedActor = mesh
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

    plt.renderer.RemoveActor(mesh)
    plt.renderer.AddActor(act0)
    plt.renderer.AddActor(act1)
    plt.renderer.ResetCamera()

    def selectPolygons(vobj, event):
        vobj.GetPlanes(planes)

    boxWidget = vtk.vtkBoxWidget()
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
    boxWidget.GetOutlineProperty().SetColor(0.2, 0.2, 0.2)
    boxWidget.GetOutlineProperty().SetOpacity(0.8)
    boxWidget.SetPlaceFactor(1.05)
    boxWidget.SetInteractor(plt.interactor)
    boxWidget.SetInputData(apd)
    boxWidget.PlaceWidget()
    boxWidget.AddObserver("InteractionEvent", selectPolygons)
    boxWidget.On()

    plt.cutterWidget = boxWidget
    plt.clickedActor = act0
    if mesh in plt.actors:
        ia = plt.actors.index(mesh)
        plt.actors[ia] = act0

    printc("Mesh Cutter Tool:", c="m", invert=1)
    printc("  Move gray handles to cut away parts of the mesh", c="m")
    printc("  Press X to save file to: clipped.vtk", c="m")
    plt.interactor.Start()

    boxWidget.Off()
    plt.widgets.append(boxWidget)

    plt.interactor.Start()  # allow extra interaction
    return act0

def _addVolumeCutterTool(vol):
    plt = settings.plotter_instance
    if not plt.renderer:
        save_int = plt.interactive
        plt.show(interactive=0)
        plt.interactive = save_int

    boxWidget = vtk.vtkBoxWidget()
    boxWidget.SetInteractor(plt.interactor)
    boxWidget.SetPlaceFactor(1.0)

    plt.cutterWidget = boxWidget

    plt.renderer.AddVolume(vol)

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

    plt.interactor.Render()
    boxWidget.On()

    plt.interactor.Start()
    boxWidget.Off()
    plt.widgets.append(boxWidget)

#####################################################################
def addIcon(mesh, pos=3, size=0.08):
    """Add an inset icon mesh into the renderer.

    :param pos: icon position in the range [1-4] indicating one of the 4 corners,
                or it can be a tuple (x,y) as a fraction of the renderer size.
    :param float size: size of the icon space.

    |icon| |icon.py|_
    """
    plt = settings.plotter_instance
    if not plt.renderer:
        printc("\lightningWarning: Use addIcon() after first rendering the scene.", c='y')
        save_int = plt.interactive
        plt.show(interactive=0)
        plt.interactive = save_int
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(mesh)
    widget.SetInteractor(plt.interactor)
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
    plt.widgets.append(widget)
    if mesh in plt.actors:
        plt.actors.remove(mesh)
    return widget

#####################################################################
def computeVisibleBounds():
    """Calculate max meshs bounds and sizes."""
    bns = []
    for a in settings.plotter_instance.actors:
        if a and a.GetUseBounds():
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
def buildRulerAxes(
    inputobj,
    xtitle="",
    ytitle="",
    ztitle="",
    xlabel="",
    ylabel="",
    zlabel="",
    xpad=0.05,
    ypad=0.04,
    zpad=0,
    font="Normografo",
    s=None,
    italic=0,
    units="",
    c=(0.2, 0, 0),
    alpha=1,
    lw=1,
    precision=3,
    labelRotation=0,
    axisRotation=0,
    xycross=True,
):
    """
    Build a 3D ruler to indicate the distance of two points p1 and p2.

    :param str xtitle: name of the axis or title
    :param str xlabel: alternative fixed label to be shown instead of the distance
    :param float xpad: gap distance from the x-axis
    :param float s: size of the label
    :param str font: font name
    :param float italic: italicness of the font [0,1]
    :param str units: string to be appended to the numeric value
    :param float lw: line width in pixel units
    :param int precision: nr of significant digits to be shown
    :param float labelRotation: initial rotation of the label around the z-axis
    :param float axisRotation: initial rotation of the line around the main axis
    :param bool xycross: show two back crossing lines in the xy plane

    |goniometer| |goniometer.py|_
    """
    if isSequence(inputobj):
        x0,x1,y0,y1,z0,z1 = inputobj
    else:
        x0,x1,y0,y1,z0,z1 = inputobj.GetBounds()
    dx,dy,dz = (y1-y0)*xpad, (x1-x0)*ypad, (y1-y0)*zpad
    d = np.sqrt((y1-y0)**2+(x1-x0)**2+(z1-z0)**2)

    if s is None:
        s = d/75

    acts, rx, ry = [], None, None
    if xtitle is not None and (x1-x0)/d>0.1:
        rx = Ruler( [x0,y0-dx,z0],
                    [x1,y0-dx,z0],
                    s=s, font=font, precision=precision,
                    labelRotation=labelRotation, axisRotation=axisRotation,
                    lw=lw, italic=italic, prefix=xtitle, label=xlabel, units=units)
        acts.append(rx)
    if ytitle is not None and (y1-y0)/d>0.1:
        ry = Ruler( [x1+dy,y0,z0],
                    [x1+dy,y1,z0],
                    s=s, font=font, precision=precision,
                    labelRotation=labelRotation, axisRotation=axisRotation,
                    lw=lw, italic=italic, prefix=ytitle, label=ylabel, units=units)
        acts.append(ry)
    if ztitle is not None and (z1-z0)/d>0.1:
        rz = Ruler( [x0-dy,y0+dz,z0],
                    [x0-dy,y0+dz,z1],
                    s=s, font=font, precision=precision,
                    labelRotation=labelRotation, axisRotation=axisRotation+90,
                    lw=lw, italic=italic, prefix=ztitle, label=zlabel, units=units)
        acts.append(rz)

    ncolls = len(settings.collectable_actors)
    if xycross and rx and ry:
        lx = shapes.Line([x0,y0,z0],    [x0,y1+dx,z0])
        ly = shapes.Line([x0-dy,y1,z0], [x1,y1,z0])
        d = min((x1-x0), (y1-y0))/200
        cxy = shapes.Circle([x0,y1,z0], r=d, res=15)
        acts.extend([lx,ly,cxy])

    macts = merge(acts).c(c).alpha(alpha).bc('t')
    macts.UseBoundsOff()
    settings.collectable_actors = settings.collectable_actors[:ncolls]
    return macts


#####################################################################
def buildAxes(obj=None,
              xtitle=None, ytitle=None, ztitle=None,
              xrange=None, yrange=None, zrange=None,
              c=None,
              numberOfDivisions=None,
              digits=None,
              limitRatio=0.04,
              axesLineWidth=1,
              gridLineWidth=1,
              reorientShortTitle=True,
              titleDepth=0,
              titleFont="", # grab settings.defaultFont
              textScale=1.0,
              xTitlePosition=0.95, yTitlePosition=0.95, zTitlePosition=0.95,
              xTitleOffset=0.02,   yTitleOffset=0.025,   zTitleOffset=0.02,
              xTitleJustify="top-right", yTitleJustify="bottom-right", zTitleJustify="bottom-right",
              xTitleRotation=0, yTitleRotation=0, zTitleRotation=0,
              xTitleBox=False,  yTitleBox=False,
              xTitleSize=0.025, yTitleSize=0.025, zTitleSize=0.025,
              xTitleColor=None, yTitleColor=None, zTitleColor=None,
              xTitleBackfaceColor=None, yTitleBackfaceColor=None, zTitleBackfaceColor=None,
              xTitleItalic=0, yTitleItalic=0, zTitleItalic=0,
              xyGrid=True, yzGrid=True, zxGrid=False,
              xyGrid2=False, yzGrid2=False, zxGrid2=False,
              xyGridTransparent=False, yzGridTransparent=False, zxGridTransparent=False,
              xyGrid2Transparent=False, yzGrid2Transparent=False, zxGrid2Transparent=False,
              xyPlaneColor=None, yzPlaneColor=None, zxPlaneColor=None,
              xyGridColor=None, yzGridColor=None, zxGridColor=None,
              xyAlpha=0.05, yzAlpha=0.05, zxAlpha=0.05,
              xyFrameLine=None, yzFrameLine=None, zxFrameLine=None,
              xLineColor=None, yLineColor=None, zLineColor=None,
              xHighlightZero=False, yHighlightZero=False, zHighlightZero=False,
              xHighlightZeroColor='r', yHighlightZeroColor='g', zHighlightZeroColor='b',
              showTicks=True,
              xTickLength=0.015, yTickLength=0.015, zTickLength=0.015,
              xTickThickness=0.0025, yTickThickness=0.0025, zTickThickness=0.0025,
              xMinorTicks=1, yMinorTicks=1, zMinorTicks=1,
              tipSize=None,
              labelFont="", # grab settings.defaultFont
              xLabelColor=None, yLabelColor=None, zLabelColor=None,
              xLabelSize=0.016, yLabelSize=0.016, zLabelSize=0.016,
              xLabelOffset=0.015, yLabelOffset=0.015, zLabelOffset=0.010,
              xLabelRotation=0, yLabelRotation=None, zLabelRotation=None,
              xFlipText=False, yFlipText=False, zFlipText=False,
              xValuesAndLabels=None, yValuesAndLabels=None, zValuesAndLabels=None,
              useGlobal=False,
              tol=0.0001,
    ):
    """
    Draw axes for the input object. Returns an ``Assembly`` object.

    - `xtitle`,                ['x'], x-axis title text
    - `xrange`,               [None], x-axis range in format (xmin, ymin), default is automatic.
    - `numberOfDivisions`,    [None], approximate number of divisions on the longest axis
    - `axesLineWidth`,           [1], width of the axes lines
    - `gridLineWidth`,           [1], width of the grid lines
    - `reorientShortTitle`,   [True], titles shorter than 2 letter are placed horizontally
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
    - `digits`,               [None], use this number of significant digits in scientific notation
    - `titleFont`,              [''], font for axes titles
    - `labelFont`,              [''], font for numeric labels
    - `textScale`,             [1.0], global scaling factor for text elements (titles, labels)
    - `xTitlePosition`,       [0.32], title fractional positions along axis
    - `xTitleOffset`,         [0.05], title fractional offset distance from axis line
    - `xTitleJustify`, ["top-right"], title justification
    - `xTitleRotation`,          [0], add a rotation of the axis title
    - `xTitleBox`,           [False], add a box around title text
    - `xLineColor`,      [automatic], color of the x-axis
    - `xTitleColor`,     [automatic], color of the axis title
    - `xTitleBackfaceColor`,  [None],  color of axis title on its backface
    - `xTitleSize`,          [0.025], size of the axis title
    - 'xTitleItalic',            [0], a bool or float to make the font italic
    - `xHighlightZero`,       [True], draw a line highlighting zero position if in range
    - `xHighlightZeroColor`, [autom], color of the line highlighting the zero position
    - `xTickLength`,         [0.005], radius of the major ticks
    - `xTickThickness`,     [0.0025], thickness of the major ticks along their axis
    - `xMinorTicks`,             [1], number of minor ticks between two major ticks
    - `xValuesAndLabels`          [], assign custom tick positions and labels [(pos1, label1), ...]
    - `xLabelColor`,     [automatic], color of numeric labels and ticks
    - `xLabelPrecision`,         [2], nr. of significative digits to be shown
    - `xLabelSize`,          [0.015], size of the numeric labels along axis
    - 'xLabelRotation',          [0], rotate clockwise [1] or anticlockwise [-1] by 90 degrees
    - 'xFlipText',           [False], flip axis title and numeric labels orientation
    - `xLabelOffset`,        [0.025], offset of numeric labels
    - `tipSize`,              [0.01], size of the arrow tip
    - `limitRatio`,           [0.04], below this ratio don't plot small axis

    :Example:

        .. code-block:: python

            from vedo import Box, show
            b = Box(pos=(1,2,3), length=8, width=9, height=7).alpha(0)
            bax = buildAxes(b, c='k')  # returns Assembly object
            show(b, bax)

    |customAxes| |customAxes.py|_
    """
    ncolls = len(settings.collectable_actors)

    if not titleFont:
        titleFont = settings.defaultFont
    if not labelFont:
        labelFont = settings.defaultFont

    if c is None:  # automatic black or white
        c = (0.1, 0.1, 0.1)
        plt = settings.plotter_instance
        if plt and plt.renderer:
            bgcol = plt.renderer.GetBackground()
        else:
            bgcol = (1,1,1)
        if np.sum(bgcol) < 1.5:
            c = (0.9, 0.9, 0.9)
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
    if not xLabelColor:  xLabelColor = xLineColor
    if not yLabelColor:  yLabelColor = yLineColor
    if not zLabelColor:  zLabelColor = zLineColor

    # vtk version<9 dont like depthpeeling
    if settings.useDepthPeeling and not vtkVersionIsAtLeast(9):
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

    if xtitle:
        xticks_float, xticks_str = make_ticks(vbb[0], vbb[1], rx,
                                              xValuesAndLabels, digits)
    if ytitle:
        yticks_float, yticks_str = make_ticks(vbb[2], vbb[3], ry,
                                              yValuesAndLabels, digits)
        if yLabelRotation is None: #automatic rotation of labels
            maxlentxt=0
            for i in range(1,len(yticks_str)-1):
                maxlentxt = max(maxlentxt, len(yticks_str[i].replace('.','')))
            if maxlentxt < 4:
                yLabelRotation = 1 # rotate clockwise
    if ztitle:
        zticks_float, zticks_str = make_ticks(vbb[4], vbb[5], rz,
                                              zValuesAndLabels, digits)
        if zLabelRotation is None: #automatic rotation of labels
            maxlentxt=0
            for i in range(1,len(zticks_str)-1):
                maxlentxt = max(maxlentxt, len(zticks_str[i].replace('.','')))
            if maxlentxt < 4:
                zLabelRotation = 1 # rotate clockwise

    ################################## calculate aspect ratio scales
    x_aspect_ratios = (1,1,1)
    y_aspect_ratios = (1,1,1)
    z_aspect_ratios = (1,1,1)
    if xtitle:
        if ss[1] and ss[0] > ss[1]:
            x_aspect_ratios = (1, ss[0]/ss[1], 1)
        elif ss[0]:
            x_aspect_ratios = (ss[1]/ss[0], 1, 1)
    if ytitle:
        if ss[1] and ss[0] > ss[1]:
            y_aspect_ratios = (ss[0]/ss[1], 1, 1)
        elif ss[0]:
            y_aspect_ratios = (1, ss[1]/ss[0], 1)
    if ztitle:
        smean = (ss[0]+ss[1])/2
        if smean and ss[2]>smean:
            textScale *= np.sqrt(ssmax/smean) # equalize text size for large z ranges
        if ss[2] and smean:
            if ss[2] > smean:
                zarfact = smean/ss[2]
                z_aspect_ratios = (zarfact, zarfact*ss[2]/smean, zarfact)
            else:
                z_aspect_ratios = (smean/ss[2], 1, 1)

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
                xmajticks = merge(xticks).c(xLabelColor)
                xmajticks.name = "xMajorTicks"
                majorticks.append(xmajticks)
        if ytitle:
            for i in range(1, len(yticks_float)-1):
                v1 = (-yTickLength/2, yticks_float[i]-yTickThickness/2, 0)
                v2 = ( yTickLength/2, yticks_float[i]+yTickThickness/2, 0)
                yticks.append(shapes.Rectangle(v1, v2))
            if len(yticks)>1:
                ymajticks = merge(yticks).c(yLabelColor)
                ymajticks.name = "yMajorTicks"
                majorticks.append(ymajticks)
        if ztitle:
            for i in range(1, len(zticks_float)-1):
                v1 = (zticks_float[i]-zTickThickness/2, -zTickLength/2.84, 0)
                v2 = (zticks_float[i]+zTickThickness/2,  zTickLength/2.84, 0)
                zticks.append(shapes.Rectangle(v1, v2))
            if len(zticks)>1:
                zmajticks = merge(zticks).c(zLabelColor)
                zmajticks.RotateZ(-45)
                zmajticks.RotateY(-90)
                zmajticks.name = "zMajorTicks"
                majorticks.append(zmajticks)

        ################################################ MINOR ticks
        if xtitle and xMinorTicks and len(xticks)>1:
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
                xminticks = merge(ticks).c(xLabelColor)
                xminticks.name = "xMinorTicks"
                minorticks.append(xminticks)

        if ytitle and yMinorTicks and len(yticks)>1:
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
                yminticks = merge(ticks).c(yLabelColor)
                yminticks.name = "yMinorTicks"
                minorticks.append(yminticks)

        if ztitle and zMinorTicks and len(zticks)>1:
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
                zminticks = merge(ticks).c(zLabelColor)
                zminticks.RotateZ(-45)
                zminticks.RotateY(-90)
                zminticks.name = "zMinorTicks"
                minorticks.append(zminticks)


    ################################################ axes tick NUMERIC text labels
    labels = []
    xlab, ylab, zlab = None, None, None
    if xLabelSize and xtitle:
        jus ="center-top"
        if xLabelRotation:
            jus = "center-left"
            if int(xLabelRotation)<0:
                jus = "center-right"
        elif xFlipText:
            jus ="center-bottom"
        for i in range(1, len(xticks_str)):
            t = xticks_str[i]
            if not t: continue
            v = (xticks_float[i], -xLabelOffset, 0)
            xlab = shapes.Text(t, pos=v, s=xLabelSize*textScale, font=labelFont, justify=jus)
            xlab.SetScale(x_aspect_ratios)
            if xLabelRotation:
                xlab.RotateZ(-90*int(xLabelRotation))
                f = max(x_aspect_ratios)
                xlab.SetScale(f/x_aspect_ratios[0], f/x_aspect_ratios[1],1)
            elif xFlipText:
                xlab.RotateZ(180)
            xlab.name = "xNumericLabel"+str(i)+" "+t
            xlab.UseBoundsOff()
            labels.append(xlab.c(xLabelColor))

    if yLabelSize and ytitle:
        jus = "center-bottom"
        if yLabelRotation:
            jus = "center-right"
            if int(yLabelRotation)<0:
                jus = "center-left"
        elif yFlipText:
            jus = "center-top"
        for i in range(1,len(yticks_str)):
            t = yticks_str[i]
            if not t: continue
            v = (-yLabelOffset, yticks_float[i], 0)
            ylab = shapes.Text(t, s=yLabelSize*textScale, font=labelFont, justify=jus)
            ylab.RotateZ(90+yTitleRotation)
            ylab.SetScale(y_aspect_ratios)
            if yLabelRotation:
                ylab.RotateZ(-90*int(yLabelRotation))
                f = max(y_aspect_ratios)
                ylab.SetScale(f/y_aspect_ratios[0], f/y_aspect_ratios[1],1)
            elif yFlipText:
                ylab.RotateZ(180)
            ylab.pos(v)
            ylab.UseBoundsOff()
            ylab.name = "yNumericLabel"+str(i)+" "+t
            labels.append(ylab.c(yLabelColor))

    if zLabelSize and ztitle:
        jus = "center-bottom"
        if zLabelRotation:
            jus = "center-right"
            if int(zLabelRotation)<0:
                jus = "center-left"
        elif zFlipText:
            jus = "center-top"
        for i in range(1, len(zticks_str)):
            t = zticks_str[i]
            if not t: continue
            v = (-zLabelOffset, -zLabelOffset, zticks_float[i])
            zlab = shapes.Text(t, s=zLabelSize*textScale, font=labelFont, justify=jus)
            zlab.RotateY(-90)
            zlab.RotateX(135+zTitleRotation)
            zlab.SetScale(z_aspect_ratios)
            if zLabelRotation:
                zlab.RotateZ(-90*int(zLabelRotation))
                f = max(z_aspect_ratios)*z_aspect_ratios[2]
                zlab.SetScale(f/z_aspect_ratios[0], f/z_aspect_ratios[1],1)
            elif zFlipText:
                zlab.RotateZ(180)
            zlab.pos(v)
            zlab.UseBoundsOff()
            zlab.name = "zNumericLabel"+str(i)+" "+t
            labels.append(zlab.c(zLabelColor))

    ################################################ axes titles
    titles = []
    if xtitle:
        if xFlipText: xTitleJustify = 'bottom-left'
        xt = shapes.Text(xtitle, s=xTitleSize*textScale, font=titleFont,
                         c=xTitleColor, justify=xTitleJustify, depth=titleDepth,
                         italic=xTitleItalic)
        if xTitleBackfaceColor: xt.backColor(xTitleBackfaceColor)
        shift = 0
        if xlab: # this is the last created one..
            lt0, lt1 = xlab.GetBounds()[2:4]
            shift =  lt1 - lt0
        wpos = [xTitlePosition, -xTitleOffset -shift, 0]
        xt.SetScale(x_aspect_ratios)
        xt.RotateX(xTitleRotation)
        if xFlipText:
            xt.RotateZ(180)
        xt.pos(wpos)
        # xt.UseBoundsOff()
        xt.name = "xtitle "+str(xtitle)
        titles.append(xt)
        if xTitleBox: titles.append(xt.box().useBounds(False))

    if ytitle:
        if yFlipText: yTitleJustify = 'top-left'
        yt = shapes.Text(ytitle, s=yTitleSize*textScale, font=titleFont,
                         c=yTitleColor, justify=yTitleJustify, depth=titleDepth,
                         italic=yTitleItalic)
        if yTitleBackfaceColor: yt.backColor(yTitleBackfaceColor)
        shift = 0
        if ylab:  # this is the last created one..
            lt0, lt1 = ylab.GetBounds()[0:2]
            shift = lt1-lt0
        if reorientShortTitle and len(ytitle) < 4:  # title is short
            yTitlePosition *= 0.975
            yt.SetScale(x_aspect_ratios) #x!
        else:
            yt.SetScale(y_aspect_ratios)
            yt.RotateZ(90+yTitleRotation)
            if yFlipText: yt.RotateZ(180)
        wpos = [-yTitleOffset -shift, yTitlePosition, 0]
        yt.pos(wpos)
        yt.UseBoundsOff()
        yt.name = "ytitle "+str(ytitle)
        titles.append(yt)
        if yTitleBox: titles.append(yt.box().useBounds(False))

    if ztitle:
        if zFlipText: zTitleJustify = 'top-left'
        zt = shapes.Text(ztitle, s=zTitleSize*textScale, font=titleFont,
                         c=zTitleColor, justify=zTitleJustify, depth=titleDepth,
                         italic=yTitleItalic)
        if zTitleBackfaceColor: zt.backColor(zTitleBackfaceColor)
        shift = 0
        if zlab: # this is the last created one..
            lt0, lt1 = zlab.GetBounds()[0:2]
            shift =  (lt1 - lt0)/1.25
        wpos = [-zTitleOffset-shift, -zTitleOffset-shift, zTitlePosition]
        if reorientShortTitle and len(ztitle) < 4:  # title is short
            zTitlePosition *= 0.975
            zt.SetScale(z_aspect_ratios[1],
                        z_aspect_ratios[0],
                        z_aspect_ratios[2])
            zt.RotateX(90)
            zt.RotateY(45)
        else:
            zt.SetScale(z_aspect_ratios)
            zt.RotateY(-90)
            zt.RotateX(135+zTitleRotation)
        if zFlipText:
            zt.RotateZ(180)
        zt.pos(wpos)
        zt.UseBoundsOff()
        zt.name = "ztitle "+str(ztitle)
        titles.append(zt)
    ###################################################

    acts = titles + lines + labels + grids + grids2 + highlights
    acts += framelines + majorticks + minorticks + cones

    tol *= mag(ss)
    orig = np.array([min_bns[0], min_bns[2], min_bns[4]]) - tol
    for a in acts:
        a.PickableOff()
        a.SetPosition(a.GetPosition() + orig)
        a.GetProperty().LightingOff()
    asse = Assembly(acts)
    asse.SetOrigin(orig)
    asse.SetScale(ss)
    asse.PickableOff()
    # throw away all extra created obj in collectable_actors
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
        - 7,  draw a 3D ruler at each side of the cartesian axes
        - 8,  show the ``vtkCubeAxesActor`` object
        - 9,  show the bounding box outLine
        - 10, show three circles representing the maximum bounding box
        - 11, show a large grid on the x-y plane (use with zoom=8)
        - 12, show polar axes
        - 13, draw a simple ruler at the bottom of the window

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
        - `xLabelColor`,      [automatic], color of major ticks
        - `xMinorTicks`,             [1], number of minor ticks between two major ticks
        - `tipSize`,              [0.01], size of the arrow tip
        - `xValuesAndLabels`       [], assign custom tick positions and labels [(pos1, label1), ...]
        - `xLabelPrecision`,         [2], nr. of significative digits to be shown
        - `xLabelSize`,          [0.015], size of the numeric labels along axis
        - `xLabelOffset`,        [0.025], offset of numeric labels
        - `limitRatio`,           [0.04], below this ratio don't plot small axis

        :Example:

            .. code-block:: python

                from vedo import Box, show
                b = Box(pos=(0,0,0), length=80, width=90, height=70).alpha(0)
                show(b, axes={ 'xtitle':'Some long variable [a.u.]',
                               'numberOfDivisions':4,
                               # ...
                             })

    |customAxes| |customAxes.py|_
    """
    plt = settings.plotter_instance
    if axtype is not None:
        plt.axes = axtype  # override

    r = plt.renderers.index(plt.renderer)

    if not plt.axes:
        return

    if c is None:  # automatic black or white
        c = (0.9, 0.9, 0.9)
        if np.sum(plt.renderer.GetBackground()) > 1.5:
            c = (0.1, 0.1, 0.1)
    else:
        c = getColor(c) # for speed

    if not plt.renderer:
        return

    if plt.axes_instances[r]:
        return

    ############################################################
    # custom grid walls
    if plt.axes == 1 or plt.axes is True or isinstance(plt.axes, dict):

        if isinstance(plt.axes, dict):
            plt.axes.update({'useGlobal':True})
            asse = buildAxes(None, **plt.axes)
        else:
            asse = buildAxes(None, useGlobal=True)

        plt.renderer.AddActor(asse)
        plt.axes_instances[r] = asse


    elif plt.axes == 2 or plt.axes == 3:
        x0, x1, y0, y1, z0, z1 = plt.renderer.ComputeVisiblePropBounds()
        xcol, ycol, zcol = "dr", "dg", "db"
        s = 1
        alpha = 1
        centered = False
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        aves = np.sqrt(dx * dx + dy * dy + dz * dz) / 2
        x0, x1 = min(x0, 0), max(x1, 0)
        y0, y1 = min(y0, 0), max(y1, 0)
        z0, z1 = min(z0, 0), max(z1, 0)

        if plt.axes == 3:
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

        if len(plt.xtitle) and dx > aves/100:
            xl = shapes.Cylinder([[x0, 0, 0], [x1, 0, 0]], r=aves/250*s, c=xcol, alpha=alpha)
            xc = shapes.Cone(pos=[x1, 0, 0], c=xcol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[1, 0, 0], res=10)
            wpos = [x1-(len(plt.xtitle)+1)*aves/40*s, -aves/25*s, 0]  # aligned to arrow tip
            if centered:
                wpos = [(x0 + x1) / 2 - len(plt.xtitle) / 2 * aves / 40 * s, -aves / 25 * s, 0]
            xt = shapes.Text(plt.xtitle, pos=wpos, s=aves / 40 * s, c=xcol)
            acts += [xl, xc, xt]

        if len(plt.ytitle) and dy > aves/100:
            yl = shapes.Cylinder([[0, y0, 0], [0, y1, 0]], r=aves/250*s, c=ycol, alpha=alpha)
            yc = shapes.Cone(pos=[0, y1, 0], c=ycol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[0, 1, 0], res=10)
            wpos = [-aves/40*s, y1-(len(plt.ytitle)+1)*aves/40*s, 0]
            if centered:
                wpos = [-aves / 40 * s, (y0 + y1) / 2 - len(plt.ytitle) / 2 * aves / 40 * s, 0]
            yt = shapes.Text(plt.ytitle, pos=(0, 0, 0), s=aves / 40 * s, c=ycol)
            yt.pos(wpos).RotateZ(90)
            acts += [yl, yc, yt]

        if len(plt.ztitle) and dz > aves/100:
            zl = shapes.Cylinder([[0, 0, z0], [0, 0, z1]], r=aves/250*s, c=zcol, alpha=alpha)
            zc = shapes.Cone(pos=[0, 0, z1], c=zcol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[0, 0, 1], res=10)
            wpos = [-aves/50*s, -aves/50*s, z1 - (len(plt.ztitle)+1)*aves/40*s]
            if centered:
                wpos = [-aves/50*s, -aves/50*s, (z0+z1)/2-len(plt.ztitle)/2*aves/40*s]
            zt = shapes.Text(plt.ztitle, pos=(0,0,0), s=aves/40*s, c=zcol)
            zt.pos(wpos).RotateZ(45)
            zt.RotateX(90)
            acts += [zl, zc, zt]
        for a in acts:
            a.PickableOff()
        ass = Assembly(acts)
        ass.PickableOff()
        plt.renderer.AddActor(ass)
        plt.axes_instances[r] = ass

    elif plt.axes == 4:
        axact = vtk.vtkAxesActor()
        axact.SetShaftTypeToCylinder()
        axact.SetCylinderRadius(0.03)
        axact.SetXAxisLabelText(plt.xtitle)
        axact.SetYAxisLabelText(plt.ytitle)
        axact.SetZAxisLabelText(plt.ztitle)
        axact.GetXAxisShaftProperty().SetColor(1, 0, 0)
        axact.GetYAxisShaftProperty().SetColor(0, 1, 0)
        axact.GetZAxisShaftProperty().SetColor(0, 0, 1)
        axact.GetXAxisTipProperty().SetColor(1, 0, 0)
        axact.GetYAxisTipProperty().SetColor(0, 1, 0)
        axact.GetZAxisTipProperty().SetColor(0, 0, 1)
        bc = np.array(plt.renderer.GetBackground())
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
        plt.axes_instances[r] = icn

    elif plt.axes == 5:
        axact = vtk.vtkAnnotatedCubeActor()
        axact.GetCubeProperty().SetColor(getColor(settings.annotatedCubeColor))
        axact.SetTextEdgesVisibility(0)
        axact.SetFaceTextScale(settings.annotatedCubeTextScale)
        axact.SetXPlusFaceText (settings.annotatedCubeXPlusText)
        axact.SetXMinusFaceText(settings.annotatedCubeXMinusText)
        axact.SetYPlusFaceText (settings.annotatedCubeYPlusText)
        axact.SetYMinusFaceText(settings.annotatedCubeYMinusText)
        axact.SetZPlusFaceText (settings.annotatedCubeZPlusText)
        axact.SetZMinusFaceText(settings.annotatedCubeZMinusText)
        axact.SetZFaceTextRotation(90)

        if settings.annotatedCubeTextColor is None: # use default
            axact.GetXPlusFaceProperty().SetColor( getColor("r"))
            axact.GetXMinusFaceProperty().SetColor(getColor("dr"))
            axact.GetYPlusFaceProperty().SetColor( getColor("g"))
            axact.GetYMinusFaceProperty().SetColor(getColor("dg"))
            axact.GetZPlusFaceProperty().SetColor( getColor("b"))
            axact.GetZMinusFaceProperty().SetColor(getColor("db"))
        else: # use single user color
            ac = getColor(getColor(settings.annotatedCubeTextColor))
            axact.GetXPlusFaceProperty().SetColor(ac)
            axact.GetXMinusFaceProperty().SetColor(ac)
            axact.GetYPlusFaceProperty().SetColor(ac)
            axact.GetYMinusFaceProperty().SetColor(ac)
            axact.GetZPlusFaceProperty().SetColor(ac)
            axact.GetZMinusFaceProperty().SetColor(ac)

        axact.PickableOff()
        icn = addIcon(axact, size=0.06)
        plt.axes_instances[r] = icn

    elif plt.axes == 6:
        ocf = vtk.vtkOutlineCornerFilter()
        ocf.SetCornerFactor(0.1)
        largestact, sz = None, -1
        for a in plt.actors:
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
        bc = np.array(plt.renderer.GetBackground())
        if np.sum(bc) < 1.5:
            lc = (1, 1, 1)
        else:
            lc = (0, 0, 0)
        ocActor.GetProperty().SetColor(lc)
        ocActor.PickableOff()
        ocActor.UseBoundsOff()
        plt.renderer.AddActor(ocActor)
        plt.axes_instances[r] = ocActor

    elif plt.axes == 7:
        vbb = computeVisibleBounds()[0]
        rulax = buildRulerAxes(vbb, c=c,
                               xtitle=plt.xtitle+' - ',
                               ytitle=plt.ytitle+' - ',
                               ztitle=plt.ztitle+' - ')
        rulax.UseBoundsOff()
        rulax.PickableOff()
        plt.renderer.AddActor(rulax)
        plt.axes_instances[r] = rulax

    elif plt.axes == 8:
        vbb = computeVisibleBounds()[0]
        ca = vtk.vtkCubeAxesActor()
        ca.SetBounds(vbb)
        ca.SetCamera(plt.renderer.GetActiveCamera())
        ca.GetXAxesLinesProperty().SetColor(c)
        ca.GetYAxesLinesProperty().SetColor(c)
        ca.GetZAxesLinesProperty().SetColor(c)
        for i in range(3):
            ca.GetLabelTextProperty(i).SetColor(c)
            ca.GetTitleTextProperty(i).SetColor(c)
        ca.SetTitleOffset(5)
        ca.SetFlyMode(3)
        ca.SetXTitle(plt.xtitle)
        ca.SetYTitle(plt.ytitle)
        ca.SetZTitle(plt.ztitle)
        if plt.xtitle == "":
            ca.SetXAxisVisibility(0)
            ca.XAxisLabelVisibilityOff()
        if plt.ytitle == "":
            ca.SetYAxisVisibility(0)
            ca.YAxisLabelVisibilityOff()
        if plt.ztitle == "":
            ca.SetZAxisVisibility(0)
            ca.ZAxisLabelVisibilityOff()
        ca.PickableOff()
        ca.UseBoundsOff()
        plt.renderer.AddActor(ca)
        plt.axes_instances[r] = ca

    elif plt.axes == 9:
        vbb = computeVisibleBounds()[0]
        src = vtk.vtkCubeSource()
        src.SetXLength(vbb[1] - vbb[0])
        src.SetYLength(vbb[3] - vbb[2])
        src.SetZLength(vbb[5] - vbb[4])
        src.Update()
        ca = Mesh(src.GetOutput(), c, 0.5).wireframe(True)
        ca.pos((vbb[0] + vbb[1]) / 2, (vbb[3] + vbb[2]) / 2, (vbb[5] + vbb[4]) / 2)
        ca.PickableOff()
        ca.UseBoundsOff()
        plt.renderer.AddActor(ca)
        plt.axes_instances[r] = ca

    elif plt.axes == 10:
        vbb = computeVisibleBounds()[0]
        x0 = (vbb[0] + vbb[1]) / 2, (vbb[3] + vbb[2]) / 2, (vbb[5] + vbb[4]) / 2
        rx, ry, rz = (vbb[1]-vbb[0])/2, (vbb[3]-vbb[2])/2, (vbb[5]-vbb[4])/2
        rm = max(rx, ry, rz)
        xc = shapes.Disc(x0, r1=rm, r2=rm, c='lr', res=(1,72))
        yc = shapes.Disc(x0, r1=rm, r2=rm, c='lg', res=(1,72))
        yc.RotateX(90)
        zc = shapes.Disc(x0, r1=rm, r2=rm, c='lb', res=(1,72))
        yc.RotateY(90)
        xc.clean().alpha(0.2).wireframe().lineWidth(2.5).PickableOff()
        yc.clean().alpha(0.2).wireframe().lineWidth(2.5).PickableOff()
        zc.clean().alpha(0.2).wireframe().lineWidth(2.5).PickableOff()
        ca = xc + yc + zc
        ca.PickableOff()
        ca.UseBoundsOff()
        plt.renderer.AddActor(ca)
        plt.axes_instances[r] = ca

    elif plt.axes == 11:
        vbb, ss = computeVisibleBounds()[0:2]
        xpos, ypos = (vbb[1] + vbb[0]) /2, (vbb[3] + vbb[2]) /2
        gs = sum(ss)*3
        gr = shapes.Grid((xpos, ypos, vbb[4]), sx=gs, sy=gs,
                          resx=11, resy=11, c=c, alpha=0.1)
        gr.lighting('off').PickableOff()
        gr.UseBoundsOff()
        plt.renderer.AddActor(gr)
        plt.axes_instances[r] = gr

    elif plt.axes == 12:
        polaxes = vtk.vtkPolarAxesActor()
        vbb = computeVisibleBounds()[0]

        if plt.xtitle == 'x':
            polaxes.SetPolarAxisTitle('radial distance')
        else:
            polaxes.SetPolarAxisTitle(plt.xtitle)
        polaxes.SetPole(0,0, vbb[4])
        rd = max(abs(vbb[0]), abs(vbb[2]), abs(vbb[1]), abs(vbb[3]))
        polaxes.SetMaximumRadius(rd)
        polaxes.AutoSubdividePolarAxisOff()
        polaxes.SetNumberOfPolarAxisTicks(10)
        polaxes.SetCamera(plt.renderer.GetActiveCamera())
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
        polaxes.UseBoundsOff()
        polaxes.PickableOff()
        plt.renderer.AddActor(polaxes)
        plt.axes_instances[r] = polaxes

    elif plt.axes == 13:
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
        plt.renderer.AddActor(ls)
        plt.axes_instances[r] = ls

    else:
        printc('\bomb Keyword axes type must be in range [0-13].', c='r')
        printc('''
  \target Available axes types:
  0 = no axes,
  1 = draw three customizable gray grid walls
  2 = show cartesian axes from (0,0,0)
  3 = show positive range of cartesian axes from (0,0,0)
  4 = show a triad at bottom left
  5 = show a cube at bottom left
  6 = mark the corners of the bounding box
  7 = draw a 3D ruler at each side of the cartesian axes
  8 = show the vtkCubeAxesActor object
  9 = show the bounding box outline
  10 = show three circles representing the maximum bounding box
  11 = show a large grid on the x-y plane (use with zoom=8)
  12 = show polar axes.
  13 = draw a simple ruler at the bottom of the window
  ''', c='r', bold=0)

    if not plt.axes_instances[r]:
        plt.axes_instances[r] = True
    return


#####################################################################
def addRendererFrame(c=None, alpha=None, lw=None):

    if c is None:  # automatic black or white
        c = (0.9, 0.9, 0.9)
        if np.sum(settings.plotter_instance.renderer.GetBackground())>1.5:
            c = (0.1, 0.1, 0.1)
    c = getColor(c)

    if alpha is None:
        alpha = settings.rendererFrameAlpha

    if lw is None:
        lw = settings.rendererFrameWidth

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

    plt = settings.plotter_instance
    if not isSequence(plt._legend):
        return

    # remove old legend if present on current renderer:
    acs = plt.renderer.GetActors2D()
    acs.InitTraversal()
    for i in range(acs.GetNumberOfItems()):
        a = acs.GetNextItem()
        if isinstance(a, vtk.vtkLegendBoxActor):
            plt.renderer.RemoveActor(a)

    meshs = plt.getMeshes()
    acts, texts = [], []
    for i, a in enumerate(meshs):
        if i < len(plt._legend) and plt._legend[i] != "":
            if isinstance(plt._legend[i], str):
                texts.append(plt._legend[i])
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
    pr = vtklegend.GetEntryTextProperty()
    pr.SetFontFamily(vtk.VTK_FONT_FILE)
    if 'LogoType' in settings.legendFont: # special case of big file
        fl = vedo.io.download("https://vedo.embl.es/fonts/LogoType.ttf",
                              verbose=False, force=False)
    else:
        if settings.legendFont == "":
            settings.legendFont = settings.defaultFont
        fl = settings.fonts_path + settings.legendFont + '.ttf'
    pr.SetFontFile(fl)
    pr.ShadowOff()
    pr.BoldOff()

    for i in range(NT):
        ti = texts[i]
        if not ti:
            continue
        a = acts[i]
        c = a.GetProperty().GetColor()
        if c == (1, 1, 1):
            c = (0.2, 0.2, 0.2)
        vtklegend.SetEntry(i, a.polydata(), "  " + ti, c)
    pos = settings.legendPos
    width = settings.legendSize
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
    vtklegend.SetBackgroundColor(getColor(settings.legendBC))
    vtklegend.SetBackgroundOpacity(0.6)
    vtklegend.LockBorderOn()
    plt.renderer.AddActor(vtklegend)


