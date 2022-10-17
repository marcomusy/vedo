#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import vtkmodules.all as vtk

import vedo
from vedo import settings
from vedo import utils
from vedo import shapes
from vedo.assembly import Assembly
from vedo.colors import get_color
from vedo.colors import printc
from vedo.mesh import merge, Mesh
from vedo.pointcloud import Points
from vedo.tetmesh import TetMesh
from vedo.volume import Volume

__doc__ = """
Create additional objects like axes, legends, lights, etc.

.. image:: https://vedo.embl.es/images/pyplot/customAxes2.png
"""

__all__ = [
    "ScalarBar",
    "ScalarBar3D",
    "add_slider",
    "add_slider3d",
    "add_button",
    "add_cutter_tool",
    "add_icon",
    "LegendBox",
    "Light",
    "Axes",
    "Ruler",
    "RulerAxes",
    "Goniometer",
]


###########################################################################################
class LegendBox(vtk.vtkLegendBoxActor, shapes.TextBase):
    """
    Create a 2D legend box for the list of specified objects

    Parameters
    ----------
    nmax : int
        max number of legend entries

    c : color
        text color, leave as None to pick the mesh color automatically

    width : float
        width of the box as fraction of the window width

    height : float
        height of the box as fraction of the window height

    padding : int
        padding space in units of pixels

    bg : color
        background color of the box

    alpha: float
        opacity of the box

    pos : str, list
        position of the box, can be either a string or a (x,y) screen position in range [0,1]

    .. hint:: examples/basic/legendbox.py, examples/other/flag_labels.py
        .. image:: https://vedo.embl.es/images/other/flag_labels.png
    """

    def __init__(
        self,
        entries=(),
        nmax=12,
        c=None,
        font="",
        width=0.18,
        height=None,
        padding=2,
        bg="k8",
        alpha=0.25,
        pos="top-right",
        markers=None,
    ):
        shapes.TextBase.__init__(self)
        vtk.vtkLegendBoxActor.__init__(self)

        self.name = "LegendBox"
        self.entries = entries[:nmax]

        n = 0
        texts = []
        for e in self.entries:
            ename = e.name
            if "legend" in e.info.keys():
                if not e.info["legend"]:
                    ename = ""
                else:
                    ename = str(e.info["legend"])

            if not isinstance(e, vtk.vtkActor):
                ename = ""
            if ename:
                n += 1
            texts.append(ename)
        self.SetNumberOfEntries(n)

        if not n:
            return

        self.ScalarVisibilityOff()
        self.PickableOff()
        self.SetPadding(padding)

        self.property = self.GetEntryTextProperty()
        self.property.ShadowOff()
        self.property.BoldOff()

        # self.property.SetJustificationToLeft() # no effect
        # self.property.SetVerticalJustificationToTop()

        self.font(font)

        n = 0
        for i in range(len(self.entries)):
            ti = texts[i]
            if not ti:
                continue
            e = entries[i]
            if c is None:
                col = e.GetProperty().GetColor()
                if col == (1, 1, 1):
                    col = (0.2, 0.2, 0.2)
            else:
                col = get_color(c)
            if markers is None:  # default
                poly = e.inputdata()
            else:
                marker = markers[i] if utils.is_sequence(markers) else markers
                if isinstance(marker, vedo.Points):
                    poly = (
                        marker.clone(deep=False).normalize().shift(0, 1, 0).polydata()
                    )
                else:  # assume string marker
                    poly = vedo.shapes.Marker(marker, s=1).shift(0, 1, 0).polydata()

            self.SetEntry(n, poly, ti, col)
            n += 1

        self.SetWidth(width)
        if height is None:
            self.SetHeight(width / 3.0 * n)
        else:
            self.SetHeight(height)

        sx, sy = 1 - self.GetWidth(), 1 - self.GetHeight()
        if pos == 1 or ("top" in pos and "left" in pos):
            self.GetPositionCoordinate().SetValue(0, sy)
        elif pos == 2 or ("top" in pos and "right" in pos):
            self.GetPositionCoordinate().SetValue(sx, sy)
        elif pos == 3 or ("bottom" in pos and "left" in pos):
            self.GetPositionCoordinate().SetValue(0, 0)
        elif pos == 4 or ("bottom" in pos and "right" in pos):
            self.GetPositionCoordinate().SetValue(sx, 0)
        if alpha:
            self.UseBackgroundOn()
            self.SetBackgroundColor(get_color(bg))
            self.SetBackgroundOpacity(alpha)
        else:
            self.UseBackgroundOff()
        self.LockBorderOn()


class Button:
    """
    Build a Button object to be shown in the rendering window.

    .. hint:: examples/basic/buttons.py, examples/advanced/timer_callback2.py
        .. image:: https://vedo.embl.es/images/advanced/timer_callback1.jpg
    """

    def __init__(
        self,
        fnc,
        states,
        c,
        bc,
        pos,
        size,
        font,
        bold,
        italic,
        alpha,
        angle,
    ):
        self.statusIdx = 0
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
            if not font:
                font = settings.default_font
            self.textproperty.SetFontFamily(vtk.VTK_FONT_FILE)
            self.textproperty.SetFontFile(utils.get_font_path(font))
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
            return self.states[self.statusIdx]

        if isinstance(s, str):
            s = self.states.index(s)
        self.statusIdx = s
        self.textproperty.SetLineOffset(self.offset)
        self.actor.SetInput(self.spacer + self.states[s] + self.spacer)
        s = s % len(self.colors)  # to avoid mismatch
        self.textproperty.SetColor(get_color(self.colors[s]))
        bcc = np.array(get_color(self.bcolors[s]))
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
        self.statusIdx = (self.statusIdx + 1) % len(self.states)
        self.status(self.statusIdx)
        return self


#####################################################################
class SplineTool(vtk.vtkContourWidget):
    """
    Spline tool, to be used with ``plotter.add_spline_tool()``.

    Parameters
    ----------
    points : list, Points
        initial set of points.

    pc : str
        point color.

    ps : int
        point size.

    lc : str
        line color.

    ac : str
        active point color.

    lw : int
        line width.

    closed : bool
        spline is closed or open.

    ontop : bool
        show it always on top of other objects.

    .. hint:: examples/basic/spline_tool.py
        .. image:: https://vedo.embl.es/images/basic/spline_tool.png
    """
    def __init__(
            self, points, pc="k", ps=8, lc="r4", ac="g5", lw=2, closed=False, ontop=True
        ):
        vtk.vtkContourWidget.__init__(self)

        self.representation = vtk.vtkOrientedGlyphContourRepresentation()
        self.representation.SetAlwaysOnTop(ontop)

        self.representation.GetLinesProperty().SetColor(get_color(lc))
        self.representation.GetLinesProperty().SetLineWidth(lw)

        self.representation.GetProperty().SetColor(get_color(pc))
        self.representation.GetProperty().SetPointSize(ps)
        self.representation.GetProperty().RenderPointsAsSpheresOn()

        self.representation.GetActiveProperty().SetColor(get_color(ac))
        self.representation.GetActiveProperty().SetLineWidth(lw + 1)

        self.SetRepresentation(self.representation)

        if utils.is_sequence(points):
            self.points = Points(points)
        else:
            self.points = points

        self.closed = closed

    def add(self, pt):
        """
        Add one point at a specified position in space if 3D,
        or 2D screen-display position if 2D.
        """
        if len(pt) == 2:
            self.representation.AddNodeAtDisplayPosition(int(pt[0]), int(pt[1]))
        else:
            self.representation.AddNodeAtWorldPosition(pt)
        return self

    def remove(self, i):
        """Remove specific node by its index"""
        self.representation.DeleteNthNode(i)
        return self

    def on(self):
        """Activate/Enable the tool"""
        self.On()
        self.Render()
        return self

    def off(self):
        """Disactivate/Disable the tool"""
        self.Off()
        self.Render()
        return self

    def render(self):
        """Render the spline"""
        self.Render()
        return self

    def bounds(self):
        """Retrieve the bounding box of the spline as [x0,x1, y0,y1, z0,z1]"""
        return self.GetBounds()

    def spline(self):
        """Return the vedo.Spline object."""
        self.representation.SetClosedLoop(self.closed)
        self.representation.BuildRepresentation()
        pd = self.representation.GetContourRepresentationAsPolyData()
        pts = utils.vtk2numpy(pd.GetPoints().GetData())
        ln = vedo.Line(pts, lw=2, c="k")
        return ln

    def nodes(self, onscreen=False):
        """Return the current position in space (or on 2D screen-display) of the spline nodes."""
        n = self.representation.GetNumberOfNodes()
        pts = []
        for i in range(n):
            p = [0.0, 0.0, 0.0]
            if onscreen:
                self.representation.GetNthNodeDisplayPosition(i, p)
            else:
                self.representation.GetNthNodeWorldPosition(i, p)
            pts.append(p)
        return np.array(pts)


#####################################################################
class SliderWidget(vtk.vtkSliderWidget):
    """Helper class for vtkSliderWidget"""
    def __init__(self):
        vtk.vtkSliderWidget.__init__(self)

    @property
    def value(self):
        return self.GetRepresentation().GetValue()

    @value.setter
    def value(self, val):
        self.GetRepresentation().SetValue(val)

    @property
    def renderer(self):
        return self.GetCurrentRenderer()

    @property
    def title(self):
        self.GetRepresentation().GetTitleText()

    @title.setter
    def title(self, txt):
        self.GetRepresentation().SetTitleText(str(txt))

    @property
    def range(self):
        xmin = self.GetRepresentation().GetMinimumValue()
        xmax = self.GetRepresentation().GetMaximumValue()
        return [xmin, xmax]

    @range.setter
    def range(self, vals):
        if vals[0] is not None:
            self.GetRepresentation().SetMinimumValue(vals[0])
        if vals[1] is not None:
            self.GetRepresentation().SetMaximumValue(vals[1])

    def on(self):
        self.EnabledOn()

    def off(self):
        self.EnabledOff()


#####################################################################
def Goniometer(
    p1,
    p2,
    p3,
    font="",
    arc_size=0.4,
    s=1,
    italic=0,
    rotation=0,
    prefix="",
    lc="k2",
    c="white",
    alpha=1,
    lw=2,
    precision=3,
):
    """
    Build a graphical goniometer to measure the angle formed by 3 points in space.

    Parameters
    ----------
    p1 : list
        first point 3D coordinates.

    p2 : list
        the vertex point.

    p3 : list
        the last point defining the angle.

    font : str
        Font name to be used.

    arc_size : float
        dimension of the arc wrt the smallest axis.

    s : float
        size of the text.

    italic : float, bool
        italic text.

    rotation : float
        rotation of text in degrees.

    prefix : str
        append this string to the numeric value of the angle.

    lc : list
        color of the goniometer lines.

    c : str
        color of the goniometer angle filling. Set alpha=0 to remove it.

    alpha : float
        transparency level.

    lw : float
        line width.

    precision : int
        number of significant digits.

    .. hint:: examples/pyplot/goniometer.py
        .. image:: https://vedo.embl.es/images/pyplot/goniometer.png
    """
    if isinstance(p1, Points): p1 = p1.GetPosition()
    if isinstance(p2, Points): p2 = p2.GetPosition()
    if isinstance(p3, Points): p3 = p3.GetPosition()
    if len(p1)==2: p1=[p1[0], p1[1], 0.0]
    if len(p2)==2: p2=[p2[0], p2[1], 0.0]
    if len(p3)==2: p3=[p3[0], p3[1], 0.0]
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

    acts = []
    ln = shapes.Line([p1, p2, p3], lw=lw, c=lc)
    acts.append(ln)

    va = utils.versor(p1 - p2)
    vb = utils.versor(p3 - p2)
    r = min(utils.mag(p3 - p2), utils.mag(p1 - p2)) * arc_size
    ptsarc = []
    res = 120
    imed = int(res / 2)
    for i in range(res + 1):
        vi = utils.versor(vb * i / res + va * (res - i) / res)
        if i == imed:
            vc = np.array(vi)
        ptsarc.append(p2 + vi * r)
    arc = shapes.Line(ptsarc).lw(lw).c(lc)
    acts.append(arc)

    angle = np.arccos(np.dot(va, vb)) * 180 / np.pi

    lb = shapes.Text3D(
        prefix + utils.precision(angle, precision) + "º",
        s=r / 12 * s,
        font=font,
        italic=italic,
        justify="center",
    )
    cr = np.cross(va, vb)
    lb.pos(p2 + vc * r / 1.75).orientation(cr * np.sign(cr[2]), rotation=rotation)
    lb.c(c).bc("tomato").lighting("off")
    acts.append(lb)

    if alpha > 0:
        pts = [p2] + arc.points().tolist() + [p2]
        msh = Mesh([pts, [list(range(arc.npoints + 2))]], c=lc, alpha=alpha)
        msh.lighting("off")
        msh.triangulate()
        msh.shift(0, 0, -r / 10000)  # to resolve 2d conflicts..
        acts.append(msh)

    asse = Assembly(acts)
    return asse


def Light(
        pos,
        focal_point=(0, 0, 0),
        angle=180,
        c=None,
        intensity=1,
        remove_others=False,
    ):
    """
    Generate a source of light placed at pos, directed to focal point.
    Returns a ``vtkLight`` object.

    Parameters
    ----------
    focal_point : list
        focal point, if this is a vedo object use its position.

    angle : float
        aperture angle of the light source, in degrees

    c : color
        set the light color

    intensity : float
        intensity value between 0 and 1.

    remove_others : bool
        remove all other lights in the scene
        (in this case a Plotter object must already exist)

    .. hint:: examples/basic/lights.py
        .. image:: https://vedo.embl.es/images/basic/lights.png
    """
    if c is None:
        try:
            c = pos.color()
        except AttributeError:
            c = "white"

    if isinstance(pos, vedo.Base3DProp):
        pos = pos.pos()

    if isinstance(focal_point, vedo.Base3DProp):
        focal_point = focal_point.pos()

    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(pos)
    light.SetConeAngle(angle)
    light.SetFocalPoint(focal_point)
    light.SetIntensity(intensity)
    light.SetColor(get_color(c))

    # light.SetPositional(1) ##??
    # if ambientColor is not None: # doesnt work anyway
    #     light.SetAmbientColor(get_color(ambientColor))
    # if diffuseColor is not None:
    #     light.SetDiffuseColor(get_color(diffuseColor))
    # if specularColor is not None:
    #     light.SetSpecularColor(get_color(specularColor))

    if remove_others:
        if vedo.plotter_instance and vedo.plotter_instance.renderer:
            vedo.plotter_instance.renderer.RemoveAllLights()
        else:
            vedo.logger.error("in Light(remove_others=True): scene does not exist.")

    return light


#####################################################################
def ScalarBar(
        obj,
        title="",
        pos=(0.8,0.05),
        title_yoffset=15,
        font_size=12,
        size=(None,None),
        nlabels=None,
        c='k',
        horizontal=False,
        use_alpha=True,
        label_format=':6.3g',
    ):
    """
    A 2D scalar bar for the specified obj.

    Parameters
    ----------
    pos : list
        fractional x and y position in the 2D window

    size : list
        size of the scalarbar in pixel units (width, height)

    nlabels : int
        number of numeric labels to be shown

    use_alpha : bool
        retain transparency in scalarbar

    horizontal : bool
        show in horizontal layout

    .. hint:: examples/basic/scalarbars.py
        .. image:: https://user-images.githubusercontent.com/32848391/62940174-4bdc7900-bdd3-11e9-9713-e4f3e2fdab63.png
    """
    if not hasattr(obj, "mapper"):
        vedo.logger.error(f"in add_scalarbar(): input is invalid {type(obj)}. Skip.")
        return None

    if isinstance(obj, Points):
        vtkscalars = obj.inputdata().GetPointData().GetScalars()
        if vtkscalars is None:
            vtkscalars = obj.inputdata().GetCellData().GetScalars()
        if not vtkscalars:
            return None
        lut = vtkscalars.GetLookupTable()
        if not lut:
            lut = obj.mapper().GetLookupTable()
            if not lut:
                return None

    elif isinstance(obj, (Volume, TetMesh)):
        lut = utils.ctf2lut(obj)

    else:
        return None

    c = get_color(c)
    sb = vtk.vtkScalarBarActor()

    # print(sb.GetLabelFormat())
    label_format = label_format.replace(":","%-#")
    sb.SetLabelFormat(label_format)

    sb.SetLookupTable(lut)
    sb.SetUseOpacity(use_alpha)
    sb.SetDrawFrame(0)
    sb.SetDrawBackground(0)
    if lut.GetUseBelowRangeColor():
        sb.DrawBelowRangeSwatchOn()
        sb.SetBelowRangeAnnotation("")
    if lut.GetUseAboveRangeColor():
        sb.DrawAboveRangeSwatchOn()
        sb.SetAboveRangeAnnotation("")
    if lut.GetNanColor() != (0.5, 0.0, 0.0, 1.0):
        sb.DrawNanAnnotationOn()
        sb.SetNanAnnotation("nan")

    if title:
        if "\\" in repr(title):
            for r in shapes._reps:
                title = title.replace(r[0], r[1])
        titprop = sb.GetTitleTextProperty()
        titprop.BoldOn()
        titprop.ItalicOff()
        titprop.ShadowOff()
        titprop.SetColor(c)
        titprop.SetVerticalJustificationToTop()
        titprop.SetFontSize(font_size)
        titprop.SetFontFamily(vtk.VTK_FONT_FILE)
        titprop.SetFontFile(utils.get_font_path(settings.default_font))
        sb.SetTitle(title)
        sb.SetVerticalTitleSeparation(title_yoffset)
        sb.SetTitleTextProperty(titprop)

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
        sb.SetPosition(pos[0] + 0.09, pos[1])
        sb.SetMaximumWidthInPixels(60)
        sb.SetMaximumHeightInPixels(250)

    if size[0] is not None:
        sb.SetMaximumWidthInPixels(size[0])
    if size[1] is not None:
        sb.SetMaximumHeightInPixels(size[1])

    if nlabels is not None:
        sb.SetNumberOfLabels(nlabels)

    sctxt = sb.GetLabelTextProperty()
    sctxt.SetFontFamily(vtk.VTK_FONT_FILE)
    sctxt.SetFontFile(utils.get_font_path(settings.default_font))
    sctxt.SetColor(c)
    sctxt.SetShadow(0)
    sctxt.SetFontSize(font_size - 2)
    sb.SetAnnotationTextProperty(sctxt)
    sb.PickableOff()
    return sb


#####################################################################
def ScalarBar3D(
        obj,
        title='',
        pos=None,
        s=(None, None),
        title_font="",
        title_xoffset=-1.5,
        title_yoffset=0.0,
        title_size=1.5,
        title_rotation=0.0,
        nlabels=8,
        label_font="",
        label_size=1,
        label_offset=0.375,
        label_rotation=0,
        label_format="",
        italic=0,
        c=None,
        draw_box=True,
        above_text=None,
        below_text=None,
        nan_text='NaN',
        categories=None,
    ):
    """
    Create a 3D scalar bar for the specified object.

    Arguments
    ---------
    ``obj`` input can be:
        - a list of numbers,
        - a list of two numbers in the form `(min, max)`,
        - a ``Mesh`` already containing a set of scalars associated to vertices or cells,
        - if ``None`` the last object in the list of actors will be used.

    Parameters
    ----------
    s : list
        (thickness, length) of scalarbar

    title : str
        scalar bar title

    title_xoffset : float
        horizontal space btw title and color scalarbar

    title_yoffset : float
        vertical space offset

    title_size : float
        size of title wrt numeric labels

    title_rotation : float
        title rotation in degrees

    nlabels : int
        number of numeric labels

    label_font : str
        font type for labels

    label_size : float
        label scale factor

    label_offset : float
        space btw numeric labels and scale

    label_rotation : float
        label rotation in degrees

    draw_box : bool
        draw a box around the colorbar

    categories : list
        make a categorical scalarbar,
        the input list will have the format [value, color, alpha, textlabel]

    .. hint:: examples/basic/scalarbars.py
    """

    if isinstance(obj, Points):
        lut = obj.mapper().GetLookupTable()
        if not lut or lut.GetTable().GetNumberOfTuples() == 0:
            # create the most similar to the default
            obj.cmap("jet_r")
            lut = obj.mapper().GetLookupTable()
        vmin, vmax = lut.GetRange()

    elif isinstance(obj, (Volume, TetMesh)):
        lut = utils.ctf2lut(obj)
        vmin, vmax = lut.GetRange()

    elif utils.is_sequence(obj):
        vmin, vmax = np.min(obj), np.max(obj)

    else:
        vedo.logger.error("in ScalarBar3D(): input must be a vedo object with bounds.")
        return obj

    bns = obj.GetBounds()
    sx, sy = s
    if sy is None:
        sy = bns[3] - bns[2]
    if sx is None:
        sx = sy / 18

    if categories is not None:  ################################
        ncats = len(categories)
        scale = shapes.Grid([-sx * label_offset, 0, 0], c=c, alpha=1, s=(sx,sy), res=(1,ncats))
        cols, alphas= [], []
        ticks_pos, ticks_txt = [0.0], ['']
        for i, cat in enumerate(categories):
            cl = get_color(cat[1])
            cols.append(cl)
            if len(cat) > 2:
                alphas.append(cat[2])
            else:
                alphas.append(1)
            if len(cat) > 3:
                ticks_txt.append(cat[3])
            else:
                ticks_txt.append("")
            ticks_pos.append((i + 0.5) / ncats)
        ticks_pos.append(1.0)
        ticks_txt.append("")
        rgba = np.c_[np.array(cols) * 255, np.array(alphas) * 255]
        scale.cell_individual_colors(rgba)

    else:  ########################################################

        # build the color scale part
        scale = shapes.Grid([-sx * label_offset, 0, 0], c=c, alpha=1, s=(sx,sy),
                            res=(1, lut.GetTable().GetNumberOfTuples()))
        cscals = np.linspace(vmin, vmax, lut.GetTable().GetNumberOfTuples())

        if lut.GetScale():  # logarithmic scale
            lut10 = vtk.vtkLookupTable()
            lut10.DeepCopy(lut)
            lut10.SetScaleToLinear()
            scale.cmap(lut10, cscals, on="cells")
            tk = utils.make_ticks(vmin, vmax, nlabels, logscale=True, useformat=label_format)
        else:
            scale.cmap(lut, cscals, on="cells")
            tk = utils.make_ticks(vmin, vmax, nlabels, logscale=False, useformat=label_format)
        ticks_pos, ticks_txt = tk
    scale.lw(0).wireframe(False).lighting("off")

    scales = [scale]

    xbns = scale.xbounds()
    if pos is None:
        d = sx / 2
        if title:
            d = np.sqrt((bns[1] - bns[0]) ** 2 + sy * sy) / 20
        pos = (bns[1] - xbns[0] + d, (bns[2] + bns[3]) / 2, bns[4])

    lsize = sy / 60 * label_size

    tacts = []
    for i, p in enumerate(ticks_pos):
        tx = ticks_txt[i]
        if i and tx:
            # build numeric text
            y = (p - 0.5) * sy
            if label_rotation:
                a = shapes.Text3D(
                    tx,
                    pos=[sx * label_offset, y, 0],
                    s=lsize,
                    justify="center-top",
                    c=c,
                    italic=italic,
                    font=label_font,
                )
                a.RotateZ(label_rotation)
            else:
                a = shapes.Text3D(
                    tx,
                    pos=[sx * label_offset, y, 0],
                    s=lsize,
                    justify="center-left",
                    c=c,
                    italic=italic,
                    font=label_font,
                )

            tacts.append(a)

            # build ticks
            tic = shapes.Line(
                [xbns[1], y, 0], [xbns[1] + sx * label_offset / 4, y, 0], lw=2, c=c
            )
            tacts.append(tic)

    # build title
    if title:
        t = shapes.Text3D(
            title,
            (0, 0, 0),
            s=sy / 50 * title_size,
            c=c,
            justify="centered",
            italic=italic,
            font=title_font,
        )
        t.RotateZ(90 + title_rotation)
        t.pos(sx * title_xoffset, title_yoffset, 0)
        tacts.append(t)

    # build below scale
    if lut.GetUseBelowRangeColor():
        r, g, b, alfa = lut.GetBelowRangeColor()
        brect = shapes.Rectangle(
            [-sx * label_offset - sx / 2, -sy / 2 - sx - sx * 0.1, 0],
            [-sx * label_offset + sx / 2, -sy / 2 - sx * 0.1, 0],
            c=(r, g, b),
            alpha=alfa,
        )
        brect.lw(1).lc(c).lighting("off")
        scales += [brect]
        if below_text is None:
            below_text = " <" + str(vmin)
        if below_text:
            if label_rotation:
                btx = shapes.Text3D(
                    below_text,
                    (0, 0, 0),
                    s=lsize,
                    c=c,
                    justify="center-top",
                    italic=italic,
                    font=label_font,
                )
                btx.RotateZ(label_rotation)
            else:
                btx = shapes.Text3D(
                    below_text,
                    (0, 0, 0),
                    s=lsize,
                    c=c,
                    justify="center-left",
                    italic=italic,
                    font=label_font,
                )

            btx.pos(sx * label_offset, -sy / 2 - sx * 0.66, 0)
            tacts.append(btx)

    # build above scale
    if lut.GetUseAboveRangeColor():
        r, g, b, alfa = lut.GetAboveRangeColor()
        arect = shapes.Rectangle(
            [-sx * label_offset - sx / 2, sy / 2 + sx * 0.1, 0],
            [-sx * label_offset + sx / 2, sy / 2 + sx + sx * 0.1, 0],
            c=(r, g, b),
            alpha=alfa,
        )
        arect.lw(1).lc(c).lighting("off")
        scales += [arect]
        if above_text is None:
            above_text = " >" + str(vmax)
        if above_text:
            if label_rotation:
                atx = shapes.Text3D(
                    above_text,
                    (0, 0, 0),
                    s=lsize,
                    c=c,
                    justify="center-top",
                    italic=italic,
                    font=label_font,
                )
                atx.RotateZ(label_rotation)
            else:
                atx = shapes.Text3D(
                    above_text,
                    (0, 0, 0),
                    s=lsize,
                    c=c,
                    justify="center-left",
                    italic=italic,
                    font=label_font,
                )

            atx.pos(sx * label_offset, sy / 2 + sx * 0.66, 0)
            tacts.append(atx)

    # build NaN scale
    if lut.GetNanColor() != (0.5, 0.0, 0.0, 1.0):
        nanshift = sx * 0.1
        if brect:
            nanshift += sx
        r, g, b, alfa = lut.GetNanColor()
        nanrect = shapes.Rectangle(
            [-sx * label_offset - sx / 2, -sy / 2 - sx - sx * 0.1 - nanshift, 0],
            [-sx * label_offset + sx / 2, -sy / 2 - sx * 0.1 - nanshift, 0],
            c=(r, g, b),
            alpha=alfa,
        )
        nanrect.lw(1).lc(c).lighting("off")
        scales += [nanrect]
        if label_rotation:
            nantx = shapes.Text3D(
                nan_text,
                (0, 0, 0),
                s=lsize,
                c=c,
                justify="center-left",
                italic=italic,
                font=label_font,
            )
            nantx.RotateZ(label_rotation)
        else:
            nantx = shapes.Text3D(
                nan_text,
                (0, 0, 0),
                s=lsize,
                c=c,
                justify="center-left",
                italic=italic,
                font=label_font,
            )
        nantx.pos(sx * label_offset, -sy / 2 - sx * 0.66 - nanshift, 0)
        tacts.append(nantx)

    if draw_box:
        tacts.append(scale.box().lw(1))

    for a in tacts:
        a.PickableOff()

    mtacts = merge(tacts).lighting("off")
    mtacts.PickableOff()
    scale.PickableOff()

    sact = Assembly(scales + tacts)
    sact.SetPosition(pos)
    sact.PickableOff()
    sact.UseBoundsOff()
    sact.name = "ScalarBar3D"
    return sact


#####################################################################
def add_slider(
        sliderfunc,
        xmin,
        xmax,
        value=None,
        pos=4,
        title="",
        font="",
        title_size=1,
        c=None,
        show_value=True,
        delayed=False,
        **options,
    ):
    """
    Add a slider widget which can call an external custom function.

    Set any value as float to increase the number of significant digits above the slider.

    Parameters
    ----------
    sliderfunc : function
        external function to be called by the widget

    xmin : float
        lower value of the slider

    xmax : float
        upper value

    value : float
        current value

    pos : list, str
        position corner number: horizontal [1-5] or vertical [11-15]
        it can also be specified by corners coordinates [(x1,y1), (x2,y2)]
        and also by a string descriptor (eg. "bottom-left")

    title : str
        title text

    font : str
        title font face

    title_size : float
        title text scale [1.0]

    show_value : bool
        if true current value is shown

    delayed : bool
        if True the callback is delayed until when the mouse button is released

    alpha : float
        opacity of the scalar bar texts

    slider_length : float
        slider length

    slider_width : float
        slider width

    end_cap_length : float
        length of the end cap

    end_cap_width : float
        width of the end cap

    tube_width : float
        width of the tube

    title_height : float
        width of the title

    tformat : str
        format of the title

    .. hint:: examples/basic/sliders1.py, sliders2.py
        ..image:: https://user-images.githubusercontent.com/32848391/50738848-be033480-11d8-11e9-9b1a-c13105423a79.jpg
    """
    options = dict(options)
    value = options.pop("value", value)
    pos = options.pop("pos", pos)
    title = options.pop("title", title)
    font = options.pop("font", font)
    title_size = options.pop("title_size", title_size)
    c = options.pop("c", c)
    show_value = options.pop("show_value", show_value)
    delayed = options.pop("delayed", delayed)
    alpha = options.pop("alpha", 1)
    slider_length = options.pop("slider_length", 0.015)
    slider_width  = options.pop("slider_width", 0.025)
    end_cap_length= options.pop("end_cap_length", 0.0015)
    end_cap_width = options.pop("end_cap_width", 0.0125)
    tube_width    = options.pop("tube_width", 0.0075)
    title_height  = options.pop("title_height", 0.022)

    plt = vedo.plotter_instance
    if c is None:  # automatic black or white
        c = (0.8, 0.8, 0.8)
        if np.sum(get_color(plt.backgrcol)) > 1.5:
            c = (0.2, 0.2, 0.2)
    c = get_color(c)

    if value is None or value < xmin:
        value = xmin

    sliderRep = vtk.vtkSliderRepresentation2D()
    sliderRep.SetMinimumValue(xmin)
    sliderRep.SetMaximumValue(xmax)
    sliderRep.SetValue(value)
    sliderRep.SetSliderLength(slider_length)
    sliderRep.SetSliderWidth(slider_width)
    sliderRep.SetEndCapLength(end_cap_length)
    sliderRep.SetEndCapWidth(end_cap_width)
    sliderRep.SetTubeWidth(tube_width)
    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()

    if isinstance(pos, str):
        if "top" in pos:
            if "left" in pos:
                if "vert" in pos:
                    pos = 11
                else:
                    pos = 1
            elif "right" in pos:
                if "vert" in pos:
                    pos = 12
                else:
                    pos = 2
        elif "bott" in pos:
            if "left" in pos:
                if "vert" in pos:
                    pos = 13
                else:
                    pos = 3
            elif "right" in pos:
                if "vert" in pos:
                    if "span" in pos:
                        pos = 15
                    else:
                        pos = 14
                else:
                    pos = 4
            elif "span" in pos:
                pos = 5

    if utils.is_sequence(pos):
        sliderRep.GetPoint1Coordinate().SetValue(pos[0][0], pos[0][1])
        sliderRep.GetPoint2Coordinate().SetValue(pos[1][0], pos[1][1])
    elif pos == 1:  # top-left horizontal
        sliderRep.GetPoint1Coordinate().SetValue(0.04, 0.93)
        sliderRep.GetPoint2Coordinate().SetValue(0.45, 0.93)
    elif pos == 2:
        sliderRep.GetPoint1Coordinate().SetValue(0.55, 0.93)
        sliderRep.GetPoint2Coordinate().SetValue(0.95, 0.93)
    elif pos == 3:
        sliderRep.GetPoint1Coordinate().SetValue(0.05, 0.06)
        sliderRep.GetPoint2Coordinate().SetValue(0.45, 0.06)
    elif pos == 4:  # bottom-right
        sliderRep.GetPoint1Coordinate().SetValue(0.55, 0.06)
        sliderRep.GetPoint2Coordinate().SetValue(0.95, 0.06)
    elif pos == 5:  # bottom span horizontal
        sliderRep.GetPoint1Coordinate().SetValue(0.04, 0.06)
        sliderRep.GetPoint2Coordinate().SetValue(0.95, 0.06)
    elif pos == 11:  # top-left vertical
        sliderRep.GetPoint1Coordinate().SetValue(0.065, 0.54)
        sliderRep.GetPoint2Coordinate().SetValue(0.065, 0.9)
    elif pos == 12:
        sliderRep.GetPoint1Coordinate().SetValue(0.94, 0.54)
        sliderRep.GetPoint2Coordinate().SetValue(0.94, 0.9)
    elif pos == 13:
        sliderRep.GetPoint1Coordinate().SetValue(0.065, 0.1)
        sliderRep.GetPoint2Coordinate().SetValue(0.065, 0.54)
    elif pos == 14:  # bottom-right vertical
        sliderRep.GetPoint1Coordinate().SetValue(0.94, 0.1)
        sliderRep.GetPoint2Coordinate().SetValue(0.94, 0.54)
    elif pos == 15:  # right margin vertical
        sliderRep.GetPoint1Coordinate().SetValue(0.95, 0.1)
        sliderRep.GetPoint2Coordinate().SetValue(0.95, 0.9)
    else:  # bottom-right
        sliderRep.GetPoint1Coordinate().SetValue(0.55, 0.06)
        sliderRep.GetPoint2Coordinate().SetValue(0.95, 0.06)

    if show_value:
        if isinstance(xmin, int) and isinstance(xmax, int) and isinstance(value, int):
            frm = "%0.0f"
        else:
            frm = "%0.2f"

        frm = options.pop("tformat", frm)

        sliderRep.SetLabelFormat(frm)  # default is '%0.3g'
        sliderRep.GetLabelProperty().SetShadow(0)
        sliderRep.GetLabelProperty().SetBold(0)
        sliderRep.GetLabelProperty().SetOpacity(alpha)
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

    sliderRep.SetTitleHeight(title_height * title_size)
    sliderRep.GetTitleProperty().SetShadow(0)
    sliderRep.GetTitleProperty().SetColor(c)
    sliderRep.GetTitleProperty().SetOpacity(alpha)
    sliderRep.GetTitleProperty().SetBold(0)
    if font.lower() == "courier":
        sliderRep.GetTitleProperty().SetFontFamilyToCourier()
    elif font.lower() == "times":
        sliderRep.GetTitleProperty().SetFontFamilyToTimes()
    elif font.lower() == "arial":
        sliderRep.GetTitleProperty().SetFontFamilyToArial()
    else:
        if font == "":
            font = utils.get_font_path(settings.default_font)
        else:
            font = utils.get_font_path(font)
        sliderRep.GetTitleProperty().SetFontFamily(vtk.VTK_FONT_FILE)
        sliderRep.GetLabelProperty().SetFontFamily(vtk.VTK_FONT_FILE)
        sliderRep.GetTitleProperty().SetFontFile(font)
        sliderRep.GetLabelProperty().SetFontFile(font)

    if title:
        sliderRep.SetTitleText(title)
        if not utils.is_sequence(pos):
            if isinstance(pos, int) and pos > 10:
                sliderRep.GetTitleProperty().SetOrientation(90)
        else:
            if abs(pos[0][0] - pos[1][0]) < 0.1:
                sliderRep.GetTitleProperty().SetOrientation(90)

    sliderWidget = SliderWidget()
    sliderWidget.SetInteractor(plt.interactor)
    sliderWidget.SetAnimationModeToJump()
    sliderWidget.SetRepresentation(sliderRep)
    if delayed:
        sliderWidget.AddObserver("EndInteractionEvent", sliderfunc)
    else:
        sliderWidget.AddObserver("InteractionEvent", sliderfunc)
    if plt.renderer:
        sliderWidget.SetCurrentRenderer(plt.renderer)
    sliderWidget.on()
    plt.sliders.append([sliderWidget, sliderfunc])
    return sliderWidget


#####################################################################
def add_slider3d(
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
        show_value=True,
    ):
    """
    Add a 3D slider widget which can call an external custom function.

    Parameters
    ----------
    sliderfunc : function
        external function to be called by the widget

    pos1 : list
        first position 3D coordinates

    pos2 : list
        second position 3D coordinates

    xmin : float
        lower value

    xmax : float
        upper value

    value : float
        initial value

    s : float
        label scaling factor

    t : float
        tube scaling factor

    title : str
        title text

    c : color
        slider color

    rotation : float
        title rotation around slider axis

    show_value : bool
        if True current value is shown on top of the slider

    .. hint:: examples/basic/sliders3d.py
    """
    plt = vedo.plotter_instance
    if c is None:  # automatic black or white
        c = (0.8, 0.8, 0.8)
        if np.sum(get_color(plt.backgrcol)) > 1.5:
            c = (0.2, 0.2, 0.2)
    else:
        c = get_color(c)

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

    if not show_value:
        sliderRep.ShowSliderLabelOff()

    sliderRep.SetTitleText(title)
    sliderRep.SetTitleHeight(s * t)
    sliderRep.SetLabelHeight(s * t * 0.85)

    sliderRep.GetTubeProperty().SetColor(c)

    sliderWidget = SliderWidget()
    sliderWidget.SetInteractor(plt.interactor)
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.SetAnimationModeToJump()
    sliderWidget.AddObserver("InteractionEvent", sliderfunc)
    sliderWidget.on()
    plt.sliders.append([sliderWidget, sliderfunc])
    return sliderWidget


#####################################################################
def add_button(
        fnc,
        states=("On", "Off"),
        c=("w", "w"),
        bc=("dg", "dr"),
        pos=(0.7, 0.05),
        size=24,
        font="Normografo",
        bold=False,
        italic=False,
        alpha=1,
        angle=0,
    ):
    """
    Add a button to the renderer window.

    Parameters
    ----------
    fnc : function
        external function to be called by the widget

    states : list
        the list of possible states, eg. ['On', 'Off']

    c : list
        the list of colors for each state eg. ['red3', 'green5']

    bc : list
        the list of background colors for each state

    pos : list, str
        2D position in pixels from left-bottom corner

    size : int
        size of button font

    font : str
        font type (arial, courier, times)

    bold : bool
        set bold font face

    italic :
        italic font face

    alpha : float
        opacity level

    angle : float
        anticlockwise rotation in degrees

    .. hint:: examples/basic/buttons.py
        .. image:: https://user-images.githubusercontent.com/32848391/50738870-c0fe2500-11d8-11e9-9b78-92754f5c5968.jpg
    """
    plt = vedo.plotter_instance
    if not plt.renderer:
        vedo.logger.error("Use addButton() only after rendering the scene.")
        return None
    bu = Button(fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle)
    plt.renderer.AddActor2D(bu.actor)
    plt.window.Render()
    plt.buttons.append(bu)
    return bu


def add_cutter_tool(obj=None, mode="box", invert=False):
    """
    Create an interactive tool to cut away parts of a mesh or volume.

    Parameters
    ----------
    mode : str
        either "box", "plane" or "sphere"

    invert : bool
        invert selection (inside-out)


    .. hint:: examples/basic/cutter.py
        .. image:: https://user-images.githubusercontent.com/32848391/50738866-c0658e80-11d8-11e9-955b-551d4d8b0db5.jpg
    """
    if obj is None:
        obj = vedo.plotter_instance.actors[0]
    try:
        if isinstance(obj, vedo.Volume):
            return _addCutterToolVolumeWithBox(obj, invert)

        if mode == "box":
            return _addCutterToolMeshWithBox(obj, invert)
        elif mode == "plane":
            return _addCutterToolMeshWithPlane(obj, invert)
        elif mode == "sphere":
            return _addCutterToolMeshWithSphere(obj, invert)
        else:
            raise RuntimeError(f"Unknown mode: {mode}")
    except:
        return None


def _addCutterToolMeshWithSphere(mesh, invert):
    plt = vedo.plotter_instance

    sph = vtk.vtkSphere()
    cm = mesh.center_of_mass()
    sph.SetCenter(cm)
    aves = mesh.average_size()
    sph.SetRadius(aves)
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(mesh.polydata())
    clipper.SetClipFunction(sph)
    clipper.GenerateClippedOutputOn()
    clipper.GenerateClipScalarsOff()
    clipper.SetInsideOut(not invert)
    clipper.Update()
    mesh.mapper().SetInputConnection(clipper.GetOutputPort())

    act0 = Mesh(clipper.GetOutput(), alpha=mesh.alpha())  # the main cut part
    act0.mapper().SetLookupTable(mesh.mapper().GetLookupTable())
    act0.mapper().SetScalarRange(mesh.mapper().GetScalarRange())

    act1 = Mesh()
    act1.mapper().SetInputConnection(clipper.GetClippedOutputPort())  # needs OutputPort
    act1.color((0.5, 0.5, 0.5), 0.04).wireframe()

    plt.remove(mesh, render=False)
    plt.add([act0, act1])

    def myCallback(obj, event):
        obj.GetSphere(sph)

    sphereWidget = vtk.vtkSphereWidget()
    sphereWidget.SetThetaResolution(120)
    sphereWidget.SetPhiResolution(60)
    sphereWidget.SetRadius(aves)
    sphereWidget.SetCenter(cm)
    sphereWidget.SetRepresentation(2)
    sphereWidget.HandleVisibilityOff()
    sphereWidget.GetSphereProperty().SetOpacity(0.2)
    sphereWidget.GetSelectedSphereProperty().SetOpacity(0.1)
    sphereWidget.SetInteractor(plt.interactor)
    sphereWidget.SetCurrentRenderer(plt.renderer)
    sphereWidget.SetInputData(mesh.inputdata())
    sphereWidget.AddObserver("InteractionEvent", myCallback)
    plt.interactor.Render()
    sphereWidget.On()
    plt.widgets.append(sphereWidget)

    plt.cutterWidget = sphereWidget
    plt.clickedActor = act0
    if mesh in plt.actors:
        ia = plt.actors.index(mesh)
        plt.actors[ia] = act0

    printc("Mesh Cutter Tool:", c="m", invert=1)
    printc("  Move gray handles to cut away parts of the mesh", c="m")
    printc("  Press X to save file to: clipped.vtk", c="m")
    printc("  [Press space bar to continue]", c="m")
    plt.interactor.Start()
    sphereWidget.Off()
    plt.interactor.Start()  # allow extra interaction
    return act0


def _addCutterToolMeshWithBox(mesh, invert):
    plt = vedo.plotter_instance
    if not plt:
        vedo.logger.error("in add_cutter_tool() scene must be first rendered.")
        raise RuntimeError()

    plt.clickedActor = mesh
    apd = mesh.polydata()

    planes = vtk.vtkPlanes()
    planes.SetBounds(apd.GetBounds())

    clipper = vtk.vtkClipPolyData()
    clipper.GenerateClipScalarsOff()
    clipper.SetInputData(apd)
    clipper.SetClipFunction(planes)
    clipper.SetInsideOut(not invert)
    clipper.GenerateClippedOutputOn()
    clipper.Update()
    cpoly = clipper.GetOutput()

    act0 = Mesh(cpoly, alpha=mesh.alpha())  # the main cut part
    act0.mapper().SetLookupTable(mesh.mapper().GetLookupTable())
    act0.mapper().SetScalarRange(mesh.mapper().GetScalarRange())

    act1 = Mesh()
    act1.mapper().SetInputConnection(clipper.GetClippedOutputPort())  # needs OutputPort
    act1.alpha(0.04).color((0.5, 0.5, 0.5)).wireframe()

    plt.remove(mesh, render=False)
    plt.add([act0, act1])

    def selectPolygons(vobj, event):
        vobj.GetPlanes(planes)

    boxWidget = vtk.vtkBoxWidget()
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
    boxWidget.GetOutlineProperty().SetColor(0.2, 0.2, 0.2)
    boxWidget.GetOutlineProperty().SetOpacity(1)
    boxWidget.SetPlaceFactor(1.025)
    boxWidget.SetInteractor(plt.interactor)
    boxWidget.SetCurrentRenderer(plt.renderer)
    boxWidget.SetInputData(apd)
    boxWidget.PlaceWidget()
    boxWidget.AddObserver("InteractionEvent", selectPolygons)
    boxWidget.On()
    plt.widgets.append(boxWidget)

    plt.cutterWidget = boxWidget
    plt.clickedActor = act0
    if mesh in plt.actors:
        ia = plt.actors.index(mesh)
        plt.actors[ia] = act0

    printc("Mesh Cutter Tool:", c="m", invert=1)
    printc("  Move gray handles to cut away parts of the mesh", c="m")
    printc("  Press X to save file to: clipped.vtk", c="m")
    printc("  [Press space bar to continue]", c="m")
    plt.interactor.Start()
    boxWidget.Off()
    plt.interactor.Start()  # allow extra interaction
    return act0


def _addCutterToolMeshWithPlane(mesh, invert):
    plt = vedo.plotter_instance

    plane = vtk.vtkPlane()
    plane.SetNormal(mesh.center_of_mass())
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(mesh.polydata())
    clipper.SetClipFunction(plane)
    clipper.GenerateClipScalarsOff()
    clipper.GenerateClippedOutputOn()
    clipper.SetInsideOut(invert)
    clipper.Update()
    mesh.mapper().SetInputConnection(clipper.GetOutputPort())

    act0 = Mesh(clipper.GetOutput(), alpha=mesh.alpha())  # the main cut part
    act0.mapper().SetLookupTable(mesh.mapper().GetLookupTable())
    act0.mapper().SetScalarRange(mesh.mapper().GetScalarRange())

    act1 = Mesh()
    act1.mapper().SetInputConnection(clipper.GetClippedOutputPort())  # needs OutputPort
    act1.alpha(0.04).color((0.5, 0.5, 0.5)).wireframe()

    plt.remove(mesh, render=False)
    plt.add([act0, act1])

    def myCallback(obj, event):
        obj.GetPlane(plane)

    planeWidget = vtk.vtkImplicitPlaneWidget()
    planeWidget.SetNormal(1, 0, 0)
    planeWidget.SetPlaceFactor(1.25)
    planeWidget.SetInteractor(plt.interactor)
    planeWidget.SetCurrentRenderer(plt.renderer)
    planeWidget.SetInputData(mesh.inputdata())
    planeWidget.PlaceWidget(mesh.GetBounds())
    planeWidget.AddObserver("InteractionEvent", myCallback)
    planeWidget.GetPlaneProperty().SetColor(get_color("grey"))
    planeWidget.GetPlaneProperty().SetOpacity(0.5)
    planeWidget.SetTubing(False)
    planeWidget.SetOutlineTranslation(True)
    planeWidget.SetOriginTranslation(True)
    planeWidget.SetDrawPlane(False)
    planeWidget.GetPlaneProperty().LightingOff()
    plt.interactor.Render()
    planeWidget.On()
    plt.widgets.append(planeWidget)

    plt.cutterWidget = planeWidget
    plt.clickedActor = act0
    if mesh in plt.actors:
        ia = plt.actors.index(mesh)
        plt.actors[ia] = act0

    printc("Mesh Cutter Tool:", c="m", invert=1)
    printc("  Move gray handles to cut away parts of the mesh", c="m")
    printc("  Press X to save file to: clipped.vtk", c="m")
    printc("  [Press space bar to continue]", c="m")
    plt.interactor.Start()
    planeWidget.Off()
    plt.interactor.Start()  # allow extra interaction
    return act0


def _addCutterToolVolumeWithBox(vol, invert):
    plt = vedo.plotter_instance

    boxWidget = vtk.vtkBoxWidget()
    boxWidget.SetInteractor(plt.interactor)
    boxWidget.SetPlaceFactor(1.0)

    plt.cutterWidget = boxWidget

    plt.renderer.AddVolume(vol)

    planes = vtk.vtkPlanes()

    def _clip(obj, event):
        obj.GetPlanes(planes)
        vol.mapper().SetClippingPlanes(planes)

    boxWidget.SetInputData(vol.inputdata())
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
    boxWidget.GetOutlineProperty().SetColor(0.2, 0.2, 0.2)
    boxWidget.GetOutlineProperty().SetOpacity(0.7)
    boxWidget.SetPlaceFactor(1.0)
    boxWidget.PlaceWidget()
    boxWidget.SetInsideOut(not invert)
    boxWidget.AddObserver("InteractionEvent", _clip)

    printc("Volume Cutter Tool:", c="m", invert=1)
    printc("  Move gray handles to cut parts of the volume", c="m")

    plt.interactor.Render()
    boxWidget.On()
    plt.interactor.Start()
    boxWidget.Off()
    plt.widgets.append(boxWidget)
    return vol


#####################################################################
def add_renderer_frame(plotter_instance, c=None, alpha=None, lw=None, padding=None):
    """
    Add a line around the renderer subwindow.

    Parameters
    ----------
    c : color
        color of the line.
    alpha : float
        opacity.
    lw : int
        line width in pixels.
    padding : int
        padding in pixel units.
    """

    if lw is None:
        lw = settings.renderer_frame_width
    if lw == 0:
        return None

    if padding is None:
        padding = settings.renderer_frame_padding

    if c is None:  # automatic black or white
        c = (0.9, 0.9, 0.9)
        if np.sum(plotter_instance.renderer.GetBackground()) > 1.5:
            c = (0.1, 0.1, 0.1)
    c = get_color(c)

    if alpha is None:
        alpha = settings.renderer_frame_alpha

    ppoints = vtk.vtkPoints()  # Generate the polyline
    xy = 1 - padding
    psqr = [
        [padding, padding],
        [padding, xy],
        [xy, xy],
        [xy, padding],
        [padding, padding],
    ]
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

    plotter_instance.renderer.AddActor(fractor)
    return fractor


#####################################################################
def add_icon(mesh, pos=3, size=0.08):
    """
    Add an inset icon mesh into the renderer.

    Parameters
    ----------
    pos : list, int
        icon position in the range [1-4] indicating one of the 4 corners,
        or it can be a tuple (x,y) as a fraction of the renderer size.

    size : float
        size of the icon space as fraction of the window size.

    .. hint:: examples/other/icon.py
    """
    plt = vedo.plotter_instance
    if not plt.renderer:
        vedo.logger.warning("Use add_icon() after first rendering the scene.")

        save_int = plt.interactive
        plt.show(interactive=0)
        plt.interactive = save_int
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(mesh)
    widget.SetInteractor(plt.interactor)
    if utils.is_sequence(pos):
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
def compute_visible_bounds(actors=None):
    """Calculate max meshes bounds and sizes."""
    bns = []

    if actors is None:
        actors = vedo.plotter_instance.actors
    elif not utils.is_sequence(actors):
        actors = [actors]

    try:
        # this block fails for VolumeSlice as vtkImageSlice.GetBounds() returns a pointer..
        # in any case we dont need axes for that one.
        for a in actors:
            if a and a.GetUseBounds():
                b = a.GetBounds()
                if b:
                    bns.append(b)
        if bns:
            max_bns = np.max(bns, axis=0)
            min_bns = np.min(bns, axis=0)
            vbb = [min_bns[0], max_bns[1], min_bns[2], max_bns[3], min_bns[4], max_bns[5]]
        else:
            vbb = vedo.plotter_instance.renderer.ComputeVisiblePropBounds()
            max_bns = vbb
            min_bns = vbb
        sizes = np.array(
            [max_bns[1] - min_bns[0], max_bns[3] - min_bns[2], max_bns[5] - min_bns[4]]
        )
        return [vbb, sizes, min_bns, max_bns]
    except:
        return [(0, 0, 0, 0, 0, 0), (0, 0, 0), 0, 0]


#####################################################################
def Ruler(
        p1, p2,
        units_scale=1,
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
        label_rotation=0,
        axis_rotation=0,
        tick_angle=90,
    ):
    """
    Build a 3D ruler to indicate the distance of two points p1 and p2.

    Parameters
    ----------
    label : str
        alternative fixed label to be shown

    units_scale : float
        factor to scale units (e.g. μm to mm)

    s : float
        size of the label

    font : str
        font face

    italic : float
        italicness of the font in the range [0,1]

    units : str
        string to be appended to the numeric value

    lw : int
        line width in pixel units

    precision : int
        nr of significant digits to be shown

    label_rotation : float
        initial rotation of the label around the z-axis

    axis_rotation : float
        initial rotation of the line around the main axis

    tick_angle : float
        initial rotation of the line around the main axis

    .. hint:: examples/pyplot/goniometer.py
        .. image:: https://vedo.embl.es/images/pyplot/goniometer.png
    """
    if units_scale != 1.0 and units == "":
        raise ValueError(
            "When setting 'units_scale' to a value other than 1, "
            + "a 'units' arguments must be specified."
        )

    if isinstance(p1, Points):
        p1 = p1.GetPosition()
    if isinstance(p2, Points):
        p2 = p2.GetPosition()

    if len(p1)==2:
        p1=[p1[0],p1[1],0.0]
    if len(p2)==2:
        p2=[p2[0],p2[1],0.0]

    p1, p2 = np.array(p1), np.array(p2)
    q1, q2 = [0, 0, 0], [utils.mag(p2 - p1), 0, 0]
    q1, q2 = np.array(q1), np.array(q2)
    v = q2 - q1
    d = utils.mag(v) * units_scale

    if s is None:
        s = d * 0.02 * (1 / units_scale)

    if not label:
        label = str(d)
        if precision:
            label = utils.precision(d, precision)
    if prefix:
        label = prefix + "~" + label
    if units:
        label += "~" + units

    lb = shapes.Text3D(
        label, pos=(q1 + q2) / 2, s=s, font=font, italic=italic, justify="center"
    )
    if label_rotation:
        lb.RotateZ(label_rotation)

    x0, x1 = lb.xbounds()
    gap = [(x1 - x0) / 2, 0, 0]
    pc1 = (v / 2 - gap) * 0.9 + q1
    pc2 = q2 - (v / 2 - gap) * 0.9

    lc1 = shapes.Line(q1 - v / 50, pc1)
    lc2 = shapes.Line(q2 + v / 50, pc2)

    zs = np.array([0, d / 50 * (1 / units_scale), 0])
    ml1 = shapes.Line(-zs, zs).pos(q1)
    ml2 = shapes.Line(-zs, zs).pos(q2)
    ml1.RotateZ(tick_angle - 90)
    ml2.RotateZ(tick_angle - 90)

    c1 = shapes.Circle(q1, r=d / 180 * (1 / units_scale), res=20)
    c2 = shapes.Circle(q2, r=d / 180 * (1 / units_scale), res=20)

    acts = [lb, lc1, lc2, c1, c2, ml1, ml2]
    macts = merge(acts).pos(p1).c(c).alpha(alpha)
    macts.GetProperty().LightingOff()
    macts.GetProperty().SetLineWidth(lw)
    macts.UseBoundsOff()
    macts.base = q1
    macts.top = q2
    macts.orientation(p2 - p1, rotation=axis_rotation).bc("t").pickable(False)
    return macts


def RulerAxes(
        inputobj,
        xtitle="",
        ytitle="",
        ztitle="",
        xlabel="",
        ylabel="",
        zlabel="",
        xpadding=0.05,
        ypadding=0.04,
        zpadding=0,
        font="Normografo",
        s=None,
        italic=0,
        units="",
        c=(0.2, 0, 0),
        alpha=1,
        lw=1,
        precision=3,
        label_rotation=0,
        axis_rotation=0,
        xycross=True,
    ):
    """
    A 3D ruler axes to indicate the sizes of the input scene or object.

    Parameters
    ----------
    xtitle : str
        name of the axis or title

    xlabel : str
        alternative fixed label to be shown instead of the distance

    s : float
        size of the label

    font : str
        font face

    italic : float
        italicness of the font in the range [0,1]

    units : str
        string to be appended to the numeric value

    lw : int
        line width in pixel units

    precision : int
        nr of significant digits to be shown

    label_rotation : float
        initial rotation of the label around the z-axis

    axis_rotation : float
        initial rotation of the line around the main axis

    xycross : bool
        show two back crossing lines in the xy plane

    .. hint:: examples/pyplot/goniometer.py
    """
    if utils.is_sequence(inputobj):
        x0, x1, y0, y1, z0, z1 = inputobj
    else:
        x0, x1, y0, y1, z0, z1 = inputobj.GetBounds()
    dx, dy, dz = (y1 - y0) * xpadding, (x1 - x0) * ypadding, (y1 - y0) * zpadding
    d = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2 + (z1 - z0) ** 2)

    if not d:
        return None

    if s is None:
        s = d / 75

    acts, rx, ry = [], None, None
    if xtitle is not None and (x1 - x0) / d > 0.1:
        rx = Ruler(
            [x0, y0 - dx, z0],
            [x1, y0 - dx, z0],
            s=s,
            font=font,
            precision=precision,
            label_rotation=label_rotation,
            axis_rotation=axis_rotation,
            lw=lw,
            italic=italic,
            prefix=xtitle,
            label=xlabel,
            units=units,
        )
        acts.append(rx)
    if ytitle is not None and (y1 - y0) / d > 0.1:
        ry = Ruler(
            [x1 + dy, y0, z0],
            [x1 + dy, y1, z0],
            s=s,
            font=font,
            precision=precision,
            label_rotation=label_rotation,
            axis_rotation=axis_rotation,
            lw=lw,
            italic=italic,
            prefix=ytitle,
            label=ylabel,
            units=units,
        )
        acts.append(ry)
    if ztitle is not None and (z1 - z0) / d > 0.1:
        rz = Ruler(
            [x0 - dy, y0 + dz, z0],
            [x0 - dy, y0 + dz, z1],
            s=s,
            font=font,
            precision=precision,
            label_rotation=label_rotation,
            axis_rotation=axis_rotation + 90,
            lw=lw,
            italic=italic,
            prefix=ztitle,
            label=zlabel,
            units=units,
        )
        acts.append(rz)

    if xycross and rx and ry:
        lx = shapes.Line([x0, y0, z0], [x0, y1 + dx, z0])
        ly = shapes.Line([x0 - dy, y1, z0], [x1, y1, z0])
        d = min((x1 - x0), (y1 - y0)) / 200
        cxy = shapes.Circle([x0, y1, z0], r=d, res=15)
        acts.extend([lx, ly, cxy])

    macts = merge(acts)
    if not macts:
        return None
    macts.c(c).alpha(alpha).bc("t")
    macts.UseBoundsOff()
    macts.PickableOff()
    return macts


#####################################################################
def add_scale_indicator(pos=(0.7, 0.05), s=0.02, length=2, lw=4, c="k", units=""):
    """
    Add a Scale Indicator.

    Parameters
    ----------
    pos : list
        fractional (x,y) position on the screen.

    s : float
        size of the text.

    length : float
        length of the line.

    units : str
        units. The default is ''.
    """
    ppoints = vtk.vtkPoints()  # Generate the polyline
    psqr = [[0.0, 0.05], [length / 10, 0.05]]
    dd = psqr[1][0] - psqr[0][0]
    for i, pt in enumerate(psqr):
        ppoints.InsertPoint(i, pt[0], pt[1], 0)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(psqr))
    for i in range(len(psqr)):
        lines.InsertCellPoint(i)
    pd = vtk.vtkPolyData()
    pd.SetPoints(ppoints)
    pd.SetLines(lines)

    plt = vedo.plotter_instance
    wsx, wsy = plt.window.GetSize()
    if not plt.renderer.GetActiveCamera().GetParallelProjection():
        vedo.logger.warning(
            "add_scale_indicator() is called with use_parallel_projection OFF."
        )

    rlabel = vtk.vtkVectorText()
    rlabel.SetText("")
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputConnection(rlabel.GetOutputPort())
    t = vtk.vtkTransform()
    t.Scale(s, s * wsx / wsy, 1)
    tf.SetTransform(t)

    app = vtk.vtkAppendPolyData()
    app.AddInputConnection(tf.GetOutputPort())
    app.AddInputData(pd)

    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputConnection(app.GetOutputPort())
    cs = vtk.vtkCoordinate()
    cs.SetCoordinateSystem(1)
    mapper.SetTransformCoordinate(cs)

    fractor = vtk.vtkActor2D()
    csys = fractor.GetPositionCoordinate()
    csys.SetCoordinateSystem(3)
    fractor.SetPosition(pos)
    fractor.SetMapper(mapper)
    fractor.GetProperty().SetColor(get_color(c))
    fractor.GetProperty().SetOpacity(1)
    fractor.GetProperty().SetLineWidth(lw)
    fractor.GetProperty().SetDisplayLocationToForeground()

    def sifunc(iren, ev):
        wsx, wsy = plt.window.GetSize()
        ps = plt.camera.GetParallelScale()
        newtxt = utils.precision(ps / wsy * wsx * length * dd, 3)
        if units:
            newtxt += " " + units
        rlabel.SetText(newtxt)

    plt.renderer.AddActor(fractor)
    plt.interactor.AddObserver("MouseWheelBackwardEvent", sifunc)
    plt.interactor.AddObserver("MouseWheelForwardEvent", sifunc)
    plt.interactor.AddObserver("InteractionEvent", sifunc)
    sifunc(0, 0)

    return fractor


#####################################################################
def Axes(
        obj=None,
        xtitle='x', ytitle='y', ztitle='z',
        xrange=None, yrange=None, zrange=None,
        c=None,
        number_of_divisions=None,
        digits=None,
        limit_ratio=0.04,
        title_depth=0,
        title_font="", # grab settings.default_font
        text_scale=1.0,
        x_values_and_labels=None, y_values_and_labels=None, z_values_and_labels=None,
        htitle="",
        htitle_size=0.03,
        htitle_font=None,
        htitle_italic=False,
        htitle_color=None,
        htitle_justify='bottom-center',
        htitle_rotation=0,
        htitle_offset=(0, 0.01, 0),
        xtitle_position=0.95, ytitle_position=0.95, ztitle_position=0.95,
        xtitle_offset=0.025,  ytitle_offset=0.0275, ztitle_offset=0.02, # can be a list (dx,dy,dz)
        xtitle_justify=None,  ytitle_justify=None,  ztitle_justify=None,
        xtitle_rotation=0, ytitle_rotation=0, ztitle_rotation=0,         # can be a list (rx,ry,rz)
        xtitle_box=False,  ytitle_box=False,
        xtitle_size=0.025, ytitle_size=0.025, ztitle_size=0.025,
        xtitle_color=None, ytitle_color=None, ztitle_color=None,
        xtitle_backface_color=None, ytitle_backface_color=None, ztitle_backface_color=None,
        xtitle_italic=0, ytitle_italic=0, ztitle_italic=0,
        grid_linewidth=1,
        xygrid=True,   yzgrid=False,  zxgrid=False,
        xygrid2=False, yzgrid2=False, zxgrid2=False,
        xygrid_transparent=False,  yzgrid_transparent=False,  zxgrid_transparent=False,
        xygrid2_transparent=False, yzgrid2_transparent=False, zxgrid2_transparent=False,
        xyplane_color=None, yzplane_color=None, zxplane_color=None,
        xygrid_color=None, yzgrid_color=None, zxgrid_color=None,
        xyalpha=0.075, yzalpha=0.075, zxalpha=0.075,
        xyframe_line=None, yzframe_line=None, zxframe_line=None,
        xyframe_color=None, yzframe_color=None, zxframe_color=None,
        axes_linewidth=1,
        xline_color=None, yline_color=None, zline_color=None,
        xhighlight_zero=False, yhighlight_zero=False, zhighlight_zero=False,
        xhighlight_zero_color='red4', yhighlight_zero_color='green4', zhighlight_zero_color='blue4',
        show_ticks=True,
        xtick_length=0.015, ytick_length=0.015, ztick_length=0.015,
        xtick_thickness=0.0025, ytick_thickness=0.0025, ztick_thickness=0.0025,
        xminor_ticks=1, yminor_ticks=1, zminor_ticks=1,
        tip_size=None,
        label_font="", # grab settings.default_font
        xlabel_color=None, ylabel_color=None, zlabel_color=None,
        xlabel_size=0.016, ylabel_size=0.016, zlabel_size=0.016,
        xlabel_offset=0.8, ylabel_offset=0.8, zlabel_offset=0.8, # each can be a list (dx,dy,dz)
        xlabel_justify=None, ylabel_justify=None, zlabel_justify=None,
        xlabel_rotation=0, ylabel_rotation=0, zlabel_rotation=0, # each can be a list (rx,ry,rz)
        xaxis_rotation=0, yaxis_rotation=0, zaxis_rotation=0,    # rotate all elements around axis
        xyshift=0, yzshift=0, zxshift=0,
        xshift_along_y=0, xshift_along_z=0,
        yshift_along_x=0, yshift_along_z=0,
        zshift_along_x=0, zshift_along_y=0,
        x_use_bounds=True, y_use_bounds=True, z_use_bounds=False,
        x_inverted=False, y_inverted=False, z_inverted=False,
        use_global=False,
        tol=0.001,
        **options,
    ):
    """
    Draw axes for the input object.

    Returns an `Assembly` object.

    Parameters
    ----------

    - `xtitle`,                  ['x'], x-axis title text
    - `xrange`,                 [None], x-axis range in format (xmin, ymin), default is automatic.
    - `number_of_divisions`,    [None], approximate number of divisions on the longest axis
    - `axes_linewidth`,           [1], width of the axes lines
    - `grid_linewidth`,           [1], width of the grid lines
    - `title_depth`,              [0], extrusion fractional depth of title text
    - `x_values_and_labels`        [], assign custom tick positions and labels [(pos1, label1), ...]
    - `xygrid`,                [True], show a gridded wall on plane xy
    - `yzgrid`,                [True], show a gridded wall on plane yz
    - `zxgrid`,                [True], show a gridded wall on plane zx
    - `zxgrid2`,              [False], show zx plane on opposite side of the bounding box
    - `xygrid_transparent`    [False], make grid plane completely transparent
    - `xygrid2_transparent`   [False], make grid plane completely transparent on opposite side box
    - `xyplane_color`,       ['None'], color of the plane
    - `xygrid_color`,        ['None'], grid line color
    - `xyalpha`,               [0.15], grid plane opacity
    - `xyframe_line`,             [0], add a frame for the plane, use value as the thickness
    - `xyframe_color`,         [None], color for the frame of the plane
    - `show_ticks`,            [True], show major ticks
    - `digits`,                [None], use this number of significant digits in scientific notation
    - `title_font`,              [''], font for axes titles
    - `label_font`,              [''], font for numeric labels
    - `text_scale`,             [1.0], global scaling factor for all text elements (titles, labels)
    - `htitle`,                  [''], header title
    - `htitle_size`,           [0.03], header title size
    - `htitle_font`,           [None], header font (defaults to `title_font`)
    - `htitle_italic`,         [True], header font is italic
    - `htitle_color`,          [None], header title color (defaults to `xtitle_color`)
    - `htitle_justify`, ['bottom-center'], origin of the title justification
    - `htitle_offset`,   [(0,0.01,0)], control offsets of header title in x, y and z
    - `xtitle_position`,       [0.32], title fractional positions along axis
    - `xtitle_offset`,         [0.05], title fractional offset distance from axis line, can be a list
    - `xtitle_justify`,        [None], choose the origin of the bounding box of title
    - `xtitle_rotation`,          [0], add a rotation of the axis title, can be a list (rx,ry,rz)
    - `xtitle_box`,           [False], add a box around title text
    - `xline_color`,      [automatic], color of the x-axis
    - `xtitle_color`,     [automatic], color of the axis title
    - `xtitle_backface_color`, [None], color of axis title on its backface
    - `xtitle_size`,          [0.025], size of the axis title
    - `xtitle_italic`,            [0], a bool or float to make the font italic
    - `xhighlight_zero`,       [True], draw a line highlighting zero position if in range
    - `xhighlight_zero_color`, [autom], color of the line highlighting the zero position
    - `xtick_length`,         [0.005], radius of the major ticks
    - `xtick_thickness`,     [0.0025], thickness of the major ticks along their axis
    - `xminor_ticks`,             [1], number of minor ticks between two major ticks
    - `xlabel_color`,     [automatic], color of numeric labels and ticks
    - `xLabelPrecision`,          [2], nr. of significative digits to be shown
    - `xlabel_size`,          [0.015], size of the numeric labels along axis
    - `xlabel_rotation`,          [0], numeric labels rotation (can be a list of 3 rotations)
    - `xlabel_offset`,          [0.8], offset of the numeric labels (can be a list of 3 offsets)
    - `xlabel_justify`,        [None], choose the origin of the bounding box of labels
    - `xaxis_rotation`,           [0], rotate the X axis elements (ticks and labels) around this same axis
    - `xyshift`                 [0.0], slide the xy-plane along z (the range is [0,1])
    - `xshift_along_y`           [0.0], slide x-axis along the y-axis (the range is [0,1])
    - `tip_size`,               [0.01], size of the arrow tip
    - `limit_ratio`,            [0.04], below this ratio don't plot smaller axis
    - `x_use_bounds`,           [True], keep into account space occupied by labels when setting camera
    - `x_inverted`,            [False], invert labels order and direction (only visually!)
    - `use_global`,             [False], try to compute the global bounding box of visible actors

    Example:
        .. code-block:: python

            from vedo import Axes, Box, show
            box = Box(pos=(1,2,3), length=8, width=9, height=7).alpha(0.1)
            axs = Axes(box, c='k')  # returns an Assembly object
            for a in axs.unpack():
                print(a.name)
            show(box, axs).close()

    .. hint::
        examples/pyplot/customAxes1.py, customAxes2.py customAxes3.py, customIndividualAxes.py

        .. image:: https://vedo.embl.es/images/pyplot/customAxes3.png
    """
    # make sure options are correct and user passed snake_case instead of camelCase
    if len(options):
        for k in options:
            if k.lower() == k:
                vedo.logger.warning(f"Unrecognised keyword '{k}' is ignored")
            else:
                vedo.logger.warning(f"Unrecognised keyword '{k}'. Please use snake_case notation")

    if not title_font:
        title_font = settings.default_font
    if not label_font:
        label_font = settings.default_font

    if c is None:  # automatic black or white
        c = (0.1, 0.1, 0.1)
        plt = vedo.plotter_instance
        if plt and plt.renderer:
            bgcol = plt.renderer.GetBackground()
        else:
            bgcol = (1, 1, 1)
        if np.sum(bgcol) < 1.5:
            c = (0.9, 0.9, 0.9)
    else:
        c = get_color(c)

    if use_global:
        vbb, drange, min_bns, max_bns = compute_visible_bounds()
    else:
        if obj is not None:
            vbb, drange, min_bns, max_bns = compute_visible_bounds(obj)
        else:
            vbb = np.zeros(6)
            drange = np.zeros(3)
            if zrange is None:
                zrange = (0, 0)
            if xrange is None or yrange is None:
                vedo.logger.error("in Axes() must specify axes ranges!")
                raise RuntimeError()

    if xrange is not None:
        if xrange[1] < xrange[0]:
            x_inverted = True
            xrange = [xrange[1], xrange[0]]
        vbb[0], vbb[1] = xrange
        drange[0] = vbb[1] - vbb[0]
        min_bns = vbb
        max_bns = vbb
    if yrange is not None:
        if yrange[1] < yrange[0]:
            y_inverted = True
            yrange = [yrange[1], yrange[0]]
        vbb[2], vbb[3] = yrange
        drange[1] = vbb[3] - vbb[2]
        min_bns = vbb
        max_bns = vbb
    if zrange is not None:
        if zrange[1] < zrange[0]:
            z_inverted = True
            zrange = [zrange[1], zrange[0]]
        vbb[4], vbb[5] = zrange
        drange[2] = vbb[5] - vbb[4]
        min_bns = vbb
        max_bns = vbb

    drangemax = max(drange)
    if not drangemax:
        return None

    if drange[0] / drangemax < limit_ratio:
        drange[0] = 0
        xtitle = ""
    if drange[1] / drangemax < limit_ratio:
        drange[1] = 0
        ytitle = ""
    if drange[2] / drangemax < limit_ratio:
        drange[2] = 0
        ztitle = ""

    x0, x1, y0, y1, z0, z1 = vbb
    dx, dy, dz = drange

    gscale = np.sqrt(dx * dx + dy * dy + dz * dz) * 0.75

    if not xyplane_color: xyplane_color = c
    if not yzplane_color: yzplane_color = c
    if not zxplane_color: zxplane_color = c
    if not xygrid_color:  xygrid_color = c
    if not yzgrid_color:  yzgrid_color = c
    if not zxgrid_color:  zxgrid_color = c
    if not xtitle_color:  xtitle_color = c
    if not ytitle_color:  ytitle_color = c
    if not ztitle_color:  ztitle_color = c
    if not xline_color:   xline_color = c
    if not yline_color:   yline_color = c
    if not zline_color:   zline_color = c
    if not xlabel_color:  xlabel_color = xline_color
    if not ylabel_color:  ylabel_color = yline_color
    if not zlabel_color:  zlabel_color = zline_color

    # vtk version<9 dont like depthpeeling: force switching off grids
    if settings.use_depth_peeling and not utils.vtk_version_at_least(9):
        xygrid = False
        yzgrid = False
        zxgrid = False
        xygrid2 = False
        yzgrid2 = False
        zxgrid2 = False

    if tip_size is None:
        tip_size = 0.005 * gscale
        if not ztitle:
            tip_size = 0  # switch off in xy 2d

    ndiv = 4
    if not ztitle or not ytitle or not xtitle:  # make more default ticks if 2D
        ndiv = 6
        if not ztitle:
            if xyframe_line is None:
                xyframe_line = True
            if tip_size is None:
                tip_size = False

    if utils.is_sequence(number_of_divisions):
        rx, ry, rz = number_of_divisions
    else:
        if not number_of_divisions:
            number_of_divisions = ndiv

    rx, ry, rz = np.ceil(drange / drangemax * number_of_divisions).astype(int)

    if xtitle:
        xticks_float, xticks_str = utils.make_ticks(x0, x1, rx, x_values_and_labels, digits)
        xticks_float = xticks_float * dx
        if x_inverted:
            xticks_float = np.flip(-(xticks_float - xticks_float[-1]))
            xticks_str = list(reversed(xticks_str))
            xticks_str[-1] = ""
            xhighlight_zero = False
    if ytitle:
        yticks_float, yticks_str = utils.make_ticks(y0, y1, ry, y_values_and_labels, digits)
        yticks_float = yticks_float * dy
        if y_inverted:
            yticks_float = np.flip(-(yticks_float - yticks_float[-1]))
            yticks_str = list(reversed(yticks_str))
            yticks_str[-1] = ""
            yhighlight_zero = False
    if ztitle:
        zticks_float, zticks_str = utils.make_ticks(z0, z1, rz, z_values_and_labels, digits)
        zticks_float = zticks_float * dz
        if z_inverted:
            zticks_float = np.flip(-(zticks_float - zticks_float[-1]))
            zticks_str = list(reversed(zticks_str))
            zticks_str[-1] = ""
            zhighlight_zero = False

    ################################################ axes lines
    lines = []
    if xtitle:
        axlinex = shapes.Line([0,0,0], [dx,0,0], c=xline_color, lw=axes_linewidth)
        if xyshift: axlinex.shift(0,0,xyshift*dz)
        if zxshift: axlinex.shift(0,zxshift*dy,0)
        if xshift_along_y: axlinex.shift(0,xshift_along_y*dy,0)
        if xshift_along_z: axlinex.shift(0,0,xshift_along_z*dz)
        axlinex.name = 'xAxis'
        lines.append(axlinex)
    if ytitle:
        axliney = shapes.Line([0,0,0], [0,dy,0], c=yline_color, lw=axes_linewidth)
        if xyshift: axliney.shift(0,0,xyshift*dz)
        if yzshift: axliney.shift(yzshift*dx,0,0)
        if yshift_along_x: axliney.shift(yshift_along_x*dx,0,0)
        if yshift_along_z: axliney.shift(0,0,yshift_along_z*dz)
        axliney.name = 'yAxis'
        lines.append(axliney)
    if ztitle:
        axlinez = shapes.Line([0,0,0], [0,0,dz], c=zline_color, lw=axes_linewidth)
        if yzshift: axlinez.shift(yzshift*dx,0,0)
        if zxshift: axlinez.shift(0,zxshift*dy,0)
        if zshift_along_x: axlinez.shift(zshift_along_x*dx,0,0)
        if zshift_along_y: axlinez.shift(0,zshift_along_y*dy,0)
        axlinez.name = 'zAxis'
        lines.append(axlinez)

    ################################################ grid planes
    # all shapes have a name to keep track of them in the Assembly
    # if user wants to unpack it
    grids = []
    if xygrid and xtitle and ytitle:
        if not xygrid_transparent:
            gxy = shapes.Grid(s=(xticks_float, yticks_float))
            gxy.alpha(xyalpha).c(xyplane_color).lw(0)
            if xyshift: gxy.shift(0,0,xyshift*dz)
            elif tol:   gxy.shift(0,0,-tol*gscale)
            gxy.name = "xyGrid"
            grids.append(gxy)
        if grid_linewidth:
            gxy_lines = shapes.Grid(s=(xticks_float, yticks_float))
            gxy_lines.c(xyplane_color).lw(grid_linewidth).alpha(xyalpha)
            if xyshift: gxy_lines.shift(0,0,xyshift*dz)
            elif tol:   gxy_lines.shift(0,0,-tol*gscale)
            gxy_lines.name = "xyGridLines"
            grids.append(gxy_lines)

    if yzgrid and ytitle and ztitle:
        if not yzgrid_transparent:
            gyz = shapes.Grid(s=(zticks_float, yticks_float))
            gyz.alpha(yzalpha).c(yzplane_color).lw(0).RotateY(-90)
            if yzshift: gyz.shift(yzshift*dx,0,0)
            elif tol:   gyz.shift(-tol*gscale,0,0)
            gyz.name = "yzGrid"
            grids.append(gyz)
        if grid_linewidth:
            gyz_lines = shapes.Grid(s=(zticks_float, yticks_float))
            gyz_lines.c(yzplane_color).lw(grid_linewidth).alpha(yzalpha).RotateY(-90)
            if yzshift: gyz_lines.shift(yzshift*dx,0,0)
            elif tol:   gyz_lines.shift(-tol*gscale,0,0)
            gyz_lines.name = "yzGridLines"
            grids.append(gyz_lines)

    if zxgrid and ztitle and xtitle:
        if not zxgrid_transparent:
            gzx = shapes.Grid(s=(xticks_float, zticks_float))
            gzx.alpha(zxalpha).c(zxplane_color).lw(0).RotateX(90)
            if zxshift: gzx.shift(0,zxshift*dy,0)
            elif tol:   gzx.shift(0,-tol*gscale,0)
            gzx.name = "zxGrid"
            grids.append(gzx)
        if grid_linewidth:
            gzx_lines = shapes.Grid(s=(xticks_float, zticks_float))
            gzx_lines.c(zxplane_color).lw(grid_linewidth).alpha(zxalpha).RotateX(90)
            if zxshift: gzx_lines.shift(0,zxshift*dy,0)
            elif tol:   gzx_lines.shift(0,-tol*gscale,0)
            gzx_lines.name = "zxGridLines"
            grids.append(gzx_lines)

    # Grid2
    if xygrid2 and xtitle and ytitle:
        if not xygrid2_transparent:
            gxy2 = shapes.Grid(s=(xticks_float, yticks_float)).z(dz)
            gxy2.alpha(xyalpha).c(xyplane_color).lw(0)
            if tol: gxy2.shift(0,tol*gscale,0)
            gxy2.name = "xyGrid2"
            grids.append(gxy2)
        if grid_linewidth:
            gxy2_lines = shapes.Grid(s=(xticks_float, yticks_float)).z(dz)
            gxy2_lines.c(xyplane_color).lw(grid_linewidth).alpha(xyalpha)
            if tol: gxy2_lines.shift(0,tol*gscale,0)
            gxy2_lines.name = "xygrid2Lines"
            grids.append(gxy2_lines)

    if yzgrid2 and ytitle and ztitle:
        if not yzgrid2_transparent:
            gyz2 = shapes.Grid(s=(zticks_float, yticks_float)).x(dx)
            gyz2.alpha(yzalpha).c(yzplane_color).lw(0).RotateY(-90)
            if tol: gyz2.shift(tol*gscale,0,0)
            gyz2.name = "yzGrid2"
            grids.append(gyz2)
        if grid_linewidth:
            gyz2_lines = shapes.Grid(s=(zticks_float, yticks_float)).x(dx)
            gyz2_lines.c(yzplane_color).lw(grid_linewidth).alpha(yzalpha).RotateY(-90)
            if tol: gyz2_lines.shift(tol*gscale,0,0)
            gyz2_lines.name = "yzGrid2Lines"
            grids.append(gyz2_lines)

    if zxgrid2 and ztitle and xtitle:
        if not zxgrid2_transparent:
            gzx2 = shapes.Grid(s=(xticks_float, zticks_float)).y(dy)
            gzx2.alpha(zxalpha).c(zxplane_color).lw(0).RotateX(90)
            if tol: gzx2.shift(0,tol*gscale,0)
            gzx2.name = "zxGrid2"
            grids.append(gzx2)
        if grid_linewidth:
            gzx2_lines = shapes.Grid(s=(xticks_float, zticks_float)).y(dy)
            gzx2_lines.c(zxplane_color).lw(grid_linewidth).alpha(zxalpha).RotateX(90)
            if tol: gzx2_lines.shift(0,tol*gscale,0)
            gzx2_lines.name = "zxGrid2Lines"
            grids.append(gzx2_lines)

    ################################################ frame lines
    framelines = []
    if xyframe_line and xtitle and ytitle:
        if not xyframe_color:
            xyframe_color = xygrid_color
        frxy = shapes.Line([[0,dy,0],[dx,dy,0],[dx,0,0],[0,0,0],[0,dy,0]],
                           c=xyframe_color, lw=xyframe_line)
        if xyshift: frxy.shift(0,0,xyshift*dz)
        frxy.name = 'xyFrameLine'
        framelines.append(frxy)
    if yzframe_line and ytitle and ztitle:
        if not yzframe_color:
            yzframe_color = yzgrid_color
        fryz = shapes.Line([[0,0,dz],[0,dy,dz],[0,dy,0],[0,0,0],[0,0,dz]],
                           c=yzframe_color, lw=yzframe_line)
        if yzshift: fryz.shift(yzshift*dx,0,0)
        fryz.name = 'yzFrameLine'
        framelines.append(fryz)
    if zxframe_line and ztitle and xtitle:
        if not zxframe_color:
            zxframe_color = zxgrid_color
        frzx = shapes.Line([[0,0,dz],[dx,0,dz],[dx,0,0],[0,0,0],[0,0,dz]],
                           c=zxframe_color, lw=zxframe_line)
        if zxshift: frzx.shift(0,zxshift*dy,0)
        frzx.name = 'zxFrameLine'
        framelines.append(frzx)

    ################################################ zero lines highlights
    highlights = []
    if xygrid and xtitle and ytitle:
        if xhighlight_zero and min_bns[0] <= 0 and max_bns[1] > 0:
            xhl = -min_bns[0]
            hxy = shapes.Line([xhl,0,0], [xhl,dy,0], c=xhighlight_zero_color)
            hxy.alpha(np.sqrt(xyalpha)).lw(grid_linewidth*2)
            if xyshift: hxy.shift(0,0,xyshift*dz)
            hxy.name = "xyHighlightZero"
            highlights.append(hxy)
        if yhighlight_zero and min_bns[2] <= 0 and max_bns[3] > 0:
            yhl = -min_bns[2]
            hyx = shapes.Line([0,yhl,0], [dx,yhl,0], c=yhighlight_zero_color)
            hyx.alpha(np.sqrt(yzalpha)).lw(grid_linewidth*2)
            if xyshift: hyx.shift(0,0,xyshift*dz)
            hyx.name = "yxHighlightZero"
            highlights.append(hyx)

    if yzgrid and ytitle and ztitle:
        if yhighlight_zero and min_bns[2] <= 0 and max_bns[3] > 0:
            yhl = -min_bns[2]
            hyz = shapes.Line([0,yhl,0], [0,yhl,dz], c=yhighlight_zero_color)
            hyz.alpha(np.sqrt(yzalpha)).lw(grid_linewidth*2)
            if yzshift: hyz.shift(yzshift*dx,0,0)
            hyz.name = "yzHighlightZero"
            highlights.append(hyz)
        if zhighlight_zero and min_bns[4] <= 0 and max_bns[5] > 0:
            zhl = -min_bns[4]
            hzy = shapes.Line([0,0,zhl], [0,dy,zhl], c=zhighlight_zero_color)
            hzy.alpha(np.sqrt(yzalpha)).lw(grid_linewidth*2)
            if yzshift: hzy.shift(yzshift*dx,0,0)
            hzy.name = "zyHighlightZero"
            highlights.append(hzy)

    if zxgrid and ztitle and xtitle:
        if zhighlight_zero and min_bns[4] <= 0 and max_bns[5] > 0:
            zhl = -min_bns[4]
            hzx = shapes.Line([0,0,zhl], [dx,0,zhl], c=zhighlight_zero_color)
            hzx.alpha(np.sqrt(zxalpha)).lw(grid_linewidth*2)
            if zxshift: hzx.shift(0,zxshift*dy,0)
            hzx.name = "zxHighlightZero"
            highlights.append(hzx)
        if xhighlight_zero and min_bns[0] <= 0 and max_bns[1] > 0:
            xhl = -min_bns[0]
            hxz = shapes.Line([xhl,0,0], [xhl,0,dz], c=xhighlight_zero_color)
            hxz.alpha(np.sqrt(zxalpha)).lw(grid_linewidth*2)
            if zxshift: hxz.shift(0,zxshift*dy,0)
            hxz.name = "xzHighlightZero"
            highlights.append(hxz)

    ################################################ arrow cone
    cones = []

    if tip_size:

        if xtitle:
            if x_inverted:
                cx = shapes.Cone(r=tip_size, height=tip_size*2,
                                 axis=(-1,0,0), c=xline_color, res=12)
            else:
                cx = shapes.Cone((dx,0,0), r=tip_size, height=tip_size*2,
                                 axis=(1,0,0), c=xline_color, res=12)
            if xyshift: cx.shift(0,0,xyshift*dz)
            if zxshift: cx.shift(0,zxshift*dy,0)
            if xshift_along_y: cx.shift(0,xshift_along_y*dy,0)
            if xshift_along_z: cx.shift(0,0,xshift_along_z*dz)
            cx.name = "xTipCone"
            cones.append(cx)

        if ytitle:
            if y_inverted:
                cy = shapes.Cone(r=tip_size, height=tip_size*2,
                                 axis=(0,-1,0), c=yline_color, res=12)
            else:
                cy = shapes.Cone((0,dy,0), r=tip_size, height=tip_size*2,
                                 axis=(0,1,0), c=yline_color, res=12)
            if xyshift: cy.shift(0,0,xyshift*dz)
            if yzshift: cy.shift(yzshift*dx,0,0)
            if yshift_along_x: cy.shift(yshift_along_x*dx,0,0)
            if yshift_along_z: cy.shift(0,0,yshift_along_z*dz)
            cy.name = "yTipCone"
            cones.append(cy)

        if ztitle:
            if z_inverted:
                cz = shapes.Cone(r=tip_size, height=tip_size*2,
                                 axis=(0,0,-1), c=zline_color, res=12)
            else:
                cz = shapes.Cone((0,0,dz), r=tip_size, height=tip_size*2,
                                 axis=(0,0,1), c=zline_color, res=12)
            if yzshift: cz.shift(yzshift*dx,0,0)
            if zxshift: cz.shift(0,zxshift*dy,0)
            if zshift_along_x: cz.shift(zshift_along_x*dx,0,0)
            if zshift_along_y: cz.shift(0,zshift_along_y*dy,0)
            cz.name = "zTipCone"
            cones.append(cz)

    ################################################################# MAJOR ticks
    majorticks, minorticks = [], []
    xticks, yticks, zticks = [], [], []
    if show_ticks:
        if xtitle:
            tickThickness = xtick_thickness * gscale / 2
            tickLength = xtick_length * gscale / 2
            for i in range(1, len(xticks_float) - 1):
                v1 = (xticks_float[i] - tickThickness, -tickLength, 0)
                v2 = (xticks_float[i] + tickThickness, tickLength, 0)
                xticks.append(shapes.Rectangle(v1, v2))
            if len(xticks) > 1:
                xmajticks = merge(xticks).c(xlabel_color)
                if xaxis_rotation:
                    xmajticks.RotateX(xaxis_rotation)
                if xyshift: xmajticks.shift(0,0,xyshift*dz)
                if zxshift: xmajticks.shift(0,zxshift*dy,0)
                if xshift_along_y: xmajticks.shift(0,xshift_along_y*dy,0)
                if xshift_along_z: xmajticks.shift(0,0,xshift_along_z*dz)
                xmajticks.name = "xMajorTicks"
                majorticks.append(xmajticks)
        if ytitle:
            tickThickness = ytick_thickness * gscale / 2
            tickLength = ytick_length * gscale / 2
            for i in range(1, len(yticks_float) - 1):
                v1 = (-tickLength, yticks_float[i] - tickThickness, 0)
                v2 = (tickLength, yticks_float[i] + tickThickness, 0)
                yticks.append(shapes.Rectangle(v1, v2))
            if len(yticks) > 1:
                ymajticks = merge(yticks).c(ylabel_color)
                if yaxis_rotation:
                    ymajticks.RotateY(yaxis_rotation)
                if xyshift: ymajticks.shift(0,0,xyshift*dz)
                if yzshift: ymajticks.shift(yzshift*dx,0,0)
                if yshift_along_x: ymajticks.shift(yshift_along_x*dx,0,0)
                if yshift_along_z: ymajticks.shift(0,0,yshift_along_z*dz)
                ymajticks.name = "yMajorTicks"
                majorticks.append(ymajticks)
        if ztitle:
            tickThickness = ztick_thickness * gscale / 2
            tickLength = ztick_length * gscale / 2.85
            for i in range(1, len(zticks_float) - 1):
                v1 = (zticks_float[i] - tickThickness, -tickLength, 0)
                v2 = (zticks_float[i] + tickThickness, tickLength, 0)
                zticks.append(shapes.Rectangle(v1, v2))
            if len(zticks) > 1:
                zmajticks = merge(zticks).c(zlabel_color)
                zmajticks.RotateZ(-45 + zaxis_rotation)
                zmajticks.RotateY(-90)
                if yzshift: zmajticks.shift(yzshift*dx,0,0)
                if zxshift: zmajticks.shift(0,zxshift*dy,0)
                if zshift_along_x: zmajticks.shift(zshift_along_x*dx,0,0)
                if zshift_along_y: zmajticks.shift(0,zshift_along_y*dy,0)
                zmajticks.name = "zMajorTicks"
                majorticks.append(zmajticks)

        ############################################################# MINOR ticks
        if xtitle and xminor_ticks and len(xticks) > 1:
            tickThickness = xtick_thickness * gscale / 4
            tickLength = xtick_length * gscale / 4
            xminor_ticks += 1
            ticks = []
            for i in range(1, len(xticks)):
                t0, t1 = xticks[i - 1].pos(), xticks[i].pos()
                dt = t1 - t0
                for j in range(1, xminor_ticks):
                    mt = dt * (j / xminor_ticks) + t0
                    v1 = (mt[0] - tickThickness, -tickLength, 0)
                    v2 = (mt[0] + tickThickness, tickLength, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = xticks[0].pos(), xticks[1].pos()
            dt = t1 - t0
            for j in range(1, xminor_ticks):
                mt = t0 - dt * (j / xminor_ticks)
                if mt[0] < 0:
                    break
                v1 = (mt[0] - tickThickness, -tickLength, 0)
                v2 = (mt[0] + tickThickness, tickLength, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = xticks[-2].pos(), xticks[-1].pos()
            dt = t1 - t0
            for j in range(1, xminor_ticks):
                mt = t1 + dt * (j / xminor_ticks)
                if mt[0] > dx:
                    break
                v1 = (mt[0] - tickThickness, -tickLength, 0)
                v2 = (mt[0] + tickThickness, tickLength, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if ticks:
                xminticks = merge(ticks).c(xlabel_color)
                if xaxis_rotation:
                    xminticks.RotateX(xaxis_rotation)
                if xyshift: xminticks.shift(0,0,xyshift*dz)
                if zxshift: xminticks.shift(0,zxshift*dy,0)
                if xshift_along_y: xminticks.shift(0,xshift_along_y*dy,0)
                if xshift_along_z: xminticks.shift(0,0,xshift_along_z*dz)
                xminticks.name = "xMinorTicks"
                minorticks.append(xminticks)

        if ytitle and yminor_ticks and len(yticks) > 1:  ##### y
            tickThickness = ytick_thickness * gscale / 4
            tickLength = ytick_length * gscale / 4
            yminor_ticks += 1
            ticks = []
            for i in range(1, len(yticks)):
                t0, t1 = yticks[i - 1].pos(), yticks[i].pos()
                dt = t1 - t0
                for j in range(1, yminor_ticks):
                    mt = dt * (j / yminor_ticks) + t0
                    v1 = (-tickLength, mt[1] - tickThickness, 0)
                    v2 = (tickLength, mt[1] + tickThickness, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = yticks[0].pos(), yticks[1].pos()
            dt = t1 - t0
            for j in range(1, yminor_ticks):
                mt = t0 - dt * (j / yminor_ticks)
                if mt[1] < 0:
                    break
                v1 = (-tickLength, mt[1] - tickThickness, 0)
                v2 = (tickLength, mt[1] + tickThickness, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = yticks[-2].pos(), yticks[-1].pos()
            dt = t1 - t0
            for j in range(1, yminor_ticks):
                mt = t1 + dt * (j / yminor_ticks)
                if mt[1] > dy:
                    break
                v1 = (-tickLength, mt[1] - tickThickness, 0)
                v2 = (tickLength, mt[1] + tickThickness, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if ticks:
                yminticks = merge(ticks).c(ylabel_color)
                if yaxis_rotation:
                    yminticks.RotateY(yaxis_rotation)
                if xyshift: yminticks.shift(0,0,xyshift*dz)
                if yzshift: yminticks.shift(yzshift*dx,0,0)
                if yshift_along_x: yminticks.shift(yshift_along_x*dx,0,0)
                if yshift_along_z: yminticks.shift(0,0,yshift_along_z*dz)
                yminticks.name = "yMinorTicks"
                minorticks.append(yminticks)

        if ztitle and zminor_ticks and len(zticks) > 1:  ##### z
            tickThickness = ztick_thickness * gscale / 4
            tickLength = ztick_length * gscale / 5
            zminor_ticks += 1
            ticks = []
            for i in range(1, len(zticks)):
                t0, t1 = zticks[i - 1].pos(), zticks[i].pos()
                dt = t1 - t0
                for j in range(1, zminor_ticks):
                    mt = dt * (j / zminor_ticks) + t0
                    v1 = (mt[0] - tickThickness, -tickLength, 0)
                    v2 = (mt[0] + tickThickness, tickLength, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = zticks[0].pos(), zticks[1].pos()
            dt = t1 - t0
            for j in range(1, zminor_ticks):
                mt = t0 - dt * (j / zminor_ticks)
                if mt[0] < 0:
                    break
                v1 = (mt[0] - tickThickness, -tickLength, 0)
                v2 = (mt[0] + tickThickness, tickLength, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = zticks[-2].pos(), zticks[-1].pos()
            dt = t1 - t0
            for j in range(1, zminor_ticks):
                mt = t1 + dt * (j / zminor_ticks)
                if mt[0] > dz:
                    break
                v1 = (mt[0] - tickThickness, -tickLength, 0)
                v2 = (mt[0] + tickThickness, tickLength, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if ticks:
                zminticks = merge(ticks).c(zlabel_color)
                zminticks.RotateZ(-45 + zaxis_rotation)
                zminticks.RotateY(-90)
                if yzshift: zminticks.shift(yzshift*dx,0,0)
                if zxshift: zminticks.shift(0,zxshift*dy,0)
                if zshift_along_x: zminticks.shift(zshift_along_x*dx,0,0)
                if zshift_along_y: zminticks.shift(0,zshift_along_y*dy,0)
                zminticks.name = "zMinorTicks"
                minorticks.append(zminticks)

    ################################################ axes NUMERIC text labels
    labels = []
    xlab, ylab, zlab = None, None, None

    if xlabel_size and xtitle:

        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(xlabel_rotation):  # unpck 3 rotations
            zRot, xRot, yRot = xlabel_rotation
        else:
            zRot = xlabel_rotation
        if zRot < 0:  # deal with negative angles
            zRot += 360

        jus = "center-top"
        if zRot:
            if zRot >  24: jus = "top-right"
            if zRot >  67: jus = "center-right"
            if zRot > 112: jus = "right-bottom"
            if zRot > 157: jus = "center-bottom"
            if zRot > 202: jus = "bottom-left"
            if zRot > 247: jus = "center-left"
            if zRot > 292: jus = "top-left"
            if zRot > 337: jus = "top-center"
        if xlabel_justify is not None:
            jus = xlabel_justify

        for i in range(1, len(xticks_str)):
            t = xticks_str[i]
            if not t:
                continue
            if utils.is_sequence(xlabel_offset):
                xoffs, yoffs, zoffs = xlabel_offset
            else:
                xoffs, yoffs, zoffs = 0, xlabel_offset, 0
            xlab = shapes.Text3D(
                t, s=xlabel_size * text_scale * gscale, font=label_font, justify=jus
            )
            tb = xlab.ybounds()  # must be ybounds: height of char
            v = (xticks_float[i], 0, 0)
            offs = -np.array([xoffs, yoffs, zoffs]) * (tb[1] - tb[0])
            xlab.pos(v + offs)
            if xaxis_rotation:
                xlab.rotate_x(xaxis_rotation)
            if zRot: xlab.RotateZ(zRot)
            if xRot: xlab.RotateX(xRot)
            if yRot: xlab.RotateY(yRot)
            if xyshift: xlab.shift(0,0,xyshift*dz)
            if zxshift: xlab.shift(0,zxshift*dy,0)
            if xshift_along_y: xlab.shift(0,xshift_along_y*dy,0)
            if xshift_along_z: xlab.shift(0,0,xshift_along_z*dz)
            xlab.name = f"xNumericLabel{i}"
            xlab.SetUseBounds(x_use_bounds)
            labels.append(xlab.c(xlabel_color))

    if ylabel_size and ytitle:

        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(ylabel_rotation):  # unpck 3 rotations
            zRot, yRot, xRot = ylabel_rotation
        else:
            zRot = ylabel_rotation
        if zRot < 0:
            zRot += 360  # deal with negative angles

        jus = "center-right"
        if zRot:
            if zRot >  24: jus = "bottom-right"
            if zRot >  67: jus = "center-bottom"
            if zRot > 112: jus = "left-bottom"
            if zRot > 157: jus = "center-left"
            if zRot > 202: jus = "top-left"
            if zRot > 247: jus = "center-top"
            if zRot > 292: jus = "top-right"
            if zRot > 337: jus = "right-center"
        if ylabel_justify is not None:
            jus = ylabel_justify

        for i in range(1, len(yticks_str)):
            t = yticks_str[i]
            if not t:
                continue
            if utils.is_sequence(ylabel_offset):
                xoffs, yoffs, zoffs = ylabel_offset
            else:
                xoffs, yoffs, zoffs = ylabel_offset, 0, 0
            ylab = shapes.Text3D(
                t, s=ylabel_size * text_scale * gscale, font=label_font, justify=jus
            )
            tb = ylab.ybounds()  # must be ybounds: height of char
            v = (0, yticks_float[i], 0)
            offs = -np.array([xoffs, yoffs, zoffs]) * (tb[1] - tb[0])
            ylab.pos(v + offs)
            if yaxis_rotation:
                ylab.rotate_y(yaxis_rotation)
            if zRot: ylab.RotateZ(zRot)
            if yRot: ylab.RotateY(yRot)
            if xRot: ylab.RotateX(xRot)
            if xyshift: ylab.shift(0,0,xyshift*dz)
            if yzshift: ylab.shift(yzshift*dx,0,0)
            if yshift_along_x: ylab.shift(yshift_along_x*dx,0,0)
            if yshift_along_z: ylab.shift(0,0,yshift_along_z*dz)
            ylab.name = f"yNumericLabel{i}"
            ylab.SetUseBounds(y_use_bounds)
            labels.append(ylab.c(ylabel_color))

    if zlabel_size and ztitle:

        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(zlabel_rotation):  # unpck 3 rotations
            xRot, yRot, zRot = zlabel_rotation
        else:
            xRot = zlabel_rotation
        if xRot < 0: xRot += 360 # deal with negative angles

        jus = "center-right"
        if xRot:
            if xRot >  24: jus = "bottom-right"
            if xRot >  67: jus = "center-bottom"
            if xRot > 112: jus = "left-bottom"
            if xRot > 157: jus = "center-left"
            if xRot > 202: jus = "top-left"
            if xRot > 247: jus = "center-top"
            if xRot > 292: jus = "top-right"
            if xRot > 337: jus = "right-center"
        if zlabel_justify is not None:
            jus = zlabel_justify

        for i in range(1, len(zticks_str)):
            t = zticks_str[i]
            if not t:
                continue
            if utils.is_sequence(zlabel_offset):
                xoffs, yoffs, zoffs = zlabel_offset
            else:
                xoffs, yoffs, zoffs = zlabel_offset, zlabel_offset, 0
            zlab = shapes.Text3D(
                t, s=zlabel_size * text_scale * gscale, font=label_font, justify=jus
            )
            tb = zlab.ybounds()  # must be ybounds: height of char
            v = (0, 0, zticks_float[i])
            offs = -np.array([xoffs, yoffs, zoffs]) * (tb[1] - tb[0]) / 1.5
            angle = 90
            if dx:
                angle = np.arctan2(dy, dx) * 57.3
            zlab.RotateZ(angle + yRot)  # vtk inverts order of rotations
            if xRot:
                zlab.RotateY(-xRot)  # ..second
            zlab.RotateX(90 + zRot)  # ..first
            zlab.pos(v + offs)
            if zaxis_rotation:
                zlab.rotate_z(zaxis_rotation)
            if yzshift: zlab.shift(yzshift*dx,0,0)
            if zxshift: zlab.shift(0,zxshift*dy,0)
            if zshift_along_x: zlab.shift(zshift_along_x*dx,0,0)
            if zshift_along_y: zlab.shift(0,zshift_along_y*dy,0)
            zlab.SetUseBounds(z_use_bounds)
            zlab.name = f"zNumericLabel{i}"
            labels.append(zlab.c(zlabel_color))

    ################################################ axes titles
    titles = []

    if xtitle:

        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(xtitle_rotation):  # unpack 3 rotations
            zRot, xRot, yRot = xtitle_rotation
        else:
            zRot = xtitle_rotation
        if zRot < 0:  # deal with negative angles
            zRot += 360

        if utils.is_sequence(xtitle_offset):
            xoffs, yoffs, zoffs = xtitle_offset
        else:
            xoffs, yoffs, zoffs = 0, xtitle_offset, 0

        if xtitle_justify is not None:
            jus = xtitle_justify
        else:
            # find best justfication for given rotation(s)
            jus = "right-top"
            if zRot:
                if zRot >  24: jus = "center-right"
                if zRot >  67: jus = "right-bottom"
                if zRot > 157: jus = "bottom-left"
                if zRot > 202: jus = "center-left"
                if zRot > 247: jus = "top-left"
                if zRot > 337: jus = "top-right"

        xt = shapes.Text3D(
            xtitle,
            s=xtitle_size * text_scale * gscale,
            font=title_font,
            c=xtitle_color,
            justify=jus,
            depth=title_depth,
            italic=xtitle_italic,
        )
        if xtitle_backface_color:
            xt.backcolor(xtitle_backface_color)
        if zRot:
            xt.RotateZ(zRot)
        if xRot:
            xt.RotateX(xRot)
        if yRot:
            xt.RotateY(yRot)
        shift = 0
        if xlab:  # xlab is the last created numeric text label..
            lt0, lt1 = xlab.GetBounds()[2:4]
            shift = lt1 - lt0
        xt.pos(
            [
                (xoffs + xtitle_position) * dx,
                -(yoffs + xtick_length / 2) * dy - shift,
                zoffs * dz,
            ]
        )
        if xaxis_rotation:
            xt.rotate_x(xaxis_rotation)
        if xyshift:
            xt.shift(0, 0, xyshift * dz)
        if xshift_along_y:
            xt.shift(0, xshift_along_y * dy, 0)
        if xshift_along_z:
            xt.shift(0, 0, xshift_along_z * dz)
        xt.SetUseBounds(x_use_bounds)
        if xtitle == " ":
            xt.SetUseBounds(False)
        xt.name = f"xtitle {xtitle}"
        titles.append(xt)
        if xtitle_box:
            titles.append(xt.box(scale=1.1).use_bounds(x_use_bounds))

    if ytitle:

        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(ytitle_rotation):  # unpck 3 rotations
            zRot, yRot, xRot = ytitle_rotation
        else:
            zRot = ytitle_rotation
            if len(ytitle) > 3:
                zRot += 90
                ytitle_position *= 0.975
        if zRot < 0:
            zRot += 360  # deal with negative angles

        if utils.is_sequence(ytitle_offset):
            xoffs, yoffs, zoffs = ytitle_offset
        else:
            xoffs, yoffs, zoffs = ytitle_offset, 0, 0

        if ytitle_justify is not None:
            jus = ytitle_justify
        else:
            jus = "center-right"
            if zRot:
                if zRot >  24: jus = "bottom-right"
                if zRot > 112: jus = "left-bottom"
                if zRot > 157: jus = "center-left"
                if zRot > 202: jus = "top-left"
                if zRot > 292: jus = "top-right"
                if zRot > 337: jus = "right-center"

        yt = shapes.Text3D(
            ytitle,
            s=ytitle_size * text_scale * gscale,
            font=title_font,
            c=ytitle_color,
            justify=jus,
            depth=title_depth,
            italic=ytitle_italic,
        )
        if ytitle_backface_color:
            yt.backcolor(ytitle_backface_color)

        if zRot: yt.RotateZ(zRot)
        if yRot: yt.RotateY(yRot)
        if xRot: yt.RotateX(xRot)

        shift = 0
        if ylab:  # this is the last created num label..
            lt0, lt1 = ylab.GetBounds()[0:2]
            shift = lt1 - lt0

        yt.pos(
            -(xoffs + ytick_length / 2) * dx - shift,
            (yoffs + ytitle_position) * dy,
            zoffs * dz,
        )
        if yaxis_rotation:
            yt.rotate_y(yaxis_rotation)
        if xyshift:      yt.shift(0, 0, xyshift*dz)
        if yshift_along_x: yt.shift(yshift_along_x*dx, 0, 0)
        if yshift_along_z: yt.shift(0, 0, yshift_along_z*dz)
        yt.SetUseBounds(y_use_bounds)
        if ytitle == " ":
            yt.SetUseBounds(False)
        yt.name = f"ytitle {ytitle}"
        titles.append(yt)
        if ytitle_box:
            titles.append(yt.box(scale=1.1).use_bounds(y_use_bounds))

    if ztitle:

        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(ztitle_rotation):  # unpck 3 rotations
            xRot, yRot, zRot = ztitle_rotation
        else:
            xRot = ztitle_rotation
            if len(ztitle) > 3:
                xRot += 90
                ztitle_position *= 0.975
        if xRot < 0:
            xRot += 360  # deal with negative angles

        if ztitle_justify is not None:
            jus = ztitle_justify
        else:
            jus = "center-right"
            if xRot:
                if xRot >  24: jus = "bottom-right"
                if xRot > 112: jus = "left-bottom"
                if xRot > 157: jus = "center-left"
                if xRot > 202: jus = "top-left"
                if xRot > 292: jus = "top-right"
                if xRot > 337: jus = "right-center"

        zt = shapes.Text3D(
            ztitle,
            s=ztitle_size * text_scale * gscale,
            font=title_font,
            c=ztitle_color,
            justify=jus,
            depth=title_depth,
            italic=ztitle_italic,
        )
        if ztitle_backface_color:
            zt.backcolor(ztitle_backface_color)

        angle = 90
        if dx:
            angle = np.arctan2(dy, dx) * 57.3
        zt.RotateZ(angle + yRot)  # vtk inverts order of rotations
        if xRot:
            zt.RotateY(-xRot)  # ..second
        zt.RotateX(90 + zRot)  # ..first

        shift = 0
        if zlab:  # this is the last created one..
            lt0, lt1 = zlab.GetBounds()[0:2]
            shift = lt1 - lt0
        zt.pos(
            -(ztitle_offset + ztick_length / 5) * dx - shift,
            -(ztitle_offset + ztick_length / 5) * dy - shift,
            ztitle_position * dz,
        )
        if zaxis_rotation:
            zt.rotate_z(zaxis_rotation)
        if zxshift: zt.shift(0,zxshift*dy,0)
        if zshift_along_x: zt.shift(zshift_along_x*dx,0,0)
        if zshift_along_y: zt.shift(0,zshift_along_y*dy,0)
        zt.SetUseBounds(z_use_bounds)
        if ztitle == " ":
            zt.SetUseBounds(False)
        zt.name = f"ztitle {ztitle}"
        titles.append(zt)

    ################################################### header title
    if htitle:
        if htitle_font is None:
            htitle_font = title_font
        if htitle_color is None:
            htitle_color = xtitle_color
        htit = shapes.Text3D(
            htitle,
            s=htitle_size * gscale * text_scale,
            font=htitle_font,
            c=htitle_color,
            justify=htitle_justify,
            depth=title_depth,
            italic=htitle_italic,
        )
        if htitle_rotation:
            htit.RotateX(htitle_rotation)
        wpos = [
            (0.5 + htitle_offset[0]) * dx,
            (1 + htitle_offset[1]) * dy,
            htitle_offset[2] * dz,
        ]
        htit.pos(wpos)
        if xyshift:
            htit.shift(0, 0, xyshift * dz)
        htit.name = f"htitle {htitle}"
        titles.append(htit)

    ######
    acts = titles + lines + labels + grids + framelines
    acts += highlights + majorticks + minorticks + cones
    orig = (min_bns[0], min_bns[2], min_bns[4])
    for a in acts:
        a.PickableOff()
        a.AddPosition(orig)
        a.GetProperty().LightingOff()
    asse = Assembly(acts)
    asse.SetOrigin(orig)
    asse.PickableOff()
    asse.name = "Axes"
    return asse


def add_global_axes(axtype=None, c=None):
    """
    Draw axes on scene. Available axes types are

    Parameters
    ----------
    axtype : int
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
        - 14, show the vtk default vtkCameraOrientationWidget object

    Axis type-1 can be fully customized by passing a dictionary ``axes=dict()``,
    see ``Axes`` for the complete list of options.

    Example
    -------
        .. code-block:: python

            from vedo import Box, show
            b = Box(pos=(0, 0, 0), length=80, width=90, height=70).alpha(0.1)
            show(
                b,
                axes={
                    "xtitle": "Some long variable [a.u.]",
                    "number_of_divisions": 4,
                    # ...
                },
            )
    """
    plt = vedo.plotter_instance
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
        c = get_color(c)  # for speed

    if not plt.renderer:
        return None

    if plt.axes_instances[r]:
        return None

    ############################################################
    # custom grid walls
    if plt.axes == 1 or plt.axes is True or isinstance(plt.axes, dict):

        if isinstance(plt.axes, dict):
            plt.axes.update({"use_global": True})
            # protect from invalid camelCase options from vedo<=2.3
            for k in plt.axes:
                if k.lower() != k:
                    return
            asse = Axes(None, **plt.axes)
        else:
            asse = Axes(None, use_global=True)

        plt.renderer.AddActor(asse)
        plt.axes_instances[r] = asse

    elif plt.axes in (2, 3):
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

        if dx > aves / 100:
            xl = shapes.Cylinder(
                [[x0, 0, 0], [x1, 0, 0]], r=aves / 250 * s, c=xcol, alpha=alpha
            )
            xc = shapes.Cone(
                pos=[x1, 0, 0],
                c=xcol,
                alpha=alpha,
                r=aves / 100 * s,
                height=aves / 25 * s,
                axis=[1, 0, 0],
                res=10,
            )
            wpos = [x1, -aves / 25 * s, 0]  # aligned to arrow tip
            if centered:
                wpos = [(x0 + x1) / 2, -aves / 25 * s, 0]
            xt = shapes.Text3D("x", pos=wpos, s=aves / 40 * s, c=xcol)
            acts += [xl, xc, xt]

        if dy > aves / 100:
            yl = shapes.Cylinder(
                [[0, y0, 0], [0, y1, 0]], r=aves / 250 * s, c=ycol, alpha=alpha
            )
            yc = shapes.Cone(
                pos=[0, y1, 0],
                c=ycol,
                alpha=alpha,
                r=aves / 100 * s,
                height=aves / 25 * s,
                axis=[0, 1, 0],
                res=10,
            )
            wpos = [-aves / 40 * s, y1, 0]
            if centered:
                wpos = [-aves / 40 * s, (y0 + y1) / 2, 0]
            yt = shapes.Text3D("y", pos=(0, 0, 0), s=aves / 40 * s, c=ycol)
            yt.pos(wpos).RotateZ(90)
            acts += [yl, yc, yt]

        if dz > aves / 100:
            zl = shapes.Cylinder(
                [[0, 0, z0], [0, 0, z1]], r=aves / 250 * s, c=zcol, alpha=alpha
            )
            zc = shapes.Cone(
                pos=[0, 0, z1],
                c=zcol,
                alpha=alpha,
                r=aves / 100 * s,
                height=aves / 25 * s,
                axis=[0, 0, 1],
                res=10,
            )
            wpos = [-aves / 50 * s, -aves / 50 * s, z1]
            if centered:
                wpos = [-aves / 50 * s, -aves / 50 * s, (z0 + z1) / 2]
            zt = shapes.Text3D("z", pos=(0, 0, 0), s=aves / 40 * s, c=zcol)
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
        axact.SetXAxisLabelText("x")
        axact.SetYAxisLabelText("y")
        axact.SetZAxisLabelText("z")
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
        icn = add_icon(axact, size=0.1)
        plt.axes_instances[r] = icn

    elif plt.axes == 5:
        axact = vtk.vtkAnnotatedCubeActor()
        axact.GetCubeProperty().SetColor(get_color(settings.annotated_cube_color))
        axact.SetTextEdgesVisibility(0)
        axact.SetFaceTextScale(settings.annotated_cube_text_scale)
        axact.SetXPlusFaceText(settings.annotated_cube_texts[0])  # XPlus
        axact.SetXMinusFaceText(settings.annotated_cube_texts[1])  # XMinus
        axact.SetYPlusFaceText(settings.annotated_cube_texts[2])  # YPlus
        axact.SetYMinusFaceText(settings.annotated_cube_texts[3])  # YMinus
        axact.SetZPlusFaceText(settings.annotated_cube_texts[4])  # ZPlus
        axact.SetZMinusFaceText(settings.annotated_cube_texts[5])  # ZMinus
        axact.SetZFaceTextRotation(90)

        if settings.annotated_cube_text_color is None:  # use default
            axact.GetXPlusFaceProperty().SetColor(get_color("r"))
            axact.GetXMinusFaceProperty().SetColor(get_color("dr"))
            axact.GetYPlusFaceProperty().SetColor(get_color("g"))
            axact.GetYMinusFaceProperty().SetColor(get_color("dg"))
            axact.GetZPlusFaceProperty().SetColor(get_color("b"))
            axact.GetZMinusFaceProperty().SetColor(get_color("db"))
        else:  # use single user color
            ac = get_color(settings.annotated_cube_text_color)
            axact.GetXPlusFaceProperty().SetColor(ac)
            axact.GetXMinusFaceProperty().SetColor(ac)
            axact.GetYPlusFaceProperty().SetColor(ac)
            axact.GetYMinusFaceProperty().SetColor(ac)
            axact.GetZPlusFaceProperty().SetColor(ac)
            axact.GetZMinusFaceProperty().SetColor(ac)

        axact.PickableOff()
        icn = add_icon(axact, size=0.06)
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
                d = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
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
        vbb = compute_visible_bounds()[0]
        rulax = RulerAxes(vbb, c=c, xtitle="x - ", ytitle="y - ", ztitle="z - ")
        plt.axes_instances[r] = rulax
        if not rulax:
            return None
        rulax.UseBoundsOff()
        rulax.PickableOff()
        plt.renderer.AddActor(rulax)

    elif plt.axes == 8:
        vbb = compute_visible_bounds()[0]
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
        ca.SetXTitle("x")
        ca.SetYTitle("y")
        ca.SetZTitle("z")
        #        if plt.xtitle == "":
        #            ca.SetXAxisVisibility(0)
        #            ca.XAxisLabelVisibilityOff()
        #        if plt.ytitle == "":
        #            ca.SetYAxisVisibility(0)
        #            ca.YAxisLabelVisibilityOff()
        #        if plt.ztitle == "":
        #            ca.SetZAxisVisibility(0)
        #            ca.ZAxisLabelVisibilityOff()
        ca.PickableOff()
        ca.UseBoundsOff()
        plt.renderer.AddActor(ca)
        plt.axes_instances[r] = ca

    elif plt.axes == 9:
        vbb = compute_visible_bounds()[0]
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
        vbb = compute_visible_bounds()[0]
        x0 = (vbb[0] + vbb[1]) / 2, (vbb[3] + vbb[2]) / 2, (vbb[5] + vbb[4]) / 2
        rx, ry, rz = (vbb[1] - vbb[0]) / 2, (vbb[3] - vbb[2]) / 2, (vbb[5] - vbb[4]) / 2
        rm = max(rx, ry, rz)
        xc = shapes.Disc(x0, r1=rm, r2=rm, c="lr", res=(1, 72))
        yc = shapes.Disc(x0, r1=rm, r2=rm, c="lg", res=(1, 72))
        yc.RotateX(90)
        zc = shapes.Disc(x0, r1=rm, r2=rm, c="lb", res=(1, 72))
        yc.RotateY(90)
        xc.clean().alpha(0.5).wireframe().linewidth(2).PickableOff()
        yc.clean().alpha(0.5).wireframe().linewidth(2).PickableOff()
        zc.clean().alpha(0.5).wireframe().linewidth(2).PickableOff()
        ca = xc + yc + zc
        ca.PickableOff()
        ca.UseBoundsOff()
        plt.renderer.AddActor(ca)
        plt.axes_instances[r] = ca

    elif plt.axes == 11:
        vbb, ss = compute_visible_bounds()[0:2]
        xpos, ypos = (vbb[1] + vbb[0]) / 2, (vbb[3] + vbb[2]) / 2
        gs = sum(ss) * 3
        gr = shapes.Grid((xpos, ypos, vbb[4]), s=(gs, gs), res=(11, 11), c=c, alpha=0.1)
        gr.lighting("off").PickableOff()
        gr.UseBoundsOff()
        plt.renderer.AddActor(gr)
        plt.axes_instances[r] = gr

    elif plt.axes == 12:
        polaxes = vtk.vtkPolarAxesActor()
        vbb = compute_visible_bounds()[0]

        polaxes.SetPolarAxisTitle("radial distance")
        polaxes.SetPole(0, 0, vbb[4])
        rd = max(abs(vbb[0]), abs(vbb[2]), abs(vbb[1]), abs(vbb[3]))
        polaxes.SetMaximumRadius(rd)
        polaxes.AutoSubdividePolarAxisOff()
        polaxes.SetNumberOfPolarAxisTicks(10)
        polaxes.SetCamera(plt.renderer.GetActiveCamera())
        polaxes.SetPolarLabelFormat("%6.1f")
        polaxes.PolarLabelVisibilityOff()  # due to bad overlap of labels

        polaxes.GetPolarArcsProperty().SetColor(c)
        polaxes.GetPolarAxisProperty().SetColor(c)
        polaxes.GetPolarAxisTitleTextProperty().SetColor(c)
        polaxes.GetPolarAxisLabelTextProperty().SetColor(c)
        polaxes.GetLastRadialAxisTextProperty().SetColor(c)
        polaxes.GetSecondaryRadialAxesTextProperty().SetColor(c)
        polaxes.GetSecondaryRadialAxesProperty().SetColor(c)
        polaxes.GetSecondaryPolarArcsProperty().SetColor(c)

        polaxes.SetMinimumAngle(0.0)
        polaxes.SetMaximumAngle(315.0)
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
        pr = ls.GetBottomAxis().GetLabelTextProperty()
        pr.SetFontFamily(vtk.VTK_FONT_FILE)
        pr.SetFontFile(utils.get_font_path(settings.default_font))
        ls.PickableOff()
        plt.renderer.AddActor(ls)
        plt.axes_instances[r] = ls

    elif plt.axes == 14:
        try:
            cow = vtk.vtkCameraOrientationWidget()
            cow.SetParentRenderer(plt.renderer)
            cow.On()
            plt.axes_instances[r] = cow
        except AttributeError:
            vedo.logger.warning("axes mode 14 is unavailable in this vtk version")

    else:
        e = "\bomb Keyword axes type must be in range [0-13]."
        e += "Available axes types are:\n\n"
        e += "0 = no axes\n"
        e += "1 = draw three customizable gray grid walls\n"
        e += "2 = show cartesian axes from (0,0,0)\n"
        e += "3 = show positive range of cartesian axes from (0,0,0)\n"
        e += "4 = show a triad at bottom left\n"
        e += "5 = show a cube at bottom left\n"
        e += "6 = mark the corners of the bounding box\n"
        e += "7 = draw a 3D ruler at each side of the cartesian axes\n"
        e += "8 = show the vtk default vtkCubeAxesActor object\n"
        e += "9 = show the bounding box outline\n"
        e += "10 = show three circles representing the maximum bounding box\n"
        e += "11 = show a large grid on the x-y plane (use with zoom=8)\n"
        e += "12 = show polar axes\n"
        e += "13 = draw a simple ruler at the bottom of the window\n"
        e += "14 = show the vtk default vtkCameraOrientationWidget object"
        vedo.logger.warning(e)

    if not plt.axes_instances[r]:
        plt.axes_instances[r] = True
    return None
