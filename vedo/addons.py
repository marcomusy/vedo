#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import vedo.vtkclasses as vtki   # a wrapper for lazy imports

import vedo
from vedo import settings
from vedo import utils
from vedo import shapes
from vedo.transformations import LinearTransform
from vedo.assembly import Assembly, Group
from vedo.colors import get_color, build_lut, color_map, printc
from vedo.mesh import Mesh
from vedo.pointcloud import Points, Point, merge
from vedo.grids import TetMesh
from vedo.volume import Volume

__docformat__ = "google"

__doc__ = """
Create additional objects like axes, legends, lights, etc.

![](https://vedo.embl.es/images/pyplot/customAxes2.png)
"""

__all__ = [
    "ScalarBar",
    "ScalarBar3D",
    "Slider2D",
    "Slider3D",
    "Icon",
    "LegendBox",
    "Light",
    "Axes",
    "RendererFrame",
    "Ruler2D",
    "Ruler3D",
    "RulerAxes",
    "DistanceTool",
    "SplineTool",
    "Goniometer",
    "Button",
    "Flagpost",
    "ProgressBarWidget",
    "BoxCutter",
    "PlaneCutter",
    "SphereCutter",
]

########################################################################################
class Flagpost(vtki.vtkFlagpoleLabel):
    """
    Create a flag post style element to describe an object.
    """

    def __init__(
        self,
        txt="",
        base=(0, 0, 0),
        top=(0, 0, 1),
        s=1,
        c="k9",
        bc="k1",
        alpha=1,
        lw=0,
        font="Calco",
        justify="center-left",
        vspacing=1,
    ):
        """
        Create a flag post style element to describe an object.

        Arguments:
            txt : (str)
                Text to display. The default is the filename or the object name.
            base : (list)
                position of the flag anchor point.
            top : (list)
                a 3D displacement or offset.
            s : (float)
                size of the text to be shown
            c : (list)
                color of text and line
            bc : (list)
                color of the flag background
            alpha : (float)
                opacity of text and box.
            lw : (int)
                line with of box frame. The default is 0.
            font : (str)
                font name. Use a monospace font for better rendering. The default is "Calco".
                Type `vedo -r fonts` for a font demo.
                Check [available fonts here](https://vedo.embl.es/fonts).
            justify : (str)
                internal text justification. The default is "center-left".
            vspacing : (float)
                vertical spacing between lines.

        Examples:
            - [flag_labels2.py](https://github.com/marcomusy/vedo/tree/master/examples/examples/other/flag_labels2.py)

            ![](https://vedo.embl.es/images/other/flag_labels2.png)
        """

        super().__init__()

        base = utils.make3d(base)
        top = utils.make3d(top)

        self.SetBasePosition(*base)
        self.SetTopPosition(*top)

        self.SetFlagSize(s)
        self.SetInput(txt)
        self.PickableOff()

        self.GetProperty().LightingOff()
        self.GetProperty().SetLineWidth(lw + 1)

        prop = self.GetTextProperty()
        if bc is not None:
            prop.SetBackgroundColor(get_color(bc))

        prop.SetOpacity(alpha)
        prop.SetBackgroundOpacity(alpha)
        if bc is not None and len(bc) == 4:
            prop.SetBackgroundRGBA(alpha)

        c = get_color(c)
        prop.SetColor(c)
        self.GetProperty().SetColor(c)

        prop.SetFrame(bool(lw))
        prop.SetFrameWidth(lw)
        prop.SetFrameColor(prop.GetColor())

        prop.SetFontFamily(vtki.VTK_FONT_FILE)
        fl = utils.get_font_path(font)
        prop.SetFontFile(fl)
        prop.ShadowOff()
        prop.BoldOff()
        prop.SetOpacity(alpha)
        prop.SetJustificationToLeft()
        if "top" in justify:
            prop.SetVerticalJustificationToTop()
        if "bottom" in justify:
            prop.SetVerticalJustificationToBottom()
        if "cent" in justify:
            prop.SetVerticalJustificationToCentered()
            prop.SetJustificationToCentered()
        if "left" in justify:
            prop.SetJustificationToLeft()
        if "right" in justify:
            prop.SetJustificationToRight()
        prop.SetLineSpacing(vspacing * 1.2)
        self.SetUseBounds(False)

    def text(self, value):
        self.SetInput(value)
        return self

    def on(self):
        self.VisibilityOn()
        return self

    def off(self):
        self.VisibilityOff()
        return self

    def toggle(self):
        self.SetVisibility(not self.GetVisibility())
        return self

    def use_bounds(self, value=True):
        self.SetUseBounds(value)
        return self

    def color(self, c):
        c = get_color(c)
        self.GetTextProperty().SetColor(c)
        self.GetProperty().SetColor(c)
        return self

    def pos(self, p):
        p = np.asarray(p)
        self.top = self.top - self.base + p
        self.base = p
        return self

    @property
    def base(self):
        return np.array(self.GetBasePosition())

    @property
    def top(self):
        return np.array(self.GetTopPosition())

    @base.setter
    def base(self, value):
        self.SetBasePosition(*value)

    @top.setter
    def top(self, value):
        self.SetTopPosition(*value)



###########################################################################################
class LegendBox(shapes.TextBase, vtki.vtkLegendBoxActor):
    """
    Create a 2D legend box.
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
        """
        Create a 2D legend box for the list of specified objects.

        Arguments:
            nmax : (int)
                max number of legend entries
            c : (color)
                text color, leave as None to pick the mesh color automatically
            font : (str)
                Check [available fonts here](https://vedo.embl.es/fonts)
            width : (float)
                width of the box as fraction of the window width
            height : (float)
                height of the box as fraction of the window height
            padding : (int)
                padding space in units of pixels
            bg : (color)
                background color of the box
            alpha: (float)
                opacity of the box
            pos : (str, list)
                position of the box, can be either a string or a (x,y) screen position in range [0,1]

        Examples:
            - [legendbox.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/legendbox.py)
            - [flag_labels1.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels1.py)
            - [flag_labels2.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels2.py)

                ![](https://vedo.embl.es/images/other/flag_labels.png)
        """
        super().__init__()

        self.name = "LegendBox"
        self.entries = entries[:nmax]
        self.properties = self.GetEntryTextProperty()

        n = 0
        texts = []
        for e in self.entries:
            ename = e.name
            if "legend" in e.info.keys():
                if not e.info["legend"]:
                    ename = ""
                else:
                    ename = str(e.info["legend"])
            if ename:
                n += 1
            texts.append(ename)
        self.SetNumberOfEntries(n)

        if not n:
            return

        self.ScalarVisibilityOff()
        self.PickableOff()
        self.SetPadding(padding)

        self.properties.ShadowOff()
        self.properties.BoldOff()

        # self.properties.SetJustificationToLeft() # no effect
        # self.properties.SetVerticalJustificationToTop()

        if not font:
            font = settings.default_font

        self.font(font)

        n = 0
        for i in range(len(self.entries)):
            ti = texts[i]
            if not ti:
                continue
            e = entries[i]
            if c is None:
                col = e.properties.GetColor()
                if col == (1, 1, 1):
                    col = (0.2, 0.2, 0.2)
            else:
                col = get_color(c)
            if markers is None:  # default
                poly = e.dataset
            else:
                marker = markers[i] if utils.is_sequence(markers) else markers
                if isinstance(marker, Points):
                    poly = marker.clone(deep=False).normalize().shift(0, 1, 0).dataset
                else:  # assume string marker
                    poly = vedo.shapes.Marker(marker, s=1).shift(0, 1, 0).dataset

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


class Button(vedo.shapes.Text2D):
    """
    Build a Button object.
    """
    def __init__(
            self,
            fnc=None,
            states=("Button"),
            c=("white"),
            bc=("green4"),
            pos=(0.7, 0.1),
            size=24,
            font="Courier",
            bold=True,
            italic=False,
            alpha=1,
            angle=0,
        ):
        """
        Build a Button object to be shown in the rendering window.

        Arguments:
            fnc : (function)
                external function to be called by the widget
            states : (list)
                the list of possible states, eg. ['On', 'Off']
            c : (list)
                the list of colors for each state eg. ['red3', 'green5']
            bc : (list)
                the list of background colors for each state
            pos : (list, str)
                2D position in pixels from left-bottom corner
            size : (int)
                size of button font
            font : (str)
                font type
            bold : (bool)
                set bold font face
            italic : (bool)
                italic font face
            alpha : (float)
                opacity level
            angle : (float)
                anticlockwise rotation in degrees

        Examples:
            - [buttons1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/buttons1.py)
            - [buttons2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/buttons2.py)

                ![](https://user-images.githubusercontent.com/32848391/50738870-c0fe2500-11d8-11e9-9b78-92754f5c5968.jpg)

            - [timer_callback2.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/timer_callback2.py)

                ![](https://vedo.embl.es/images/advanced/timer_callback1.jpg)
        """
        super().__init__()

        self.status_idx = 0

        self.spacer = " "

        self.states = states

        if not utils.is_sequence(c):
            c = [c]
        self.colors = c

        if not utils.is_sequence(bc):
            bc = [bc]
        self.bcolors = bc

        assert len(c) == len(bc), "in Button color number mismatch!"

        self.function = fnc
        self.function_id = None

        self.status(0)

        if font == "courier":
            font = font.capitalize()
        self.font(font).bold(bold).italic(italic)

        self.alpha(alpha).angle(angle)
        self.size(size/20)
        self.pos(pos, "center")
        self.PickableOn()


    def status(self, s=None):
        """
        Set/Get the status of the button.
        """
        if s is None:
            return self.states[self.status_idx]

        if isinstance(s, str):
            s = self.states.index(s)
        self.status_idx = s
        self.text(self.spacer + self.states[s] + self.spacer)
        s = s % len(self.bcolors)
        self.color(self.colors[s])
        self.background(self.bcolors[s])
        return self

    def switch(self):
        """
        Change/cycle button status to the next defined status in states list.
        """
        self.status_idx = (self.status_idx + 1) % len(self.states)
        self.status(self.status_idx)
        return self


#####################################################################
class SplineTool(vtki.vtkContourWidget):
    """
    Spline tool, draw a spline through a set of points interactively.
    """

    def __init__(self, points, pc="k", ps=8, lc="r4", ac="g5",
                 lw=2, alpha=1, closed=False, ontop=True, can_add_nodes=True):
        """
        Spline tool, draw a spline through a set of points interactively.

        Arguments:
            points : (list), Points
                initial set of points.
            pc : (str)
                point color.
            ps : (int)
                point size.
            lc : (str)
                line color.
            ac : (str)
                active point color.
            lw : (int)
                line width.
            alpha : (float)
                line transparency level.
            closed : (bool)
                spline is closed or open.
            ontop : (bool)
                show it always on top of other objects.
            can_add_nodes : (bool)
                allow to add (or remove) new nodes interactively.

        Examples:
            - [spline_tool.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/spline_tool.py)

                ![](https://vedo.embl.es/images/basic/spline_tool.png)
        """
        super().__init__()

        self.representation = self.GetRepresentation()
        self.representation.SetAlwaysOnTop(ontop)
        self.SetAllowNodePicking(can_add_nodes)


        self.representation.GetLinesProperty().SetColor(get_color(lc))
        self.representation.GetLinesProperty().SetLineWidth(lw)
        self.representation.GetLinesProperty().SetOpacity(alpha)
        if lw == 0 or alpha == 0:
            self.representation.GetLinesProperty().SetOpacity(0)

        self.representation.GetActiveProperty().SetLineWidth(lw + 1)
        self.representation.GetActiveProperty().SetColor(get_color(ac))

        self.representation.GetProperty().SetColor(get_color(pc))
        self.representation.GetProperty().SetPointSize(ps)
        self.representation.GetProperty().RenderPointsAsSpheresOn()

        # self.representation.BuildRepresentation() # crashes

        self.SetRepresentation(self.representation)

        if utils.is_sequence(points):
            self.points = Points(points)
        else:
            self.points = points

        self.closed = closed

    @property
    def interactor(self):
        """Return the current interactor."""
        return self.GetInteractor()
    
    @interactor.setter
    def interactor(self, iren):
        """Set the current interactor."""
        self.SetInteractor(iren)

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
    
    def add_observer(self, event, func, priority=1):
        """Add an observer to the widget."""
        event = utils.get_vtk_name_event(event)
        cid = self.AddObserver(event, func, priority)
        return cid

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
        ln = vedo.Line(pd, lw=2, c="k")
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
class SliderWidget(vtki.vtkSliderWidget):
    """Helper class for `vtkSliderWidget`"""

    def __init__(self):
        super().__init__()

    @property
    def interactor(self):
        return self.GetInteractor()

    @interactor.setter
    def interactor(self, iren):
        self.SetInteractor(iren)

    @property
    def representation(self):
        return self.GetRepresentation()

    @property
    def value(self):
        return self.GetRepresentation().GetValue()

    @value.setter
    def value(self, val):
        self.GetRepresentation().SetValue(val)

    @property
    def renderer(self):
        return self.GetCurrentRenderer()

    @renderer.setter
    def renderer(self, ren):
        self.SetCurrentRenderer(ren)

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

    def toggle(self):
        self.SetEnabled(not self.GetEnabled())

    def add_observer(self, event, func, priority=1):
        """Add an observer to the widget."""
        event = utils.get_vtk_name_event(event)
        cid = self.widget.AddObserver(event, func, priority)
        return cid


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

    Arguments:
        p1 : (list)
            first point 3D coordinates.
        p2 : (list)
            the vertex point.
        p3 : (list)
            the last point defining the angle.
        font : (str)
            Font face. Check [available fonts here](https://vedo.embl.es/fonts).
        arc_size : (float)
            dimension of the arc wrt the smallest axis.
        s : (float)
            size of the text.
        italic : (float, bool)
            italic text.
        rotation : (float)
            rotation of text in degrees.
        prefix : (str)
            append this string to the numeric value of the angle.
        lc : (list)
            color of the goniometer lines.
        c : (str)
            color of the goniometer angle filling. Set alpha=0 to remove it.
        alpha : (float)
            transparency level.
        lw : (float)
            line width.
        precision : (int)
            number of significant digits.

    Examples:
        - [goniometer.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/goniometer.py)

            ![](https://vedo.embl.es/images/pyplot/goniometer.png)
    """
    if isinstance(p1, Points): p1 = p1.pos()
    if isinstance(p2, Points): p2 = p2.pos()
    if isinstance(p3, Points): p3 = p3.pos()
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
        s=r/12 * s,
        font=font,
        italic=italic,
        justify="center",
    )
    cr = np.cross(va, vb)
    lb.reorient([0,0,1], cr * np.sign(cr[2]), rotation=rotation, xyplane=False)
    lb.pos(p2 + vc * r / 1.75)
    lb.c(c).bc("tomato").lighting("off")
    acts.append(lb)

    if alpha > 0:
        pts = [p2] + arc.vertices.tolist() + [p2]
        msh = Mesh([pts, [list(range(arc.npoints + 2))]], c=lc, alpha=alpha)
        msh.lighting("off")
        msh.triangulate()
        msh.shift(0, 0, -r / 10000)  # to resolve 2d conflicts..
        acts.append(msh)

    asse = Assembly(acts)
    asse.name = "Goniometer"
    return asse


def Light(pos, focal_point=(0, 0, 0), angle=180, c=None, intensity=1):
    """
    Generate a source of light placed at `pos` and directed to `focal point`.
    Returns a `vtkLight` object.

    Arguments:
        focal_point : (list)
            focal point, if a `vedo` object is passed then will grab its position.
        angle : (float)
            aperture angle of the light source, in degrees
        c : (color)
            set the light color
        intensity : (float)
            intensity value between 0 and 1.

    Check also:
        `plotter.Plotter.remove_lights()`

    Examples:
        - [light_sources.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/light_sources.py)

            ![](https://vedo.embl.es/images/basic/lights.png)
    """
    if c is None:
        try:
            c = pos.color()
        except AttributeError:
            c = "white"

    try:
        pos = pos.pos()
    except AttributeError:
        pass
    
    try:
        focal_point = focal_point.pos()
    except AttributeError:
        pass

    light = vtki.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(pos)
    light.SetConeAngle(angle)
    light.SetFocalPoint(focal_point)
    light.SetIntensity(intensity)
    light.SetColor(get_color(c))
    return light


#####################################################################
def ScalarBar(
    obj,
    title="",
    pos=(0.775, 0.05),
    title_yoffset=15,
    font_size=12,
    size=(None, None),
    nlabels=None,
    c="k",
    horizontal=False,
    use_alpha=True,
    label_format=":6.3g",
):
    """
    A 2D scalar bar for the specified obj.

    Arguments:
        title : (str)
            scalar bar title
        pos : (float,float)
            position coordinates of the bottom left corner
        title_yoffset : (float)
            vertical space offset between title and color scalarbar
        font_size : (float)
            size of font for title and numeric labels
        size : (float,float)
            size of the scalarbar in number of pixels (width, height)
        nlabels : (int)
            number of numeric labels
        c : (list)
            color of the scalar bar text
        horizontal : (bool)
            lay the scalarbar horizontally
        use_alpha : (bool)
            render transparency in the color bar itself
        label_format : (str)
            c-style format string for numeric labels

    Examples:
        - [scalarbars.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/scalarbars.py)

        ![](https://user-images.githubusercontent.com/32848391/62940174-4bdc7900-bdd3-11e9-9713-e4f3e2fdab63.png)
    """

    if isinstance(obj, (Points, TetMesh, vedo.UnstructuredGrid)):
        vtkscalars = obj.dataset.GetPointData().GetScalars()
        if vtkscalars is None:
            vtkscalars = obj.dataset.GetCellData().GetScalars()
        if not vtkscalars:
            return None
        lut = vtkscalars.GetLookupTable()
        if not lut:
            lut = obj.mapper.GetLookupTable()
            if not lut:
                return None

    elif isinstance(obj, Volume):
        lut = utils.ctf2lut(obj)

    elif utils.is_sequence(obj) and len(obj) == 2:
        x = np.linspace(obj[0], obj[1], 256)
        data = []
        for i in range(256):
            rgb = color_map(i, c, 0, 256)
            data.append([x[i], rgb])
        lut = build_lut(data)

    elif not hasattr(obj, "mapper"):
        vedo.logger.error(f"in add_scalarbar(): input is invalid {type(obj)}. Skip.")
        return None

    else:
        return None

    c = get_color(c)
    sb = vtki.vtkScalarBarActor()
    #sb.SetTextPosition(0)

    # print("GetLabelFormat", sb.GetLabelFormat())
    label_format = label_format.replace(":", "%-#")
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
        titprop.SetFontFamily(vtki.VTK_FONT_FILE)
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

    if not horizontal:
        if size[0] is not None:
            sb.SetMaximumWidthInPixels(size[0])
        if size[1] is not None:
            sb.SetMaximumHeightInPixels(size[1])
    else:
        if size[0] is not None:
            sb.SetMaximumHeightInPixels(size[0])
        if size[1] is not None:
            sb.SetMaximumWidthInPixels(size[1])

    if nlabels is not None:
        sb.SetNumberOfLabels(nlabels)

    sctxt = sb.GetLabelTextProperty()
    sctxt.SetFontFamily(vtki.VTK_FONT_FILE)
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
    title="",
    pos=None,
    size=(0, 0),
    title_font="",
    title_xoffset=-1.2,
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
    c='k',
    draw_box=True,
    above_text=None,
    below_text=None,
    nan_text="NaN",
    categories=None,
):
    """
    Create a 3D scalar bar for the specified object.

    Input `obj` input can be:

        - a list of numbers,
        - a list of two numbers in the form (min, max),
        - a Mesh already containing a set of scalars associated to vertices or cells,
        - if None the last object in the list of actors will be used.

    Arguments:
        size : (list)
            (thickness, length) of scalarbar
        title : (str)
            scalar bar title
        title_xoffset : (float)
            horizontal space btw title and color scalarbar
        title_yoffset : (float)
            vertical space offset
        title_size : (float)
            size of title wrt numeric labels
        title_rotation : (float)
            title rotation in degrees
        nlabels : (int)
            number of numeric labels
        label_font : (str)
            font type for labels
        label_size : (float)
            label scale factor
        label_offset : (float)
            space btw numeric labels and scale
        label_rotation : (float)
            label rotation in degrees
        draw_box : (bool)
            draw a box around the colorbar
        categories : (list)
            make a categorical scalarbar,
            the input list will have the format [value, color, alpha, textlabel]

    Examples:
        - [scalarbars.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/scalarbars.py)
        - [plot_fxy2.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_fxy2.py)
    """

    if isinstance(obj, (Points, TetMesh, vedo.UnstructuredGrid)):
        lut = obj.mapper.GetLookupTable()
        if not lut or lut.GetTable().GetNumberOfTuples() == 0:
            # create the most similar to the default
            obj.cmap("jet_r")
            lut = obj.mapper.GetLookupTable()
        vmin, vmax = lut.GetRange()

    elif isinstance(obj, Volume):
        lut = utils.ctf2lut(obj)
        vmin, vmax = lut.GetRange()

    else:
        vedo.logger.error("in ScalarBar3D(): input must be a vedo object with bounds.")
        return None

    bns = obj.bounds()
    sx, sy = size
    if sy == 0 or sy is None:
        sy = bns[3] - bns[2]
    if sx == 0 or sx is None:
        sx = sy / 18

    if categories is not None:  ################################
        ncats = len(categories)
        scale = shapes.Grid([-float(sx) * label_offset, 0, 0],
                            c=c, alpha=1, s=(sx, sy), res=(1, ncats))
        cols, alphas = [], []
        ticks_pos, ticks_txt = [0.0], [""]
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
        scale.cellcolors = rgba

    else:  ########################################################

        # build the color scale part
        scale = shapes.Grid(
            [-float(sx) * label_offset, 0, 0],
            c=c,
            s=(sx, sy),
            res=(1, lut.GetTable().GetNumberOfTuples()),
        )
        cscals = np.linspace(vmin, vmax, lut.GetTable().GetNumberOfTuples(), endpoint=True)

        if lut.GetScale():  # logarithmic scale
            lut10 = vtki.vtkLookupTable()
            lut10.DeepCopy(lut)
            lut10.SetScaleToLinear()
            lut10.Build()
            scale.cmap(lut10, cscals, on="cells")
            tk = utils.make_ticks(vmin, vmax, nlabels, logscale=True, useformat=label_format)
        else:
            # for i in range(lut.GetTable().GetNumberOfTuples()):
            #     print("LUT i=", i, lut.GetTableValue(i))
            scale.cmap(lut, cscals, on="cells")
            tk = utils.make_ticks(vmin, vmax, nlabels, logscale=False, useformat=label_format)
        ticks_pos, ticks_txt = tk
    
    scale.lw(0).wireframe(False).lighting("off")

    scales = [scale]

    xbns = scale.xbounds()

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
                    s=lsize,
                    justify="center-top",
                    c=c,
                    italic=italic,
                    font=label_font,
                )
                a.rotate_z(label_rotation)
                a.pos(sx * label_offset, y, 0)
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
            tic = shapes.Line([xbns[1], y, 0], [xbns[1] + sx * label_offset / 4, y, 0], lw=2, c=c)
            tacts.append(tic)

    # build title
    if title:
        t = shapes.Text3D(
            title,
            pos=(0, 0, 0),
            s=sy / 50 * title_size,
            c=c,
            justify="centered-bottom",
            italic=italic,
            font=title_font,
        )
        t.rotate_z(90 + title_rotation)
        t.pos(sx * title_xoffset, title_yoffset, 0)
        tacts.append(t)

    if pos is None:
        tsize = 0
        if title:
            bbt = t.bounds()
            tsize = bbt[1] - bbt[0]
        pos = (bns[1] + tsize + sx*1.5, (bns[2]+bns[3])/2, bns[4])

    # build below scale
    if lut.GetUseBelowRangeColor():
        r, g, b, alfa = lut.GetBelowRangeColor()
        sx = float(sx)
        sy = float(sy)
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
                    pos=(0, 0, 0),
                    s=lsize,
                    c=c,
                    justify="center-top",
                    italic=italic,
                    font=label_font,
                )
                btx.rotate_z(label_rotation)
            else:
                btx = shapes.Text3D(
                    below_text,
                    pos=(0, 0, 0),
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
                    pos=(0, 0, 0),
                    s=lsize,
                    c=c,
                    justify="center-top",
                    italic=italic,
                    font=label_font,
                )
                atx.rotate_z(label_rotation)
            else:
                atx = shapes.Text3D(
                    above_text,
                    pos=(0, 0, 0),
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
                pos=(0, 0, 0),
                s=lsize,
                c=c,
                justify="center-left",
                italic=italic,
                font=label_font,
            )
            nantx.rotate_z(label_rotation)
        else:
            nantx = shapes.Text3D(
                nan_text,
                pos=(0, 0, 0),
                s=lsize,
                c=c,
                justify="center-left",
                italic=italic,
                font=label_font,
            )
        nantx.pos(sx * label_offset, -sy / 2 - sx * 0.66 - nanshift, 0)
        tacts.append(nantx)

    if draw_box:
        tacts.append(scale.box().lw(1).c(c))

    for m in tacts + scales:
        m.shift(pos)
        m.actor.PickableOff()
        m.properties.LightingOff()

    asse = Assembly(scales + tacts)

    # asse.transform = LinearTransform().shift(pos)

    bb = asse.GetBounds()
    # print("ScalarBar3D pos",pos, bb)
    # asse.SetOrigin(pos)

    asse.SetOrigin(bb[0], bb[2], bb[4])
    # asse.SetOrigin(bb[0],0,0) #in pyplot line 1312

    asse.PickableOff()
    asse.UseBoundsOff()
    asse.name = "ScalarBar3D"
    return asse


#####################################################################
class Slider2D(SliderWidget):
    """
    Add a slider which can call an external custom function.
    """
    def __init__(
        self,
        sliderfunc,
        xmin,
        xmax,
        value=None,
        pos=4,
        title="",
        font="Calco",
        title_size=1,
        c="k",
        alpha=1,
        show_value=True,
        delayed=False,
        **options,
    ):
        """
        Add a slider which can call an external custom function.
        Set any value as float to increase the number of significant digits above the slider.

        Use `play()` to start an animation between the current slider value and the last value.

        Arguments:
            sliderfunc : (function)
                external function to be called by the widget
            xmin : (float)
                lower value of the slider
            xmax : (float)
                upper value
            value : (float)
                current value
            pos : (list, str)
                position corner number: horizontal [1-5] or vertical [11-15]
                it can also be specified by corners coordinates [(x1,y1), (x2,y2)]
                and also by a string descriptor (eg. "bottom-left")
            title : (str)
                title text
            font : (str)
                title font face. Check [available fonts here](https://vedo.embl.es/fonts).
            title_size : (float)
                title text scale [1.0]
            show_value : (bool)
                if True current value is shown
            delayed : (bool)
                if True the callback is delayed until when the mouse button is released
            alpha : (float)
                opacity of the scalar bar texts
            slider_length : (float)
                slider length
            slider_width : (float)
                slider width
            end_cap_length : (float)
                length of the end cap
            end_cap_width : (float)
                width of the end cap
            tube_width : (float)
                width of the tube
            title_height : (float)
                height of the title
            tformat : (str)
                format of the title

        Examples:
            - [sliders1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders1.py)
            - [sliders2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders2.py)

            ![](https://user-images.githubusercontent.com/32848391/50738848-be033480-11d8-11e9-9b1a-c13105423a79.jpg)
        """
        slider_length = options.pop("slider_length",  0.015)
        slider_width  = options.pop("slider_width",   0.025)
        end_cap_length= options.pop("end_cap_length", 0.0015)
        end_cap_width = options.pop("end_cap_width",  0.0125)
        tube_width    = options.pop("tube_width",     0.0075)
        title_height  = options.pop("title_height",   0.025)
        tformat       = options.pop("tformat",        None)

        if options:
            vedo.logger.warning(f"in Slider2D unknown option(s): {options}")

        c = get_color(c)

        if value is None or value < xmin:
            value = xmin

        slider_rep = vtki.new("SliderRepresentation2D")
        slider_rep.SetMinimumValue(xmin)
        slider_rep.SetMaximumValue(xmax)
        slider_rep.SetValue(value)
        slider_rep.SetSliderLength(slider_length)
        slider_rep.SetSliderWidth(slider_width)
        slider_rep.SetEndCapLength(end_cap_length)
        slider_rep.SetEndCapWidth(end_cap_width)
        slider_rep.SetTubeWidth(tube_width)
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()

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
            slider_rep.GetPoint1Coordinate().SetValue(pos[0][0], pos[0][1])
            slider_rep.GetPoint2Coordinate().SetValue(pos[1][0], pos[1][1])
        elif pos == 1:  # top-left horizontal
            slider_rep.GetPoint1Coordinate().SetValue(0.04, 0.93)
            slider_rep.GetPoint2Coordinate().SetValue(0.45, 0.93)
        elif pos == 2:
            slider_rep.GetPoint1Coordinate().SetValue(0.55, 0.93)
            slider_rep.GetPoint2Coordinate().SetValue(0.95, 0.93)
        elif pos == 3:
            slider_rep.GetPoint1Coordinate().SetValue(0.05, 0.06)
            slider_rep.GetPoint2Coordinate().SetValue(0.45, 0.06)
        elif pos == 4:  # bottom-right
            slider_rep.GetPoint1Coordinate().SetValue(0.55, 0.06)
            slider_rep.GetPoint2Coordinate().SetValue(0.95, 0.06)
        elif pos == 5:  # bottom span horizontal
            slider_rep.GetPoint1Coordinate().SetValue(0.04, 0.06)
            slider_rep.GetPoint2Coordinate().SetValue(0.95, 0.06)
        elif pos == 11:  # top-left vertical
            slider_rep.GetPoint1Coordinate().SetValue(0.065, 0.54)
            slider_rep.GetPoint2Coordinate().SetValue(0.065, 0.9)
        elif pos == 12:
            slider_rep.GetPoint1Coordinate().SetValue(0.94, 0.54)
            slider_rep.GetPoint2Coordinate().SetValue(0.94, 0.9)
        elif pos == 13:
            slider_rep.GetPoint1Coordinate().SetValue(0.065, 0.1)
            slider_rep.GetPoint2Coordinate().SetValue(0.065, 0.54)
        elif pos == 14:  # bottom-right vertical
            slider_rep.GetPoint1Coordinate().SetValue(0.94, 0.1)
            slider_rep.GetPoint2Coordinate().SetValue(0.94, 0.54)
        elif pos == 15:  # right margin vertical
            slider_rep.GetPoint1Coordinate().SetValue(0.95, 0.1)
            slider_rep.GetPoint2Coordinate().SetValue(0.95, 0.9)
        else:  # bottom-right
            slider_rep.GetPoint1Coordinate().SetValue(0.55, 0.06)
            slider_rep.GetPoint2Coordinate().SetValue(0.95, 0.06)

        if show_value:
            if tformat is None:
                if isinstance(xmin, int) and isinstance(xmax, int) and isinstance(value, int):
                    tformat = "%0.0f"
                else:
                    tformat = "%0.2f"

            slider_rep.SetLabelFormat(tformat)  # default is '%0.3g'
            slider_rep.GetLabelProperty().SetShadow(0)
            slider_rep.GetLabelProperty().SetBold(0)
            slider_rep.GetLabelProperty().SetOpacity(alpha)
            slider_rep.GetLabelProperty().SetColor(c)
            if isinstance(pos, int) and pos > 10:
                slider_rep.GetLabelProperty().SetOrientation(90)
        else:
            slider_rep.ShowSliderLabelOff()
        slider_rep.GetTubeProperty().SetColor(c)
        slider_rep.GetTubeProperty().SetOpacity(0.75)
        slider_rep.GetSliderProperty().SetColor(c)
        slider_rep.GetSelectedProperty().SetColor(np.sqrt(np.array(c)))
        slider_rep.GetCapProperty().SetColor(c)

        slider_rep.SetTitleHeight(title_height * title_size)
        slider_rep.GetTitleProperty().SetShadow(0)
        slider_rep.GetTitleProperty().SetColor(c)
        slider_rep.GetTitleProperty().SetOpacity(alpha)
        slider_rep.GetTitleProperty().SetBold(0)
        if font.lower() == "courier":
            slider_rep.GetTitleProperty().SetFontFamilyToCourier()
        elif font.lower() == "times":
            slider_rep.GetTitleProperty().SetFontFamilyToTimes()
        elif font.lower() == "arial":
            slider_rep.GetTitleProperty().SetFontFamilyToArial()
        else:
            if font == "":
                font = utils.get_font_path(settings.default_font)
            else:
                font = utils.get_font_path(font)
            slider_rep.GetTitleProperty().SetFontFamily(vtki.VTK_FONT_FILE)
            slider_rep.GetLabelProperty().SetFontFamily(vtki.VTK_FONT_FILE)
            slider_rep.GetTitleProperty().SetFontFile(font)
            slider_rep.GetLabelProperty().SetFontFile(font)

        if title:
            slider_rep.SetTitleText(title)
            if not utils.is_sequence(pos):
                if isinstance(pos, int) and pos > 10:
                    slider_rep.GetTitleProperty().SetOrientation(90)
            else:
                if abs(pos[0][0] - pos[1][0]) < 0.1:
                    slider_rep.GetTitleProperty().SetOrientation(90)

        super().__init__()

        self.SetAnimationModeToJump()
        self.SetRepresentation(slider_rep)
        if delayed:
            self.AddObserver("EndInteractionEvent", sliderfunc)
        else:
            self.AddObserver("InteractionEvent", sliderfunc)


#####################################################################
class Slider3D(SliderWidget):
    """
    Add a 3D slider which can call an external custom function.
    """

    def __init__(
        self,
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
        Add a 3D slider which can call an external custom function.

        Arguments:
            sliderfunc : (function)
                external function to be called by the widget
            pos1 : (list)
                first position 3D coordinates
            pos2 : (list)
                second position 3D coordinates
            xmin : (float)
                lower value
            xmax : (float)
                upper value
            value : (float)
                initial value
            s : (float)
                label scaling factor
            t : (float)
                tube scaling factor
            title : (str)
                title text
            c : (color)
                slider color
            rotation : (float)
                title rotation around slider axis
            show_value : (bool)
                if True current value is shown on top of the slider

        Examples:
            - [sliders3d.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders3d.py)
        """
        c = get_color(c)

        if value is None or value < xmin:
            value = xmin

        slider_rep = vtki.new("SliderRepresentation3D")
        slider_rep.SetMinimumValue(xmin)
        slider_rep.SetMaximumValue(xmax)
        slider_rep.SetValue(value)

        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToWorld()
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToWorld()
        slider_rep.GetPoint1Coordinate().SetValue(pos2)
        slider_rep.GetPoint2Coordinate().SetValue(pos1)

        # slider_rep.SetPoint1InWorldCoordinates(pos2[0], pos2[1], pos2[2])
        # slider_rep.SetPoint2InWorldCoordinates(pos1[0], pos1[1], pos1[2])

        slider_rep.SetSliderWidth(0.03 * t)
        slider_rep.SetTubeWidth(0.01 * t)
        slider_rep.SetSliderLength(0.04 * t)
        slider_rep.SetSliderShapeToCylinder()
        slider_rep.GetSelectedProperty().SetColor(np.sqrt(np.array(c)))
        slider_rep.GetSliderProperty().SetColor(np.array(c) / 1.5)
        slider_rep.GetCapProperty().SetOpacity(0)
        slider_rep.SetRotation(rotation)

        if not show_value:
            slider_rep.ShowSliderLabelOff()

        slider_rep.SetTitleText(title)
        slider_rep.SetTitleHeight(s * t)
        slider_rep.SetLabelHeight(s * t * 0.85)

        slider_rep.GetTubeProperty().SetColor(c)

        super().__init__()

        self.SetRepresentation(slider_rep)
        self.SetAnimationModeToJump()
        self.AddObserver("InteractionEvent", sliderfunc)

class BaseCutter:
    """
    Base class for Cutter widgets.
    """
    def __init__(self):
        self._implicit_func = None
        self.widget = None
        self.clipper = None
        self.cutter = None
        self.mesh = None
        self.remnant = None
        self._alpha = 0.5
        self._keypress_id = None

    def invert(self):
        """Invert selection."""
        self.clipper.SetInsideOut(not self.clipper.GetInsideOut())
        return self

    def bounds(self, value=None):
        """Set or get the bounding box."""
        if value is None:
            return self.cutter.GetBounds()
        else:
            self._implicit_func.SetBounds(value)
            return self

    def on(self):
        """Switch the widget on or off."""
        self.widget.On()
        return self

    def off(self):
        """Switch the widget on or off."""
        self.widget.Off()
        return self

    def add_to(self, plt):
        """Assign the widget to the provided `Plotter` instance."""
        self.widget.SetInteractor(plt.interactor)
        self.widget.SetCurrentRenderer(plt.renderer)
        if self.widget not in plt.widgets:
            plt.widgets.append(self.widget)

        cpoly = self.clipper.GetOutput()
        self.mesh._update(cpoly)

        out = self.clipper.GetClippedOutputPort()
        if self._alpha:
            self.remnant.mapper.SetInputConnection(out)
            self.remnant.alpha(self._alpha).color((0.5, 0.5, 0.5))
            self.remnant.lighting('off').wireframe()
            plt.add(self.mesh, self.remnant)
        else:
            plt.add(self.mesh)

        self._keypress_id = plt.interactor.AddObserver(
            "KeyPressEvent", self._keypress
        )
        if plt.interactor and plt.interactor.GetInitialized():
            self.widget.On()
            self._select_polygons(self.widget, "InteractionEvent")
            plt.interactor.Render()
        return self

    def remove_from(self, plt):
        """Remove the widget to the provided `Plotter` instance."""
        self.widget.Off()
        self.widget.RemoveAllObservers() ### NOT SURE
        plt.remove(self.remnant)
        if self.widget in plt.widgets:
            plt.widgets.remove(self.widget)
        if self._keypress_id:
            plt.interactor.RemoveObserver(self._keypress_id)
        return self

    def add_observer(self, event, func, priority=1):
        """Add an observer to the widget."""
        event = utils.get_vtk_name_event(event)
        cid = self.widget.AddObserver(event, func, priority)
        return cid


class PlaneCutter(vtki.vtkPlaneWidget, BaseCutter):
    """
    Create a box widget to cut away parts of a Mesh.
    """
    def __init__(
            self,
            mesh,
            invert=False,
            can_translate=True,
            can_scale=True,
            origin=(),
            normal=(),
            padding=0.05,
            delayed=False,
            c=(0.25, 0.25, 0.25),
            alpha=0.05,
    ):
        """
        Create a box widget to cut away parts of a Mesh.

        Arguments:
            mesh : (Mesh)
                the input mesh
            invert : (bool)
                invert the clipping plane
            can_translate : (bool)
                enable translation of the widget
            can_scale : (bool)
                enable scaling of the widget
            origin : (list)
                origin of the plane
            normal : (list)
                normal to the plane
            padding : (float)
                padding around the input mesh
            delayed : (bool)
                if True the callback is delayed until
                when the mouse button is released (useful for large meshes)
            c : (color)
                color of the box cutter widget
            alpha : (float)
                transparency of the cut-off part of the input mesh
        
        Examples:
            - [slice_plane3.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slice_plane3.py)
        """
        super().__init__()

        self.mesh = mesh
        self.remnant = Mesh()
        self.remnant.name = mesh.name + "Remnant"
        self.remnant.pickable(False)

        self._alpha = alpha
        self._keypress_id = None

        self._implicit_func = vtki.new("Plane")

        poly = mesh.dataset
        self.clipper = vtki.new("ClipPolyData")
        self.clipper.GenerateClipScalarsOff()
        self.clipper.SetInputData(poly)
        self.clipper.SetClipFunction(self._implicit_func)
        self.clipper.SetInsideOut(invert)
        self.clipper.GenerateClippedOutputOn()
        self.clipper.Update()

        self.widget = vtki.new("ImplicitPlaneWidget")

        # self.widget.KeyPressActivationOff()
        # self.widget.SetKeyPressActivationValue('i')

        self.widget.SetOriginTranslation(can_translate)
        self.widget.SetOutlineTranslation(can_translate)
        self.widget.SetScaleEnabled(can_scale)

        self.widget.GetOutlineProperty().SetColor(get_color(c))
        self.widget.GetOutlineProperty().SetOpacity(0.25)
        self.widget.GetOutlineProperty().SetLineWidth(1)
        self.widget.GetOutlineProperty().LightingOff()

        self.widget.GetSelectedOutlineProperty().SetColor(get_color("red3"))

        self.widget.SetTubing(0)
        self.widget.SetDrawPlane(bool(alpha))
        self.widget.GetPlaneProperty().LightingOff()
        self.widget.GetPlaneProperty().SetOpacity(alpha)
        self.widget.GetSelectedPlaneProperty().SetColor(get_color("red5"))
        self.widget.GetSelectedPlaneProperty().LightingOff()

        self.widget.SetPlaceFactor(1.0 + padding)
        self.widget.SetInputData(poly)
        self.widget.PlaceWidget()
        if delayed:
            self.widget.AddObserver("EndInteractionEvent", self._select_polygons)
        else:
            self.widget.AddObserver("InteractionEvent", self._select_polygons)

        if len(origin) == 3:
            self.widget.SetOrigin(origin)
        else:
            self.widget.SetOrigin(mesh.center_of_mass())

        if len(normal) == 3:
            self.widget.SetNormal(normal)
        else:
            self.widget.SetNormal((1, 0, 0))
    
    @property
    def origin(self):
        """Get the origin of the plane."""
        return np.array(self.widget.GetOrigin())
    
    @property
    def normal(self):
        """Get the normal of the plane."""
        return np.array(self.widget.GetNormal())
    
    @origin.setter
    def origin(self, value):
        """Set the origin of the plane."""
        self.widget.SetOrigin(value)

    @normal.setter
    def normal(self, value):
        """Set the normal of the plane."""
        self.widget.SetNormal(value)

    def _select_polygons(self, vobj, event):
        vobj.GetPlane(self._implicit_func)

    def _keypress(self, vobj, event):
        if vobj.GetKeySym() == "r": # reset planes
            self.widget.GetPlane(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "u": # invert cut
            self.invert()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "x": # set normal along x
            self.widget.SetNormal((1, 0, 0))
            self.widget.GetPlane(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "y": # set normal along y
            self.widget.SetNormal((0, 1, 0))
            self.widget.GetPlane(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "z": # set normal along z
            self.widget.SetNormal((0, 0, 1))
            self.widget.GetPlane(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "s": # Ctrl+s to save mesh
            if self.widget.GetInteractor():
                if self.widget.GetInteractor().GetControlKey():
                    self.mesh.write("vedo_clipped.vtk")
                    printc(":save: saved mesh to vedo_clipped.vtk")


class BoxCutter(vtki.vtkBoxWidget, BaseCutter):
    """
    Create a box widget to cut away parts of a Mesh.
    """
    def __init__(
            self,
            mesh,
            invert=False,
            can_rotate=True,
            can_translate=True,
            can_scale=True,
            initial_bounds=(),
            padding=0.025,
            delayed=False,
            c=(0.25, 0.25, 0.25),
            alpha=0.05,
    ):
        """
        Create a box widget to cut away parts of a Mesh.

        Arguments:
            mesh : (Mesh)
                the input mesh
            invert : (bool)
                invert the clipping plane
            can_rotate : (bool)
                enable rotation of the widget
            can_translate : (bool)
                enable translation of the widget
            can_scale : (bool)
                enable scaling of the widget
            initial_bounds : (list)
                initial bounds of the box widget
            padding : (float)
                padding space around the input mesh
            delayed : (bool)
                if True the callback is delayed until
                when the mouse button is released (useful for large meshes)
            c : (color)
                color of the box cutter widget
            alpha : (float)
                transparency of the cut-off part of the input mesh
        """
        super().__init__()

        self.mesh = mesh
        self.remnant = Mesh()
        self.remnant.name = mesh.name + "Remnant"
        self.remnant.pickable(False)

        self._alpha = alpha
        self._keypress_id = None
        self._init_bounds = initial_bounds
        if len(self._init_bounds) == 0:
            self._init_bounds = mesh.bounds()
        else:
            self._init_bounds = initial_bounds

        self._implicit_func = vtki.new("Planes")
        self._implicit_func.SetBounds(self._init_bounds)

        poly = mesh.dataset
        self.clipper = vtki.new("ClipPolyData")
        self.clipper.GenerateClipScalarsOff()
        self.clipper.SetInputData(poly)
        self.clipper.SetClipFunction(self._implicit_func)
        self.clipper.SetInsideOut(not invert)
        self.clipper.GenerateClippedOutputOn()
        self.clipper.Update()

        self.widget = vtki.vtkBoxWidget()

        self.widget.SetRotationEnabled(can_rotate)
        self.widget.SetTranslationEnabled(can_translate)
        self.widget.SetScalingEnabled(can_scale)

        self.widget.OutlineCursorWiresOn()
        self.widget.GetSelectedOutlineProperty().SetColor(get_color("red3"))
        self.widget.GetSelectedHandleProperty().SetColor(get_color("red5"))

        self.widget.GetOutlineProperty().SetColor(c)
        self.widget.GetOutlineProperty().SetOpacity(1)
        self.widget.GetOutlineProperty().SetLineWidth(1)
        self.widget.GetOutlineProperty().LightingOff()

        self.widget.GetSelectedFaceProperty().LightingOff()
        self.widget.GetSelectedFaceProperty().SetOpacity(0.1)

        self.widget.SetPlaceFactor(1.0 + padding)
        self.widget.SetInputData(poly)
        self.widget.PlaceWidget()
        if delayed:
            self.widget.AddObserver("EndInteractionEvent", self._select_polygons)
        else:
            self.widget.AddObserver("InteractionEvent", self._select_polygons)

    def _select_polygons(self, vobj, event):
        vobj.GetPlanes(self._implicit_func)

    def _keypress(self, vobj, event):
        if vobj.GetKeySym() == "r":  # reset planes
            self._implicit_func.SetBounds(self._init_bounds)
            self.widget.GetPlanes(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "u":
            self.invert()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "s": # Ctrl+s to save mesh
            if self.widget.GetInteractor():
                if self.widget.GetInteractor().GetControlKey():
                    self.mesh.write("vedo_clipped.vtk")
                    printc(":save: saved mesh to vedo_clipped.vtk")


class SphereCutter(vtki.vtkSphereWidget, BaseCutter):
    """
    Create a box widget to cut away parts of a Mesh.
    """
    def __init__(
            self,
            mesh,
            invert=False,
            can_translate=True,
            can_scale=True,
            origin=(),
            radius=0,
            res=60,
            delayed=False,
            c='white',
            alpha=0.05,
    ):
        """
        Create a box widget to cut away parts of a Mesh.

        Arguments:
            mesh : Mesh
                the input mesh
            invert : bool
                invert the clipping
            can_translate : bool
                enable translation of the widget
            can_scale : bool
                enable scaling of the widget
            origin : list
                initial position of the sphere widget
            radius : float
                initial radius of the sphere widget
            res : int
                resolution of the sphere widget
            delayed : bool
                if True the cutting callback is delayed until
                when the mouse button is released (useful for large meshes)
            c : color
                color of the box cutter widget
            alpha : float
                transparency of the cut-off part of the input mesh
        """
        super().__init__()

        self.mesh = mesh
        self.remnant = Mesh()
        self.remnant.name = mesh.name + "Remnant"
        self.remnant.pickable(False)

        self._alpha = alpha
        self._keypress_id = None

        self._implicit_func = vtki.new("Sphere")

        if len(origin) == 3:
            self._implicit_func.SetCenter(origin)
        else:
            origin = mesh.center_of_mass()
            self._implicit_func.SetCenter(origin)

        if radius > 0:
            self._implicit_func.SetRadius(radius)
        else:
            radius = mesh.average_size() * 2
            self._implicit_func.SetRadius(radius)

        poly = mesh.dataset
        self.clipper = vtki.new("ClipPolyData")
        self.clipper.GenerateClipScalarsOff()
        self.clipper.SetInputData(poly)
        self.clipper.SetClipFunction(self._implicit_func)
        self.clipper.SetInsideOut(not invert)
        self.clipper.GenerateClippedOutputOn()
        self.clipper.Update()

        self.widget = vtki.vtkSphereWidget()

        self.widget.SetThetaResolution(res*2)
        self.widget.SetPhiResolution(res)
        self.widget.SetRadius(radius)
        self.widget.SetCenter(origin)
        self.widget.SetRepresentation(2)
        self.widget.HandleVisibilityOff()

        self.widget.SetTranslation(can_translate)
        self.widget.SetScale(can_scale)

        self.widget.HandleVisibilityOff()
        self.widget.GetSphereProperty().SetColor(get_color(c))
        self.widget.GetSphereProperty().SetOpacity(0.2)
        self.widget.GetSelectedSphereProperty().SetColor(get_color("red5"))
        self.widget.GetSelectedSphereProperty().SetOpacity(0.2)

        self.widget.SetPlaceFactor(1.0)
        self.widget.SetInputData(poly)
        self.widget.PlaceWidget()
        if delayed:
            self.widget.AddObserver("EndInteractionEvent", self._select_polygons)
        else:
            self.widget.AddObserver("InteractionEvent", self._select_polygons)

    def _select_polygons(self, vobj, event):
        vobj.GetSphere(self._implicit_func)

    def _keypress(self, vobj, event):
        if vobj.GetKeySym() == "r":  # reset planes
            self._implicit_func.SetBounds(self._init_bounds)
            self.widget.GetPlanes(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "u":
            self.invert()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "s": # Ctrl+s to save mesh
            if self.widget.GetInteractor():
                if self.widget.GetInteractor().GetControlKey():
                    self.mesh.write("vedo_clipped.vtk")
                    printc(":save: saved mesh to vedo_clipped.vtk")

    @property
    def center(self):
        """Get the center of the sphere."""
        return np.array(self.widget.GetCenter())
    
    @property
    def radius(self):
        """Get the radius of the sphere."""
        return self.widget.GetRadius()
    
    @center.setter
    def center(self, value):
        """Set the center of the sphere."""
        self.widget.SetCenter(value)

    @radius.setter
    def radius(self, value):
        """Set the radius of the sphere."""
        self.widget.SetRadius(value)


#####################################################################
class RendererFrame(vtki.vtkActor2D):
    """
    Add a line around the renderer subwindow.
    """

    def __init__(self, c="k", alpha=None, lw=None, padding=None):
        """
        Add a line around the renderer subwindow.

        Arguments:
            c : (color)
                color of the line.
            alpha : (float)
                opacity.
            lw : (int)
                line width in pixels.
            padding : (int)
                padding in pixel units.
        """

        if lw is None:
            lw = settings.renderer_frame_width
        if lw == 0:
            return None

        if alpha is None:
            alpha = settings.renderer_frame_alpha

        if padding is None:
            padding = settings.renderer_frame_padding

        c = get_color(c)

        ppoints = vtki.vtkPoints()  # Generate the polyline
        xy = 1 - padding
        psqr = [[padding, padding], [padding, xy], [xy, xy], [xy, padding], [padding, padding]]
        for i, pt in enumerate(psqr):
            ppoints.InsertPoint(i, pt[0], pt[1], 0)
        lines = vtki.vtkCellArray()
        lines.InsertNextCell(len(psqr))
        for i in range(len(psqr)):
            lines.InsertCellPoint(i)
        pd = vtki.vtkPolyData()
        pd.SetPoints(ppoints)
        pd.SetLines(lines)

        mapper = vtki.new("PolyDataMapper2D")
        mapper.SetInputData(pd)
        cs = vtki.new("Coordinate")
        cs.SetCoordinateSystemToNormalizedViewport()
        mapper.SetTransformCoordinate(cs)

        super().__init__()

        self.GetPositionCoordinate().SetValue(0, 0)
        self.GetPosition2Coordinate().SetValue(1, 1)
        self.SetMapper(mapper)
        self.GetProperty().SetColor(c)
        self.GetProperty().SetOpacity(alpha)
        self.GetProperty().SetLineWidth(lw)

#####################################################################
class ProgressBarWidget(vtki.vtkActor2D):
    """
    Add a progress bar in the rendering window.
    """
    def __init__(self, n=None, c="blue5", alpha=0.8, lw=10, autohide=True):
        """
        Add a progress bar window.

        Arguments:
            n : (int)
                number of iterations.
                If None, you need to call `update(fraction)` manually.
            c : (color)
                color of the line.
            alpha : (float)
                opacity of the line.
            lw : (int)
                line width in pixels.
            autohide : (bool)
                if True, hide the progress bar when completed.
        """
        self.n = 0
        self.iterations = n
        self.autohide = autohide

        ppoints = vtki.vtkPoints()  # Generate the line
        psqr = [[0, 0, 0], [1, 0, 0]]
        for i, pt in enumerate(psqr):
            ppoints.InsertPoint(i, *pt)
        lines = vtki.vtkCellArray()
        lines.InsertNextCell(len(psqr))
        for i in range(len(psqr)):
            lines.InsertCellPoint(i)
        pd = vtki.vtkPolyData()
        pd.SetPoints(ppoints)
        pd.SetLines(lines)
        self.dataset = pd

        mapper = vtki.new("PolyDataMapper2D")
        mapper.SetInputData(pd)
        cs = vtki.vtkCoordinate()
        cs.SetCoordinateSystemToNormalizedViewport()
        mapper.SetTransformCoordinate(cs)

        super().__init__()

        self.SetMapper(mapper)
        self.GetProperty().SetOpacity(alpha)
        self.GetProperty().SetColor(get_color(c))
        self.GetProperty().SetLineWidth(lw*2)


    def lw(self, value):
        """Set width."""
        self.GetProperty().SetLineWidth(value*2)
        return self

    def c(self, color):
        """Set color."""
        c = get_color(color)
        self.GetProperty().SetColor(c)
        return self

    def alpha(self, value):
        """Set opacity."""
        self.GetProperty().SetOpacity(value)
        return self

    def update(self, fraction=None):
        """Update progress bar to fraction of the window width."""
        if fraction is None:
            if self.iterations is None:
                vedo.printc("Error in ProgressBarWindow: must specify iterations", c='r')
                return self
            self.n += 1
            fraction = self.n / self.iterations

        if fraction >= 1 and self.autohide:
            fraction = 0

        psqr = [[0, 0, 0], [fraction, 0, 0]]
        vpts = utils.numpy2vtk(psqr, dtype=np.float32)
        self.dataset.GetPoints().SetData(vpts)
        return self

    def reset(self):
        """Reset progress bar."""
        self.n = 0
        self.update(0)
        return self


#####################################################################
class Icon(vtki.vtkOrientationMarkerWidget):
    """
    Add an inset icon mesh into the renderer.
    """

    def __init__(self, mesh, pos=3, size=0.08):
        """
        Arguments:
            pos : (list, int)
                icon position in the range [1-4] indicating one of the 4 corners,
                or it can be a tuple (x,y) as a fraction of the renderer size.
            size : (float)
                size of the icon space as fraction of the window size.

        Examples:
            - [icon.py](https://github.com/marcomusy/vedo/tree/master/examples/other/icon.py)
        """
        super().__init__()

        try:
            self.SetOrientationMarker(mesh.actor)
        except AttributeError:
            self.SetOrientationMarker(mesh)

        if utils.is_sequence(pos):
            self.SetViewport(pos[0] - size, pos[1] - size, pos[0] + size, pos[1] + size)
        else:
            if pos < 2:
                self.SetViewport(0, 1 - 2 * size, size * 2, 1)
            elif pos == 2:
                self.SetViewport(1 - 2 * size, 1 - 2 * size, 1, 1)
            elif pos == 3:
                self.SetViewport(0, 0, size * 2, size * 2)
            elif pos == 4:
                self.SetViewport(1 - 2 * size, 0, 1, size * 2)


#####################################################################
def compute_visible_bounds(objs=None):
    """Calculate max objects bounds and sizes."""
    bns = []

    if objs is None:
        objs = vedo.plotter_instance.actors
    elif not utils.is_sequence(objs):
        objs = [objs]

    actors = [ob.actor for ob in objs if hasattr(ob, 'actor') and ob.actor]

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
            vbb = list(vedo.plotter_instance.renderer.ComputeVisiblePropBounds())
            max_bns = vbb
            min_bns = vbb
        sizes = np.array(
            [max_bns[1] - min_bns[0], max_bns[3] - min_bns[2], max_bns[5] - min_bns[4]]
        )
        return [vbb, sizes, min_bns, max_bns]
    except:
        return [[0, 0, 0, 0, 0, 0], [0, 0, 0], 0, 0]


#####################################################################
def Ruler3D(
    p1,
    p2,
    units_scale=1,
    label="",
    s=None,
    font=None,
    italic=0,
    prefix="",
    units="",  # eg.'μm'
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

    Arguments:
        label : (str)
            alternative fixed label to be shown
        units_scale : (float)
            factor to scale units (e.g. μm to mm)
        s : (float)
            size of the label
        font : (str)
            font face.  Check [available fonts here](https://vedo.embl.es/fonts).
        italic : (float)
            italicness of the font in the range [0,1]
        units : (str)
            string to be appended to the numeric value
        lw : (int)
            line width in pixel units
        precision : (int)
            nr of significant digits to be shown
        label_rotation : (float)
            initial rotation of the label around the z-axis
        axis_rotation : (float)
            initial rotation of the line around the main axis
        tick_angle : (float)
            initial rotation of the line around the main axis

    Examples:
        - [goniometer.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/goniometer.py)

        ![](https://vedo.embl.es/images/pyplot/goniometer.png)
    """

    if units_scale != 1.0 and units == "":
        raise ValueError(
            "When setting 'units_scale' to a value other than 1, "
            + "a 'units' arguments must be specified."
        )

    try:
        p1 = p1.pos()
    except AttributeError:
        pass

    try:
        p2 = p2.pos()
    except AttributeError:
        pass

    if len(p1) == 2:
        p1 = [p1[0], p1[1], 0.0]
    if len(p2) == 2:
        p2 = [p2[0], p2[1], 0.0]
    

    p1, p2 = np.asarray(p1), np.asarray(p2)
    q1, q2 = [0, 0, 0], [utils.mag(p2 - p1), 0, 0]
    q1, q2 = np.array(q1), np.array(q2)
    v = q2 - q1
    d = utils.mag(v) * units_scale

    pos = np.array(p1)
    p1 = p1 - pos
    p2 = p2 - pos

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

    lb = shapes.Text3D(label, s=s, font=font, italic=italic, justify="center")
    if label_rotation:
        lb.rotate_z(label_rotation)
    lb.pos((q1 + q2) / 2)

    x0, x1 = lb.xbounds()
    gap = [(x1 - x0) / 2, 0, 0]
    pc1 = (v / 2 - gap) * 0.9 + q1
    pc2 = q2 - (v / 2 - gap) * 0.9

    lc1 = shapes.Line(q1 - v / 50, pc1).lw(lw)
    lc2 = shapes.Line(q2 + v / 50, pc2).lw(lw)

    zs = np.array([0, d / 50 * (1 / units_scale), 0])
    ml1 = shapes.Line(-zs, zs).lw(lw)
    ml2 = shapes.Line(-zs, zs).lw(lw)
    ml1.rotate_z(tick_angle - 90).pos(q1)
    ml2.rotate_z(tick_angle - 90).pos(q2)

    c1 = shapes.Circle(q1, r=d / 180 * (1 / units_scale), res=24)
    c2 = shapes.Circle(q2, r=d / 180 * (1 / units_scale), res=24)

    macts = merge(lb, lc1, lc2, c1, c2, ml1, ml2)
    macts.c(c).alpha(alpha)
    macts.properties.SetLineWidth(lw)
    macts.properties.LightingOff()
    macts.actor.UseBoundsOff()
    macts.rotate_x(axis_rotation)
    macts.reorient(q2 - q1, p2 - p1)
    macts.pos(pos)
    macts.bc("tomato").pickable(False)
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
    xaxis_rotation=0,
    yaxis_rotation=0,
    zaxis_rotation=0,
    xycross=True,
):
    """
    A 3D ruler axes to indicate the sizes of the input scene or object.

    Arguments:
        xtitle : (str)
            name of the axis or title
        xlabel : (str)
            alternative fixed label to be shown instead of the distance
        s : (float)
            size of the label
        font : (str)
            font face. Check [available fonts here](https://vedo.embl.es/fonts).
        italic : (float)
            italicness of the font in the range [0,1]
        units : (str)
            string to be appended to the numeric value
        lw : (int)
            line width in pixel units
        precision : (int)
            nr of significant digits to be shown
        label_rotation : (float)
            initial rotation of the label around the z-axis
        [x,y,z]axis_rotation : (float)
            initial rotation of the line around the main axis in degrees
        xycross : (bool)
            show two back crossing lines in the xy plane

    Examples:
        - [goniometer.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/goniometer.py)
    """
    if utils.is_sequence(inputobj):
        x0, x1, y0, y1, z0, z1 = inputobj
    else:
        x0, x1, y0, y1, z0, z1 = inputobj.bounds()
    dx, dy, dz = (y1 - y0) * xpadding, (x1 - x0) * ypadding, (y1 - y0) * zpadding
    d = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2 + (z1 - z0) ** 2)

    if not d:
        return None

    if s is None:
        s = d / 75

    acts, rx, ry = [], None, None
    if xtitle is not None and (x1 - x0) / d > 0.1:
        rx = Ruler3D(
            [x0, y0 - dx, z0],
            [x1, y0 - dx, z0],
            s=s,
            font=font,
            precision=precision,
            label_rotation=label_rotation,
            axis_rotation=xaxis_rotation,
            lw=lw,
            italic=italic,
            prefix=xtitle,
            label=xlabel,
            units=units,
        )
        acts.append(rx)

    if ytitle is not None and (y1 - y0) / d > 0.1:
        ry = Ruler3D(
            [x1 + dy, y0, z0],
            [x1 + dy, y1, z0],
            s=s,
            font=font,
            precision=precision,
            label_rotation=label_rotation,
            axis_rotation=yaxis_rotation,
            lw=lw,
            italic=italic,
            prefix=ytitle,
            label=ylabel,
            units=units,
        )
        acts.append(ry)

    if ztitle is not None and (z1 - z0) / d > 0.1:
        rz = Ruler3D(
            [x0 - dy, y0 + dz, z0],
            [x0 - dy, y0 + dz, z1],
            s=s,
            font=font,
            precision=precision,
            label_rotation=label_rotation,
            axis_rotation=zaxis_rotation + 90,
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
    macts.actor.UseBoundsOff()
    macts.actor.PickableOff()
    return macts


#####################################################################
class Ruler2D(vtki.vtkAxisActor2D):
    """
    Create a ruler with tick marks, labels and a title.
    """

    def __init__(
        self,
        lw=2,
        ticks=True,
        labels=False,
        c="k",
        alpha=1,
        title="",
        font="Calco",
        font_size=24,
        bc=None,
    ):
        """
        Create a ruler with tick marks, labels and a title.

        Ruler2D is a 2D actor; that is, it is drawn on the overlay
        plane and is not occluded by 3D geometry.
        To use this class, specify two points defining the start and end
        with update_points() as 3D points.

        This class decides decides how to create reasonable tick
        marks and labels.

        Labels are drawn on the "right" side of the axis.
        The "right" side is the side of the axis on the right.
        The way the labels and title line up with the axis and tick marks
        depends on whether the line is considered horizontal or vertical.

        Arguments:
            lw : (int)
                width of the line in pixel units
            ticks : (bool)
                control if drawing the tick marks
            labels : (bool)
                control if drawing the numeric labels
            c : (color)
                color of the object
            alpha : (float)
                opacity of the object
            title : (str)
                title of the ruler
            font : (str)
                font face name. Check [available fonts here](https://vedo.embl.es/fonts).
            font_size : (int)
                font size
            bc : (color)
                background color of the title

        Example:
            ```python
            from vedo  import *
            plt = Plotter(axes=1, interactive=False)
            plt.show(Cube())
            rul = Ruler2D()
            rul.set_points([0,0,0], [0.5,0.5,0.5])
            plt.add(rul)
            plt.interactive().close()
            ```
            ![](https://vedo.embl.es/images/feats/dist_tool.png)
        """
        super().__init__()

        plt = vedo.plotter_instance
        if not plt:
            vedo.logger.error("Ruler2D need to initialize Plotter first.")
            raise RuntimeError()

        self.p0 = [0, 0, 0]
        self.p1 = [0, 0, 0]
        self.distance = 0
        self.title = title

        prop = self.GetProperty()
        tprop = self.GetTitleTextProperty()

        self.SetTitle(title)
        self.SetNumberOfLabels(9)

        if not font:
            font = settings.default_font
        if font.lower() == "courier":
            tprop.SetFontFamilyToCourier()
        elif font.lower() == "times":
            tprop.SetFontFamilyToTimes()
        elif font.lower() == "arial":
            tprop.SetFontFamilyToArial()
        else:
            tprop.SetFontFamily(vtki.VTK_FONT_FILE)
            tprop.SetFontFile(utils.get_font_path(font))
        tprop.SetFontSize(font_size)
        tprop.BoldOff()
        tprop.ItalicOff()
        tprop.ShadowOff()
        tprop.SetColor(get_color(c))
        tprop.SetOpacity(alpha)
        if bc is not None:
            bc = get_color(bc)
            tprop.SetBackgroundColor(bc)
            tprop.SetBackgroundOpacity(alpha)

        lprop = vtki.vtkTextProperty()
        lprop.ShallowCopy(tprop)
        self.SetLabelTextProperty(lprop)

        self.SetLabelFormat("%0.3g")
        self.SetTickVisibility(ticks)
        self.SetLabelVisibility(labels)
        prop.SetLineWidth(lw)
        prop.SetColor(get_color(c))

        self.renderer = plt.renderer
        self.cid = plt.interactor.AddObserver("RenderEvent", self._update_viz, 1.0)

    def color(self, c):
        """Assign a new color."""
        c = get_color(c)
        self.GetTitleTextProperty().SetColor(c)
        self.GetLabelTextProperty().SetColor(c)
        self.GetProperty().SetColor(c)
        return self

    def off(self):
        """Switch off the ruler completely."""
        self.renderer.RemoveObserver(self.cid)
        self.renderer.RemoveActor(self)

    def set_points(self, p0, p1):
        """Set new values for the ruler start and end points."""
        self.p0 = np.asarray(p0)
        self.p1 = np.asarray(p1)
        self._update_viz(0, 0)
        return self

    def _update_viz(self, evt, name):
        ren = self.renderer
        view_size = np.array(ren.GetSize())

        ren.SetWorldPoint(*self.p0, 1)
        ren.WorldToDisplay()
        disp_point1 = ren.GetDisplayPoint()[:2]
        disp_point1 = np.array(disp_point1) / view_size

        ren.SetWorldPoint(*self.p1, 1)
        ren.WorldToDisplay()
        disp_point2 = ren.GetDisplayPoint()[:2]
        disp_point2 = np.array(disp_point2) / view_size

        self.SetPoint1(*disp_point1)
        self.SetPoint2(*disp_point2)
        self.distance = np.linalg.norm(self.p1 - self.p0)
        self.SetRange(0, self.distance)
        if not self.title:
            self.SetTitle(utils.precision(self.distance, 3))


#####################################################################
class DistanceTool(Group):
    """
    Create a tool to measure the distance between two clicked points.
    """

    def __init__(self, plotter=None, c="k", lw=2):
        """
        Create a tool to measure the distance between two clicked points.

        Example:
            ```python
            from vedo import *
            mesh = ParametricShape("RandomHills").c("red5")
            plt = Plotter(axes=1)
            dtool = DistanceTool()
            dtool.on()
            plt.show(mesh, dtool)
            dtool.off()
            ```
            ![](https://vedo.embl.es/images/feats/dist_tool.png)
        """
        super().__init__()

        self.p0 = [0, 0, 0]
        self.p1 = [0, 0, 0]
        self.distance = 0
        if plotter is None:
            plotter = vedo.plotter_instance
        self.plotter = plotter
        self.callback = None
        self.cid = None
        self.color = c
        self.linewidth = lw
        self.toggle = True
        self.ruler = None
        self.title = ""

    def on(self):
        """Switch tool on."""
        self.cid = self.plotter.add_callback("click", self._onclick)
        self.VisibilityOn()
        self.plotter.render()
        return self

    def off(self):
        """Switch tool off."""
        self.plotter.remove_callback(self.cid)
        self.VisibilityOff()
        self.ruler.off()
        self.plotter.render()

    def _onclick(self, event):
        if not event.actor:
            return

        self.clear()

        acts = []
        if self.toggle:
            self.p0 = event.picked3d
            acts.append(Point(self.p0, c=self.color))
        else:
            self.p1 = event.picked3d
            self.distance = np.linalg.norm(self.p1 - self.p0)
            acts.append(Point(self.p0, c=self.color))
            acts.append(Point(self.p1, c=self.color))
            self.ruler = Ruler2D(c=self.color)
            self.ruler.set_points(self.p0, self.p1)
            acts.append(self.ruler)

            if self.callback is not None:
                self.callback(event)

        self += acts
        self.toggle = not self.toggle


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
        htitle_color=None, htitle_backface_color=None,
        htitle_justify='bottom-left',
        htitle_rotation=0,
        htitle_offset=(0, 0.01, 0),
        xtitle_position=0.95, ytitle_position=0.95, ztitle_position=0.95,
        # xtitle_offset can be a list (dx,dy,dz)
        xtitle_offset=0.025,  ytitle_offset=0.0275, ztitle_offset=0.02,
        xtitle_justify=None,  ytitle_justify=None,  ztitle_justify=None,
        # xtitle_rotation can be a list (rx,ry,rz)
        xtitle_rotation=0, ytitle_rotation=0, ztitle_rotation=0,
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
        xlabel_backface_color=None, ylabel_backface_color=None, zlabel_backface_color=None,
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
    ):
    """
    Draw axes for the input object.
    Check [available fonts here](https://vedo.embl.es/fonts).

    Returns an `vedo.Assembly` object.

    Parameters
    ----------

    - `xtitle`,                 ['x'], x-axis title text
    - `xrange`,                [None], x-axis range in format (xmin, ymin), default is automatic.
    - `number_of_divisions`,   [None], approximate number of divisions on the longest axis
    - `axes_linewidth`,           [1], width of the axes lines
    - `grid_linewidth`,           [1], width of the grid lines
    - `title_depth`,              [0], extrusion fractional depth of title text
    - `x_values_and_labels`        [], assign custom tick positions and labels [(pos1, label1), ...]
    - `xygrid`,                [True], show a gridded wall on plane xy
    - `yzgrid`,                [True], show a gridded wall on plane yz
    - `zxgrid`,                [True], show a gridded wall on plane zx
    - `yzgrid2`,              [False], show yz plane on opposite side of the bounding box
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
    - `htitle_backface_color`, [None], header title color on its backface
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
    - `xhighlight_zero_color`, [auto], color of the line highlighting the zero position
    - `xtick_length`,         [0.005], radius of the major ticks
    - `xtick_thickness`,     [0.0025], thickness of the major ticks along their axis
    - `xminor_ticks`,             [1], number of minor ticks between two major ticks
    - `xlabel_color`,     [automatic], color of numeric labels and ticks
    - `xlabel_backface_color`, [auto], back face color of numeric labels and ticks
    - `xlabel_size`,          [0.015], size of the numeric labels along axis
    - `xlabel_rotation`,     [0,list], numeric labels rotation (can be a list of 3 rotations)
    - `xlabel_offset`,     [0.8,list], offset of the numeric labels (can be a list of 3 offsets)
    - `xlabel_justify`,        [None], choose the origin of the bounding box of labels
    - `xaxis_rotation`,           [0], rotate the X axis elements (ticks and labels) around this same axis
    - `xyshift`                 [0.0], slide the xy-plane along z (the range is [0,1])
    - `xshift_along_y`          [0.0], slide x-axis along the y-axis (the range is [0,1])
    - `tip_size`,              [0.01], size of the arrow tip as a fraction of the bounding box diagonal
    - `limit_ratio`,           [0.04], below this ratio don't plot smaller axis
    - `x_use_bounds`,          [True], keep into account space occupied by labels when setting camera
    - `x_inverted`,           [False], invert labels order and direction (only visually!)
    - `use_global`,           [False], try to compute the global bounding box of visible actors

    Example:
        ```python
        from vedo import Axes, Box, show
        box = Box(pos=(1,2,3), length=8, width=9, height=7).alpha(0.1)
        axs = Axes(box, c='k')  # returns an Assembly object
        for a in axs.unpack():
            print(a.name)
        show(box, axs).close()
        ```
        ![](https://vedo.embl.es/images/feats/axes1.png)

    Examples:
        - [custom_axes1.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/custom_axes1.py)
        - [custom_axes2.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/custom_axes2.py)
        - [custom_axes3.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/custom_axes3.py)
        - [custom_axes4.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/custom_axes4.py)

        ![](https://vedo.embl.es/images/pyplot/customAxes3.png)
    """
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

    # Check if obj has bounds, if so use those
    if obj is not None:
        try:
            bb = obj.bounds()
        except AttributeError:
            try:
                bb = obj.GetBounds()
                if xrange is None: xrange = (bb[0], bb[1])
                if yrange is None: yrange = (bb[2], bb[3])
                if zrange is None: zrange = (bb[4], bb[5])
                obj = None # dont need it anymore
            except AttributeError:
                pass
        if utils.is_sequence(obj) and len(obj)==6 and utils.is_number(obj[0]):
            # passing a list of numeric bounds
            if xrange is None: xrange = (obj[0], obj[1])
            if yrange is None: yrange = (obj[2], obj[3])
            if zrange is None: zrange = (obj[4], obj[5])

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
                return None  ###########################################

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
        axlinex.shift([0, zxshift*dy + xshift_along_y*dy, xyshift*dz + xshift_along_z*dz])
        axlinex.name = 'xAxis'
        lines.append(axlinex)
    if ytitle:
        axliney = shapes.Line([0,0,0], [0,dy,0], c=yline_color, lw=axes_linewidth)
        axliney.shift([yzshift*dx + yshift_along_x*dx, 0, xyshift*dz + yshift_along_z*dz])
        axliney.name = 'yAxis'
        lines.append(axliney)
    if ztitle:
        axlinez = shapes.Line([0,0,0], [0,0,dz], c=zline_color, lw=axes_linewidth)
        axlinez.shift([yzshift*dx + zshift_along_x*dx, zxshift*dy + zshift_along_y*dy, 0])
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
            if xyshift: gxy.shift([0,0,xyshift*dz])
            elif tol:   gxy.shift([0,0,-tol*gscale])
            gxy.name = "xyGrid"
            grids.append(gxy)
        if grid_linewidth:
            gxy_lines = shapes.Grid(s=(xticks_float, yticks_float))
            gxy_lines.c(xyplane_color).lw(grid_linewidth).alpha(xyalpha)
            if xyshift: gxy_lines.shift([0,0,xyshift*dz])
            elif tol:   gxy_lines.shift([0,0,-tol*gscale])
            gxy_lines.name = "xyGridLines"
            grids.append(gxy_lines)

    if yzgrid and ytitle and ztitle:
        if not yzgrid_transparent:
            gyz = shapes.Grid(s=(zticks_float, yticks_float))
            gyz.alpha(yzalpha).c(yzplane_color).lw(0).rotate_y(-90)
            if yzshift: gyz.shift([yzshift*dx,0,0])
            elif tol:   gyz.shift([-tol*gscale,0,0])
            gyz.name = "yzGrid"
            grids.append(gyz)
        if grid_linewidth:
            gyz_lines = shapes.Grid(s=(zticks_float, yticks_float))
            gyz_lines.c(yzplane_color).lw(grid_linewidth).alpha(yzalpha).rotate_y(-90)
            if yzshift: gyz_lines.shift([yzshift*dx,0,0])
            elif tol:   gyz_lines.shift([-tol*gscale,0,0])
            gyz_lines.name = "yzGridLines"
            grids.append(gyz_lines)

    if zxgrid and ztitle and xtitle:
        if not zxgrid_transparent:
            gzx = shapes.Grid(s=(xticks_float, zticks_float))
            gzx.alpha(zxalpha).c(zxplane_color).lw(0).rotate_x(90)
            if zxshift: gzx.shift([0,zxshift*dy,0])
            elif tol:   gzx.shift([0,-tol*gscale,0])
            gzx.name = "zxGrid"
            grids.append(gzx)
        if grid_linewidth:
            gzx_lines = shapes.Grid(s=(xticks_float, zticks_float))
            gzx_lines.c(zxplane_color).lw(grid_linewidth).alpha(zxalpha).rotate_x(90)
            if zxshift: gzx_lines.shift([0,zxshift*dy,0])
            elif tol:   gzx_lines.shift([0,-tol*gscale,0])
            gzx_lines.name = "zxGridLines"
            grids.append(gzx_lines)

    # Grid2
    if xygrid2 and xtitle and ytitle:
        if not xygrid2_transparent:
            gxy2 = shapes.Grid(s=(xticks_float, yticks_float)).z(dz)
            gxy2.alpha(xyalpha).c(xyplane_color).lw(0)
            gxy2.shift([0,tol*gscale,0])
            gxy2.name = "xyGrid2"
            grids.append(gxy2)
        if grid_linewidth:
            gxy2_lines = shapes.Grid(s=(xticks_float, yticks_float)).z(dz)
            gxy2_lines.c(xyplane_color).lw(grid_linewidth).alpha(xyalpha)
            gxy2_lines.shift([0,tol*gscale,0])
            gxy2_lines.name = "xygrid2Lines"
            grids.append(gxy2_lines)

    if yzgrid2 and ytitle and ztitle:
        if not yzgrid2_transparent:
            gyz2 = shapes.Grid(s=(zticks_float, yticks_float))
            gyz2.alpha(yzalpha).c(yzplane_color).lw(0)
            gyz2.rotate_y(-90).x(dx).shift([tol*gscale,0,0])
            gyz2.name = "yzGrid2"
            grids.append(gyz2)
        if grid_linewidth:
            gyz2_lines = shapes.Grid(s=(zticks_float, yticks_float))
            gyz2_lines.c(yzplane_color).lw(grid_linewidth).alpha(yzalpha)
            gyz2_lines.rotate_y(-90).x(dx).shift([tol*gscale,0,0])
            gyz2_lines.name = "yzGrid2Lines"
            grids.append(gyz2_lines)

    if zxgrid2 and ztitle and xtitle:
        if not zxgrid2_transparent:
            gzx2 = shapes.Grid(s=(xticks_float, zticks_float))
            gzx2.alpha(zxalpha).c(zxplane_color).lw(0)
            gzx2.rotate_x(90).y(dy).shift([0,tol*gscale,0])
            gzx2.name = "zxGrid2"
            grids.append(gzx2)
        if grid_linewidth:
            gzx2_lines = shapes.Grid(s=(xticks_float, zticks_float))
            gzx2_lines.c(zxplane_color).lw(grid_linewidth).alpha(zxalpha)
            gzx2_lines.rotate_x(90).y(dy).shift([0,tol*gscale,0])
            gzx2_lines.name = "zxGrid2Lines"
            grids.append(gzx2_lines)

    ################################################ frame lines
    framelines = []
    if xyframe_line and xtitle and ytitle:
        if not xyframe_color:
            xyframe_color = xygrid_color
        frxy = shapes.Line([[0,dy,0],[dx,dy,0],[dx,0,0],[0,0,0],[0,dy,0]],
                           c=xyframe_color, lw=xyframe_line)
        frxy.shift([0,0,xyshift*dz])
        frxy.name = 'xyFrameLine'
        framelines.append(frxy)
    if yzframe_line and ytitle and ztitle:
        if not yzframe_color:
            yzframe_color = yzgrid_color
        fryz = shapes.Line([[0,0,dz],[0,dy,dz],[0,dy,0],[0,0,0],[0,0,dz]],
                           c=yzframe_color, lw=yzframe_line)
        fryz.shift([yzshift*dx,0,0])
        fryz.name = 'yzFrameLine'
        framelines.append(fryz)
    if zxframe_line and ztitle and xtitle:
        if not zxframe_color:
            zxframe_color = zxgrid_color
        frzx = shapes.Line([[0,0,dz],[dx,0,dz],[dx,0,0],[0,0,0],[0,0,dz]],
                           c=zxframe_color, lw=zxframe_line)
        frzx.shift([0,zxshift*dy,0])
        frzx.name = 'zxFrameLine'
        framelines.append(frzx)

    ################################################ zero lines highlights
    highlights = []
    if xygrid and xtitle and ytitle:
        if xhighlight_zero and min_bns[0] <= 0 and max_bns[1] > 0:
            xhl = -min_bns[0]
            hxy = shapes.Line([xhl,0,0], [xhl,dy,0], c=xhighlight_zero_color)
            hxy.alpha(np.sqrt(xyalpha)).lw(grid_linewidth*2)
            hxy.shift([0,0,xyshift*dz])
            hxy.name = "xyHighlightZero"
            highlights.append(hxy)
        if yhighlight_zero and min_bns[2] <= 0 and max_bns[3] > 0:
            yhl = -min_bns[2]
            hyx = shapes.Line([0,yhl,0], [dx,yhl,0], c=yhighlight_zero_color)
            hyx.alpha(np.sqrt(yzalpha)).lw(grid_linewidth*2)
            hyx.shift([0,0,xyshift*dz])
            hyx.name = "yxHighlightZero"
            highlights.append(hyx)

    if yzgrid and ytitle and ztitle:
        if yhighlight_zero and min_bns[2] <= 0 and max_bns[3] > 0:
            yhl = -min_bns[2]
            hyz = shapes.Line([0,yhl,0], [0,yhl,dz], c=yhighlight_zero_color)
            hyz.alpha(np.sqrt(yzalpha)).lw(grid_linewidth*2)
            hyz.shift([yzshift*dx,0,0])
            hyz.name = "yzHighlightZero"
            highlights.append(hyz)
        if zhighlight_zero and min_bns[4] <= 0 and max_bns[5] > 0:
            zhl = -min_bns[4]
            hzy = shapes.Line([0,0,zhl], [0,dy,zhl], c=zhighlight_zero_color)
            hzy.alpha(np.sqrt(yzalpha)).lw(grid_linewidth*2)
            hzy.shift([yzshift*dx,0,0])
            hzy.name = "zyHighlightZero"
            highlights.append(hzy)

    if zxgrid and ztitle and xtitle:
        if zhighlight_zero and min_bns[4] <= 0 and max_bns[5] > 0:
            zhl = -min_bns[4]
            hzx = shapes.Line([0,0,zhl], [dx,0,zhl], c=zhighlight_zero_color)
            hzx.alpha(np.sqrt(zxalpha)).lw(grid_linewidth*2)
            hzx.shift([0,zxshift*dy,0])
            hzx.name = "zxHighlightZero"
            highlights.append(hzx)
        if xhighlight_zero and min_bns[0] <= 0 and max_bns[1] > 0:
            xhl = -min_bns[0]
            hxz = shapes.Line([xhl,0,0], [xhl,0,dz], c=xhighlight_zero_color)
            hxz.alpha(np.sqrt(zxalpha)).lw(grid_linewidth*2)
            hxz.shift([0,zxshift*dy,0])
            hxz.name = "xzHighlightZero"
            highlights.append(hxz)

    ################################################ arrow cone
    cones = []

    if tip_size:

        if xtitle:
            if x_inverted:
                cx = shapes.Cone(
                    r=tip_size, height=tip_size * 2, axis=(-1, 0, 0), c=xline_color, res=12
                )
            else:
                cx = shapes.Cone((dx,0,0), r=tip_size, height=tip_size*2,
                                 axis=(1,0,0), c=xline_color, res=12)
            T = LinearTransform()
            T.translate([0, zxshift*dy + xshift_along_y*dy, xyshift*dz + xshift_along_z*dz])
            cx.apply_transform(T)
            cx.name = "xTipCone"
            cones.append(cx)

        if ytitle:
            if y_inverted:
                cy = shapes.Cone(r=tip_size, height=tip_size*2,
                                 axis=(0,-1,0), c=yline_color, res=12)
            else:
                cy = shapes.Cone((0,dy,0), r=tip_size, height=tip_size*2,
                                 axis=(0,1,0), c=yline_color, res=12)
            T = LinearTransform()
            T.translate([yzshift*dx + yshift_along_x*dx, 0, xyshift*dz + yshift_along_z*dz])
            cy.apply_transform(T)
            cy.name = "yTipCone"
            cones.append(cy)

        if ztitle:
            if z_inverted:
                cz = shapes.Cone(r=tip_size, height=tip_size*2,
                                 axis=(0,0,-1), c=zline_color, res=12)
            else:
                cz = shapes.Cone((0,0,dz), r=tip_size, height=tip_size*2,
                                 axis=(0,0,1), c=zline_color, res=12)
            T = LinearTransform()
            T.translate([yzshift*dx + zshift_along_x*dx, zxshift*dy + zshift_along_y*dy, 0])
            cz.apply_transform(T)
            cz.name = "zTipCone"
            cones.append(cz)

    ################################################################# MAJOR ticks
    majorticks, minorticks = [], []
    xticks, yticks, zticks = [], [], []
    if show_ticks:
        if xtitle:
            tick_thickness = xtick_thickness * gscale / 2
            tick_length = xtick_length * gscale / 2
            for i in range(1, len(xticks_float) - 1):
                v1 = (xticks_float[i] - tick_thickness, -tick_length, 0)
                v2 = (xticks_float[i] + tick_thickness, tick_length, 0)
                xticks.append(shapes.Rectangle(v1, v2))
            if len(xticks) > 1:
                xmajticks = merge(xticks).c(xlabel_color)
                T = LinearTransform()
                T.rotate_x(xaxis_rotation)
                T.translate([0, zxshift*dy + xshift_along_y*dy, xyshift*dz + xshift_along_z*dz])
                xmajticks.apply_transform(T)
                xmajticks.name = "xMajorTicks"
                majorticks.append(xmajticks)
        if ytitle:
            tick_thickness = ytick_thickness * gscale / 2
            tick_length = ytick_length * gscale / 2
            for i in range(1, len(yticks_float) - 1):
                v1 = (-tick_length, yticks_float[i] - tick_thickness, 0)
                v2 = ( tick_length, yticks_float[i] + tick_thickness, 0)
                yticks.append(shapes.Rectangle(v1, v2))
            if len(yticks) > 1:
                ymajticks = merge(yticks).c(ylabel_color)
                T = LinearTransform()
                T.rotate_y(yaxis_rotation)
                T.translate([yzshift*dx + yshift_along_x*dx, 0, xyshift*dz + yshift_along_z*dz])
                ymajticks.apply_transform(T)
                ymajticks.name = "yMajorTicks"
                majorticks.append(ymajticks)
        if ztitle:
            tick_thickness = ztick_thickness * gscale / 2
            tick_length = ztick_length * gscale / 2.85
            for i in range(1, len(zticks_float) - 1):
                v1 = (zticks_float[i] - tick_thickness, -tick_length, 0)
                v2 = (zticks_float[i] + tick_thickness,  tick_length, 0)
                zticks.append(shapes.Rectangle(v1, v2))
            if len(zticks) > 1:
                zmajticks = merge(zticks).c(zlabel_color)
                T = LinearTransform()
                T.rotate_y(-90).rotate_z(-45 + zaxis_rotation)
                T.translate([yzshift*dx + zshift_along_x*dx, zxshift*dy + zshift_along_y*dy, 0])
                zmajticks.apply_transform(T)
                zmajticks.name = "zMajorTicks"
                majorticks.append(zmajticks)

        ############################################################# MINOR ticks
        if xtitle and xminor_ticks and len(xticks) > 1:
            tick_thickness = xtick_thickness * gscale / 4
            tick_length = xtick_length * gscale / 4
            xminor_ticks += 1
            ticks = []
            for i in range(1, len(xticks)):
                t0, t1 = xticks[i - 1].pos(), xticks[i].pos()
                dt = t1 - t0
                for j in range(1, xminor_ticks):
                    mt = dt * (j / xminor_ticks) + t0
                    v1 = (mt[0] - tick_thickness, -tick_length, 0)
                    v2 = (mt[0] + tick_thickness, tick_length, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = xticks[0].pos(), xticks[1].pos()
            dt = t1 - t0
            for j in range(1, xminor_ticks):
                mt = t0 - dt * (j / xminor_ticks)
                if mt[0] < 0:
                    break
                v1 = (mt[0] - tick_thickness, -tick_length, 0)
                v2 = (mt[0] + tick_thickness,  tick_length, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = xticks[-2].pos(), xticks[-1].pos()
            dt = t1 - t0
            for j in range(1, xminor_ticks):
                mt = t1 + dt * (j / xminor_ticks)
                if mt[0] > dx:
                    break
                v1 = (mt[0] - tick_thickness, -tick_length, 0)
                v2 = (mt[0] + tick_thickness,  tick_length, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if ticks:
                xminticks = merge(ticks).c(xlabel_color)
                T = LinearTransform()
                T.rotate_x(xaxis_rotation)
                T.translate([0, zxshift*dy + xshift_along_y*dy, xyshift*dz + xshift_along_z*dz])
                xminticks.apply_transform(T)
                xminticks.name = "xMinorTicks"
                minorticks.append(xminticks)

        if ytitle and yminor_ticks and len(yticks) > 1:  ##### y
            tick_thickness = ytick_thickness * gscale / 4
            tick_length = ytick_length * gscale / 4
            yminor_ticks += 1
            ticks = []
            for i in range(1, len(yticks)):
                t0, t1 = yticks[i - 1].pos(), yticks[i].pos()
                dt = t1 - t0
                for j in range(1, yminor_ticks):
                    mt = dt * (j / yminor_ticks) + t0
                    v1 = (-tick_length, mt[1] - tick_thickness, 0)
                    v2 = ( tick_length, mt[1] + tick_thickness, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = yticks[0].pos(), yticks[1].pos()
            dt = t1 - t0
            for j in range(1, yminor_ticks):
                mt = t0 - dt * (j / yminor_ticks)
                if mt[1] < 0:
                    break
                v1 = (-tick_length, mt[1] - tick_thickness, 0)
                v2 = ( tick_length, mt[1] + tick_thickness, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = yticks[-2].pos(), yticks[-1].pos()
            dt = t1 - t0
            for j in range(1, yminor_ticks):
                mt = t1 + dt * (j / yminor_ticks)
                if mt[1] > dy:
                    break
                v1 = (-tick_length, mt[1] - tick_thickness, 0)
                v2 = ( tick_length, mt[1] + tick_thickness, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if ticks:
                yminticks = merge(ticks).c(ylabel_color)
                T = LinearTransform()
                T.rotate_y(yaxis_rotation)
                T.translate([yzshift*dx + yshift_along_x*dx, 0, xyshift*dz + yshift_along_z*dz])
                yminticks.apply_transform(T)
                yminticks.name = "yMinorTicks"
                minorticks.append(yminticks)

        if ztitle and zminor_ticks and len(zticks) > 1:  ##### z
            tick_thickness = ztick_thickness * gscale / 4
            tick_length = ztick_length * gscale / 5
            zminor_ticks += 1
            ticks = []
            for i in range(1, len(zticks)):
                t0, t1 = zticks[i - 1].pos(), zticks[i].pos()
                dt = t1 - t0
                for j in range(1, zminor_ticks):
                    mt = dt * (j / zminor_ticks) + t0
                    v1 = (mt[0] - tick_thickness, -tick_length, 0)
                    v2 = (mt[0] + tick_thickness,  tick_length, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = zticks[0].pos(), zticks[1].pos()
            dt = t1 - t0
            for j in range(1, zminor_ticks):
                mt = t0 - dt * (j / zminor_ticks)
                if mt[0] < 0:
                    break
                v1 = (mt[0] - tick_thickness, -tick_length, 0)
                v2 = (mt[0] + tick_thickness,  tick_length, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = zticks[-2].pos(), zticks[-1].pos()
            dt = t1 - t0
            for j in range(1, zminor_ticks):
                mt = t1 + dt * (j / zminor_ticks)
                if mt[0] > dz:
                    break
                v1 = (mt[0] - tick_thickness, -tick_length, 0)
                v2 = (mt[0] + tick_thickness,  tick_length, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if ticks:
                zminticks = merge(ticks).c(zlabel_color)
                T = LinearTransform()
                T.rotate_y(-90).rotate_z(-45 + zaxis_rotation)
                T.translate([yzshift*dx + zshift_along_x*dx, zxshift*dy + zshift_along_y*dy, 0])
                zminticks.apply_transform(T)
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
                t, s=xlabel_size * text_scale * gscale, font=label_font, justify=jus,
            )
            tb = xlab.ybounds()  # must be ybounds: height of char

            v = (xticks_float[i], 0, 0)
            offs = -np.array([xoffs, yoffs, zoffs]) * (tb[1] - tb[0])

            T = LinearTransform()
            T.rotate_x(xaxis_rotation).rotate_y(yRot).rotate_x(xRot).rotate_z(zRot)
            T.translate(v + offs)
            T.translate([0, zxshift*dy + xshift_along_y*dy, xyshift*dz + xshift_along_z*dz])
            xlab.apply_transform(T)

            xlab.use_bounds(x_use_bounds)

            xlab.c(xlabel_color)
            if xlabel_backface_color is None:
                bfc = 1 - np.array(get_color(xlabel_color))
                xlab.backcolor(bfc)
            xlab.name = f"xNumericLabel{i}"
            labels.append(xlab)

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

            T = LinearTransform()
            T.rotate_y(yaxis_rotation).rotate_x(xRot).rotate_y(yRot).rotate_z(zRot)
            T.translate(v + offs)
            T.translate([yzshift*dx + yshift_along_x*dx, 0, xyshift*dz + yshift_along_z*dz])
            ylab.apply_transform(T)

            ylab.use_bounds(y_use_bounds)

            ylab.c(ylabel_color)
            if ylabel_backface_color is None:
                bfc = 1 - np.array(get_color(ylabel_color))
                ylab.backcolor(bfc)
            ylab.name = f"yNumericLabel{i}"
            labels.append(ylab)

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
            zlab = shapes.Text3D(t, s=zlabel_size*text_scale*gscale, font=label_font, justify=jus)
            tb = zlab.ybounds()  # must be ybounds: height of char

            v = (0, 0, zticks_float[i])
            offs = -np.array([xoffs, yoffs, zoffs]) * (tb[1] - tb[0]) / 1.5
            angle = np.arctan2(dy, dx) * 57.3

            T = LinearTransform()
            T.rotate_x(90 + zRot).rotate_y(-xRot).rotate_z(angle + yRot + zaxis_rotation)
            T.translate(v + offs)
            T.translate([yzshift*dx + zshift_along_x*dx, zxshift*dy + zshift_along_y*dy, 0])
            zlab.apply_transform(T)

            zlab.use_bounds(z_use_bounds)

            zlab.c(zlabel_color)
            if zlabel_backface_color is None:
                bfc = 1 - np.array(get_color(zlabel_color))
                zlab.backcolor(bfc)
            zlab.name = f"zNumericLabel{i}"
            labels.append(zlab)

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
        if xtitle_backface_color is None:
            xtitle_backface_color = 1 - np.array(get_color(xtitle_color))
            xt.backcolor(xtitle_backface_color)

        shift = 0
        if xlab:  # xlab is the last created numeric text label..
            lt0, lt1 = xlab.bounds()[2:4]
            shift = lt1 - lt0

        T = LinearTransform()
        T.rotate_x(xRot).rotate_y(yRot).rotate_z(zRot)
        T.set_position(
            [(xoffs + xtitle_position) * dx,
            -(yoffs + xtick_length / 2) * dy - shift,
            zoffs * dz]
        )
        T.rotate_x(xaxis_rotation)
        T.translate([0, xshift_along_y*dy, xyshift*dz + xshift_along_z*dz])
        xt.apply_transform(T)

        xt.use_bounds(x_use_bounds)
        if xtitle == " ":
            xt.use_bounds(False)
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
        if ytitle_backface_color is None:
            ytitle_backface_color = 1 - np.array(get_color(ytitle_color))
            yt.backcolor(ytitle_backface_color)

        shift = 0
        if ylab:  # this is the last created num label..
            lt0, lt1 = ylab.bounds()[0:2]
            shift = lt1 - lt0

        T = LinearTransform()
        T.rotate_x(xRot).rotate_y(yRot).rotate_z(zRot)
        T.set_position(
            [-(xoffs + ytick_length / 2) * dx - shift,
            (yoffs + ytitle_position) * dy,
            zoffs * dz]
        )
        T.rotate_y(yaxis_rotation)
        T.translate([yshift_along_x*dx, 0, xyshift*dz + yshift_along_z*dz])
        yt.apply_transform(T)

        yt.use_bounds(y_use_bounds)
        if ytitle == " ":
            yt.use_bounds(False)
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
        if ztitle_backface_color is None:
            ztitle_backface_color = 1 - np.array(get_color(ztitle_color))
            zt.backcolor(ztitle_backface_color)

        angle = np.arctan2(dy, dx) * 57.3
        shift = 0
        if zlab:  # this is the last created one..
            lt0, lt1 = zlab.bounds()[0:2]
            shift = lt1 - lt0

        T = LinearTransform()
        T.rotate_x(90 + zRot).rotate_y(-xRot).rotate_z(angle + yRot)
        T.set_position([
            -(ztitle_offset + ztick_length / 5) * dx - shift,
            -(ztitle_offset + ztick_length / 5) * dy - shift,
            ztitle_position * dz]
        )
        T.rotate_z(zaxis_rotation)
        T.translate([zshift_along_x*dx, zxshift*dy + zshift_along_y*dy, 0])
        zt.apply_transform(T)

        zt.use_bounds(z_use_bounds)
        if ztitle == " ":
            zt.use_bounds(False)
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
        if htitle_backface_color is None:
            htitle_backface_color = 1 - np.array(get_color(htitle_color))
            htit.backcolor(htitle_backface_color)
        htit.rotate_x(htitle_rotation)
        wpos = [htitle_offset[0]*dx, (1 + htitle_offset[1])*dy, htitle_offset[2]*dz]
        htit.shift(np.array(wpos) + [0, 0, xyshift*dz])
        htit.name = f"htitle {htitle}"
        titles.append(htit)

    ######
    acts = titles + lines + labels + grids + framelines
    acts += highlights + majorticks + minorticks + cones
    orig = (min_bns[0], min_bns[2], min_bns[4])
    for a in acts:
        a.shift(orig)
        a.actor.PickableOff()
        a.properties.LightingOff()
    asse = Assembly(acts)
    asse.PickableOff()
    asse.name = "Axes"
    return asse


def add_global_axes(axtype=None, c=None, bounds=()):
    """
    Draw axes on scene. Available axes types are

    Parameters
    ----------
    axtype : (int)
        - 0,  no axes,
        - 1,  draw three gray grid walls
        - 2,  show cartesian axes from (0,0,0)
        - 3,  show positive range of cartesian axes from (0,0,0)
        - 4,  show a triad at bottom left
        - 5,  show a cube at bottom left
        - 6,  mark the corners of the bounding box
        - 7,  draw a 3D ruler at each side of the cartesian axes
        - 8,  show the `vtkCubeAxesActor` object
        - 9,  show the bounding box outLine
        - 10, show three circles representing the maximum bounding box
        - 11, show a large grid on the x-y plane (use with zoom=8)
        - 12, show polar axes
        - 13, draw a simple ruler at the bottom of the window
        - 14, show the vtk default `vtkCameraOrientationWidget` object

    Axis type-1 can be fully customized by passing a dictionary `axes=dict()`,
    see `vedo.Axes` for the complete list of options.

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
        return

    if plt.axes_instances[r]:
        return

    ############################################################
    # custom grid walls
    if plt.axes == 1 or plt.axes is True or isinstance(plt.axes, dict):

        if len(bounds) == 6:
            bnds = bounds
            xrange = (bnds[0], bnds[1])
            yrange = (bnds[2], bnds[3])
            zrange = (bnds[4], bnds[5])
        else:
            xrange=None
            yrange=None
            zrange=None

        if isinstance(plt.axes, dict):
            plt.axes.update({"use_global": True})
            # protect from invalid camelCase options from vedo<=2.3
            for k in plt.axes:
                if k.lower() != k:
                    return
            if "xrange" in plt.axes:
                xrange = plt.axes.pop("xrange")
            if "yrange" in plt.axes:
                yrange = plt.axes.pop("yrange")
            if "zrange" in plt.axes:
                zrange = plt.axes.pop("zrange")
            asse = Axes(**plt.axes, xrange=xrange, yrange=yrange, zrange=zrange)
        else:
            asse = Axes(xrange=xrange, yrange=yrange, zrange=zrange)
        
        plt.add(asse)
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
            xl = shapes.Cylinder([[x0, 0, 0], [x1, 0, 0]], r=aves / 250 * s, c=xcol, alpha=alpha)
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
            yl = shapes.Cylinder([[0, y0, 0], [0, y1, 0]], r=aves / 250 * s, c=ycol, alpha=alpha)
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
            yt.rotate_z(90)
            yt.pos(wpos)
            acts += [yl, yc, yt]

        if dz > aves / 100:
            zl = shapes.Cylinder([[0, 0, z0], [0, 0, z1]], r=aves / 250 * s, c=zcol, alpha=alpha)
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
            zt.rotate_z(45)
            zt.rotate_x(90)
            zt.pos(wpos)
            acts += [zl, zc, zt]
        for a in acts:
            a.actor.PickableOff()
        asse = Assembly(acts)
        asse.actor.PickableOff()
        plt.add(asse)
        plt.axes_instances[r] = asse

    elif plt.axes == 4:
        axact = vtki.vtkAxesActor()
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
        icn = Icon(axact, size=0.1)
        plt.axes_instances[r] = icn
        icn.SetInteractor(plt.interactor)
        icn.EnabledOn()
        icn.InteractiveOff()
        plt.widgets.append(icn)

    elif plt.axes == 5:
        axact = vtki.new("AnnotatedCubeActor")
        axact.GetCubeProperty().SetColor(get_color(settings.annotated_cube_color))
        axact.SetTextEdgesVisibility(0)
        axact.SetFaceTextScale(settings.annotated_cube_text_scale)
        axact.SetXPlusFaceText(settings.annotated_cube_texts[0])  # XPlus
        axact.SetXMinusFaceText(settings.annotated_cube_texts[1]) # XMinus
        axact.SetYPlusFaceText(settings.annotated_cube_texts[2])  # YPlus
        axact.SetYMinusFaceText(settings.annotated_cube_texts[3]) # YMinus
        axact.SetZPlusFaceText(settings.annotated_cube_texts[4])  # ZPlus
        axact.SetZMinusFaceText(settings.annotated_cube_texts[5]) # ZMinus
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
        icn = Icon(axact, size=0.06)
        plt.axes_instances[r] = icn
        icn.SetInteractor(plt.interactor)
        icn.EnabledOn()
        icn.InteractiveOff()
        plt.widgets.append(icn)

    elif plt.axes == 6:
        ocf = vtki.new("OutlineCornerFilter")
        ocf.SetCornerFactor(0.1)
        largestact, sz = None, -1
        for a in plt.objects:
            try:
                if a.pickable():
                    b = a.bounds()
                    if b is None:
                        return
                    d = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
                    if sz < d:
                        largestact = a
                        sz = d
            except AttributeError:
                pass
        
        try:
            ocf.SetInputData(largestact)
        except TypeError:
            try:
                ocf.SetInputData(largestact.dataset)
            except (TypeError, AttributeError):
                return
        ocf.Update()

        oc_mapper = vtki.new("HierarchicalPolyDataMapper")
        oc_mapper.SetInputConnection(0, ocf.GetOutputPort(0))
        oc_actor = vtki.vtkActor()
        oc_actor.SetMapper(oc_mapper)
        bc = np.array(plt.renderer.GetBackground())
        if np.sum(bc) < 1.5:
            lc = (1, 1, 1)
        else:
            lc = (0, 0, 0)
        oc_actor.GetProperty().SetColor(lc)
        oc_actor.PickableOff()
        oc_actor.UseBoundsOn()
        plt.axes_instances[r] = oc_actor
        plt.add(oc_actor)

    elif plt.axes == 7:
        vbb = compute_visible_bounds()[0]
        rulax = RulerAxes(vbb, c=c, xtitle="x - ", ytitle="y - ", ztitle="z - ")
        plt.axes_instances[r] = rulax
        if not rulax:
            return
        rulax.actor.UseBoundsOn()
        rulax.actor.PickableOff()
        plt.add(rulax)

    elif plt.axes == 8:
        vbb = compute_visible_bounds()[0]
        ca = vtki.new("CubeAxesActor")
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
        ca.PickableOff()
        ca.UseBoundsOff()
        plt.axes_instances[r] = ca
        plt.renderer.AddActor(ca)

    elif plt.axes == 9:
        vbb = compute_visible_bounds()[0]
        src = vtki.new("CubeSource")
        src.SetXLength(vbb[1] - vbb[0])
        src.SetYLength(vbb[3] - vbb[2])
        src.SetZLength(vbb[5] - vbb[4])
        src.Update()
        ca = Mesh(src.GetOutput(), c, 0.5).wireframe(True)
        ca.pos((vbb[0] + vbb[1]) / 2, (vbb[3] + vbb[2]) / 2, (vbb[5] + vbb[4]) / 2)
        ca.actor.PickableOff()
        ca.actor.UseBoundsOff()
        plt.axes_instances[r] = ca
        plt.add(ca)

    elif plt.axes == 10:
        vbb = compute_visible_bounds()[0]
        x0 = (vbb[0] + vbb[1]) / 2, (vbb[3] + vbb[2]) / 2, (vbb[5] + vbb[4]) / 2
        rx, ry, rz = (vbb[1] - vbb[0]) / 2, (vbb[3] - vbb[2]) / 2, (vbb[5] - vbb[4]) / 2
        rm = max(rx, ry, rz)
        xc = shapes.Disc(x0, r1=rm, r2=rm, c="lr", res=(1, 72))
        yc = shapes.Disc(x0, r1=rm, r2=rm, c="lg", res=(1, 72))
        yc.rotate_x(90)
        zc = shapes.Disc(x0, r1=rm, r2=rm, c="lb", res=(1, 72))
        yc.rotate_y(90)
        xc.clean().alpha(0.5).wireframe().linewidth(2).actor.PickableOff()
        yc.clean().alpha(0.5).wireframe().linewidth(2).actor.PickableOff()
        zc.clean().alpha(0.5).wireframe().linewidth(2).actor.PickableOff()
        ca = xc + yc + zc
        ca.PickableOff()
        ca.UseBoundsOn()
        plt.renderer.AddActor(ca)
        plt.axes_instances[r] = ca

    elif plt.axes == 11:
        vbb, ss = compute_visible_bounds()[0:2]
        xpos, ypos = (vbb[1] + vbb[0]) / 2, (vbb[3] + vbb[2]) / 2
        gs = sum(ss) * 3
        gr = shapes.Grid((xpos, ypos, vbb[4]), s=(gs, gs), res=(11, 11), c=c, alpha=0.1)
        gr.lighting("off").actor.PickableOff()
        gr.actor.UseBoundsOff()
        plt.axes_instances[r] = gr
        plt.add(gr)

    elif plt.axes == 12:
        polaxes = vtki.new("PolarAxesActor")
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
        polaxes.UseBoundsOn()
        polaxes.PickableOff()
        plt.axes_instances[r] = polaxes
        plt.renderer.AddActor(polaxes)

    elif plt.axes == 13:
        # draws a simple ruler at the bottom of the window
        ls = vtki.new("LegendScaleActor")
        ls.RightAxisVisibilityOff()
        ls.TopAxisVisibilityOff()
        ls.LeftAxisVisibilityOff()
        ls.LegendVisibilityOff()
        ls.SetBottomBorderOffset(50)
        ls.GetBottomAxis().SetNumberOfMinorTicks(1)
        ls.GetBottomAxis().SetFontFactor(1.1)
        ls.GetBottomAxis().GetProperty().SetColor(c)
        ls.GetBottomAxis().GetProperty().SetOpacity(1.0)
        ls.GetBottomAxis().GetProperty().SetLineWidth(2)
        ls.GetBottomAxis().GetLabelTextProperty().SetColor(c)
        ls.GetBottomAxis().GetLabelTextProperty().BoldOff()
        ls.GetBottomAxis().GetLabelTextProperty().ItalicOff()
        pr = ls.GetBottomAxis().GetLabelTextProperty()
        pr.SetFontFamily(vtki.VTK_FONT_FILE)
        pr.SetFontFile(utils.get_font_path(settings.default_font))
        ls.PickableOff()
        # if not plt.renderer.GetActiveCamera().GetParallelProjection():
        #     vedo.logger.warning("Axes type 13 should be used with parallel projection")
        plt.axes_instances[r] = ls
        plt.renderer.AddActor(ls)

    elif plt.axes == 14:
        try:
            cow = vtki.new("CameraOrientationWidget")
            cow.SetParentRenderer(plt.renderer)
            cow.On()
            plt.axes_instances[r] = cow
        except ImportError:
            vedo.logger.warning("axes mode 14 is unavailable in this vtk version")

    else:
        e = "Keyword axes type must be in range [0-13]."
        e += "Available axes types are:\n\n"
        e += "0 = no axes\n"
        e += "1 = draw three customizable gray grid walls\n"
        e += "2 = show cartesian axes from (0,0,0)\n"
        e += "3 = show positive range of cartesian axes from (0,0,0)\n"
        e += "4 = show a triad at bottom left\n"
        e += "5 = show a cube at bottom left\n"
        e += "6 = mark the corners of the bounding box\n"
        e += "7 = draw a 3D ruler at each side of the cartesian axes\n"
        e += "8 = show the vtkCubeAxesActor object\n"
        e += "9 = show the bounding box outline\n"
        e += "10 = show three circles representing the maximum bounding box\n"
        e += "11 = show a large grid on the x-y plane (use with zoom=8)\n"
        e += "12 = show polar axes\n"
        e += "13 = draw a simple ruler at the bottom of the window\n"
        e += "14 = show the CameraOrientationWidget object"
        vedo.logger.warning(e)

    if not plt.axes_instances[r]:
        plt.axes_instances[r] = True
    return
