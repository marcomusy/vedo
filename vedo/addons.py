#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import vedo
import vedo.shapes as shapes
import vedo.utils as utils
import vtk
from vedo import settings
from vedo.assembly import Assembly
from vedo.colors import getColor
from vedo.colors import printc
from vedo.mesh import merge
from vedo.mesh import Mesh
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
            "addSlider2D",
            "addSlider3D",
            "addButton",
            "addCutterTool",
            "addIcon",
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
    def __init__( self,
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
            if 'legend' in e.info.keys():
                if not e.info['legend']:
                    ename = ''
                else:
                    ename = str(e.info['legend'])

            if not isinstance(e, vtk.vtkActor):
                ename = ''
            if ename:
                n+=1
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
                col = getColor(c)
            if markers is None: # default
                poly = e.inputdata()
            else:
                marker = markers[i] if utils.isSequence(markers) else markers
                if isinstance(marker, vedo.Points):
                    poly = marker.clone(deep=False).normalize().shift(0,1,0).polydata()
                else: # assume string marker
                    poly = vedo.shapes.Marker(marker, s=1).shift(0,1,0).polydata()

            self.SetEntry(n, poly, ti, col)
            n += 1

        self.SetWidth(width)
        if height is None:
            self.SetHeight(width / 3.0 * n)
        else:
            self.SetHeight(height)

        sx, sy = 1 - self.GetWidth(), 1 - self.GetHeight()
        if   pos == 1 or ("top" in pos and "left" in pos):
            self.GetPositionCoordinate().SetValue(0, sy)
        elif pos == 2 or ("top" in pos and "right" in pos):
            self.GetPositionCoordinate().SetValue(sx, sy)
        elif pos == 3 or ("bottom" in pos and "left" in pos):
            self.GetPositionCoordinate().SetValue(0, 0)
        elif pos == 4 or ("bottom" in pos and "right" in pos):
            self.GetPositionCoordinate().SetValue(sx, 0)
        if alpha:
            self.UseBackgroundOn()
            self.SetBackgroundColor(getColor(bg))
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

    def __init__(self,
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
                font = settings.defaultFont
            self.textproperty.SetFontFamily(vtk.VTK_FONT_FILE)
            self.textproperty.SetFontFile(utils.getFontPath(font))
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
        self.statusIdx = (self.statusIdx + 1) % len(self.states)
        self.status(self.statusIdx)
        return self


#####################################################################
class SplineTool(vtk.vtkContourWidget):

    def __init__(self, points, pc='k', ps=8, lc='r4', ac='g5', lw=2, closed=False, ontop=True):
        """
        Spline tool, to be used with ``plotter.addSplineTool()``.

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
        vtk.vtkContourWidget.__init__(self)

        self.representation = vtk.vtkOrientedGlyphContourRepresentation()
        self.representation.SetAlwaysOnTop(ontop)

        self.representation.GetLinesProperty().SetColor(getColor(lc))
        self.representation.GetLinesProperty().SetLineWidth(lw)

        self.representation.GetProperty().SetColor(getColor(pc))
        self.representation.GetProperty().SetPointSize(ps)
        self.representation.GetProperty().RenderPointsAsSpheresOn()

        self.representation.GetActiveProperty().SetColor(getColor(ac))
        self.representation.GetActiveProperty().SetLineWidth(lw+1)

        self.SetRepresentation(self.representation)

        if utils.isSequence(points):
            self.points = Points(points)
        else:
            self.points = points

        self.closed = closed

    def add(self, pt):
        """
        Add one point at a specified position in space if 3D,
        or 2D screen-display position if 2D.
        """
        if len(pt)==2:
            self.representation.AddNodeAtDisplayPosition(int(pt[0]), int(pt[1]))
        else:
            self.representation.AddNodeAtWorldPosition(pt)
        return self

    def remove(self, i):
        self.representation.DeleteNthNode(i)
        return self

    def on(self):
        self.On()
        self.Render()
        return self

    def off(self):
        self.Off()
        self.Render()
        return self

    def render(self):
        self.Render()
        return self

    def bounds(self):
        return self.GetBounds()

    def spline(self):
        self.representation.SetClosedLoop(self.closed)
        self.representation.BuildRepresentation()
        pd = self.representation.GetContourRepresentationAsPolyData()
        pts = utils.vtk2numpy(pd.GetPoints().GetData())
        ln = vedo.Line(pts, lw=2, c='k')
        return ln

    def nodes(self, onscreen=False):
        """Return the current position in space (or on 2D screen-display) of the spline nodes."""
        n = self.representation.GetNumberOfNodes()
        pts = []
        for i in range(n):
            p = [0.,0.,0.]
            if onscreen:
                self.representation.GetNthNodeDisplayPosition(i, p)
            else:
                self.representation.GetNthNodeWorldPosition(i, p)
            pts.append(p)
        return np.array(pts)


#####################################################################
def Goniometer(
        p1,p2,p3,
        font="",
        arcSize=0.4,
        s=1,
        italic=0,
        rotation=0,
        prefix="",
        lc='k2',
        c='white',
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

    arcSize : float
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

    acts=[]
    ln = shapes.Line([p1,p2,p3], lw=lw, c=lc)
    acts.append(ln)

    va = utils.versor(p1-p2)
    vb = utils.versor(p3-p2)
    r = min(utils.mag(p3-p2), utils.mag(p1-p2))*arcSize
    ptsarc = []
    res = 120
    imed = int(res/2)
    for i in range(res+1):
        vi = utils.versor(vb*i/res + va*(res-i)/res)
        if i==imed: vc = np.array(vi)
        ptsarc.append(p2+vi*r)
    arc = shapes.Line(ptsarc).lw(lw).c(lc)
    acts.append(arc)

    angle = np.arccos(np.dot(va,vb))*180/np.pi

    lb = shapes.Text3D(prefix+utils.precision(angle,precision)+'º', s=r/12*s,
                       font=font, italic=italic, justify="center")
    cr = np.cross(va,vb)
    lb.pos(p2+vc*r/1.75).orientation(cr*np.sign(cr[2]), rotation=rotation)
    lb.c(c).bc('tomato').lighting('off')
    acts.append(lb)

    if alpha>0:
        pts = [p2] + arc.points().tolist() + [p2]
        msh = Mesh([pts, [list(range(arc.N()+2))]], c=lc, alpha=alpha)
        msh.lighting('off')
        msh.triangulate()
        msh.shift(0,0,-r/10000) # to resolve 2d conflicts..
        acts.append(msh)

    asse = Assembly(acts)
    return asse

def Light(
        pos,
        focalPoint=(0, 0, 0),
        angle=180,
        c=None,
        intensity=1,
        removeOthers=False,
    ):
    """
    Generate a source of light placed at pos, directed to focal point.
    Returns a ``vtkLight`` object.

    Parameters
    ----------
    focalPoint : list
        focal point, if this is a vedo object use its position.

    angle : float
        aperture angle of the light source, in degrees

    c : color
        set the light color

    intensity : float
        intensity value between 0 and 1.

    removeOthers : bool
        remove all other lights in the scene
        (in this case a Plotter object must already exist)

    .. hint:: examples/basic/lights.py
        .. image:: https://vedo.embl.es/images/basic/lights.png
    """
    if hasattr(pos, "color") and c is None:
        c = pos.color()
    if c is None:
        c = 'white'

    if isinstance(pos, vedo.Base3DProp):
        pos = pos.pos()

    if isinstance(focalPoint, vedo.Base3DProp):
        focalPoint = focalPoint.pos()

    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(pos)
    light.SetConeAngle(angle)
    light.SetFocalPoint(focalPoint)
    light.SetIntensity(intensity)
    light.SetColor(getColor(c))

    # light.SetPositional(1) ##??
    # if ambientColor is not None: # doesnt work anyway
    #     light.SetAmbientColor(getColor(ambientColor))
    # if diffuseColor is not None:
    #     light.SetDiffuseColor(getColor(diffuseColor))
    # if specularColor is not None:
    #     light.SetSpecularColor(getColor(specularColor))

    if removeOthers:
        if vedo.plotter_instance and vedo.plotter_instance.renderer:
            vedo.plotter_instance.renderer.RemoveAllLights()
        else:
            vedo.logger.error("in Light(removeOthers=True): scene does not exist.")

    return light


#####################################################################
def ScalarBar(
        obj,
        title="",
        pos=(0.8,0.05),
        titleYOffset=15,
        titleFontSize=12,
        size=(None,None),
        nlabels=None,
        c='k',
        horizontal=False,
        useAlpha=True,
        tformat='%-#6.3g',
    ):
    """
    A 2D scalar bar for the specified obj.

    Parameters
    ----------
    pos : list
        fractional x and y position in the 2D window

    size : list
        size of the scalarbar in pixel units (width, heigth)

    nlabels : int
        number of numeric labels to be shown

    useAlpha : bool
        retain trasparency in scalarbar

    horizontal : bool
        show in horizontal layout

    .. hint:: examples/basic/scalarbars.py
        .. image:: https://user-images.githubusercontent.com/32848391/62940174-4bdc7900-bdd3-11e9-9713-e4f3e2fdab63.png
    """
    if not hasattr(obj, "mapper"):
        vedo.logger.error(f"in addScalarBar(): input is invalid {type(obj)}. Skip.")
        return None

    if isinstance(obj, Points):
        vtkscalars = obj._data.GetPointData().GetScalars()
        if vtkscalars is None:
            vtkscalars = obj._data.GetCellData().GetScalars()
        if not vtkscalars:
            return None
        lut = vtkscalars.GetLookupTable()
        if not lut:
            lut = obj._mapper.GetLookupTable()
            if not lut:
                return None

    elif isinstance(obj, (Volume, TetMesh)):
        lut = utils.ctf2lut(obj)

    else:
        return None

    c = getColor(c)
    sb = vtk.vtkScalarBarActor()
    #print(sb.GetLabelFormat())
    sb.SetLabelFormat(tformat)
    sb.SetLookupTable(lut)
    sb.SetUseOpacity(useAlpha)
    sb.SetDrawFrame(0)
    sb.SetDrawBackground(0)
    if lut.GetUseBelowRangeColor():
        sb.DrawBelowRangeSwatchOn()
        sb.SetBelowRangeAnnotation('')
    if lut.GetUseAboveRangeColor():
        sb.DrawAboveRangeSwatchOn()
        sb.SetAboveRangeAnnotation('')
    if lut.GetNanColor() != (0.5, 0.0, 0.0, 1.0):
        sb.DrawNanAnnotationOn()
        sb.SetNanAnnotation('nan')

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
        titprop.SetFontSize(titleFontSize)
        titprop.SetFontFamily(vtk.VTK_FONT_FILE)
        titprop.SetFontFile(utils.getFontPath(settings.defaultFont))
        sb.SetTitle(title)
        sb.SetVerticalTitleSeparation(titleYOffset)
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
        sb.SetPosition(pos[0]+0.09, pos[1])
        sb.SetMaximumWidthInPixels(60)
        sb.SetMaximumHeightInPixels(250)

    if size[0] is not None: sb.SetMaximumWidthInPixels(size[0])
    if size[1] is not None: sb.SetMaximumHeightInPixels(size[1])

    if nlabels is not None:
        sb.SetNumberOfLabels(nlabels)

    sctxt = sb.GetLabelTextProperty()
    sctxt.SetFontFamily(vtk.VTK_FONT_FILE)
    sctxt.SetFontFile(utils.getFontPath(settings.defaultFont))
    sctxt.SetColor(c)
    sctxt.SetShadow(0)
    sctxt.SetFontSize(titleFontSize-2)
    sb.SetAnnotationTextProperty(sctxt)
    sb.PickableOff()
    return sb

#####################################################################
def ScalarBar3D(
        obj,
        title='',
        pos=None,
        s=(None, None),
        titleFont="",
        titleXOffset=-1.5,
        titleYOffset=0.0,
        titleSize=1.5,
        titleRotation=0.0,
        nlabels=9,
        labelFont="",
        labelSize=1,
        labelOffset=0.375,
        labelRotation=0,
        italic=0,
        c=None,
        useAlpha=True,
        drawBox=True,
        aboveText=None,
        belowText=None,
        nanText='NaN',
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

    titleXOffset : float
        horizontal space btw title and color scalarbar

    titleYOffset : float
        vertical space offset

    titleSize : float
        size of title wrt numeric labels

    titleRotation : float
        title rotation in degrees

    nlabels : int
        number of numeric labels

    labelFont : str
        font type for labels

    labelSize : float
        label scale factor

    labelOffset : float
        space btw numeric labels and scale

    labelRotation : float
        label rotation in degrees

    useAlpha : bool
        render transparency of the color bar, otherwise ignore

    drawBox : bool
        draw a box around the colorbar (useful with useAlpha=True)

    categories : list
        make a categorical scalarbar,
        the input list will have the format [value, color, alpha, textlabel]

    .. hint:: examples/basic/scalarbars.py
    """

    if isinstance(obj, Points):
        lut = obj.mapper().GetLookupTable()
        if not lut or lut.GetTable().GetNumberOfTuples() == 0:
            # create the most similar to the default
            obj.cmap('jet_r')
            # todo: grab the auto created default LUT (but where is it?)
            #       cells or points?
            lut = obj.mapper().GetLookupTable()
        vmin, vmax = lut.GetRange()

    elif isinstance(obj, (Volume, TetMesh)):
        lut = utils.ctf2lut(obj)
        vmin, vmax = lut.GetRange()

    elif utils.isSequence(obj):
        vmin, vmax = np.min(obj), np.max(obj)

    else:
        vedo.logger.error("in ScalarBar3D(): input must be a vedo object with bounds.")
        return obj


    bns = obj.GetBounds()
    sx, sy = s
    if sy is None:
        sy = (bns[3]-bns[2])
    if sx is None:
        sx = sy/18


    if categories is not None: ################################
        ncats = len(categories)
        scale = shapes.Grid([-sx * labelOffset, 0, 0], c=c, alpha=1, s=(sx,sy), res=(1,ncats))
        cols, alphas= [], []
        ticks_pos, ticks_txt = [0.0], ['']
        for i, cat in enumerate(categories):
            cl = getColor(cat[1])
            cols.append(cl)
            if len(cat)>2:
                alphas.append(cat[2])
            else:
                alphas.append(1)
            if len(cat)>3:
                ticks_txt.append(cat[3])
            else:
                ticks_txt.append("")
            ticks_pos.append((i+0.5)/ncats)
        ticks_pos.append(1.0)
        ticks_txt.append('')
        rgba = np.c_[np.array(cols)*255, np.array(alphas)*255]
        scale.cellIndividualColors(rgba)

    else: ########################################################

        # build the color scale part
        scale = shapes.Grid([-sx * labelOffset, 0, 0], c=c, alpha=1, s=(sx,sy),
                            res=(1, lut.GetTable().GetNumberOfTuples()))
        cscals = np.linspace(vmin, vmax, lut.GetTable().GetNumberOfTuples())
        scale.cmap(lut, cscals, on='cells')
        ticks_pos, ticks_txt = utils.makeTicks(vmin, vmax, nlabels)

    scale.lw(0).wireframe(False).lighting('off')

    scales = [scale]

    xbns = scale.xbounds()
    if pos is None:
        d = sx/2
        if title:
            d = np.sqrt((bns[1]-bns[0])**2+sy*sy)/20
        pos = (bns[1]-xbns[0]+d, (bns[2]+bns[3])/2, bns[4])

    lsize = sy/60*labelSize

    tacts = []
    for i, p in enumerate(ticks_pos):
        tx = ticks_txt[i]
        if i and tx:
            # build numeric text
            y = (p - 0.5) *sy
            if labelRotation:
                a = shapes.Text3D(tx, pos=[sx*labelOffset, y, 0], s=lsize,
                                  justify='center-top', c=c, italic=italic, font=labelFont)
                a.RotateZ(labelRotation)
            else:
                a = shapes.Text3D(tx, pos=[sx*labelOffset, y, 0], s=lsize,
                                  justify='center-left', c=c, italic=italic, font=labelFont)

            tacts.append(a)

            # build ticks
            tic = shapes.Line([xbns[1], y, 0], [xbns[1]+sx*labelOffset/4, y, 0], lw=2, c=c)
            tacts.append(tic)

    # build title
    if title:
        t = shapes.Text3D(title, (0,0,0), s=sy/50*titleSize,
                          c=c, justify='centered', italic=italic, font=titleFont)
        t.RotateZ(90+titleRotation)
        t.pos(sx*titleXOffset, titleYOffset, 0)
        tacts.append(t)

    # build below scale
    if lut.GetUseBelowRangeColor():
        r,g,b,alfa = lut.GetBelowRangeColor()
        brect = shapes.Rectangle([-sx *labelOffset -sx/2, -sy/2-sx-sx*0.1, 0],
                                 [-sx *labelOffset +sx/2, -sy/2   -sx*0.1, 0],
                                 c=(r,g,b), alpha=alfa)
        brect.lw(1).lc(c).lighting('off')
        scales += [brect]
        if belowText is None:
           belowText = ' <'+str(vmin)
        if belowText:
            if labelRotation:
                btx = shapes.Text3D(belowText, (0,0,0), s=lsize,
                                    c=c, justify='center-top', italic=italic, font=labelFont)
                btx.RotateZ(labelRotation)
            else:
                btx = shapes.Text3D(belowText, (0,0,0), s=lsize,
                                    c=c, justify='center-left', italic=italic, font=labelFont)

            btx.pos(sx*labelOffset, -sy/2-sx*0.66, 0)
            tacts.append(btx)

    # build above scale
    if lut.GetUseAboveRangeColor():
        r,g,b,alfa = lut.GetAboveRangeColor()
        arect = shapes.Rectangle([-sx *labelOffset -sx/2, sy/2   +sx*0.1, 0],
                                 [-sx *labelOffset +sx/2, sy/2+sx+sx*0.1, 0],
                                 c=(r,g,b), alpha=alfa)
        arect.lw(1).lc(c).lighting('off')
        scales += [arect]
        if aboveText is None:
            aboveText = ' >'+str(vmax)
        if aboveText:
            if labelRotation:
                atx = shapes.Text3D(aboveText, (0,0,0), s=lsize,
                                    c=c, justify='center-top', italic=italic, font=labelFont)
                atx.RotateZ(labelRotation)
            else:
                atx = shapes.Text3D(aboveText, (0,0,0), s=lsize,
                                    c=c, justify='center-left', italic=italic, font=labelFont)

            atx.pos(sx*labelOffset, sy/2+sx*0.66, 0)
            tacts.append(atx)

    # build NaN scale
    if lut.GetNanColor() != (0.5, 0.0, 0.0, 1.0):
        nanshift = sx*0.1
        if brect:
            nanshift += sx
        r,g,b,alfa = lut.GetNanColor()
        nanrect = shapes.Rectangle([-sx *labelOffset -sx/2, -sy/2-sx-sx*0.1-nanshift, 0],
                                   [-sx *labelOffset +sx/2, -sy/2   -sx*0.1-nanshift, 0],
                                   c=(r,g,b), alpha=alfa)
        nanrect.lw(1).lc(c).lighting('off')
        scales += [nanrect]
        if labelRotation:
            nantx = shapes.Text3D(nanText, (0,0,0), s=lsize,
                                  c=c, justify='center-left', italic=italic, font=labelFont)
            nantx.RotateZ(labelRotation)
        else:
            nantx = shapes.Text3D(nanText, (0,0,0), s=lsize,
                                  c=c, justify='center-left', italic=italic, font=labelFont)
        nantx.pos(sx*labelOffset, -sy/2-sx*0.66-nanshift, 0)
        tacts.append(nantx)

    if drawBox:
        tacts.append(scale.box().lw(1))

    for a in tacts: a.PickableOff()

    mtacts = merge(tacts).lighting('off')
    mtacts.PickableOff()
    scale.PickableOff()

    sact = Assembly(scales + tacts)
    sact.SetPosition(pos)
    sact.PickableOff()
    sact.UseBoundsOff()
    sact.name = 'ScalarBar3D'
    return sact



#####################################################################
def addSlider2D(
        sliderfunc, xmin, xmax, value=None, pos=4,
        title='', font='', titleSize=1, c=None,
        showValue=True, delayed=False,
        **options
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

    titleSize : float
        title text scale [1.0]

    showValue : bool
        if true current value is shown

    delayed : bool
        if True the callback is delayed until when the mouse button is released

    alpha : float
        opacity of the scalar bar texts

    sliderLength : float
        slider length

    sliderWidth : float
        slider width

    endCapLength : float
        length of the end cap

    endCapWidth : float
        width of the end cap

    tubeWidth : float
        width of the tube

    titleHeight : float
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
    titleSize = options.pop("titleSize", titleSize)
    c = options.pop("c", c)
    showValue = options.pop("showValue", showValue)
    delayed = options.pop("delayed", delayed)
    alpha = options.pop("alpha", 1)
    sliderLength = options.pop("sliderLength", 0.015)
    sliderWidth  = options.pop("sliderWidth", 0.025)
    endCapLength = options.pop("endCapLength", 0.0015)
    endCapWidth  = options.pop("endCapWidth", 0.0125)
    tubeWidth    = options.pop("tubeWidth", 0.0075)
    titleHeight  = options.pop("titleHeight", 0.022)

    plt = vedo.plotter_instance
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
    sliderRep.SetSliderLength(sliderLength)
    sliderRep.SetSliderWidth(sliderWidth)
    sliderRep.SetEndCapLength(endCapLength)
    sliderRep.SetEndCapWidth(endCapWidth)
    sliderRep.SetTubeWidth(tubeWidth)
    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()

    if isinstance(pos, str):
        if "top" in pos:
            if "left" in pos:
                if "vert" in pos:
                    pos=11
                else:
                    pos=1
            elif "right" in pos:
                if "vert" in pos:
                    pos=12
                else:
                    pos=2
        elif "bott" in pos:
            if "left" in pos:
                if "vert" in pos:
                    pos=13
                else:
                    pos=3
            elif "right" in pos:
                if "vert" in pos:
                    if "span" in pos:
                        pos=15
                    else:
                        pos=14
                else:
                    pos=4
            elif "span":
                pos=5

    if utils.isSequence(pos):
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
    else: # bottom-right
        sliderRep.GetPoint1Coordinate().SetValue(0.55, 0.06)
        sliderRep.GetPoint2Coordinate().SetValue(0.95, 0.06)

    if showValue:
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

    sliderRep.SetTitleHeight(titleHeight * titleSize)
    sliderRep.GetTitleProperty().SetShadow(0)
    sliderRep.GetTitleProperty().SetColor(c)
    sliderRep.GetTitleProperty().SetOpacity(alpha)
    sliderRep.GetTitleProperty().SetBold(0)
    if font.lower() == 'courier':
        sliderRep.GetTitleProperty().SetFontFamilyToCourier()
    elif font.lower() == "times":
        sliderRep.GetTitleProperty().SetFontFamilyToTimes()
    elif font.lower() == "arial":
        sliderRep.GetTitleProperty().SetFontFamilyToArial()
    else:
        if font =='':
            font = utils.getFontPath(settings.defaultFont)
        else:
            font = utils.getFontPath(font)
        sliderRep.GetTitleProperty().SetFontFamily(vtk.VTK_FONT_FILE)
        sliderRep.GetLabelProperty().SetFontFamily(vtk.VTK_FONT_FILE)
        sliderRep.GetTitleProperty().SetFontFile(font)
        sliderRep.GetLabelProperty().SetFontFile(font)

    if title:
        sliderRep.SetTitleText(title)
        if not utils.isSequence(pos):
            if isinstance(pos, int) and pos > 10:
                sliderRep.GetTitleProperty().SetOrientation(90)
        else:
            if abs(pos[0][0] - pos[1][0]) < 0.1:
                sliderRep.GetTitleProperty().SetOrientation(90)

    sliderWidget = vtk.vtkSliderWidget()
    sliderWidget.SetInteractor(plt.interactor)
    sliderWidget.SetAnimationModeToJump()
    sliderWidget.SetRepresentation(sliderRep)
    if delayed:
        sliderWidget.AddObserver("EndInteractionEvent", sliderfunc)
    else:
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

    showValue : bool
        if True current value is shown on top of the slider

    .. hint:: examples/basic/sliders3d.py
    """
    plt = vedo.plotter_instance
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


def addCutterTool(obj=None, mode="box", invert=False):
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
        else:
            if mode=='box':
                return _addCutterToolMeshWithBox(obj, invert)
            elif mode=='plane':
                return _addCutterToolMeshWithPlane(obj, invert)
            elif mode=='sphere':
                return _addCutterToolMeshWithSphere(obj, invert)
            else:
                raise RuntimeError(f"Unknown mode: {mode}")
    except:
        return None

def _addCutterToolMeshWithSphere(mesh, invert):
    plt = vedo.plotter_instance

    sph = vtk.vtkSphere()
    cm = mesh.centerOfMass()
    sph.SetCenter(cm)
    aves = mesh.averageSize()
    sph.SetRadius(aves)
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(mesh.polydata())
    clipper.SetClipFunction(sph)
    clipper.GenerateClippedOutputOn()
    clipper.GenerateClipScalarsOff()
    clipper.SetInsideOut(not invert)
    clipper.Update()
    mesh.mapper().SetInputConnection(clipper.GetOutputPort())

    act0 = Mesh(clipper.GetOutput(), alpha=mesh.alpha()) # the main cut part
    act0.mapper().SetLookupTable(mesh.mapper().GetLookupTable())
    act0.mapper().SetScalarRange(mesh.mapper().GetScalarRange())

    act1 = Mesh()
    act1.mapper().SetInputConnection(clipper.GetClippedOutputPort()) # needs OutputPort
    act1.color((0.5,0.5,0.5), 0.04).wireframe()

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
    plt.interactor.Start() # allow extra interaction
    return act0

def _addCutterToolMeshWithBox(mesh, invert):
    plt = vedo.plotter_instance
    if not plt:
        vedo.logger.error("in addCutterTool() scene must be first rendered.")
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

    act0 = Mesh(cpoly, alpha=mesh.alpha()) # the main cut part
    act0.mapper().SetLookupTable(mesh.mapper().GetLookupTable())
    act0.mapper().SetScalarRange(mesh.mapper().GetScalarRange())

    act1 = Mesh()
    act1.mapper().SetInputConnection(clipper.GetClippedOutputPort()) # needs OutputPort
    act1.alpha(0.04).color((0.5,0.5,0.5)).wireframe()

    plt.remove(mesh, render=False)
    plt.add([act0, act1])

    def selectPolygons(vobj, event):
        vobj.GetPlanes(planes)

    boxWidget = vtk.vtkBoxWidget()
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1, 0, 1)
    boxWidget.GetOutlineProperty().SetColor(0.2, 0.2, 0.2)
    boxWidget.GetOutlineProperty().SetOpacity(0.8)
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
    plt.interactor.Start() # allow extra interaction
    return act0

def _addCutterToolMeshWithPlane(mesh, invert):
    plt = vedo.plotter_instance

    plane = vtk.vtkPlane()
    plane.SetNormal(mesh.centerOfMass())
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(mesh.polydata())
    clipper.SetClipFunction(plane)
    clipper.GenerateClipScalarsOff()
    clipper.GenerateClippedOutputOn()
    clipper.SetInsideOut(invert)
    clipper.Update()
    mesh.mapper().SetInputConnection(clipper.GetOutputPort())

    act0 = Mesh(clipper.GetOutput(), alpha=mesh.alpha()) # the main cut part
    act0.mapper().SetLookupTable(mesh.mapper().GetLookupTable())
    act0.mapper().SetScalarRange(mesh.mapper().GetScalarRange())

    act1 = Mesh()
    act1.mapper().SetInputConnection(clipper.GetClippedOutputPort()) # needs OutputPort
    act1.alpha(0.04).color((0.5,0.5,0.5)).wireframe()

    plt.remove(mesh, render=False)
    plt.add([act0, act1])

    def myCallback(obj, event):
        obj.GetPlane(plane)

    planeWidget = vtk.vtkImplicitPlaneWidget()
    planeWidget.SetNormal(1,0,0)
    planeWidget.SetPlaceFactor(1.25)
    planeWidget.SetInteractor(plt.interactor)
    planeWidget.SetCurrentRenderer(plt.renderer)
    planeWidget.SetInputData(mesh.inputdata())
    planeWidget.PlaceWidget(mesh.GetBounds())
    planeWidget.AddObserver("InteractionEvent", myCallback)
    planeWidget.GetPlaneProperty().SetColor(getColor('grey'))
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
    plt.interactor.Start() # allow extra interaction
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
def addRendererFrame(
        plotter_instance, c=None, alpha=None, lw=None, padding=None,
    ):
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
        lw = settings.rendererFrameWidth
    if lw==0:
        return None

    if padding is None:
        padding = settings.rendererFramePadding

    if c is None:  # automatic black or white
        c = (0.9, 0.9, 0.9)
        if np.sum(plotter_instance.renderer.GetBackground())>1.5:
            c = (0.1, 0.1, 0.1)
    c = getColor(c)

    if alpha is None:
        alpha = settings.rendererFrameAlpha

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
def addIcon(mesh, pos=3, size=0.08):
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
        vedo.logger.warning("Use addIcon() after first rendering the scene.")

        save_int = plt.interactive
        plt.show(interactive=0)
        plt.interactive = save_int
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(mesh)
    widget.SetInteractor(plt.interactor)
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
    plt.widgets.append(widget)
    if mesh in plt.actors:
        plt.actors.remove(mesh)
    return widget

#####################################################################
def computeVisibleBounds(actors=None):
    """Calculate max meshes bounds and sizes."""
    bns = []

    if actors is None:
        actors = vedo.plotter_instance.actors
    elif not utils.isSequence(actors):
        actors = [actors]

    try:
        # this block fails for VolumeSlice as vtkImageSlice.GetBounds() returns a pointer..
        # in any case we dont need axes for that one.
        for a in actors:
            if a and a.GetUseBounds():
                b = a.GetBounds()
                if b:
                    bns.append(b)
        if len(bns):
            max_bns = np.max(bns, axis=0)
            min_bns = np.min(bns, axis=0)
            vbb = [min_bns[0], max_bns[1], min_bns[2], max_bns[3], min_bns[4], max_bns[5]]
        else:
            vbb = vedo.plotter_instance.renderer.ComputeVisiblePropBounds()
            max_bns = vbb
            min_bns = vbb
        sizes = np.array([max_bns[1]-min_bns[0], max_bns[3]-min_bns[2], max_bns[5]-min_bns[4]])
        return [vbb, sizes, min_bns, max_bns]
    except:
        return [(0,0,0,0,0,0), (0,0,0), 0,0]


#####################################################################
def Ruler(
        p1, p2,
        unitScale=1,
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

    Parameters
    ----------
    label : str
        alternative fixed label to be shown

    unitScale : float
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

    labelRotation : float
        initial rotation of the label around the z-axis

    axisRotation : float
        initial rotation of the line around the main axis

    tickAngle : float
        initial rotation of the line around the main axis

    .. hint:: examples/pyplot/goniometer.py
        .. image:: https://vedo.embl.es/images/pyplot/goniometer.png
    """
    if unitScale != 1.0 and units == "":
        raise ValueError("When setting 'unitScale' to a value other than 1, " +
                         "a 'units' arguments must be specified.")

    if isinstance(p1, Points): p1 = p1.GetPosition()
    if isinstance(p2, Points): p2 = p2.GetPosition()
    if len(p1)==2: p1=[p1[0],p1[1],0.0]
    if len(p2)==2: p2=[p2[0],p2[1],0.0]
    p1, p2 = np.array(p1), np.array(p2)
    q1, q2 = [0, 0, 0], [utils.mag(p2 - p1), 0, 0]
    q1, q2 = np.array(q1), np.array(q2)
    v = q2 - q1
    d = utils.mag(v) * unitScale

    if s is None:
        s = d*0.02*(1/unitScale)

    if not label:
        label = str(d)
        if precision:
            label = utils.precision(d, precision)
    if prefix:
        label = prefix+ '~' + label
    if units:
        label += '~'+ units

    lb = shapes.Text3D(label, pos=(q1 + q2) / 2, s=s,
                       font=font, italic=italic, justify="center")
    if labelRotation:
        lb.RotateZ(labelRotation)

    x0, x1 = lb.xbounds()
    gap = [(x1 - x0) / 2, 0, 0]
    pc1 = (v / 2 - gap) * 0.9 + q1
    pc2 = q2 - (v / 2 - gap) * 0.9

    lc1 = shapes.Line(q1 - v / 50, pc1)
    lc2 = shapes.Line(q2 + v / 50, pc2)

    zs = np.array([0, d / 50 * (1/unitScale), 0])
    ml1 = shapes.Line(-zs, zs).pos(q1)
    ml2 = shapes.Line(-zs, zs).pos(q2)
    ml1.RotateZ(tickAngle-90)
    ml2.RotateZ(tickAngle-90)

    c1 = shapes.Circle(q1, r=d / 180 * (1/unitScale), res=20)
    c2 = shapes.Circle(q2, r=d / 180 * (1/unitScale), res=20)

    acts = [lb, lc1, lc2, c1, c2, ml1, ml2]
    macts = merge(acts).pos(p1).c(c).alpha(alpha)
    macts.GetProperty().LightingOff()
    macts.GetProperty().SetLineWidth(lw)
    macts.UseBoundsOff()
    macts.base = q1
    macts.top = q2
    macts.orientation(p2 - p1, rotation=axisRotation).bc('t').pickable(False)
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
        labelRotation=0,
        axisRotation=0,
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

    labelRotation : float
        initial rotation of the label around the z-axis

    axisRotation : float
        initial rotation of the line around the main axis

    xycross : bool
        show two back crossing lines in the xy plane

    .. hint:: examples/pyplot/goniometer.py
    """
    if utils.isSequence(inputobj):
        x0,x1,y0,y1,z0,z1 = inputobj
    else:
        x0,x1,y0,y1,z0,z1 = inputobj.GetBounds()
    dx,dy,dz = (y1-y0)*xpadding, (x1-x0)*ypadding, (y1-y0)*zpadding
    d = np.sqrt((y1-y0)**2+(x1-x0)**2+(z1-z0)**2)

    if not d:
        return None

    if s is None:
        s = d/75

    acts, rx, ry = [], None, None
    if xtitle is not None and (x1-x0)/d>0.1:
        rx = Ruler( [x0,y0-dx,z0],
                    [x1,y0-dx,z0],
                    s=s, font=font, precision=precision,
                    labelRotation=labelRotation, axisRotation=axisRotation,
                    lw=lw, italic=italic, prefix=xtitle, label=xlabel, units=units
        )
        acts.append(rx)
    if ytitle is not None and (y1-y0)/d>0.1:
        ry = Ruler( [x1+dy,y0,z0],
                    [x1+dy,y1,z0],
                    s=s, font=font, precision=precision,
                    labelRotation=labelRotation, axisRotation=axisRotation,
                    lw=lw, italic=italic, prefix=ytitle, label=ylabel, units=units
        )
        acts.append(ry)
    if ztitle is not None and (z1-z0)/d>0.1:
        rz = Ruler( [x0-dy,y0+dz,z0],
                    [x0-dy,y0+dz,z1],
                    s=s, font=font, precision=precision,
                    labelRotation=labelRotation, axisRotation=axisRotation+90,
                    lw=lw, italic=italic, prefix=ztitle, label=zlabel, units=units
        )
        acts.append(rz)

    if xycross and rx and ry:
        lx = shapes.Line([x0,y0,z0],    [x0,y1+dx,z0])
        ly = shapes.Line([x0-dy,y1,z0], [x1,y1,z0])
        d = min((x1-x0), (y1-y0))/200
        cxy = shapes.Circle([x0,y1,z0], r=d, res=15)
        acts.extend([lx,ly,cxy])

    macts = merge(acts)
    if not macts:
        return None
    macts.c(c).alpha(alpha).bc('t')
    macts.UseBoundsOff()
    macts.PickableOff()
    return macts


#####################################################################
def addScaleIndicator(pos=(0.7,0.05), s=0.02, length=2, lw=4, c='k', units=''):
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
    psqr = [[0.0,0.05], [length/10,0.05]]
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
        vedo.logger.warning("addScaleIndicator is called with useParallelProjection OFF.")

    rlabel = vtk.vtkVectorText()
    rlabel.SetText('')
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputConnection(rlabel.GetOutputPort())
    t = vtk.vtkTransform()
    t.Scale(s,s*wsx/wsy,1)
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
    fractor.GetProperty().SetColor(getColor(c))
    fractor.GetProperty().SetOpacity(1)
    fractor.GetProperty().SetLineWidth(lw)
    fractor.GetProperty().SetDisplayLocationToForeground()

    def sifunc(iren, ev):
        wsx, wsy = plt.window.GetSize()
        ps = plt.camera.GetParallelScale()
        newtxt = utils.precision(ps/wsy*wsx*length*dd,3)
        if units:
            newtxt += ' '+units
        rlabel.SetText(newtxt)

    plt.renderer.AddActor(fractor)
    plt.interactor.AddObserver('MouseWheelBackwardEvent', sifunc)
    plt.interactor.AddObserver('MouseWheelForwardEvent', sifunc)
    plt.interactor.AddObserver('InteractionEvent', sifunc)
    sifunc(0,0)

    return fractor

#####################################################################
def Axes(
        obj=None,
        xtitle='x', ytitle='y', ztitle='z',
        xrange=None, yrange=None, zrange=None,
        c=None,
        numberOfDivisions=None,
        digits=None,
        limitRatio=0.04,
        htitle="",
        hTitleSize=0.03,
        hTitleFont=None,
        hTitleItalic=False,
        hTitleColor=None,
        hTitleJustify='bottom-center',
        hTitleRotation=0,
        hTitleOffset=(0, 0.01, 0),
        titleDepth=0,
        titleFont="", # grab settings.defaultFont
        textScale=1.0,
        xTitlePosition=0.95, yTitlePosition=0.95, zTitlePosition=0.95,
        xTitleOffset=0.025,  yTitleOffset=0.0275,  zTitleOffset=0.02, # can be a list (dx,dy,dz)
        xTitleJustify=None, yTitleJustify=None, zTitleJustify=None,
        xTitleRotation=0, yTitleRotation=0, zTitleRotation=0,         # can be a list (rx,ry,rz)
        xTitleBox=False,  yTitleBox=False,
        xTitleSize=0.025, yTitleSize=0.025, zTitleSize=0.025,
        xTitleColor=None, yTitleColor=None, zTitleColor=None,
        xTitleBackfaceColor=None, yTitleBackfaceColor=None, zTitleBackfaceColor=None,
        xTitleItalic=0, yTitleItalic=0, zTitleItalic=0,
        gridLineWidth=1,
        xyGrid=True, yzGrid=False, zxGrid=False,
        xyGrid2=False, yzGrid2=False, zxGrid2=False,
        xyGridTransparent=False, yzGridTransparent=False, zxGridTransparent=False,
        xyGrid2Transparent=False, yzGrid2Transparent=False, zxGrid2Transparent=False,
        xyPlaneColor=None, yzPlaneColor=None, zxPlaneColor=None,
        xyGridColor=None, yzGridColor=None, zxGridColor=None,
        xyAlpha=0.075, yzAlpha=0.075, zxAlpha=0.075,
        xyFrameLine=None, yzFrameLine=None, zxFrameLine=None,
        xyFrameColor=None, yzFrameColor=None, zxFrameColor=None,
        axesLineWidth=1,
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
        xLabelOffset=0.8, yLabelOffset=0.8, zLabelOffset=0.8, # each can be a list (dx,dy,dz)
        xLabelJustify=None, yLabelJustify=None, zLabelJustify=None,
        xLabelRotation=0, yLabelRotation=0, zLabelRotation=0, # each can be a list (rx,ry,rz)
        xAxisRotation=0, yAxisRotation=0, zAxisRotation=0,    # rotate all elements around axis
        xValuesAndLabels=None, yValuesAndLabels=None, zValuesAndLabels=None,
        xyShift=0, yzShift=0, zxShift=0,
        xShiftAlongY=0, xShiftAlongZ=0,
        yShiftAlongX=0, yShiftAlongZ=0,
        zShiftAlongX=0, zShiftAlongY=0,
        xUseBounds=True, yUseBounds=True, zUseBounds=False,
        xInverted=False, yInverted=False, zInverted=False,
        useGlobal=False,
        tol=0.001,
    ):
    """
    Draw axes for the input object.

    Returns an `Assembly` object.

    Parameters
    ----------

    - `xtitle`,                ['x'], x-axis title text
    - `xrange`,               [None], x-axis range in format (xmin, ymin), default is automatic.
    - `numberOfDivisions`,    [None], approximate number of divisions on the longest axis
    - `axesLineWidth`,           [1], width of the axes lines
    - `gridLineWidth`,           [1], width of the grid lines
    - `titleDepth`,              [0], extrusion fractional depth of title text
    - `xyGrid`,               [True], show a gridded wall on plane xy
    - `yzGrid`,               [True], show a gridded wall on plane yz
    - `zxGrid`,               [True], show a gridded wall on plane zx
    - `zxGrid2`,             [False], show zx plane on opposite side of the bounding box
    - `xyGridTransparent`    [False], make grid plane completely transparent
    - `xyGrid2Transparent`   [False], make grid plane completely transparent on opposite side box
    - `xyPlaneColor`,       ['None'], color of the plane
    - `xyGridColor`,        ['None'], grid line color
    - `xyAlpha`,              [0.15], grid plane opacity
    - `xyFrameLine`,             [0], add a frame for the plane, use value as the thickness
    - `xyFrameColor`,         [None], color for the frame of the plane
    - `showTicks`,            [True], show major ticks
    - `digits`,               [None], use this number of significant digits in scientific notation
    - `titleFont`,              [''], font for axes titles
    - `labelFont`,              [''], font for numeric labels
    - `textScale`,             [1.0], global scaling factor for all text elements (titles, labels)
    - `htitle`,                 [''], header title
    - `hTitleSize`,           [0.03], header title size
    - `hTitleFont`,           [None], header font (defaults to `titleFont`)
    - `hTitleItalic`,         [True], header font is italic
    - `hTitleColor`,          [None], header title color (defaults to `xTitleColor`)
    - `hTitleJustify`, ['bottom-center'], origin of the title justification
    - `hTitleOffset`,   [(0,0.01,0)], control offsets of header title in x, y and z
    - `xTitlePosition`,       [0.32], title fractional positions along axis
    - `xTitleOffset`,         [0.05], title fractional offset distance from axis line, can be a list
    - `xTitleJustify`,        [None], choose the origin of the bounding box of title
    - `xTitleRotation`,          [0], add a rotation of the axis title, can be a list (rx,ry,rz)
    - `xTitleBox`,           [False], add a box around title text
    - `xLineColor`,      [automatic], color of the x-axis
    - `xTitleColor`,     [automatic], color of the axis title
    - `xTitleBackfaceColor`,  [None], color of axis title on its backface
    - `xTitleSize`,          [0.025], size of the axis title
    - `xTitleItalic`,            [0], a bool or float to make the font italic
    - `xHighlightZero`,       [True], draw a line highlighting zero position if in range
    - `xHighlightZeroColor`, [autom], color of the line highlighting the zero position
    - `xTickLength`,         [0.005], radius of the major ticks
    - `xTickThickness`,     [0.0025], thickness of the major ticks along their axis
    - `xMinorTicks`,             [1], number of minor ticks between two major ticks
    - `xLabelColor`,     [automatic], color of numeric labels and ticks
    - `xLabelPrecision`,         [2], nr. of significative digits to be shown
    - `xLabelSize`,          [0.015], size of the numeric labels along axis
    - `xLabelRotation`,          [0], numeric labels rotation (can be a list of 3 rotations)
    - `xLabelOffset`,          [0.8], offset of the numeric labels (can be a list of 3 offsets)
    - `xLabelJustify`,        [None], choose the origin of the bounding box of labels
    - `xAxisRotation`,           [0], rotate the X axis elements (ticks and labels) around this same axis
    - `xValuesAndLabels`          [], assign custom tick positions and labels [(pos1, label1), ...]
    - `xyShift`                [0.0], slide the xy-plane along z (the range is [0,1])
    - `xShiftAlongY`           [0.0], slide x-axis along the y-axis (the range is [0,1])
    - `tipSize`,              [0.01], size of the arrow tip
    - `limitRatio`,           [0.04], below this ratio don't plot smaller axis
    - `xUseBounds`,           [True], keep into account space occupied by labels when setting camera
    - `xInverted`,           [False], invert labels order and direction (only visually!)
    - `useGlobal`,           [False], try to compute the global bounding box of visible actors

    Example:
        .. code-block:: python

            from vedo import Axes, Box, show
            b = Box(pos=(1,2,3), length=8, width=9, height=7).alpha(0.1)
            axs = Axes(b, c='k')  # returns Assembly object
            #for a in axs.unpack(): print(a.name)
            show(axs)

    .. hint::
        examples/pyplot/customAxes1.py, customAxes2.py customAxes3.py, customIndividualAxes.py

        .. image:: https://vedo.embl.es/images/pyplot/customAxes3.png
    """
    if not titleFont:
        titleFont = settings.defaultFont
    if not labelFont:
        labelFont = settings.defaultFont

    if c is None:  # automatic black or white
        c = (0.1, 0.1, 0.1)
        plt = vedo.plotter_instance
        if plt and plt.renderer:
            bgcol = plt.renderer.GetBackground()
        else:
            bgcol = (1,1,1)
        if np.sum(bgcol) < 1.5:
            c = (0.9, 0.9, 0.9)
    else:
        c = getColor(c)

    if useGlobal:
        vbb, drange, min_bns, max_bns = computeVisibleBounds()
    else:
        if obj is not None:
            vbb, drange, min_bns, max_bns = computeVisibleBounds(obj)
        else:
            vbb = np.zeros(6)
            drange = np.zeros(3)
            if zrange is None:
                zrange=(0,0)
            if xrange is None or yrange is None:
                vedo.logger.error("in Axes() must specify axes ranges!")
                raise RuntimeError()

    if xrange is not None:
        if xrange[1] < xrange[0]:
            xInverted = True
            xrange = [xrange[1], xrange[0]]
        vbb[0], vbb[1] = xrange
        drange[0] = vbb[1] - vbb[0]
        min_bns = vbb
        max_bns = vbb
    if yrange is not None:
        if yrange[1] < yrange[0]:
            yInverted = True
            yrange = [yrange[1], yrange[0]]
        vbb[2], vbb[3] = yrange
        drange[1] = vbb[3] - vbb[2]
        min_bns = vbb
        max_bns = vbb
    if zrange is not None:
        if zrange[1] < zrange[0]:
            zInverted = True
            zrange = [zrange[1], zrange[0]]
        vbb[4], vbb[5] = zrange
        drange[2] = vbb[5] - vbb[4]
        min_bns = vbb
        max_bns = vbb

    drangemax = max(drange)
    if not drangemax:
        return

    if drange[0]/drangemax < limitRatio:
        drange[0] = 0
        xtitle = ''
    if drange[1]/drangemax < limitRatio:
        drange[1] = 0
        ytitle = ''
    if drange[2]/drangemax < limitRatio:
        drange[2] = 0
        ztitle = ''

    x0,x1, y0,y1, z0,z1 = vbb
    dx, dy, dz = drange

    gscale = np.sqrt(dx*dx + dy*dy + dz*dz)*0.75

    if not xyPlaneColor: xyPlaneColor = c
    if not yzPlaneColor: yzPlaneColor = c
    if not zxPlaneColor: zxPlaneColor = c
    if not xyGridColor:  xyGridColor = c
    if not yzGridColor:  yzGridColor = c
    if not zxGridColor:  zxGridColor = c
    if not xTitleColor:  xTitleColor = c
    if not yTitleColor:  yTitleColor = c
    if not zTitleColor:  zTitleColor = c
    if not xLineColor:   xLineColor = c
    if not yLineColor:   yLineColor = c
    if not zLineColor:   zLineColor = c
    if not xLabelColor:  xLabelColor = xLineColor
    if not yLabelColor:  yLabelColor = yLineColor
    if not zLabelColor:  zLabelColor = zLineColor

    # vtk version<9 dont like depthpeeling: force switching off grids
    if settings.useDepthPeeling and not utils.vtkVersionIsAtLeast(9):
        xyGrid = False
        yzGrid = False
        zxGrid = False
        xyGrid2 = False
        yzGrid2 = False
        zxGrid2 = False

    if tipSize is None:
        tipSize = 0.005*gscale
        if not ztitle:
            tipSize = 0 # switch off in xy 2d

    ndiv = 4
    if not ztitle or not ytitle or not xtitle: # make more default ticks if 2D
        ndiv = 6
        if not ztitle:
            if xyFrameLine is None:
                xyFrameLine = True
            if tipSize is None:
                tipSize = False

    if utils.isSequence(numberOfDivisions):
        rx, ry, rz = numberOfDivisions
    else:
        if not numberOfDivisions:
            numberOfDivisions = ndiv

    rx, ry, rz = np.ceil(drange/drangemax * numberOfDivisions).astype(int)

    if xtitle:
        xticks_float, xticks_str = utils.makeTicks(x0,x1, rx, xValuesAndLabels, digits)
        xticks_float = xticks_float * dx
        if xInverted:
            xticks_float = np.flip(-(xticks_float - xticks_float[-1]))
            xticks_str = list(reversed(xticks_str))
            xticks_str[-1] = ''
            xHighlightZero = False
    if ytitle:
        yticks_float, yticks_str = utils.makeTicks(y0,y1, ry, yValuesAndLabels, digits)
        yticks_float = yticks_float * dy
        if yInverted:
            yticks_float = np.flip(-(yticks_float - yticks_float[-1]))
            yticks_str = list(reversed(yticks_str))
            yticks_str[-1] = ''
            yHighlightZero = False
    if ztitle:
        zticks_float, zticks_str = utils.makeTicks(z0,z1, rz, zValuesAndLabels, digits)
        zticks_float = zticks_float * dz
        if zInverted:
            zticks_float = np.flip(-(zticks_float - zticks_float[-1]))
            zticks_str = list(reversed(zticks_str))
            zticks_str[-1] = ''
            zHighlightZero = False

    ################################################ axes lines
    lines = []
    if xtitle:
        axlinex = shapes.Line([0,0,0], [dx,0,0], c=xLineColor, lw=axesLineWidth)
        if xyShift: axlinex.shift(0,0,xyShift*dz)
        if zxShift: axlinex.shift(0,zxShift*dy,0)
        if xShiftAlongY: axlinex.shift(0,xShiftAlongY*dy,0)
        if xShiftAlongZ: axlinex.shift(0,0,xShiftAlongZ*dz)
        axlinex.name = 'xAxis'
        lines.append(axlinex)
    if ytitle:
        axliney = shapes.Line([0,0,0], [0,dy,0], c=yLineColor, lw=axesLineWidth)
        if xyShift: axliney.shift(0,0,xyShift*dz)
        if yzShift: axliney.shift(yzShift*dx,0,0)
        if yShiftAlongX: axliney.shift(yShiftAlongX*dx,0,0)
        if yShiftAlongZ: axliney.shift(0,0,yShiftAlongZ*dz)
        axliney.name = 'yAxis'
        lines.append(axliney)
    if ztitle:
        axlinez = shapes.Line([0,0,0], [0,0,dz], c=zLineColor, lw=axesLineWidth)
        if yzShift: axlinez.shift(yzShift*dx,0,0)
        if zxShift: axlinez.shift(0,zxShift*dy,0)
        if zShiftAlongX: axlinez.shift(zShiftAlongX*dx,0,0)
        if zShiftAlongY: axlinez.shift(0,zShiftAlongY*dy,0)
        axlinez.name = 'zAxis'
        lines.append(axlinez)

    ################################################ grid planes
    # all shapes have a name to keep track of them in the Assembly
    # if user wants to unpack it
    grids = []
    if xyGrid and xtitle and ytitle:
        gxy = shapes.Grid(s=(xticks_float, yticks_float))
        gxy.alpha(xyAlpha).wireframe(xyGridTransparent).c(xyPlaneColor)
        gxy.lc(xyGridColor).lw(gridLineWidth)
        if xyShift: gxy.shift(0,0,xyShift*dz)
        elif tol: gxy.shift(0,0,-tol*gscale)
        gxy.name = "xyGrid"
        grids.append(gxy)
    if yzGrid and ytitle and ztitle:
        gyz = shapes.Grid(s=(zticks_float, yticks_float))
        gyz.alpha(yzAlpha).wireframe(yzGridTransparent).c(yzPlaneColor)
        gyz.lc(yzGridColor).lw(gridLineWidth).RotateY(-90)
        if yzShift: gyz.shift(yzShift*dx,0,0)
        elif tol: gyz.shift(-tol*gscale,0,0)
        gyz.name = "yzGrid"
        grids.append(gyz)
    if zxGrid and ztitle and xtitle:
        gzx = shapes.Grid(s=(xticks_float, zticks_float))
        gzx.alpha(zxAlpha).wireframe(zxGridTransparent).c(zxPlaneColor)
        gzx.lc(zxGridColor).lw(gridLineWidth).RotateX(90)
        if zxShift: gzx.shift(0,zxShift*dy,0)
        elif tol: gzx.shift(0,-tol*gscale,0)
        gzx.name = "zxGrid"
        grids.append(gzx)
    #Grid2
    if xyGrid2 and xtitle and ytitle:
        gxy2 = shapes.Grid(s=(xticks_float, yticks_float)).z(dz)
        gxy2.alpha(xyAlpha).wireframe(xyGrid2Transparent).c(xyPlaneColor)
        gxy2.lc(xyGridColor).lw(gridLineWidth)
        if tol: gxy2.shift(0,tol*gscale,0)
        gxy2.name = "xyGrid2"
        grids.append(gxy2)
    if yzGrid2 and ytitle and ztitle:
        gyz2 = shapes.Grid(s=(zticks_float, yticks_float)).x(dx)
        gyz2.alpha(yzAlpha).wireframe(yzGrid2Transparent).c(yzPlaneColor)
        gyz2.lc(yzGridColor).lw(gridLineWidth).RotateY(-90)
        if tol: gyz2.shift(tol*gscale,0,0)
        gyz2.name = "yzGrid2"
        grids.append(gyz2)
    if zxGrid2 and ztitle and xtitle:
        gzx2 = shapes.Grid(s=(xticks_float, zticks_float)).y(dy)
        gzx2.alpha(zxAlpha).wireframe(zxGrid2Transparent).c(zxPlaneColor)
        gzx2.lc(zxGridColor).lw(gridLineWidth).RotateX(90)
        if tol: gzx2.shift(0,tol*gscale,0)
        gzx2.name = "zxGrid2"
        grids.append(gzx2)

    ################################################ frame lines
    framelines = []
    if xyFrameLine and xtitle and ytitle:
        if not xyFrameColor:
            xyFrameColor = xyGridColor
        frxy = shapes.Line([[0,dy,0],[dx,dy,0],[dx,0,0],[0,0,0],[0,dy,0]],
                           c=xyFrameColor, lw=xyFrameLine)
        if xyShift: frxy.shift(0,0,xyShift*dz)
        frxy.name = 'xyFrameLine'
        framelines.append(frxy)
    if yzFrameLine and ytitle and ztitle:
        if not yzFrameColor:
            yzFrameColor = yzGridColor
        fryz = shapes.Line([[0,0,dz],[0,dy,dz],[0,dy,0],[0,0,0],[0,0,dz]],
                           c=yzFrameColor, lw=yzFrameLine)
        if yzShift: fryz.shift(yzShift*dx,0,0)
        fryz.name = 'yzFrameLine'
        framelines.append(fryz)
    if zxFrameLine and ztitle and xtitle:
        if not zxFrameColor:
            zxFrameColor = zxGridColor
        frzx = shapes.Line([[0,0,dz],[dx,0,dz],[dx,0,0],[0,0,0],[0,0,dz]],
                           c=zxFrameColor, lw=zxFrameLine)
        if zxShift: frzx.shift(0,zxShift*dy,0)
        frzx.name = 'zxFrameLine'
        framelines.append(frzx)

    ################################################ zero lines highlights
    highlights = []
    if xyGrid and xtitle and ytitle:
        if xHighlightZero and min_bns[0] <= 0 and max_bns[1] > 0:
            xhl = -min_bns[0]
            hxy = shapes.Line([xhl,0,0], [xhl,dy,0], c=xHighlightZeroColor)
            hxy.alpha(np.sqrt(xyAlpha)).lw(gridLineWidth*2)
            if xyShift: hxy.shift(0,0,xyShift*dz)
            hxy.name = "xyHighlightZero"
            highlights.append(hxy)
        if yHighlightZero and min_bns[2] <= 0 and max_bns[3] > 0:
            yhl = -min_bns[2]
            hyx = shapes.Line([0,yhl,0], [dx,yhl,0], c=yHighlightZeroColor)
            hyx.alpha(np.sqrt(yzAlpha)).lw(gridLineWidth*2)
            if xyShift: hyx.shift(0,0,xyShift*dz)
            hyx.name = "yxHighlightZero"
            highlights.append(hyx)

    if yzGrid and ytitle and ztitle:
        if yHighlightZero and min_bns[2] <= 0 and max_bns[3] > 0:
            yhl = -min_bns[2]
            hyz = shapes.Line([0,yhl,0], [0,yhl,dz], c=yHighlightZeroColor)
            hyz.alpha(np.sqrt(yzAlpha)).lw(gridLineWidth*2)
            if yzShift: hyz.shift(yzShift*dx,0,0)
            hyz.name = "yzHighlightZero"
            highlights.append(hyz)
        if zHighlightZero and min_bns[4] <= 0 and max_bns[5] > 0:
            zhl = -min_bns[4]
            hzy = shapes.Line([0,0,zhl], [0,dy,zhl], c=zHighlightZeroColor)
            hzy.alpha(np.sqrt(yzAlpha)).lw(gridLineWidth*2)
            if yzShift: hzy.shift(yzShift*dx,0,0)
            hzy.name = "zyHighlightZero"
            highlights.append(hzy)

    if zxGrid and ztitle and xtitle:
        if zHighlightZero and min_bns[4] <= 0 and max_bns[5] > 0:
            zhl = -min_bns[4]
            hzx = shapes.Line([0,0,zhl], [dx,0,zhl], c=zHighlightZeroColor)
            hzx.alpha(np.sqrt(zxAlpha)).lw(gridLineWidth*2)
            if zxShift: hzx.shift(0,zxShift*dy,0)
            hzx.name = "zxHighlightZero"
            highlights.append(hzx)
        if xHighlightZero and min_bns[0] <= 0 and max_bns[1] > 0:
            xhl = -min_bns[0]
            hxz = shapes.Line([xhl,0,0], [xhl,0,dz], c=xHighlightZeroColor)
            hxz.alpha(np.sqrt(zxAlpha)).lw(gridLineWidth*2)
            if zxShift: hxz.shift(0,zxShift*dy,0)
            hxz.name = "xzHighlightZero"
            highlights.append(hxz)

    ################################################ arrow cone
    cones = []

    if tipSize:

        if xtitle:
            if xInverted:
                cx = shapes.Cone(r=tipSize, height=tipSize*2,
                                 axis=(-1,0,0), c=xLineColor, res=12)
            else:
                cx = shapes.Cone((dx,0,0), r=tipSize, height=tipSize*2,
                                 axis=(1,0,0), c=xLineColor, res=12)
            if xyShift: cx.shift(0,0,xyShift*dz)
            if zxShift: cx.shift(0,zxShift*dy,0)
            if xShiftAlongY: cx.shift(0,xShiftAlongY*dy,0)
            if xShiftAlongZ: cx.shift(0,0,xShiftAlongZ*dz)
            cx.name = "xTipCone"
            cones.append(cx)

        if ytitle:
            if yInverted:
                cy = shapes.Cone(r=tipSize, height=tipSize*2,
                                 axis=(0,-1,0), c=yLineColor, res=12)
            else:
                cy = shapes.Cone((0,dy,0), r=tipSize, height=tipSize*2,
                                 axis=(0,1,0), c=yLineColor, res=12)
            if xyShift: cy.shift(0,0,xyShift*dz)
            if yzShift: cy.shift(yzShift*dx,0,0)
            if yShiftAlongX: cy.shift(yShiftAlongX*dx,0,0)
            if yShiftAlongZ: cy.shift(0,0,yShiftAlongZ*dz)
            cy.name = "yTipCone"
            cones.append(cy)

        if ztitle:
            if zInverted:
                cz = shapes.Cone(r=tipSize, height=tipSize*2,
                                 axis=(0,0,-1), c=zLineColor, res=12)
            else:
                cz = shapes.Cone((0,0,dz), r=tipSize, height=tipSize*2,
                                 axis=(0,0,1), c=zLineColor, res=12)
            if yzShift: cz.shift(yzShift*dx,0,0)
            if zxShift: cz.shift(0,zxShift*dy,0)
            if zShiftAlongX: cz.shift(zShiftAlongX*dx,0,0)
            if zShiftAlongY: cz.shift(0,zShiftAlongY*dy,0)
            cz.name = "zTipCone"
            cones.append(cz)

    ################################################################# MAJOR ticks
    majorticks, minorticks= [], []
    xticks, yticks, zticks = [],[],[]
    if showTicks:
        if xtitle:
            tickThickness = xTickThickness * gscale/2
            tickLength = xTickLength * gscale/2
            for i in range(1, len(xticks_float)-1):
                v1 = (xticks_float[i]-tickThickness, -tickLength, 0)
                v2 = (xticks_float[i]+tickThickness,  tickLength, 0)
                xticks.append(shapes.Rectangle(v1, v2))
            if len(xticks)>1:
                xmajticks = merge(xticks).c(xLabelColor)
                if xAxisRotation:
                    xmajticks.RotateX(xAxisRotation)
                if xyShift: xmajticks.shift(0,0,xyShift*dz)
                if zxShift: xmajticks.shift(0,zxShift*dy,0)
                if xShiftAlongY: xmajticks.shift(0,xShiftAlongY*dy,0)
                if xShiftAlongZ: xmajticks.shift(0,0,xShiftAlongZ*dz)
                xmajticks.name = "xMajorTicks"
                majorticks.append(xmajticks)
        if ytitle:
            tickThickness = yTickThickness * gscale/2
            tickLength = yTickLength * gscale/2
            for i in range(1, len(yticks_float)-1):
                v1 = (-tickLength, yticks_float[i]-tickThickness, 0)
                v2 = ( tickLength, yticks_float[i]+tickThickness, 0)
                yticks.append(shapes.Rectangle(v1, v2))
            if len(yticks)>1:
                ymajticks = merge(yticks).c(yLabelColor)
                if yAxisRotation:
                    ymajticks.RotateY(yAxisRotation)
                if xyShift: ymajticks.shift(0,0,xyShift*dz)
                if yzShift: ymajticks.shift(yzShift*dx,0,0)
                if yShiftAlongX: ymajticks.shift(yShiftAlongX*dx,0,0)
                if yShiftAlongZ: ymajticks.shift(0,0,yShiftAlongZ*dz)
                ymajticks.name = "yMajorTicks"
                majorticks.append(ymajticks)
        if ztitle:
            tickThickness = zTickThickness * gscale/2
            tickLength = zTickLength * gscale/2.85
            for i in range(1, len(zticks_float)-1):
                v1 = (zticks_float[i]-tickThickness, -tickLength, 0)
                v2 = (zticks_float[i]+tickThickness,  tickLength, 0)
                zticks.append(shapes.Rectangle(v1, v2))
            if len(zticks)>1:
                zmajticks = merge(zticks).c(zLabelColor)
                zmajticks.RotateZ(-45 + zAxisRotation)
                zmajticks.RotateY(-90)
                if yzShift: zmajticks.shift(yzShift*dx,0,0)
                if zxShift: zmajticks.shift(0,zxShift*dy,0)
                if zShiftAlongX: zmajticks.shift(zShiftAlongX*dx,0,0)
                if zShiftAlongY: zmajticks.shift(0,zShiftAlongY*dy,0)
                zmajticks.name = "zMajorTicks"
                majorticks.append(zmajticks)

        ############################################################# MINOR ticks
        if xtitle and xMinorTicks and len(xticks)>1:
            tickThickness = xTickThickness * gscale/4
            tickLength = xTickLength * gscale/4
            xMinorTicks += 1
            ticks = []
            for i in range(1,len(xticks)):
                t0, t1 = xticks[i-1].pos(), xticks[i].pos()
                dt = t1 - t0
                for j in range(1, xMinorTicks):
                    mt = dt*(j/xMinorTicks) + t0
                    v1 = (mt[0]-tickThickness, -tickLength, 0)
                    v2 = (mt[0]+tickThickness,  tickLength, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = xticks[0].pos(), xticks[1].pos()
            dt = t1 - t0
            for j in range(1, xMinorTicks):
                mt = t0 - dt*(j/xMinorTicks)
                if mt[0]<0: break
                v1 = (mt[0]-tickThickness, -tickLength, 0)
                v2 = (mt[0]+tickThickness,  tickLength, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = xticks[-2].pos(), xticks[-1].pos()
            dt = t1 - t0
            for j in range(1, xMinorTicks):
                mt = t1 + dt*(j/xMinorTicks)
                if mt[0]>dx: break
                v1 = (mt[0]-tickThickness, -tickLength, 0)
                v2 = (mt[0]+tickThickness,  tickLength, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if len(ticks):
                xminticks = merge(ticks).c(xLabelColor)
                if xAxisRotation:
                    xminticks.RotateX(xAxisRotation)
                if xyShift: xminticks.shift(0,0,xyShift*dz)
                if zxShift: xminticks.shift(0,zxShift*dy,0)
                if xShiftAlongY: xminticks.shift(0,xShiftAlongY*dy,0)
                if xShiftAlongZ: xminticks.shift(0,0,xShiftAlongZ*dz)
                xminticks.name = "xMinorTicks"
                minorticks.append(xminticks)

        if ytitle and yMinorTicks and len(yticks)>1:   ##### y
            tickThickness = yTickThickness * gscale/4
            tickLength = yTickLength * gscale/4
            yMinorTicks += 1
            ticks = []
            for i in range(1,len(yticks)):
                t0, t1 = yticks[i-1].pos(), yticks[i].pos()
                dt = t1 - t0
                for j in range(1, yMinorTicks):
                    mt = dt*(j/yMinorTicks) + t0
                    v1 = (-tickLength, mt[1]-tickThickness, 0)
                    v2 = ( tickLength, mt[1]+tickThickness, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = yticks[0].pos(), yticks[1].pos()
            dt = t1 - t0
            for j in range(1, yMinorTicks):
                mt = t0 - dt*(j/yMinorTicks)
                if mt[1]<0: break
                v1 = (-tickLength, mt[1]-tickThickness, 0)
                v2 = ( tickLength, mt[1]+tickThickness, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = yticks[-2].pos(), yticks[-1].pos()
            dt = t1 - t0
            for j in range(1, yMinorTicks):
                mt = t1 + dt*(j/yMinorTicks)
                if mt[1]>dy: break
                v1 = (-tickLength, mt[1]-tickThickness, 0)
                v2 = ( tickLength, mt[1]+tickThickness, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if len(ticks):
                yminticks = merge(ticks).c(yLabelColor)
                if yAxisRotation:
                    yminticks.RotateY(yAxisRotation)
                if xyShift: yminticks.shift(0,0,xyShift*dz)
                if yzShift: yminticks.shift(yzShift*dx,0,0)
                if yShiftAlongX: yminticks.shift(yShiftAlongX*dx,0,0)
                if yShiftAlongZ: yminticks.shift(0,0,yShiftAlongZ*dz)
                yminticks.name = "yMinorTicks"
                minorticks.append(yminticks)

        if ztitle and zMinorTicks and len(zticks)>1:   ##### z
            tickThickness = zTickThickness * gscale/4
            tickLength = zTickLength * gscale/5
            zMinorTicks += 1
            ticks = []
            for i in range(1,len(zticks)):
                t0, t1 = zticks[i-1].pos(), zticks[i].pos()
                dt = t1 - t0
                for j in range(1, zMinorTicks):
                    mt = dt*(j/zMinorTicks) + t0
                    v1 = (mt[0]-tickThickness, -tickLength, 0)
                    v2 = (mt[0]+tickThickness,  tickLength, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = zticks[0].pos(), zticks[1].pos()
            dt = t1 - t0
            for j in range(1, zMinorTicks):
                mt = t0 - dt*(j/zMinorTicks)
                if mt[0]<0: break
                v1 = (mt[0]-tickThickness, -tickLength, 0)
                v2 = (mt[0]+tickThickness,  tickLength, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = zticks[-2].pos(), zticks[-1].pos()
            dt = t1 - t0
            for j in range(1, zMinorTicks):
                mt = t1 + dt*(j/zMinorTicks)
                if mt[0]>dz: break
                v1 = (mt[0]-tickThickness, -tickLength, 0)
                v2 = (mt[0]+tickThickness,  tickLength, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if len(ticks):
                zminticks = merge(ticks).c(zLabelColor)
                zminticks.RotateZ(-45 + zAxisRotation)
                zminticks.RotateY(-90)
                if yzShift: zminticks.shift(yzShift*dx,0,0)
                if zxShift: zminticks.shift(0,zxShift*dy,0)
                if zShiftAlongX: zminticks.shift(zShiftAlongX*dx,0,0)
                if zShiftAlongY: zminticks.shift(0,zShiftAlongY*dy,0)
                zminticks.name = "zMinorTicks"
                minorticks.append(zminticks)

    ################################################ axes NUMERIC text labels
    labels = []
    xlab, ylab, zlab = None, None, None

    if xLabelSize and xtitle:

        xRot,yRot,zRot = 0,0,0
        if utils.isSequence(xLabelRotation):  # unpck 3 rotations
            zRot, xRot, yRot = xLabelRotation
        else:
            zRot = xLabelRotation
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
        if xLabelJustify is not None:
            jus = xLabelJustify

        for i in range(1, len(xticks_str)):
            t = xticks_str[i]
            if not t:
                continue
            if utils.isSequence(xLabelOffset):
                xoffs, yoffs, zoffs = xLabelOffset
            else:
                xoffs, yoffs, zoffs = 0, xLabelOffset, 0
            xlab = shapes.Text3D(t, s=xLabelSize*textScale*gscale,
                                 font=labelFont, justify=jus)
            tb = xlab.ybounds() # must be ybounds: height of char
            v = (xticks_float[i], 0, 0)
            offs = -np.array([xoffs, yoffs, zoffs])*(tb[1]-tb[0])
            xlab.pos(v+offs)
            if xAxisRotation:
                xlab.rotateX(xAxisRotation)
            if zRot: xlab.RotateZ(zRot)
            if xRot: xlab.RotateX(xRot)
            if yRot: xlab.RotateY(yRot)
            if xyShift: xlab.shift(0,0,xyShift*dz)
            if zxShift: xlab.shift(0,zxShift*dy,0)
            if xShiftAlongY: xlab.shift(0,xShiftAlongY*dy,0)
            if xShiftAlongZ: xlab.shift(0,0,xShiftAlongZ*dz)
            xlab.name = f"xNumericLabel{i}"
            xlab.SetUseBounds(xUseBounds)
            labels.append(xlab.c(xLabelColor))

    if yLabelSize and ytitle:

        xRot,yRot,zRot = 0,0,0
        if utils.isSequence(yLabelRotation):  # unpck 3 rotations
            zRot, yRot, xRot = yLabelRotation
        else:
            zRot = yLabelRotation
        if zRot < 0: zRot += 360 # deal with negative angles

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
        if yLabelJustify is not None:
            jus = yLabelJustify

        for i in range(1, len(yticks_str)):
            t = yticks_str[i]
            if not t:
                continue
            if utils.isSequence(yLabelOffset):
                xoffs, yoffs, zoffs = yLabelOffset
            else:
                xoffs, yoffs, zoffs = yLabelOffset, 0, 0
            ylab = shapes.Text3D(t, s=yLabelSize*textScale*gscale,
                                 font=labelFont, justify=jus)
            tb = ylab.ybounds() # must be ybounds: height of char
            v = (0, yticks_float[i], 0)
            offs = -np.array([xoffs, yoffs, zoffs])*(tb[1]-tb[0])
            ylab.pos(v+offs)
            if yAxisRotation:
                ylab.rotateY(yAxisRotation)
            if zRot: ylab.RotateZ(zRot)
            if yRot: ylab.RotateY(yRot)
            if xRot: ylab.RotateX(xRot)
            if xyShift: ylab.shift(0,0,xyShift*dz)
            if yzShift: ylab.shift(yzShift*dx,0,0)
            if yShiftAlongX: ylab.shift(yShiftAlongX*dx,0,0)
            if yShiftAlongZ: ylab.shift(0,0,yShiftAlongZ*dz)
            ylab.name = f"yNumericLabel{i}"
            ylab.SetUseBounds(yUseBounds)
            labels.append(ylab.c(yLabelColor))

    if zLabelSize and ztitle:

        xRot,yRot,zRot = 0,0,0
        if utils.isSequence(zLabelRotation):  # unpck 3 rotations
            xRot, yRot, zRot = zLabelRotation
        else:
            xRot = zLabelRotation
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
        if zLabelJustify is not None:
            jus = zLabelJustify

        for i in range(1, len(zticks_str)):
            t = zticks_str[i]
            if not t:
                continue
            if utils.isSequence(zLabelOffset):
                xoffs, yoffs, zoffs = zLabelOffset
            else:
                xoffs, yoffs, zoffs = zLabelOffset, zLabelOffset, 0
            zlab = shapes.Text3D(t, s=zLabelSize*textScale*gscale,
                                 font=labelFont, justify=jus)
            tb = zlab.ybounds() # must be ybounds: height of char
            v = (0, 0, zticks_float[i])
            offs = -np.array([xoffs, yoffs, zoffs])*(tb[1]-tb[0])/1.5
            angle=90
            if dx: angle = np.arctan2(dy,dx)*57.3
            zlab.RotateZ(angle + yRot)    # vtk inverts order of rotations
            if xRot: zlab.RotateY(-xRot)  # ..second
            zlab.RotateX(90+zRot)         # ..first
            zlab.pos(v+offs)
            if zAxisRotation:
                zlab.rotateZ(zAxisRotation)
            if yzShift: zlab.shift(yzShift*dx,0,0)
            if zxShift: zlab.shift(0,zxShift*dy,0)
            if zShiftAlongX: zlab.shift(zShiftAlongX*dx,0,0)
            if zShiftAlongY: zlab.shift(0,zShiftAlongY*dy,0)
            zlab.SetUseBounds(zUseBounds)
            zlab.name = f"zNumericLabel{i}"
            labels.append(zlab.c(zLabelColor))

    ################################################ axes titles
    titles = []

    if xtitle:

        xRot,yRot,zRot = 0,0,0
        if utils.isSequence(xTitleRotation):  # unpack 3 rotations
            zRot, xRot, yRot = xTitleRotation
        else:
            zRot = xTitleRotation
        if zRot < 0:  # deal with negative angles
            zRot += 360

        if utils.isSequence(xTitleOffset):
            xoffs, yoffs, zoffs = xTitleOffset
        else:
            xoffs, yoffs, zoffs = 0, xTitleOffset, 0

        if xTitleJustify is not None:
            jus = xTitleJustify
        else:
            # find best justfication for given rotation(s)
            jus ="right-top"
            if zRot:
                if zRot >  24: jus = "center-right"
                if zRot >  67: jus = "right-bottom"
                if zRot > 157: jus = "bottom-left"
                if zRot > 202: jus = "center-left"
                if zRot > 247: jus = "top-left"
                if zRot > 337: jus = "top-right"

        xt = shapes.Text3D(xtitle, s=xTitleSize*textScale*gscale,
                           font=titleFont, c=xTitleColor, justify=jus,
                           depth=titleDepth, italic=xTitleItalic)
        if xTitleBackfaceColor:
            xt.backColor(xTitleBackfaceColor)
        if zRot: xt.RotateZ(zRot)
        if xRot: xt.RotateX(xRot)
        if yRot: xt.RotateY(yRot)
        shift = 0
        if xlab: # xlab is the last created numeric text label..
            lt0, lt1 = xlab.GetBounds()[2:4]
            shift =  lt1 - lt0
        xt.pos([(xoffs+xTitlePosition)*dx,
                -(yoffs+xTickLength/2)*dy -shift, zoffs*dz])
        if xAxisRotation:
            xt.rotateX(xAxisRotation)
        if xyShift: xt.shift(0,0,xyShift*dz)
        if xShiftAlongY: xt.shift(0,xShiftAlongY*dy,0)
        if xShiftAlongZ: xt.shift(0,0,xShiftAlongZ*dz)
        xt.SetUseBounds(xUseBounds)
        if xtitle == " ":
            xt.SetUseBounds(False)
        xt.name = f"xtitle {xtitle}"
        titles.append(xt)
        if xTitleBox:
            titles.append(xt.box(scale=1.1).useBounds(xUseBounds))

    if ytitle:

        xRot,yRot,zRot = 0,0,0
        if utils.isSequence(yTitleRotation):  # unpck 3 rotations
            zRot, yRot, xRot = yTitleRotation
        else:
            zRot = yTitleRotation
            if len(ytitle) > 3:
                zRot += 90
                yTitlePosition *= 0.975
        if zRot < 0:
            zRot += 360 # deal with negative angles

        if utils.isSequence(yTitleOffset):
            xoffs, yoffs, zoffs = yTitleOffset
        else:
            xoffs, yoffs, zoffs = yTitleOffset, 0, 0

        if yTitleJustify is not None:
            jus = yTitleJustify
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
            ytitle, s=yTitleSize*textScale*gscale, font=titleFont,
            c=yTitleColor, justify=jus, depth=titleDepth,
            italic=yTitleItalic,
        )
        if yTitleBackfaceColor:
            yt.backColor(yTitleBackfaceColor)

        if zRot: yt.RotateZ(zRot)
        if yRot: yt.RotateY(yRot)
        if xRot: yt.RotateX(xRot)

        shift = 0
        if ylab:  # this is the last created num label..
            lt0, lt1 = ylab.GetBounds()[0:2]
            shift = lt1 - lt0
        yt.pos(-(xoffs + yTickLength/2)*dx -shift,
               (yoffs + yTitlePosition)*dy, zoffs*dz)
        if yAxisRotation:
            yt.rotateY(yAxisRotation)
        if xyShift:      yt.shift(0, 0, xyShift*dz)
        if yShiftAlongX: yt.shift(yShiftAlongX*dx, 0, 0)
        if yShiftAlongZ: yt.shift(0, 0, yShiftAlongZ*dz)
        yt.SetUseBounds(yUseBounds)
        if ytitle == " ":
            yt.SetUseBounds(False)
        yt.name = f"ytitle {ytitle}"
        titles.append(yt)
        if yTitleBox:
            titles.append(yt.box(scale=1.1).useBounds(yUseBounds))

    if ztitle:

        xRot,yRot,zRot = 0,0,0
        if utils.isSequence(zTitleRotation):  # unpck 3 rotations
            xRot, yRot, zRot = zTitleRotation
        else:
            xRot = zTitleRotation
            if len(ztitle) > 3:
                xRot += 90
                zTitlePosition *= 0.975
        if xRot < 0:
            xRot += 360 # deal with negative angles

        if zTitleJustify is not None:
            jus = zTitleJustify
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
            ztitle, s=zTitleSize*textScale*gscale, font=titleFont,
            c=zTitleColor, justify=jus, depth=titleDepth,
            italic=yTitleItalic,
        )
        if zTitleBackfaceColor:
            zt.backColor(zTitleBackfaceColor)

        angle=90
        if dx: angle = np.arctan2(dy,dx)*57.3
        zt.RotateZ(angle+yRot) # vtk inverts order of rotations
        if xRot: zt.RotateY(-xRot)  # ..second
        zt.RotateX(90+zRot)         # ..first

        shift = 0
        if zlab: # this is the last created one..
            lt0, lt1 = zlab.GetBounds()[0:2]
            shift = lt1 - lt0
        zt.pos(-(zTitleOffset+zTickLength/5)*dx-shift,
               -(zTitleOffset+zTickLength/5)*dy-shift, zTitlePosition*dz)
        if zAxisRotation:
            zt.rotateZ(zAxisRotation)
        if zxShift: zt.shift(0,zxShift*dy,0)
        if zShiftAlongX: zt.shift(zShiftAlongX*dx,0,0)
        if zShiftAlongY: zt.shift(0,zShiftAlongY*dy,0)
        zt.SetUseBounds(zUseBounds)
        if ztitle == " ":
            zt.SetUseBounds(False)
        zt.name = f"ztitle {ztitle}"
        titles.append(zt)

    ################################################### header title
    if htitle:
        if hTitleFont is None:
            hTitleFont = titleFont
        if hTitleColor is None:
            hTitleColor = xTitleColor
        htit = shapes.Text3D(
            htitle, s=hTitleSize*gscale*textScale, font=hTitleFont,
            c=hTitleColor, justify=hTitleJustify, depth=titleDepth,
            italic=hTitleItalic,
        )
        if hTitleRotation:
            htit.RotateX(hTitleRotation)
        wpos = [(0.5+hTitleOffset[0])*dx,
                (1+hTitleOffset[1])*dy, hTitleOffset[2]*dz]
        htit.pos(wpos)
        if xyShift: htit.shift(0,0,xyShift*dz)
        htit.name = f"htitle {htitle}"
        titles.append(htit)

    ######
    acts = titles + lines + labels + grids + framelines
    acts+= highlights + majorticks + minorticks + cones
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


def addGlobalAxes(axtype=None, c=None):
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
            b = Box(pos=(0,0,0), length=80, width=90, height=70).alpha(0.1)
            show(b, axes={ 'xtitle':'Some long variable [a.u.]',
                           'numberOfDivisions':4,
                           # ...
                         }
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
            asse = Axes(None, **plt.axes)
        else:
            asse = Axes(None, useGlobal=True)

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

        if dx > aves/100:
            xl = shapes.Cylinder([[x0, 0, 0], [x1, 0, 0]], r=aves/250*s, c=xcol, alpha=alpha)
            xc = shapes.Cone(pos=[x1, 0, 0], c=xcol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[1, 0, 0], res=10)
            wpos = [x1, -aves/25*s, 0]  # aligned to arrow tip
            if centered:
                wpos = [(x0 + x1) / 2, -aves / 25 * s, 0]
            xt = shapes.Text3D('x', pos=wpos, s=aves / 40 * s, c=xcol)
            acts += [xl, xc, xt]

        if dy > aves/100:
            yl = shapes.Cylinder([[0, y0, 0], [0, y1, 0]], r=aves/250*s, c=ycol, alpha=alpha)
            yc = shapes.Cone(pos=[0, y1, 0], c=ycol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[0, 1, 0], res=10)
            wpos = [-aves/40*s, y1, 0]
            if centered:
                wpos = [-aves / 40 * s, (y0 + y1) / 2, 0]
            yt = shapes.Text3D('y', pos=(0, 0, 0), s=aves / 40 * s, c=ycol)
            yt.pos(wpos).RotateZ(90)
            acts += [yl, yc, yt]

        if dz > aves/100:
            zl = shapes.Cylinder([[0, 0, z0], [0, 0, z1]], r=aves/250*s, c=zcol, alpha=alpha)
            zc = shapes.Cone(pos=[0, 0, z1], c=zcol, alpha=alpha,
                             r=aves/100*s, height=aves/25*s, axis=[0, 0, 1], res=10)
            wpos = [-aves/50*s, -aves/50*s, z1]
            if centered:
                wpos = [-aves/50*s, -aves/50*s, (z0+z1)/2]
            zt = shapes.Text3D('z', pos=(0,0,0), s=aves/40*s, c=zcol)
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
        axact.SetXAxisLabelText('x')
        axact.SetYAxisLabelText('y')
        axact.SetZAxisLabelText('z')
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
        axact.SetXPlusFaceText (settings.annotatedCubeTexts[0])  #XPlus
        axact.SetXMinusFaceText(settings.annotatedCubeTexts[1])  #XMinus
        axact.SetYPlusFaceText (settings.annotatedCubeTexts[2])  #YPlus
        axact.SetYMinusFaceText(settings.annotatedCubeTexts[3])  #YMinus
        axact.SetZPlusFaceText (settings.annotatedCubeTexts[4])  #ZPlus
        axact.SetZMinusFaceText(settings.annotatedCubeTexts[5])  #ZMinus
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
        rulax = RulerAxes(vbb, c=c,
                          xtitle='x - ',
                          ytitle='y - ',
                          ztitle='z - ')
        plt.axes_instances[r] = rulax
        if not rulax:
            return None
        rulax.UseBoundsOff()
        rulax.PickableOff()
        plt.renderer.AddActor(rulax)

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
        ca.SetXTitle('x')
        ca.SetYTitle('y')
        ca.SetZTitle('z')
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
        xc.clean().alpha(0.5).wireframe().lineWidth(2).PickableOff()
        yc.clean().alpha(0.5).wireframe().lineWidth(2).PickableOff()
        zc.clean().alpha(0.5).wireframe().lineWidth(2).PickableOff()
        ca = xc + yc + zc
        ca.PickableOff()
        ca.UseBoundsOff()
        plt.renderer.AddActor(ca)
        plt.axes_instances[r] = ca

    elif plt.axes == 11:
        vbb, ss = computeVisibleBounds()[0:2]
        xpos, ypos = (vbb[1] + vbb[0]) /2, (vbb[3] + vbb[2]) /2
        gs = sum(ss)*3
        gr = shapes.Grid((xpos, ypos, vbb[4]), s=(gs,gs), res=(11,11), c=c, alpha=0.1)
        gr.lighting('off').PickableOff()
        gr.UseBoundsOff()
        plt.renderer.AddActor(gr)
        plt.axes_instances[r] = gr

    elif plt.axes == 12:
        polaxes = vtk.vtkPolarAxesActor()
        vbb = computeVisibleBounds()[0]

        polaxes.SetPolarAxisTitle('radial distance')
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
        pr = ls.GetBottomAxis().GetLabelTextProperty()
        pr.SetFontFamily(vtk.VTK_FONT_FILE)
        pr.SetFontFile(utils.getFontPath(settings.defaultFont))
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
        e = '\bomb Keyword axes type must be in range [0-13].'
        e+= 'Available axes types are:\n\n'
        e+= '0 = no axes\n'
        e+= '1 = draw three customizable gray grid walls\n'
        e+= '2 = show cartesian axes from (0,0,0)\n'
        e+= '3 = show positive range of cartesian axes from (0,0,0)\n'
        e+= '4 = show a triad at bottom left\n'
        e+= '5 = show a cube at bottom left\n'
        e+= '6 = mark the corners of the bounding box\n'
        e+= '7 = draw a 3D ruler at each side of the cartesian axes\n'
        e+= '8 = show the vtk default vtkCubeAxesActor object\n'
        e+= '9 = show the bounding box outline\n'
        e+= '10 = show three circles representing the maximum bounding box\n'
        e+= '11 = show a large grid on the x-y plane (use with zoom=8)\n'
        e+= '12 = show polar axes\n'
        e+= '13 = draw a simple ruler at the bottom of the window\n'
        e+= '14 = show the vtk default vtkCameraOrientationWidget object'
        vedo.logger.warning(e)

    if not plt.axes_instances[r]:
        plt.axes_instances[r] = True
    return
