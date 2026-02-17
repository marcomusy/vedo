#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Core utility functions extracted from vedo.addons."""

import numpy as np
from typing_extensions import Self

import vedo
import vedo.vtkclasses as vtki

from vedo import utils
from vedo import shapes
from vedo.colors import get_color, build_lut
from vedo.assembly import Assembly
from vedo.mesh import Mesh
from vedo.pointcloud import Points
from vedo.grids import TetMesh
from vedo.volume import Volume
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
        s=r / 12 * s,
        font=font,
        italic=italic,
        justify="center",
    )
    cr = np.cross(va, vb)
    lb.reorient([0, 0, 1], cr * np.sign(cr[2]), rotation=rotation, xyplane=False)
    lb.pos(p2 + vc * r / 1.75)
    lb.c(c).bc("tomato").lighting("off")
    acts.append(lb)

    if alpha > 0:
        pts = [p2] + arc.coordinates.tolist() + [p2]
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
    pos=(),
    size=(80, 400),
    font_size=14,
    title_yoffset=20,
    nlabels=None,
    c="k",
    horizontal=False,
    use_alpha=True,
    label_format=":6.3g",
) -> vtki.vtkScalarBarActor | None:
    """
    A 2D scalar bar for the specified object.

    Arguments:
        title : (str)
            scalar bar title
        pos : (list)
            position coordinates of the bottom left corner.
            Can also be a pair of (x,y) values in the range [0,1]
            to indicate the position of the bottom-left and top-right corners.
        size : (float,float)
            size of the scalarbar in number of pixels (width, height)
        font_size : (float)
            size of font for title and numeric labels
        title_yoffset : (float)
            vertical space offset between title and color scalarbar
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
        titprop.SetFontFile(utils.get_font_path(vedo.settings.default_font))
        sb.SetTitle(title)
        sb.SetVerticalTitleSeparation(title_yoffset)
        sb.SetTitleTextProperty(titprop)

    sb.SetTextPad(0)
    sb.UnconstrainedFontSizeOn()
    sb.DrawAnnotationsOn()
    sb.DrawTickLabelsOn()
    sb.SetMaximumNumberOfColors(256)
    if nlabels is not None:
        sb.SetNumberOfLabels(nlabels)

    if len(pos) == 0 or utils.is_sequence(pos[0]):
        if len(pos) == 0:
            pos = ((0.87, 0.05), (0.97, 0.5))
            if horizontal:
                pos = ((0.5, 0.05), (0.97, 0.15))
        sb.SetTextPositionToPrecedeScalarBar()
        if horizontal:
            if not nlabels: sb.SetNumberOfLabels(3)
            sb.SetOrientationToHorizontal()
            sb.SetTextPositionToSucceedScalarBar()
    else:

        if horizontal:
            size = (size[1], size[0])  # swap size
            sb.SetPosition(pos[0]-0.7, pos[1])
            if not nlabels: sb.SetNumberOfLabels(3)
            sb.SetOrientationToHorizontal()
            sb.SetTextPositionToSucceedScalarBar()
        else:
            sb.SetPosition(pos[0], pos[1])
            if not nlabels: sb.SetNumberOfLabels(7)
            sb.SetTextPositionToPrecedeScalarBar()
        sb.SetHeight(1)
        sb.SetWidth(1)
        if size[0] is not None: sb.SetMaximumWidthInPixels(size[0])
        if size[1] is not None: sb.SetMaximumHeightInPixels(size[1])

    sb.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    sb.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()

    s = np.array(pos[1]) - np.array(pos[0])
    sb.GetPositionCoordinate().SetValue(pos[0][0], pos[0][1])
    sb.GetPosition2Coordinate().SetValue(s[0], s[1]) # size !!??

    sctxt = sb.GetLabelTextProperty()
    sctxt.SetFontFamily(vtki.VTK_FONT_FILE)
    sctxt.SetFontFile(utils.get_font_path(vedo.settings.default_font))
    sctxt.SetColor(c)
    sctxt.SetShadow(0)
    sctxt.SetFontSize(font_size)
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
    c="k",
    draw_box=True,
    above_text=None,
    below_text=None,
    nan_text="NaN",
    categories=None,
) -> Assembly | None:
    """
    Create a 3D scalar bar for the specified object.

    Input `obj` input can be:

        - a look-up-table,
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

    elif isinstance(obj, vtki.vtkLookupTable):
        lut = obj
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
        pos = (bns[1] + tsize + sx * 1.5, (bns[2] + bns[3]) / 2, bns[4])

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

    bb = asse.actor.GetBounds()
    # print("ScalarBar3D pos",pos, bb)
    # asse.SetOrigin(pos)

    asse.actor.SetOrigin(bb[0], bb[2], bb[4])
    # asse.SetOrigin(bb[0],0,0) #in pyplot line 1312

    asse.actor.PickableOff()
    asse.actor.UseBoundsOff()
    asse.name = "ScalarBar3D"
    return asse


#####################################################################
