#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Statistical plotting helpers."""

from typing_extensions import Self
import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import settings
from vedo.transformations import cart2spher, spher2cart
from vedo import addons
from vedo import colors
from vedo import utils
from vedo import shapes
from vedo.pointcloud import Points, merge
from vedo.mesh import Mesh
from vedo.assembly import Assembly

from .functions import _histogram_polar

def pie_chart(
    fractions,
    title="",
    tsize=0.3,
    r1=1.7,
    r2=1,
    phigap=0,
    lpos=0.8,
    lsize=0.15,
    c=None,
    bc="k",
    alpha=1,
    labels=(),
    show_disc=False,
) -> Assembly:
    """
    Donut plot or pie chart.

    Arguments:
        title : (str)
            plot title
        tsize : (float)
            title size
        r1 : (float) inner radius
        r2 : (float)
            outer radius, starting from r1
        phigap : (float)
            gap angle btw 2 radial bars, in degrees
        lpos : (float)
            label gap factor along radius
        lsize : (float)
            label size
        c : (color)
            color of the plot slices
        bc : (color)
            color of the disc frame
        alpha : (float)
            opacity of the disc frame
        labels : (list)
            list of labels
        show_disc : (bool)
            show the outer ring axis

    Examples:
        - [donut.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/donut.py)

            ![](https://vedo.embl.es/images/pyplot/donut.png)
    """
    fractions = np.array(fractions, dtype=float)
    angles = np.add.accumulate(2 * np.pi * fractions)
    angles[-1] = 2 * np.pi
    if angles[-2] > 2 * np.pi:
        print("Error in donut(): fractions must sum to 1.")
        raise RuntimeError

    cols = []
    for i, th in enumerate(np.linspace(0, 2 * np.pi, 360, endpoint=False)):
        for ia, a in enumerate(angles):
            if th < a:
                cols.append(c[ia])
                break
    labs = []
    if labels:
        angles = np.concatenate([[0], angles])
        labs = [""] * 360
        for i in range(len(labels)):
            a = (angles[i + 1] + angles[i]) / 2
            j = int(a / np.pi * 180)
            labs[j] = labels[i]

    data = np.linspace(0, 2 * np.pi, 360, endpoint=False) + 0.005
    dn = _histogram_polar(
        data,
        title=title,
        bins=360,
        r1=r1,
        r2=r2,
        phigap=phigap,
        lpos=lpos,
        lsize=lsize,
        tsize=tsize,
        c=cols,
        bc=bc,
        alpha=alpha,
        vmin=0,
        vmax=1,
        labels=labs,
        show_disc=show_disc,
        show_lines=0,
        show_angles=0,
        show_errors=0,
    )
    dn.name = "Donut"
    return dn


def violin(
    values,
    bins=10,
    vlim=None,
    x=0,
    width=3,
    splined=True,
    fill=True,
    c="violet",
    alpha=1,
    outline=True,
    centerline=True,
    lc="darkorchid",
    lw=3,
) -> Assembly:
    """
    Violin style histogram.

    Arguments:
        bins : (int)
            number of bins
        vlim : (list)
            input value limits. Crop values outside range
        x : (float)
            x-position of the violin axis
        width : (float)
            width factor of the normalized distribution
        splined : (bool)
            spline the outline
        fill : (bool)
            fill violin with solid color
        outline : (bool)
            add the distribution outline
        centerline : (bool)
            add the vertical centerline at x
        lc : (color)
            line color

    Examples:
        - [histo_violin.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_violin.py)

            ![](https://vedo.embl.es/images/pyplot/histo_violin.png)
    """
    fs, edges = np.histogram(values, bins=bins, range=vlim)
    mine, maxe = np.min(edges), np.max(edges)
    fs = fs.astype(float) / len(values) * width

    rs = []

    if splined:
        lnl, lnr = [(0, edges[0], 0)], [(0, edges[0], 0)]
        for i in range(bins):
            xc = (edges[i] + edges[i + 1]) / 2
            yc = fs[i]
            lnl.append([-yc, xc, 0])
            lnr.append([yc, xc, 0])
        lnl.append((0, edges[-1], 0))
        lnr.append((0, edges[-1], 0))
        spl = shapes.KSpline(lnl).x(x)
        spr = shapes.KSpline(lnr).x(x)
        spl.color(lc).alpha(alpha).lw(lw)
        spr.color(lc).alpha(alpha).lw(lw)
        if outline:
            rs.append(spl)
            rs.append(spr)
        if fill:
            rb = shapes.Ribbon(spl, spr, c=c, alpha=alpha).lighting("off")
            rs.append(rb)

    else:
        lns1 = [[0, mine, 0]]
        for i in range(bins):
            lns1.append([fs[i], edges[i], 0])
            lns1.append([fs[i], edges[i + 1], 0])
        lns1.append([0, maxe, 0])

        lns2 = [[0, mine, 0]]
        for i in range(bins):
            lns2.append([-fs[i], edges[i], 0])
            lns2.append([-fs[i], edges[i + 1], 0])
        lns2.append([0, maxe, 0])

        if outline:
            rs.append(shapes.Line(lns1, c=lc, alpha=alpha, lw=lw).x(x))
            rs.append(shapes.Line(lns2, c=lc, alpha=alpha, lw=lw).x(x))

        if fill:
            for i in range(bins):
                p0 = (-fs[i], edges[i], 0)
                p1 = (fs[i], edges[i + 1], 0)
                r = shapes.Rectangle(p0, p1).x(p0[0] + x)
                r.color(c).alpha(alpha).lighting("off")
                rs.append(r)

    if centerline:
        cl = shapes.Line([0, mine, 0.01], [0, maxe, 0.01], c=lc, alpha=alpha, lw=2).x(x)
        rs.append(cl)

    asse = Assembly(rs)
    asse.name = "Violin"
    return asse


def whisker(data, s=0.25, c="k", lw=2, bc="blue", alpha=0.25, r=5, jitter=True, horizontal=False) -> Assembly:
    """
    Generate a "whisker" bar from a 1-dimensional dataset.

    Arguments:
        s : (float)
            size of the box
        c : (color)
            color of the lines
        lw : (float)
            line width
        bc : (color)
            color of the box
        alpha : (float)
            transparency of the box
        r : (float)
            point radius in pixels (use value 0 to disable)
        jitter : (bool)
            add some randomness to points to avoid overlap
        horizontal : (bool)
            set horizontal layout

    Examples:
        - [whiskers.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/whiskers.py)

            ![](https://vedo.embl.es/images/pyplot/whiskers.png)
    """
    xvals = np.zeros_like(np.asarray(data))
    if jitter:
        xjit = np.random.randn(len(xvals)) * s / 9
        xjit = np.clip(xjit, -s / 2.1, s / 2.1)
        xvals += xjit

    dmean = np.mean(data)
    dq05 = np.quantile(data, 0.05)
    dq25 = np.quantile(data, 0.25)
    dq75 = np.quantile(data, 0.75)
    dq95 = np.quantile(data, 0.95)

    pts = None
    if r:
        pts = Points(np.array([xvals, data]).T, c=c, r=r)

    rec = shapes.Rectangle([-s / 2, dq25], [s / 2, dq75], c=bc, alpha=alpha)
    rec.properties.LightingOff()
    rl = shapes.Line([[-s / 2, dq25], [s / 2, dq25], [s / 2, dq75], [-s / 2, dq75]], closed=True)
    l1 = shapes.Line([0, dq05, 0], [0, dq25, 0], c=c, lw=lw)
    l2 = shapes.Line([0, dq75, 0], [0, dq95, 0], c=c, lw=lw)
    lm = shapes.Line([-s / 2, dmean], [s / 2, dmean])
    lns = merge(l1, l2, lm, rl)
    asse = Assembly([lns, rec, pts])
    if horizontal:
        asse.rotate_z(-90)
    asse.name = "Whisker"
    asse.info["mean"] = dmean
    asse.info["quantile_05"] = dq05
    asse.info["quantile_25"] = dq25
    asse.info["quantile_75"] = dq75
    asse.info["quantile_95"] = dq95
    return asse


def matrix(
    M,
    title="Matrix",
    xtitle="",
    ytitle="",
    xlabels=(),
    ylabels=(),
    xrotation=0,
    cmap="Reds",
    vmin=None,
    vmax=None,
    precision=2,
    font="Theemim",
    scale=0,
    scalarbar=True,
    lc="white",
    lw=0,
    c="black",
    alpha=1,
) -> Assembly:
    """
    Generate a matrix, or a 2D color-coded plot with bin labels.

    Returns an `Assembly` object.

    Arguments:
        M : (list, numpy array)
            the input array to visualize
        title : (str)
            title of the plot
        xtitle : (str)
            title of the horizontal colmuns
        ytitle : (str)
            title of the vertical rows
        xlabels : (list)
            individual string labels for each column. Must be of length m
        ylabels : (list)
            individual string labels for each row. Must be of length n
        xrotation : (float)
            rotation of the horizontal labels
        cmap : (str)
            color map name
        vmin : (float)
            minimum value of the colormap range
        vmax : (float)
            maximum value of the colormap range
        precision : (int)
            number of digits for the matrix entries or bins
        font : (str)
            font name. Check [available fonts here](https://vedo.embl.es/fonts).

        scale : (float)
            size of the numeric entries or bin values
        scalarbar : (bool)
            add a scalar bar to the right of the plot
        lc : (str)
            color of the line separating the bins
        lw : (float)
            Width of the line separating the bins
        c : (str)
            text color
        alpha : (float)
            plot transparency

    Examples:
        - [np_matrix.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/np_matrix.py)

            ![](https://vedo.embl.es/images/pyplot/np_matrix.png)
    """
    M = np.asarray(M)
    n, m = M.shape
    gr = shapes.Grid(res=(m, n), s=(m / (m + n) * 2, n / (m + n) * 2), c=c, alpha=alpha)
    gr.wireframe(False).lc(lc).lw(lw)

    matr = np.flip(np.flip(M), axis=1).ravel(order="C")
    gr.cmap(cmap, matr, on="cells", vmin=vmin, vmax=vmax)
    sbar = None
    if scalarbar:
        gr.add_scalarbar3d(title_font=font, label_font=font)
        sbar = gr.scalarbar
    labs = None
    if scale != 0:
        gr.compute_normals(points=False)
        labs = gr.labels(
            on="cells",
            scale=scale / max(m, n),
            precision=precision,
            font=font,
            justify="center",
            c=c,
        )
        labs.z(0.001)
    t = None
    if title:
        if title == "Matrix":
            title += " " + str(n) + "x" + str(m)
        t = shapes.Text3D(title, font=font, s=0.04, justify="bottom-center", c=c)
        t.shift(0, n / (m + n) * 1.05)

    xlabs = None
    if len(xlabels) == m:
        xlabs = []
        jus = "top-center"
        if xrotation > 44:
            jus = "right-center"
        for i in range(m):
            xl = shapes.Text3D(xlabels[i], font=font, s=0.02, justify=jus, c=c).rotate_z(xrotation)
            xl.shift((2 * i - m + 1) / (m + n), -n / (m + n) * 1.05)
            xlabs.append(xl)

    ylabs = None
    if len(ylabels) == n:
        ylabels = list(reversed(ylabels))
        ylabs = []
        for i in range(n):
            yl = shapes.Text3D(ylabels[i], font=font, s=0.02, justify="right-center", c=c)
            yl.shift(-m / (m + n) * 1.05, (2 * i - n + 1) / (m + n))
            ylabs.append(yl)

    xt = None
    if xtitle:
        xt = shapes.Text3D(xtitle, font=font, s=0.035, justify="top-center", c=c)
        xt.shift(0, -n / (m + n) * 1.05)
        if xlabs is not None:
            y0, y1 = xlabs[0].ybounds()
            xt.shift(0, -(y1 - y0) - 0.55 / (m + n))
    yt = None
    if ytitle:
        yt = shapes.Text3D(ytitle, font=font, s=0.035, justify="bottom-center", c=c).rotate_z(90)
        yt.shift(-m / (m + n) * 1.05, 0)
        if ylabs is not None:
            x0, x1 = ylabs[0].xbounds()
            yt.shift(-(x1 - x0) - 0.55 / (m + n), 0)
    asse = Assembly(gr, sbar, labs, t, xt, yt, xlabs, ylabs)
    asse.name = "Matrix"
    return asse


def CornerPlot(points, pos=1, s=0.2, title="", c="b", bg="k", lines=True, dots=True):
    """
    Return a `vtkXYPlotActor` that is a plot of `x` versus `y`,
    where `points` is a list of `(x,y)` points.

    Assign position following this convention:

        - 1, topleft,
        - 2, topright,
        - 3, bottomleft,
        - 4, bottomright.
    """
    if len(points) == 2:  # passing [allx, ally]
        points = np.stack((points[0], points[1]), axis=1)

    c = colors.get_color(c)  # allow different codings
    array_x = vtki.vtkFloatArray()
    array_y = vtki.vtkFloatArray()
    array_x.SetNumberOfTuples(len(points))
    array_y.SetNumberOfTuples(len(points))
    for i, p in enumerate(points):
        array_x.InsertValue(i, p[0])
        array_y.InsertValue(i, p[1])
    field = vtki.vtkFieldData()
    field.AddArray(array_x)
    field.AddArray(array_y)
    data = vtki.vtkDataObject()
    data.SetFieldData(field)

    xyplot = vtki.new("XYPlotActor")
    xyplot.AddDataObjectInput(data)
    xyplot.SetDataObjectXComponent(0, 0)
    xyplot.SetDataObjectYComponent(0, 1)
    xyplot.SetXValuesToValue()
    xyplot.SetAdjustXLabels(0)
    xyplot.SetAdjustYLabels(0)
    xyplot.SetNumberOfXLabels(3)

    xyplot.GetProperty().SetPointSize(5)
    xyplot.GetProperty().SetLineWidth(2)
    xyplot.GetProperty().SetColor(colors.get_color(bg))
    xyplot.SetPlotColor(0, c[0], c[1], c[2])

    xyplot.SetXTitle(title)
    xyplot.SetYTitle("")
    xyplot.ExchangeAxesOff()
    xyplot.SetPlotPoints(dots)

    if not lines:
        xyplot.PlotLinesOff()

    if isinstance(pos, str):
        spos = 2
        if "top" in pos:
            if "left" in pos:
                spos = 1
            elif "right" in pos:
                spos = 2
        elif "bottom" in pos:
            if "left" in pos:
                spos = 3
            elif "right" in pos:
                spos = 4
        pos = spos
    if pos == 1:
        xyplot.GetPositionCoordinate().SetValue(0.0, 0.8, 0)
    elif pos == 2:
        xyplot.GetPositionCoordinate().SetValue(0.76, 0.8, 0)
    elif pos == 3:
        xyplot.GetPositionCoordinate().SetValue(0.0, 0.0, 0)
    elif pos == 4:
        xyplot.GetPositionCoordinate().SetValue(0.76, 0.0, 0)
    else:
        xyplot.GetPositionCoordinate().SetValue(pos[0], pos[1], 0)

    xyplot.GetPosition2Coordinate().SetValue(s, s, 0)
    return xyplot


def CornerHistogram(
    values,
    bins=20,
    vrange=None,
    minbin=0,
    logscale=False,
    title="",
    c="g",
    bg="k",
    alpha=1,
    pos="bottom-left",
    s=0.175,
    lines=True,
    dots=False,
    nmax=None,
):
    """
    Build a histogram from a list of values in n bins.
    The resulting object is a 2D actor.

    Use `vrange` to restrict the range of the histogram.

    Use `nmax` to limit the sampling to this max nr of entries

    Use `pos` to assign its position:
        - 1, topleft,
        - 2, topright,
        - 3, bottomleft,
        - 4, bottomright,
        - (x, y), as fraction of the rendering window
    """
    if hasattr(values, "dataset"):
        values = utils.vtk2numpy(values.dataset.GetPointData().GetScalars())

    n = values.shape[0]
    if nmax and nmax < n:
        # subsample:
        idxs = np.linspace(0, n, num=int(nmax), endpoint=False).astype(int)
        values = values[idxs]

    fs, edges = np.histogram(values, bins=bins, range=vrange)

    if minbin:
        fs = fs[minbin:-1]
    if logscale:
        fs = np.log10(fs + 1)
    pts = []
    for i in range(len(fs)):
        pts.append([(edges[i] + edges[i + 1]) / 2, fs[i]])

    cplot = CornerPlot(pts, pos, s, title, c, bg, lines, dots)
    cplot.SetNumberOfYLabels(2)
    cplot.SetNumberOfXLabels(3)
    tprop = vtki.vtkTextProperty()
    tprop.SetColor(colors.get_color(bg))
    tprop.SetFontFamily(vtki.VTK_FONT_FILE)
    tprop.SetFontFile(utils.get_font_path("Calco"))
    tprop.SetOpacity(alpha)
    cplot.SetAxisTitleTextProperty(tprop)
    cplot.GetProperty().SetOpacity(alpha)
    cplot.GetXAxisActor2D().SetLabelTextProperty(tprop)
    cplot.GetXAxisActor2D().SetTitleTextProperty(tprop)
    cplot.GetXAxisActor2D().SetFontFactor(0.55)
    cplot.GetYAxisActor2D().SetLabelFactor(0.0)
    cplot.GetYAxisActor2D().LabelVisibilityOff()
    return cplot



__all__ = ["pie_chart", "violin", "whisker", "matrix", "CornerPlot", "CornerHistogram"]
