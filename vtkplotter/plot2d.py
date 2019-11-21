from __future__ import division, print_function
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk

import vtkplotter.docs as docs
import vtkplotter.settings as settings
import vtkplotter.utils as utils
import vtkplotter.colors as colors
import vtkplotter.shapes as shapes
from vtkplotter.actors import Actor, Assembly, merge


__doc__ = (
    """
Utilities to plot in 2D.
"""
    + docs._defs
)

__all__ = [
    "plotxy",
    "fxy",
    "cornerPlot",
    "cornerHistogram",
    "histogram",
    "hexHistogram",
    "polarHistogram",
    "polarPlot",
    "donutPlot",
]


def plotxy(
    data,
    xerrors=None,
    yerrors=None,
    xlimits=None,
    ylimits=None,
    xscale=1,
    yscale=None,
    xlogscale=False,
    ylogscale=False,
    c="k",
    alpha=1,
    xtitle="x",
    ytitle="y",
    title="",
    titleSize=None,
    ec=None,
    lc="k",
    lw=2,
    line=True,
    dashed=False,
    splined=False,
    marker=None,
    ms=None,
    mc=None,
    ma=None,
):
    """Draw a 2D plot of variable x vs y.

    :param list data: input format can be [allx, ally] or [(x1,y1), (x2,y2), ...]

    :param list xerrors: set uncertainties for the x variable, shown as error bars.
    :param list yerrors: set uncertainties for the y variable, shown as error bars.
    :param list xlimits: set limits to the range for the x variable
    :param list ylimits: set limits to the range for the y variable
    :param float xscale: set scaling factor in x. Default is 1.
    :param float yscale: set scaling factor in y. Automatically calculated to get
        a reasonable aspect ratio. Scaling factor is saved in `info['yscale']`.

    :param bool xlogscale: set x logarithmic scale.
    :param bool ylogscale: set y logarithmic scale.
    :param str c: color of frame and text.
    :param float alpha: opacity of frame and text.
    :param str xtitle: title label along x-axis.
    :param str ytitle: title label along y-axis.
    :param str title: histogram title on top.
    :param float titleSize: size of title
    :param str ec: color of error bar, by default the same as marker color
    :param str lc: color of line
    :param float lw: width of line
    :param bool line: join points with line
    :param bool dashed: use a dashed line style
    :param bool splined: spline the line joining the point as a countinous curve
    :param str,int marker: use a marker shape for the data points
    :param float ms: marker size.
    :param str mc: color of marker
    :param float ma: opacity of marker

    |plotxy| |plotxy.py|_
    """
    if len(data) == 2 and len(data[0])>1 and len(data[0]) == len(data[1]):
        #format is [allx, ally], convert it:
        data = np.c_[data[0], data[1]]

    if xlimits is not None:
        cdata = []
        x0lim = xlimits[0]
        x1lim = xlimits[1]
        for d in data:
            if d[0] > x0lim and d[0] < x1lim:
                cdata.append(d)
        data = cdata
        if not len(data):
            colors.printc("Error in plotxy(): no points within xlimits", c=1)
            return None

    if ylimits is not None:
        cdata = []
        y0lim = ylimits[0]
        y1lim = ylimits[1]
        for d in data:
            if d[1] > y0lim and d[1] < y1lim:
                cdata.append(d)
        data = cdata
        if not len(data):
            colors.printc("Error in plotxy(): no points within ylimits", c=1)
            return None

    data = np.array(data)[:, [0, 1]]

    if xlogscale:
        data[:, 0] = np.log(data[:, 0])

    if ylogscale:
        data[:, 1] = np.log(data[:, 1])

    x0, y0 = np.min(data, axis=0)
    x1, y1 = np.max(data, axis=0)

    if yscale is None:
        yscale = (x1 - x0) / (y1 - y0) * 0.75  # default 3/4 aspect ratio
        yscale = float(utils.precision(yscale, 1))
        if abs(yscale - 1) > 0.2:
            ytitle += " *" + str(yscale)
            y0 *= yscale
            y1 *= yscale
        else:
            yscale = 1

    scale = np.array([[xscale, yscale]])
    data = np.multiply(data, scale)

    acts = []
    if dashed:
        l = shapes.DashedLine(data, lw=lw, spacing=20)
        acts.append(l)
    elif splined:
        l = shapes.KSpline(data).lw(lw).c(lc)
        acts.append(l)
    elif line:
        l = shapes.Line(data, lw=lw, c=lc)
        acts.append(l)

    if marker:
        if ms is None:
            ms = (x1 - x0) / 75.0
        if mc is None:
            mc = lc
        mk = shapes.Marker(marker, s=ms, alpha=ma)
        pts = shapes.Points(data)
        marked = shapes.Glyph(pts, glyphObj=mk, c=mc)
        acts.append(marked)

    if ec is None:
        if mc is not None:
            ec = mc
        else:
            ec = lc
        offs = (x1-x0)/1000

    if yerrors is not None:
        if len(yerrors) != len(data):
            colors.printc("Error in plotxy(yerrors=...): mismatched array length.", c=1)
            return None
        errs = []
        for i in range(len(data)):
            xval, yval = data[i]
            yerr = yerrors[i]/2*yscale
            errs.append(shapes.Line((xval, yval-yerr, offs), (xval, yval+yerr, offs)))
        myerrs = merge(errs).c(ec).lw(lw).alpha(alpha)
        acts.append(myerrs)

    if xerrors is not None:
        if len(xerrors) != len(data):
            colors.printc("Error in plotxy(xerrors=...): mismatched array length.", c=1)
            return None
        errs = []
        for i in range(len(data)):
            xval, yval = data[i]
            xerr = xerrors[i]/2
            errs.append(shapes.Line((xval-xerr, yval, offs), (xval+xerr, yval, offs)))
        mxerrs = merge(errs).c(ec).lw(lw).alpha(alpha)
        acts.append(mxerrs)

    x0lim = x0
    x1lim = x1
    y0lim = y0*yscale
    y1lim = y1*yscale
    if xlimits is not None or ylimits is not None:
        if xlimits is not None:
            x0lim = min(xlimits[0], x0)
            x1lim = max(xlimits[1], x1)
        if ylimits is not None:
            y0lim = min(ylimits[0]*yscale, y0)
            y1lim = max(ylimits[1]*yscale, y1)
        rec = shapes.Rectangle([x0lim, y0lim, 0], [x1lim, y1lim, 0])
        rec.alpha(0).wireframe()
        acts.append(rec)

    if title:
        if titleSize is None:
            titleSize = (x1lim - x0lim) / 40.0
        tit = shapes.Text(
            title,
            s=titleSize,
            c=c,
            depth=0,
            alpha=alpha,
            pos=((x1lim + x0lim) / 2.0, y1lim, 0),
            justify="bottom-center",
        )
        tit.pickable(False)
        acts.append(tit)

    settings.xtitle = xtitle
    settings.ytitle = ytitle
    asse = Assembly(acts)
    asse.info["yscale"] = yscale
    return asse


def cornerPlot(points, pos=1, s=0.2, title="", c="b", bg="k", lines=True):
    """
    Return a ``vtkXYPlotActor`` that is a plot of `x` versus `y`,
    where `points` is a list of `(x,y)` points.

    :param int pos: assign position:

        - 1, topleft,
        - 2, topright,
        - 3, bottomleft,
        - 4, bottomright.
    """
    if len(points) == 2:  # passing [allx, ally]
        points = list(zip(points[0], points[1]))

    c = colors.getColor(c)  # allow different codings
    array_x = vtk.vtkFloatArray()
    array_y = vtk.vtkFloatArray()
    array_x.SetNumberOfTuples(len(points))
    array_y.SetNumberOfTuples(len(points))
    for i, p in enumerate(points):
        array_x.InsertValue(i, p[0])
        array_y.InsertValue(i, p[1])
    field = vtk.vtkFieldData()
    field.AddArray(array_x)
    field.AddArray(array_y)
    data = vtk.vtkDataObject()
    data.SetFieldData(field)

    plot = vtk.vtkXYPlotActor()
    plot.AddDataObjectInput(data)
    plot.SetDataObjectXComponent(0, 0)
    plot.SetDataObjectYComponent(0, 1)
    plot.SetXValuesToValue()
    plot.SetAdjustXLabels(0)
    plot.SetAdjustYLabels(0)
    plot.SetNumberOfXLabels(3)

    plot.GetProperty().SetPointSize(5)
    plot.GetProperty().SetLineWidth(2)
    plot.GetProperty().SetColor(colors.getColor(bg))
    plot.SetPlotColor(0, c[0], c[1], c[2])

    plot.SetXTitle(title)
    plot.SetYTitle("")
    plot.ExchangeAxesOff()
    plot.PlotPointsOn()
    if not lines:
        plot.PlotLinesOff()
    if pos == 1:
        plot.GetPositionCoordinate().SetValue(0.0, 0.8, 0)
    elif pos == 2:
        plot.GetPositionCoordinate().SetValue(0.76, 0.8, 0)
    elif pos == 3:
        plot.GetPositionCoordinate().SetValue(0.0, 0.0, 0)
    elif pos == 4:
        plot.GetPositionCoordinate().SetValue(0.76, 0.0, 0)
    else:
        plot.GetPositionCoordinate().SetValue(pos[0], pos[1], 0)
    plot.GetPosition2Coordinate().SetValue(s, s, 0)
    return plot


def cornerHistogram(
    values,
    bins=20,
    vrange=None,
    minbin=0,
    logscale=False,
    title="",
    c="g",
    bg="k",
    pos=1,
    s=0.2,
    lines=True,
):
    """
    Build a histogram from a list of values in n bins.
    The resulting object is a 2D actor.

    Use *vrange* to restrict the range of the histogram.

    Use `pos` to assign its position:
        - 1, topleft,
        - 2, topright,
        - 3, bottomleft,
        - 4, bottomright,
        - (x, y), as fraction of the rendering window
    """
    fs, edges = np.histogram(values, bins=bins, range=vrange)
    if minbin:
        fs = fs[minbin:-1]
    if logscale:
        fs = np.log10(fs + 1)
    pts = []
    for i in range(len(fs)):
        pts.append([(edges[i] + edges[i + 1]) / 2, fs[i]])

    plot = cornerPlot(pts, pos, s, title, c, bg, lines)
    plot.SetNumberOfYLabels(2)
    plot.SetNumberOfXLabels(3)
    tprop = vtk.vtkTextProperty()
    tprop.SetColor(colors.getColor(bg))
    plot.SetAxisTitleTextProperty(tprop)
    plot.GetXAxisActor2D().SetLabelTextProperty(tprop)
    plot.GetXAxisActor2D().SetTitleTextProperty(tprop)
    plot.GetXAxisActor2D().SetFontFactor(0.5)
    plot.GetYAxisActor2D().SetLabelFactor(0.0)
    plot.GetYAxisActor2D().LabelVisibilityOff()
    return plot


def histogram(
    values,
    xtitle="",
    ytitle="",
    bins=25,
    vrange=None,
    logscale=False,
    yscale=None,
    fill=True,
    gap=0.02,
    c="olivedrab",
    alpha=1,
    outline=True,
    lw=2,
    lc="black",
    errors=False,
):
    """
    Build a histogram from a list of values in n bins.
    The resulting object is a 2D actor.

    :param int bins: number of bins.
    :param list vrange: restrict the range of the histogram.
    :param bool logscale: use logscale on y-axis.
    :param bool fill: fill bars woth solid color `c`.
    :param float gap: leave a small space btw bars.
    :param bool outline: show outline of the bins.
    :param bool errors: show error bars.

    |histogram| |histogram.py|_
    """
    if xtitle:
        from vtkplotter import settings

        settings.xtitle = xtitle
    if ytitle:
        from vtkplotter import settings

        settings.ytitle = ytitle

    fs, edges = np.histogram(values, bins=bins, range=vrange)
    if logscale:
        fs = np.log10(fs + 1)
    mine, maxe = np.min(edges), np.max(edges)
    binsize = edges[1] - edges[0]

    rs = []
    if fill:
        if outline:
            gap = 0
        for i in range(bins):
            p0 = (edges[i] + gap * binsize, 0, 0)
            p1 = (edges[i + 1] - gap * binsize, fs[i], 0)
            r = shapes.Rectangle(p0, p1)
            r.color(c).alpha(alpha).lighting("ambient")
            rs.append(r)

    if outline:
        lns = [[mine, 0, 0]]
        for i in range(bins):
            lns.append([edges[i], fs[i], 0])
            lns.append([edges[i + 1], fs[i], 0])
        lns.append([maxe, 0, 0])
        rs.append(shapes.Line(lns, c=lc, alpha=alpha, lw=lw))

    if errors:
        errs = np.sqrt(fs)
        for i in range(bins):
            x = (edges[i] + edges[i + 1]) / 2
            el = shapes.Line(
                [x, fs[i] - errs[i] / 2, 0.1 * binsize],
                [x, fs[i] + errs[i] / 2, 0.1 * binsize],
                c=lc,
                alpha=alpha,
                lw=lw,
            )
            pt = shapes.Point([x, fs[i], 0.1 * binsize], r=7, c=lc, alpha=alpha)
            rs.append(el)
            rs.append(pt)

    asse = Assembly(rs)
    if yscale is None:
        yscale = 10 / np.sum(fs) * (maxe - mine)
    asse.scale([1, yscale, 1])
    return asse


def hexHistogram(
    xvalues,
    yvalues,
    xtitle="",
    ytitle="",
    ztitle="",
    bins=12,
    norm=1,
    fill=True,
    c=None,
    cmap="terrain_r",
    alpha=1,
):
    """
    Build a hexagonal histogram from a list of x and y values.

    :param bool bins: nr of bins for the smaller range in x or y.
    :param float norm: sets a scaling factor for the z axis (freq. axis).
    :param bool fill: draw solid hexagons.
    :param str cmap: color map name for elevation.

    |histoHexagonal| |histoHexagonal.py|_
    """
    if xtitle:
        from vtkplotter import settings

        settings.xtitle = xtitle
    if ytitle:
        from vtkplotter import settings

        settings.ytitle = ytitle
    if ztitle:
        from vtkplotter import settings

        settings.ztitle = ztitle

    xmin, xmax = np.min(xvalues), np.max(xvalues)
    ymin, ymax = np.min(yvalues), np.max(yvalues)
    dx, dy = xmax - xmin, ymax - ymin

    if xmax - xmin < ymax - ymin:
        n = bins
        m = np.rint(dy / dx * n / 1.2 + 0.5).astype(int)
    else:
        m = bins
        n = np.rint(dx / dy * m * 1.2 + 0.5).astype(int)

    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(len(xvalues))
    src.Update()
    pointsPolydata = src.GetOutput()

    values = list(zip(xvalues, yvalues))
    zs = [[0.0]] * len(values)
    values = np.append(values, zs, axis=1)

    pointsPolydata.GetPoints().SetData(numpy_to_vtk(values, deep=True))
    cloud = Actor(pointsPolydata)

    col = None
    if c is not None:
        col = colors.getColor(c)

    hexs, binmax = [], 0
    ki, kj = 1.33, 1.12
    r = 0.47 / n * 1.2 * dx
    for i in range(n + 3):
        for j in range(m + 2):
            cyl = vtk.vtkCylinderSource()
            cyl.SetResolution(6)
            cyl.CappingOn()
            cyl.SetRadius(0.5)
            cyl.SetHeight(0.1)
            cyl.Update()
            t = vtk.vtkTransform()
            if not i % 2:
                p = (i / ki, j / kj, 0)
            else:
                p = (i / ki, j / kj + 0.45, 0)
            q = (p[0] / n * 1.2 * dx + xmin, p[1] / m * dy + ymin, 0)
            ids = cloud.closestPoint(q, radius=r, returnIds=True)
            ne = len(ids)
            if fill:
                t.Translate(p[0], p[1], ne / 2)
                t.Scale(1, 1, ne * 10)
            else:
                t.Translate(p[0], p[1], ne)
            t.RotateX(90)  # put it along Z
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(cyl.GetOutput())
            tf.SetTransform(t)
            tf.Update()
            if c is None:
                col = i
            h = Actor(tf.GetOutput(), c=col, alpha=alpha).flat()
            h.GetProperty().SetSpecular(0)
            h.GetProperty().SetDiffuse(1)
            h.PickableOff()
            hexs.append(h)
            if ne > binmax:
                binmax = ne

    if cmap is not None:
        for h in hexs:
            z = h.GetBounds()[5]
            col = colors.colorMap(z, cmap, 0, binmax)
            h.color(col)

    asse = Assembly(hexs)
    asse.SetScale(1.2 / n * dx, 1 / m * dy, norm / binmax * (dx + dy) / 4)
    asse.SetPosition(xmin, ymin, 0)
    return asse


def polarHistogram(
    values,
    title="",
    bins=10,
    r1=0.25,
    r2=1,
    phigap=3,
    rgap=0.05,
    lpos=1,
    lsize=0.05,
    c=None,
    bc="k",
    alpha=1,
    cmap=None,
    deg=False,
    vmin=None,
    vmax=None,
    labels=(),
    showDisc=True,
    showLines=True,
    showAngles=True,
    showErrors=False,
):
    """
    Polar histogram with errorbars.

    :param str title: histogram title
    :param int bins: number of bins in phi
    :param float r1: inner radius
    :param float r2: outer radius
    :param float phigap: gap angle btw 2 radial bars, in degrees
    :param float rgap: gap factor along radius of numeric angle labels
    :param float lpos: label gap factor along radius
    :param float lsize: label size
    :param c: color of the histogram bars, can be a list of length `bins`.
    :param bc: color of the frame and labels
    :param alpha: alpha of the frame
    :param str cmap: color map name
    :param bool deg: input array is in degrees
    :param float vmin: minimum value of the radial axis
    :param float vmax: maximum value of the radial axis
    :param list labels: list of labels, must be of length `bins`
    :param bool showDisc: show the outer ring axis
    :param bool showLines: show lines to the origin
    :param bool showAngles: show angular values
    :param bool showErrors: show error bars

    |polarHisto| |polarHisto.py|_
    """
    k = 180 / np.pi
    if deg:
        values = np.array(values) / k

    dp = np.pi / bins
    vals = []
    for v in values:  # normalize range
        t = np.arctan2(np.sin(v), np.cos(v))
        if t < 0:
            t += 2 * np.pi
        vals.append(t - dp)

    histodata, edges = np.histogram(vals, bins=bins, range=(-dp, 2 * np.pi - dp))
    thetas = []
    for i in range(bins):
        thetas.append((edges[i] + edges[i + 1]) / 2)

    if vmin is None:
        vmin = np.min(histodata)
    if vmax is None:
        vmax = np.max(histodata)

    errors = np.sqrt(histodata)
    r2e = r1 + r2
    if showErrors:
        r2e += np.max(errors) / vmax * 1.5

    back = None
    if showDisc:
        back = shapes.Disc(r1=r2e, r2=r2e * 1.01, c=bc, res=1, resphi=360)
        back.z(-0.01).lighting(diffuse=0, ambient=1).alpha(alpha)

    slices = []
    lines = []
    angles = []
    labs = []
    errbars = []

    for i, t in enumerate(thetas):
        r = histodata[i] / vmax * r2
        d = shapes.Disc((0, 0, 0), r1, r1 + r, res=1, resphi=360)
        delta = dp - np.pi / 2 - phigap / k
        d.cutWithPlane(normal=(np.cos(t + delta), np.sin(t + delta), 0))
        d.cutWithPlane(normal=(np.cos(t - delta), np.sin(t - delta), 0))
        if cmap is not None:
            cslice = colors.colorMap(histodata[i], cmap, vmin, vmax)
            d.color(cslice)
        else:
            if c is None:
                d.color(i)
            elif utils.isSequence(c) and len(c) == bins:
                d.color(c[i])
            else:
                d.color(c)
        slices.append(d)

        ct, st = np.cos(t), np.sin(t)

        if showErrors:
            showLines = False
            err = np.sqrt(histodata[i]) / vmax * r2
            errl = shapes.Line(
                ((r1 + r - err) * ct, (r1 + r - err) * st, 0.01),
                ((r1 + r + err) * ct, (r1 + r + err) * st, 0.01),
            )
            errl.alpha(alpha).lw(3).color(bc)
            errbars.append(errl)

        if showDisc:
            if showLines:
                l = shapes.Line((0, 0, -0.01), (r2e * ct * 1.03, r2e * st * 1.03, -0.01))
                lines.append(l)
            elif showAngles:  # just the ticks
                l = shapes.Line(
                    (r2e * ct * 0.98, r2e * st * 0.98, -0.01),
                    (r2e * ct * 1.03, r2e * st * 1.03, -0.01),
                )
                lines.append(l)

        if showAngles:
            if 0 <= t < np.pi / 2:
                ju = "bottom-left"
            elif t == np.pi / 2:
                ju = "bottom-center"
            elif np.pi / 2 < t <= np.pi:
                ju = "bottom-right"
            elif np.pi < t < np.pi * 3 / 2:
                ju = "top-right"
            elif t == np.pi * 3 / 2:
                ju = "top-center"
            else:
                ju = "top-left"
            a = shapes.Text(int(t * k), pos=(0, 0, 0), s=lsize, depth=0, justify=ju)
            a.pos(r2e * ct * (1 + rgap), r2e * st * (1 + rgap), -0.01)
            angles.append(a)

        if len(labels) == bins:
            lab = shapes.Text(labels[i], (0, 0, 0), s=lsize, depth=0, justify="center")
            lab.pos(r2e * ct * (1 + rgap) * lpos / 2, r2e * st * (1 + rgap) * lpos / 2, 0.01)
            labs.append(lab)

    ti = None
    if title:
        ti = shapes.Text(title, (0, 0, 0), s=lsize * 2, depth=0, justify="top-center")
        ti.pos(0, -r2e * 1.15, 0.01)

    mrg = merge(back, lines, angles, labs, ti)
    if mrg:
        mrg.color(bc).alpha(alpha).lighting(diffuse=0, ambient=1)
    rh = Assembly(slices + errbars + [mrg])
    rh.base = np.array([0, 0, 0])
    rh.top = np.array([0, 0, 1])
    return rh


def donutPlot(
    fractions,
    title="",
    r1=1.7,
    r2=1,
    phigap=0,
    lpos=0.8,
    lsize=0.15,
    c=None,
    bc="k",
    alpha=1,
    labels=(),
    showDisc=False,
):
    """
    Donut plot or pie chart.

    :param str title: plot title
    :param float r1: inner radius
    :param float r2: outer radius, starting from r1
    :param float phigap: gap angle btw 2 radial bars, in degrees
    :param float lpos: label gap factor along radius
    :param float lsize: label size
    :param c: color of the plot slices
    :param bc: color of the disc frame
    :param alpha: alpha of the disc frame
    :param list labels: list of labels
    :param bool showDisc: show the outer ring axis

    |donutPlot| |donutPlot.py|_
    """
    fractions = np.array(fractions)
    angles = np.add.accumulate(2 * np.pi * fractions)
    angles[-1] = 2 * np.pi
    if angles[-2] > 2 * np.pi:
        print("Error in donutPlot(): fractions must sum to 1.")
        raise RuntimeError

    cols = []
    for i, th in enumerate(np.linspace(0, 2 * np.pi, 360, endpoint=False)):
        for ia, a in enumerate(angles):
            if th < a:
                cols.append(c[ia])
                break
    labs = ()
    if len(labels):
        angles = np.concatenate([[0], angles])
        labs = [""] * 360
        for i in range(len(labels)):
            a = (angles[i + 1] + angles[i]) / 2
            j = int(a / np.pi * 180)
            labs[j] = labels[i]

    data = np.linspace(0, 2 * np.pi, 360, endpoint=False) + 0.005
    dn = polarHistogram(
        data,
        title=title,
        bins=360,
        r1=r1,
        r2=r2,
        phigap=phigap,
        lpos=lpos,
        lsize=lsize,
        c=cols,
        bc=bc,
        alpha=alpha,
        vmin=0,
        vmax=1,
        labels=labs,
        showDisc=showDisc,
        showLines=0,
        showAngles=0,
        showErrors=0,
    )
    return dn


def polarPlot(
    rphi,
    title="",
    r1=0,
    r2=1,
    lpos=1,
    lsize=0.03,
    c="blue",
    bc="k",
    alpha=1,
    lw=3,
    deg=False,
    vmax=None,
    fill=True,
    spline=True,
    smooth=0,
    showPoints=True,
    showDisc=True,
    showLines=True,
    showAngles=True,
):
    """
    Polar/radar plot by splining a set of points in polar coordinates.
    Input is a list of polar angles and radii.

    :param str title: histogram title
    :param int bins: number of bins in phi
    :param float r1: inner radius
    :param float r2: outer radius
    :param float lsize: label size
    :param c: color of the line
    :param bc: color of the frame and labels
    :param alpha: alpha of the frame
    :param int lw: line width in pixels
    :param bool deg: input array is in degrees
    :param bool fill: fill convex area with solid color
    :param bool spline: interpolate the set of input points
    :param bool showPoints: show data points
    :param bool showDisc: show the outer ring axis
    :param bool showLines: show lines to the origin
    :param bool showAngles: show angular values

    |polarPlot| |polarPlot.py|_
    """
    if len(rphi) == 2:
        rphi = list(zip(rphi[0], rphi[1]))
    rphi = np.array(rphi)
    thetas = rphi[:, 0]
    radii = rphi[:, 1]

    k = 180 / np.pi
    if deg:
        thetas = np.array(thetas) / k

    vals = []
    for v in thetas:  # normalize range
        t = np.arctan2(np.sin(v), np.cos(v))
        if t < 0:
            t += 2 * np.pi
        vals.append(t)
    thetas = np.array(vals)

    if vmax is None:
        vmax = np.max(radii)

    angles = []
    labs = []
    points = []
    for i in range(len(thetas)):
        t = thetas[i]
        r = (radii[i]) / vmax * r2 + r1
        ct, st = np.cos(t), np.sin(t)
        points.append([r * ct, r * st, 0])
    p0 = points[0]
    points.append(p0)

    r2e = r1 + r2
    if spline:
        lines = shapes.KSpline(points, closed=True)
    else:
        lines = shapes.Line(points)
    lines.c(c).lw(lw).alpha(alpha)

    points.pop()

    ptsact = None
    if showPoints:
        ptsact = shapes.Points(points).c(c).alpha(alpha)

    filling = None
    if fill:
        faces = []
        coords = [[0, 0, 0]] + lines.coordinates().tolist()
        for i in range(1, lines.N()):
            faces.append([0, i, i + 1])
        filling = Actor([coords, faces]).c(c).alpha(alpha)

    back = None
    if showDisc:
        back = shapes.Disc(r1=r2e, r2=r2e * 1.01, c=bc, res=1, resphi=360)
        back.z(-0.01).lighting(diffuse=0, ambient=1).alpha(alpha)

    ti = None
    if title:
        ti = shapes.Text(title, (0, 0, 0), s=lsize * 2, depth=0, justify="top-center")
        ti.pos(0, -r2e * 1.15, 0.01)

    rays = []
    if showDisc:
        rgap = 0.05
        for t in np.linspace(0, 2 * np.pi, num=8, endpoint=False):
            ct, st = np.cos(t), np.sin(t)
            if showLines:
                l = shapes.Line((0, 0, -0.01), (r2e * ct * 1.03, r2e * st * 1.03, -0.01))
                rays.append(l)
            elif showAngles:  # just the ticks
                l = shapes.Line(
                    (r2e * ct * 0.98, r2e * st * 0.98, -0.01),
                    (r2e * ct * 1.03, r2e * st * 1.03, -0.01),
                )
            if showAngles:
                if 0 <= t < np.pi / 2:
                    ju = "bottom-left"
                elif t == np.pi / 2:
                    ju = "bottom-center"
                elif np.pi / 2 < t <= np.pi:
                    ju = "bottom-right"
                elif np.pi < t < np.pi * 3 / 2:
                    ju = "top-right"
                elif t == np.pi * 3 / 2:
                    ju = "top-center"
                else:
                    ju = "top-left"
                a = shapes.Text(int(t * k), pos=(0, 0, 0), s=lsize, depth=0, justify=ju)
                a.pos(r2e * ct * (1 + rgap), r2e * st * (1 + rgap), -0.01)
                angles.append(a)

    mrg = merge(back, angles, rays, labs, ti)
    if mrg:
        mrg.color(bc).alpha(alpha).lighting(diffuse=0, ambient=1)
    rh = Assembly([lines, ptsact, filling] + [mrg])
    rh.base = np.array([0, 0, 0])
    rh.top = np.array([0, 0, 1])
    return rh


def fxy(
    z="sin(3*x)*log(x-y)/3",
    x=(0, 3),
    y=(0, 3),
    zlimits=(None, None),
    showNan=True,
    zlevels=10,
    c="b",
    bc="aqua",
    alpha=1,
    texture="paper",
    res=(100, 100),
):
    """
    Build a surface representing the function :math:`f(x,y)` specified as a string
    or as a reference to an external function.

    :param float x: x range of values.
    :param float y: y range of values.
    :param float zlimits: limit the z range of the independent variable.
    :param int zlevels: will draw the specified number of z-levels contour lines.
    :param bool showNan: show where the function does not exist as red points.
    :param list res: resolution in x and y.

    |fxy| |fxy.py|_

        Function is: :math:`f(x,y)=\sin(3x) \cdot \log(x-y)/3` in range :math:`x=[0,3], y=[0,3]`.
    """
    if isinstance(z, str):
        try:
            z = z.replace("math.", "").replace("np.", "")
            namespace = locals()
            code = "from math import*\ndef zfunc(x,y): return " + z
            exec(code, namespace)
            z = namespace["zfunc"]
        except:
            colors.printc("Syntax Error in fxy()", c=1)
            return None

    ps = vtk.vtkPlaneSource()
    ps.SetResolution(res[0], res[1])
    ps.SetNormal([0, 0, 1])
    ps.Update()
    poly = ps.GetOutput()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    todel, nans = [], []

    for i in range(poly.GetNumberOfPoints()):
        px, py, _ = poly.GetPoint(i)
        xv = (px + 0.5) * dx + x[0]
        yv = (py + 0.5) * dy + y[0]
        try:
            zv = z(xv, yv)
        except:
            zv = 0
            todel.append(i)
            nans.append([xv, yv, 0])
        poly.GetPoints().SetPoint(i, [xv, yv, zv])

    if len(todel):
        cellIds = vtk.vtkIdList()
        poly.BuildLinks()
        for i in todel:
            poly.GetPointCells(i, cellIds)
            for j in range(cellIds.GetNumberOfIds()):
                poly.DeleteCell(cellIds.GetId(j))  # flag cell
        poly.RemoveDeletedCells()
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(poly)
        cl.Update()
        poly = cl.GetOutput()

    if not poly.GetNumberOfPoints():
        colors.printc("Function is not real in the domain", c=1)
        return None

    if zlimits[0]:
        tmpact1 = Actor(poly)
        a = tmpact1.cutWithPlane((0, 0, zlimits[0]), (0, 0, 1))
        poly = a.polydata()
    if zlimits[1]:
        tmpact2 = Actor(poly)
        a = tmpact2.cutWithPlane((0, 0, zlimits[1]), (0, 0, -1))
        poly = a.polydata()

    if c is None:
        elev = vtk.vtkElevationFilter()
        elev.SetInputData(poly)
        elev.Update()
        poly = elev.GetOutput()

    actor = Actor(poly, c, alpha).computeNormals().lighting("plastic")
    if c is None:
        actor.scalars("Elevation")

    if bc:
        actor.bc(bc)

    actor.texture(texture)

    acts = [actor]
    if zlevels:
        elevation = vtk.vtkElevationFilter()
        elevation.SetInputData(poly)
        bounds = poly.GetBounds()
        elevation.SetLowPoint(0, 0, bounds[4])
        elevation.SetHighPoint(0, 0, bounds[5])
        elevation.Update()
        bcf = vtk.vtkBandedPolyDataContourFilter()
        bcf.SetInputData(elevation.GetOutput())
        bcf.SetScalarModeToValue()
        bcf.GenerateContourEdgesOn()
        bcf.GenerateValues(zlevels, elevation.GetScalarRange())
        bcf.Update()
        zpoly = bcf.GetContourEdgesOutput()
        zbandsact = Actor(zpoly, "k", alpha).lw(0.5)
        acts.append(zbandsact)

    if showNan and len(todel):
        bb = actor.GetBounds()
        if bb[4] <= 0 and bb[5] >= 0:
            zm = 0.0
        else:
            zm = (bb[4] + bb[5]) / 2
        nans = np.array(nans) + [0, 0, zm]
        nansact = shapes.Points(nans, r=2, c="red", alpha=alpha)
        nansact.GetProperty().RenderPointsAsSpheresOff()
        acts.append(nansact)

    if len(acts) > 1:
        return Assembly(acts)
    else:
        return actor
