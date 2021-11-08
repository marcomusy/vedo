import vtk
import numpy as np
import vedo
import vedo.settings as settings
import vedo.utils as utils
import vedo.colors as colors
import vedo.shapes as shapes
import vedo.addons as addons
from vedo.assembly import Assembly
from vedo.mesh import Mesh, merge
from vedo.plotter import show # not used, but useful to import this

__doc__ = """Plotting utility functions.""" + vedo.docs._defs

__all__ = [
    "plot",
    "histogram",
    "donut",
    "quiver",
    "violin",
    "whisker",
    "streamplot",
    "matrix",
    "DirectedGraph",
    "show",
]


##########################################################################
class Plot(Assembly):
    """
    Derived class of ``Assembly`` to manipulate plots.
    """

    def __init__(self, *objs):

        Assembly.__init__(self, *objs)

        self.yscale = 1
        self.aspect = 4 / 3.0
        self.cut = True  # todo
        self.xlim = None
        self.ylim = None
        self.pad = 0.05

        self._x0lim = None
        self._y0lim = None
        self._x1lim = None
        self._y1lim = None
        self.zmax = 0  # z-order
        self.fixed_scale = 1

        self.bins = []
        self.freqs = []


    def ybounds(self, scaled=True):
        if scaled:
            return (self._y0lim/self.yscale, self._y1lim/self.yscale)
        else:
            return (self._y0lim, self._y1lim)

    def __iadd__(self, *objs):
        """
        Add object to plot with taking automatically into account the correct aspect ratio.
        """
        # these types will scale proportionally to keep their native shape aspect ratio intact
        typs = (
            shapes.Text3D,
            shapes.Polygon,
            shapes.Star,
            shapes.Disc,
            shapes.Ellipsoid,
            shapes.Latex,
            shapes.Sphere,
            # shapes.Arrow2D,
            Assembly,
            vedo.Picture,
        )
        self.fixed_scale = np.min([1, self.yscale])

        objs = objs[0]  # make a list anyway
        if not utils.isSequence(objs):
            objs = [objs]

        if not utils.isSequence(objs[0]) and isinstance(objs[0], Plot):
            # is adding another whole Plot # TO BE REVISED
            plot2 = objs[0]
            plot_z = plot2.z() + (plot2._x1lim - plot2._x0lim)/1000 # add a small shift in z
            # print(plot2.yscale, self.yscale)
            elems = plot2.unpack()
            objs2 = []
            for e in elems:
                if e.name == "axes":
                    continue
                ec = e.clone()
                # remove plot2.yscale and apply self.yscale:
                ec.SetScale(1, self.yscale/plot2.yscale, 1)
                self.AddPart(ec.z(plot_z))
                objs2.append(ec)
            objs = objs2

        else:
            # print('adding individual objects', len(objs))
            for a in objs:
                if isinstance(a, typs):
                    # special scaling to preserve the aspect ratio
                    # print('adding', a.name, 'fixed scale', self.fixed_scale)
                    a.scale(self.fixed_scale)
                else:
                    # print('adding', a.name, 'yscale', self.yscale)
                    a.scale([1, self.yscale, 1])
                py = a.y()
                a.y(py * self.yscale)
                self.AddPart(a)

        if self.cut:  # todo
            for a in objs:
                if not a or a.name == "axes":
                    continue
                if self._y0lim is not None and hasattr(a, "cutWithPlane"):
                    a.cutWithPlane([0, self._y0lim, 0], [0, 1, 0])
                if self._y1lim is not None and hasattr(a, "cutWithPlane"):
                    a.cutWithPlane([0, self._y1lim, 0], [0, -1, 0])
                if self._x0lim is not None and hasattr(a, "cutWithPlane"):
                    a.cutWithPlane([self._x0lim, 0, 0], [1, 0, 0])
                if self._x1lim is not None and hasattr(a, "cutWithPlane"):
                    a.cutWithPlane([self._x1lim, 0, 0], [-1, 0, 0])

        return self

    def overlayPlot(self, *args, **kwargs):
        """Plot on top of an already existing plot."""
        kwargs['format'] = self
        plt = plot(*args, **kwargs)
        plt.format = self
        for a in plt.unpack():
            self.AddPart(a)
        return self

    def overlayHistogram(self, *args, **kwargs):
        """Plot histogram on top of an already existing plot."""
        kwargs['format'] = self
        h = histogram(*args, **kwargs)
        h.format = self
        for a in h.unpack():
            self.AddPart(a)
        return self


def plot(*args, **kwargs):
    """
    Draw a 2D line plot, or scatter plot, of variable x vs variable y.
    Input format can be either [allx], [allx, ally] or [(x1,y1), (x2,y2), ...]

    :param list xerrors: set uncertainties for the x variable, shown as error bars.
    :param list yerrors: set uncertainties for the y variable, shown as error bars.
    :param bool errorBand: represent errors on y as a filled error band.
        Use ``ec`` keyword to modify its color.

    :param list xlim: set limits to the range for the x variable
    :param list ylim: set limits to the range for the y variable
    :param float, aspect: desired aspect ratio.
        If None, it is automatically calculated to get a reasonable aspect ratio.
        Scaling factor is saved in ``Plot.yscale``.

    :param str c: color of frame and text.
    :param float alpha: opacity of frame and text.
    :param str xtitle: title label along x-axis.
    :param str ytitle: title label along y-axis.
    :param str title: histogram title on top.
    :param float titleSize: size of title
    :param str ec: color of error bar, by default the same as marker color
    :param str lc: color of line
    :param float la: transparency of line
    :param float lw: width of line
    :param bool dashed: use a dashed line style
    :param bool splined: spline the line joining the point as a countinous curve
    :param str,int marker: use a marker shape for the data points
    :param float ms: marker size.
    :param str mc: color of marker
    :param float ma: opacity of marker

    :Example:
        .. code-block:: python

            from vedo.pyplot import plot
            import numpy as np

            x = np.linspace(0, 6.28, num=50)

            plot(np.sin(x), 'r').plot(np.cos(x), 'bo-').show()

        |simpleplot|

    More examples:

    |plot_errbars| |plot_errbars.py|_

    |plot_errband| |plot_errband.py|_

    |plot_pip| |plot_pip.py|_

    |scatter1| |scatter1.py|_

    |scatter2| |scatter2.py|_


    If input is an external function or a forumula, draw the surface
    representing the function :math:`f(x,y)`.

    :param float x: x range of values.
    :param float y: y range of values.
    :param float zlimits: limit the z range of the independent variable.
    :param int zlevels: will draw the specified number of z-levels contour lines.
    :param bool showNan: show where the function does not exist as red points.
    :param list bins: number of bins in x and y.

    |plot_fxy| |plot_fxy.py|_

    Function is: :math:`f(x,y)=\sin(3x) \cdot \log(x-y)/3` in range :math:`x=[0,3], y=[0,3]`.


    If ``mode='complex'`` draw the real value of the function and color map the imaginary part.

    :param str cmap: diverging color map (white means imag(z)=0).
    :param float lw: line with of the binning
    :param list bins: binning in x and y

    |fcomplex| |plot_fxy.py|_


    If ``mode='polar'`` input arrays are interpreted as a list of polar angles and radii.
    Build a polar (radar) plot by joining the set of points in polar coordinates.

    :param str title: plot title
    :param float tsize: title size
    :param int bins: number of bins in phi
    :param float r1: inner radius
    :param float r2: outer radius
    :param float lsize: label size
    :param c: color of the line
    :param bc: color of the frame and labels
    :param alpha: alpha of the frame
    :param int ps: point size in pixels, if ps=0 no point is drawn
    :param int lw: line width in pixels, if lw=0 no line is drawn
    :param bool deg: input array is in degrees
    :param float vmax: normalize radius to this maximum value
    :param bool fill: fill convex area with solid color
    :param bool spline: interpolate the set of input points
    :param bool showDisc: draw the outer ring axis
    :param int nrays: draw this number of axis rays (continuous and dashed)
    :param bool showLines: draw lines to the origin
    :param bool showAngles: draw angle values

    |histo_polar| |histo_polar.py|_


    If ``mode='spheric'`` input input is an external function rho(theta, phi).
    A surface is created in spherical coordinates.
    Return an ``Plot(Assembly)`` of 2 objects, the unit grid
    sphere (in wireframe representation) and the surface `rho(theta, phi)`.

    :param function rfunc: handle to a user defined function.
    :param bool normalize: scale surface to fit inside the unit sphere
    :param int res: grid resolution
    :param bool scalarbar: add a 3D scalarbar to the plot for radius
    :param c: color of the unit grid
    :param alpha: transparency of the unit grid
    :param str cmap: color map of the surface

    |plot_spheric| |plot_spheric.py|_
    """
    mode = kwargs.pop("mode", "")
    if "spher" in mode:
        return _plotSpheric(args[0], **kwargs)

    if "bar" in mode:
        return _barplot(args[0], **kwargs)

    if isinstance(args[0], str) or "function" in str(type(args[0])):
        if "complex" in mode:
            return _plotFz(args[0], **kwargs)
        return _plotFxy(args[0], **kwargs)

    # grab the matplotlib-like options
    optidx = None
    for i, a in enumerate(args):
        if i > 0 and isinstance(a, str):
            optidx = i
            break
    if optidx:
        opts = args[optidx].replace(" ", "")
        if "--" in opts:
            opts = opts.replace("--", "")
            kwargs["dashed"] = True
        elif "-" in opts:
            opts = opts.replace("-", "")
        else:
            kwargs["lw"] = 0
        symbs = [".", "p", "*", "h", "D", "d", "o", "v", "^", ">", "<", "s", "x", "+", "a"]
        for ss in symbs:
            if ss in opts:
                opts = opts.replace(ss, "", 1)
                kwargs["marker"] = ss
                break
        allcols = list(colors.color_nicks.keys()) + list(colors.colors.keys())
        for cc in allcols:
            if cc in opts:
                opts = opts.replace(cc, "")
                kwargs["lc"] = cc
                kwargs["mc"] = cc
                break
        if opts:
            colors.printc("Could not understand option(s):", opts, c="y")

    if optidx == 1 or optidx is None:
        if utils.isSequence(args[0][0]):
            # print('case 1', 'plot([(x,y),..])')
            data = np.array(args[0])
            x = np.array(data[:, 0])
            y = np.array(data[:, 1])
        elif len(args) == 1 or optidx == 1:
            # print('case 2', 'plot(x)')
            x = np.linspace(0, len(args[0]), num=len(args[0]))
            y = np.array(args[0])
        elif utils.isSequence(args[1]):
            # print('case 3', 'plot(allx,ally)')
            x = np.array(args[0])
            y = np.array(args[1])
        elif utils.isSequence(args[0]) and utils.isSequence(args[0][0]):
            # print('case 4', 'plot([allx,ally])')
            x = np.array(args[0][0])
            y = np.array(args[0][1])

    elif optidx == 2:
        # print('case 5', 'plot(x,y)')
        x = np.array(args[0])
        y = np.array(args[1])

    else:
        print("plot(): Could not understand input arguments", args)
        return None

    if "polar" in mode:
        return _plotPolar(np.c_[x, y], **kwargs)

    return _plotxy(np.c_[x, y], **kwargs)


def histogram(*args, **kwargs):
    """
    Histogramming for 1D and 2D data arrays.

    For 1D arrays:

    :param int bins: number of bins.
    :param list vrange: restrict the range of the histogram.
    :param bool density: normalize the area to 1 by dividing by the nr of entries and bin size.
    :param bool logscale: use logscale on y-axis.
    :param bool fill: fill bars woth solid color `c`.
    :param float gap: leave a small space btw bars.
    :param bool outline: show outline of the bins.
    :param bool errors: show error bars.

    |histo_1D| |histo_1D.py|_


    If ``mode='polar'`` assume input is polar coordinate system (rho, theta):

    :param list weights: array of weights, of the same shape as the input.
        Each value only contributes its associated weight towards the bin count (instead of 1).

    :param str title: histogram title
    :param float tsize: title size
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
    :param int nrays: draw this number of axis rays (continuous and dashed)
    :param bool showLines: show lines to the origin
    :param bool showAngles: show angular values
    :param bool showErrors: show error bars

    |histo_polar| |histo_polar.py|_


    For 2D arrays:

    Input data formats [(x1,x2,..), (y1,y2,..)] or [(x1,y1), (x2,y2),..] are both valid.

    :param str xtitle: x axis title
    :param str ytitle: y axis title
    :param list bins: binning as (nx, ny)
    :param list vrange: range in x and y in format [(xmin,xmax), (ymin,ymax)]
    :param str cmap: color map name
    :param float lw: line width of the binning
    :param bool scalarbar: add a scalarbar

    |histo_2D| |histo_2D.py|_


    If ``mode='hexbin'``, build a hexagonal histogram from a list of x and y values.

    :param str xtitle: x axis title
    :param str ytitle: y axis title
    :param bool bins: nr of bins for the smaller range in x or y.
    :param list vrange: range in x and y in format [(xmin,xmax), (ymin,ymax)]
    :param float norm: sets a scaling factor for the z axis (freq. axis).
    :param bool fill: draw solid hexagons.
    :param str cmap: color map name for elevation.

    |histo_hexagonal| |histo_hexagonal.py|_


    If ``mode='spheric'``, build a histogram from list of theta and phi values.

    :param float rmax: maximum radial elevation of bin
    :param int res: sphere resolution
    :param cmap: color map name
    :param float lw: line width of the bin edges
    :param bool scalarbar: add a scalarbar to plot

    |histo_spheric| |histo_spheric.py|_
    """
    mode = kwargs.pop("mode", "")
    if len(args) == 2:  # x, y
        if "spher" in mode:
            return _histogramSpheric(args[0], args[1], **kwargs)
        if "hex" in mode:
            return _histogramHexBin(args[0], args[1], **kwargs)
        return _histogram2D(args[0], args[1], **kwargs)

    elif len(args) == 1:

        if isinstance(args[0], vedo.Volume):
            data = args[0].pointdata[0]
        elif isinstance(args[0], vedo.Points):
            pd0 = args[0].pointdata[0]
            if pd0:
                data = pd0.ravel()
            else:
                data = args[0].celldata[0].ravel()
        else:
            data = np.array(args[0])

        if "spher" in mode:
            return _histogramSpheric(args[0][:, 0], args[0][:, 1], **kwargs)

        if len(data.shape) == 1:
            if "polar" in mode:
                return _histogramPolar(data, **kwargs)
            return _histogram1D(data, **kwargs)
        else:
            if "hex" in mode:
                return _histogramHexBin(args[0][:, 0], args[0][:, 1], **kwargs)
            return _histogram2D(args[0], **kwargs)

    print("histogram(): Could not understand input", args[0])
    return None


def fit(points,
        deg=1,
        niter=0,
        nstd=3,
        xerrors=None,
        yerrors=None,
        vrange=None,
        res=250,
        lw=3,
        c='red4',
    ):
    """
    Polynomial fitting in 2D with parameter error and error bands calculation.

    Errors bars in both x and y are supported.

    Additional information about the fitting output can be accessed. E.g.:

        ``fit = fitPolynomial(pts)``

        - ``fit.coefficients``: contains the coefficient of the polynomial fit
        - ``fit.coefficientErrors``: errors on the fitting coefficients,
            these numbers only make sense if parameters are not correlated

        - ``fit.MonteCarloCoefficients``: fitting coefficient set from MC generation
        - ``fit.covarianceMatrix``: covariance matrix as a numpy array
        - ``fit.reducedChi2``: reduced chi-square of the fitting
        - ``fit.ndof``: number of degrees of freedom
        - ``fit.dataSigma``: mean data dispersion from the central fit assuming Chi2=1
        - ``fit.errorLines``: a ``vedo.Line`` object for the upper and lower error band
        - ``fit.errorBand``: the ``vedo.Mesh`` object representing the error band

    Errors on x and y can be specified. If left `None` an estimate is made from
    the statistical spread of the dataset itself. Errors are always assumed gaussian.

    :param int deg: degree of the polynomial to be fitted
    :param int niter: number of monte-carlo iterations to compute error bands.
        If set to 0, return the simple least-squares fit with naive error estimation
        on coefficients only. A reasonable non-zero value to set is about 500, in
        this case ``errorLines``, ``errorBand`` and the other class attributes are filled

    :param int nstd: nr. of standard deviation to use for error calculation
    :param list xerrors: array of the same length of points with the errors on x
    :param list yerrors: array of the same length of points with the errors on y
    :param list vrange: specify the domain range of the fitting line
        (only affects visualization, but can be used to extrapolate the fit
         outside the data range)

    :param int res: resolution of the output fitted line and error lines

    |fitPolynomial1| |fitPolynomial1.py|_

    |fitPolynomial2| |fitPolynomial2.py|_
    """
    if isinstance(points, vedo.pointcloud.Points):
        points = points.points()
    points = np.asarray(points)
    if len(points) == 2: # assume user is passing [x,y]
        points = np.c_[points[0],points[1]]
    x = points[:,0]
    y = points[:,1] # ignore z

    n = len(x)
    ndof = n - deg - 1
    if vrange is not None:
        x0, x1 = vrange
    else:
        x0, x1 = np.min(x), np.max(x)
        if xerrors is not None:
            x0 -= xerrors[0]/2
            x1 += xerrors[-1]/2

    tol = (x1-x0)/1000
    xr = np.linspace(x0,x1, res)

    # project x errs on y
    if xerrors is not None:
        xerrors = np.asarray(xerrors)
        if yerrors is not None:
            yerrors = np.asarray(yerrors)
            w = 1.0/yerrors
            coeffs = np.polyfit(x, y, deg, w=w, rcond=None)
        else:
            coeffs = np.polyfit(x, y, deg, rcond=None)
        # update yerrors, 1 bootstrap iteration is enough
        p1d = np.poly1d(coeffs)
        der = (p1d(x+tol)-p1d(x))/tol
        yerrors = np.sqrt(yerrors*yerrors + np.power(der*xerrors,2))

    if yerrors is not None:
        yerrors = np.asarray(yerrors)
        w = 1.0/yerrors
        coeffs, V = np.polyfit(x, y, deg, w=w, rcond=None, cov=True)
    else:
        w = 1
        coeffs, V = np.polyfit(x, y, deg, rcond=None, cov=True)

    p1d = np.poly1d(coeffs)
    theor = p1d(xr)
    l = shapes.Line(xr, theor, lw=lw, c=c).z(tol*2)
    l.coefficients = coeffs
    l.covarianceMatrix = V
    residuals2_sum = np.sum(np.power(p1d(x)-y, 2))/ndof
    sigma = np.sqrt(residuals2_sum)
    l.reducedChi2 = np.sum(np.power((p1d(x)-y)*w, 2))/ndof
    l.ndof = ndof
    l.dataSigma = sigma # worked out from data using chi2=1 hypo
    l.name = "LinePolynomialFit"

    if not niter:
        l.coefficientErrors = np.sqrt(np.diag(V))
        return l ################################

    if yerrors is not None:
        sigma = yerrors
    else:
        w = None
        l.reducedChi2 = 1

    Theors, all_coeffs = [], []
    for i in range(niter):
        noise = np.random.randn(n)*sigma
        Coeffs = np.polyfit(x, y + noise, deg, w=w, rcond=None)
        all_coeffs.append(Coeffs)
        P1d = np.poly1d(Coeffs)
        Theor = P1d(xr)
        Theors.append(Theor)
    all_coeffs = np.array(all_coeffs)
    l.MonteCarloCoefficients = all_coeffs

    stds = np.std(Theors, axis=0)
    l.coefficientErrors = np.std(all_coeffs, axis=0)

    # check distributions on the fly
    # for i in range(deg+1):
    #     vedo.pyplot.histogram(all_coeffs[:,i],title='par'+str(i)).show(new=1)
    # vedo.pyplot.histogram(all_coeffs[:,0], all_coeffs[:,1],
    #                       xtitle='param0', ytitle='param1',scalarbar=1).show(new=1)
    # vedo.pyplot.histogram(all_coeffs[:,1], all_coeffs[:,2],
    #                       xtitle='param1', ytitle='param2').show(new=1)
    # vedo.pyplot.histogram(all_coeffs[:,0], all_coeffs[:,2],
    #                       xtitle='param0', ytitle='param2').show(new=1)

    error_lines = []
    for i in [nstd, -nstd]:
        el = shapes.Line(xr, theor+stds*i, lw=1, alpha=0.2, c='k').z(tol)
        error_lines.append(el)
        el.name = "ErrorLine for sigma="+str(i)

    l.errorLines = error_lines
    l1 = error_lines[0].points().tolist()
    cband = l1 + list(reversed(error_lines[1].points().tolist())) + [l1[0]]
    l.errorBand = shapes.Line(cband).triangulate().lw(0).c('k', 0.15)
    l.errorBand.name = "PolynomialFitErrorBand"
    return l


#########################################################################################
def _plotxy(
    data,
    format=None,
    aspect=4/3,
    xlim=None,
    ylim=None,
    xerrors=None,
    yerrors=None,
    title="",
    xtitle="x",
    ytitle="y",
    titleSize=None,
    c="k",
    alpha=1,
    ec=None,
    lc="k",
    la=1,
    lw=3,
    dashed=False,
    spline=False,
    errorBand=False,
    marker="",
    ms=None,
    mc=None,
    ma=None,
    pad=0.05,
    axes={},
):
    line=False
    if lw>0:
        line=True

    if marker == "" and not line and not spline:
        line = True

    # purge NaN from data
    validIds = np.all(np.logical_not(np.isnan(data)), axis=1)
    data = data[validIds]
    offs = 0  # z offset

    if format is not None:  # reset to allow meaningful overlap
        xlim = format.xlim
        ylim = format.ylim
        aspect = format.aspect
        pad = format.pad
        title = ""
        xtitle = ""
        ytitle = ""
        offs = format.zmax

    x0, y0 = np.min(data, axis=0)
    x1, y1 = np.max(data, axis=0)
    x0lim, x1lim = x0 - pad * (x1 - x0), x1 + pad * (x1 - x0)
    y0lim, y1lim = y0 - pad * (y1 - y0), y1 + pad * (y1 - y0)
    if y0lim == y1lim:  # in case y is constant
        y0lim = y0lim - (x1lim - x0lim) / 2
        y1lim = y1lim + (x1lim - x0lim) / 2
    elif x0lim == x1lim:  # in case x is constant
        x0lim = x0lim - (y1lim - y0lim) / 2
        x1lim = x1lim + (y1lim - y0lim) / 2

    if xlim is not None and xlim[0] is not None:
        x0lim = xlim[0]
    if xlim is not None and xlim[1] is not None:
        x1lim = xlim[1]
    if ylim is not None and ylim[0] is not None:
        y0lim = ylim[0]
    if ylim is not None and ylim[1] is not None:
        y1lim = ylim[1]

    dx = x1lim - x0lim
    dy = y1lim - y0lim
    if dx == 0 and dy == 0:  # in case x and y are all constant
        x0lim = x0lim - 1
        x1lim = x1lim + 1
        y0lim = y0lim - 1
        y1lim = y1lim + 1
        dx, dy = 1, 1

    yscale = dx / dy / aspect
    y0lim, y1lim = y0lim * yscale, y1lim * yscale

    if format is not None:
        x0lim = format._x0lim
        y0lim = format._y0lim
        x1lim = format._x1lim
        y1lim = format._y1lim
        yscale = format.yscale

    dx = x1lim - x0lim
    dy = y1lim - y0lim
    offs += np.sqrt(dx * dx + dy * dy) / 10000

    scale = np.array([[1, yscale]])
    data = np.multiply(data, scale)

    acts = []

    # the line or spline
    if dashed:
        l = shapes.DashedLine(data, c=lc, alpha=la, lw=lw)
        acts.append(l)
    elif spline:
        l = shapes.KSpline(data).lw(lw).c(lc).alpha(la)
        acts.append(l)
    elif line:
        l = shapes.Line(data, c=lc, alpha=la).lw(lw)
        acts.append(l)

    if marker:

        pts = shapes.Points(data)
        if mc is None:
            mc = lc
        if ma is None:
            ma = la

        if utils.isSequence(ms):  ### variable point size
            mk = shapes.Marker(marker, s=1)
            msv = np.zeros_like(pts.points())
            msv[:, 0] = ms
            marked = shapes.Glyph(
                pts, glyphObj=mk, c=mc, orientationArray=msv, scaleByVectorSize=True
            )
        else:  ### fixed point size

            if ms is None:
                ms = dx / 100.0
                # print('automatic ms =', ms)

            if utils.isSequence(mc):
                # print('mc is sequence')
                mk = shapes.Marker(marker, s=ms).triangulate()
                msv = np.zeros_like(pts.points())
                msv[:, 0] = 1
                marked = shapes.Glyph(
                    pts, glyphObj=mk, c=mc, orientationArray=msv, scaleByVectorSize=True
                )
            else:
                # print('mc is fixed color')
                mk = shapes.Marker(marker, s=ms).triangulate()
                marked = shapes.Glyph(pts, glyphObj=mk, c=mc)

        marked.alpha(ma).z(offs)
        acts.append(marked)

    if ec is None:
        if mc is not None:
            ec = mc
        else:
            ec = lc

    if xerrors is not None and not errorBand:
        if len(xerrors) != len(data):
            colors.printc("Error in plotxy(xerrors=...): mismatched array length.", c='r')
            return None
        errs = []
        for i, dta in enumerate(data):
            xval, yval = dta
            xerr = xerrors[i] / 2
            el = shapes.Line((xval - xerr, yval, offs), (xval + xerr, yval, offs))
            errs.append(el)
        mxerrs = merge(errs).c(ec).lw(lw).alpha(alpha).z(2 * offs)
        acts.append(mxerrs)

    if yerrors is not None and not errorBand:
        if len(yerrors) != len(data):
            colors.printc("Error in plotxy(yerrors=...): mismatched array length.", c='r')
            return None
        errs = []
        for i in range(len(data)):
            xval, yval = data[i]
            yerr = yerrors[i] * yscale
            el = shapes.Line((xval, yval - yerr, offs), (xval, yval + yerr, offs))
            errs.append(el)
        myerrs = merge(errs).c(ec).lw(lw).alpha(alpha).z(3 * offs)
        acts.append(myerrs)

    if errorBand:
        epsy = np.zeros_like(data)
        epsy[:, 1] = yerrors * yscale
        data3dup = data + epsy
        data3dup = np.c_[data3dup, np.zeros_like(yerrors)]
        data3d_down = data - epsy
        data3d_down = np.c_[data3d_down, np.zeros_like(yerrors)]
        band = shapes.Ribbon(data3dup, data3d_down).z(-offs)
        if ec is None:
            band.c(lc)
        else:
            band.c(ec)
        band.alpha(la).z(2 * offs)
        acts.append(band)

    for a in acts:
        a.cutWithPlane([0, y0lim, 0], [0, 1, 0])
        a.cutWithPlane([0, y1lim, 0], [0, -1, 0])
        a.cutWithPlane([x0lim, 0, 0], [1, 0, 0])
        a.cutWithPlane([x1lim, 0, 0], [-1, 0, 0])
        a.lighting('off')

    if title:
        if titleSize is None:
            titleSize = dx / 40.0
        tit = shapes.Text3D(
            title,
            s=titleSize,
            c=c,
            depth=0,
            alpha=alpha,
            pos=((x0lim + x1lim) / 2, y1lim + (y1lim-y0lim) / 80, 0),
            justify="bottom-center",
        )
        tit.pickable(False).z(3 * offs)
        acts.append(tit)

    if axes == 1 or axes == True:
        axes = {}
    if isinstance(axes, dict):  #####################
        ndiv = 6
        if "numberOfDivisions" in axes.keys():
            ndiv = axes["numberOfDivisions"]
        tp, ts = utils.makeTicks(y0lim / yscale, y1lim / yscale, ndiv / aspect)
        labs = []
        for i in range(1, len(tp) - 1):
            ynew = utils.linInterpolate(tp[i], [0, 1], [y0lim, y1lim])
            # print(i, tp[i], ynew, ts[i])
            labs.append([ynew, ts[i]])
        if "xtitle" not in axes: axes["xtitle"] = xtitle
        if "ytitle" not in axes: axes["ytitle"] = ytitle
        axes["yValuesAndLabels"] = labs
        axes["xrange"] = (x0lim, x1lim)
        axes["yrange"] = (y0lim, y1lim)
        axes["zrange"] = (0, 0)
        # axes["c"] = "k"
        axes["yUseBounds"] = True
        axs = addons.Axes(**axes)
        axs.name = "axes"
        asse = Plot(acts, axs)
        asse.axes = axs
        asse.SetOrigin(x0lim, y0lim, 0)

    else:
#        settings.xtitle = xtitle
#        settings.ytitle = ytitle
        asse = Plot(acts)

    asse.yscale = yscale
    asse.xlim = xlim
    asse.ylim = ylim
    asse.aspect = aspect
    asse.pad = pad
    asse.title = title
    asse.xtitle = xtitle
    asse.ytitle = ytitle
    asse._x0lim = x0lim
    asse._y0lim = y0lim
    asse._x1lim = x1lim
    asse._y1lim = y1lim
    asse.zmax = offs * 3  # z-order
    asse.name = "plotxy"
    return asse


def _plotFxy(
    z,
    xlim=(0, 3),
    ylim=(0, 3),
    zlim=(None, None),
    showNan=True,
    zlevels=10,
    c=None,
    bc="aqua",
    alpha=1,
    texture="paper4",
    bins=(100, 100),
    axes=True,
):
    if isinstance(z, str):
        try:
            z = z.replace("math.", "").replace("np.", "")
            namespace = locals()
            code = "from math import*\ndef zfunc(x,y): return " + z
            exec(code, namespace)
            z = namespace["zfunc"]
        except:
            colors.printc("Syntax Error in _plotFxy()", c='r')
            return None

    if c is not None:
        texture = None # disable

    ps = vtk.vtkPlaneSource()
    ps.SetResolution(bins[0], bins[1])
    ps.SetNormal([0, 0, 1])
    ps.Update()
    poly = ps.GetOutput()
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    todel, nans = [], []

    for i in range(poly.GetNumberOfPoints()):
        px, py, _ = poly.GetPoint(i)
        xv = (px + 0.5) * dx + xlim[0]
        yv = (py + 0.5) * dy + ylim[0]
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
        colors.printc("Function is not real in the domain", c='r')
        return None

    if zlim[0]:
        tmpact1 = Mesh(poly)
        a = tmpact1.cutWithPlane((0, 0, zlim[0]), (0, 0, 1))
        poly = a.polydata()
    if zlim[1]:
        tmpact2 = Mesh(poly)
        a = tmpact2.cutWithPlane((0, 0, zlim[1]), (0, 0, -1))
        poly = a.polydata()

    cmap=''
    if c in colors.cmaps_names:
        cmap = c
        c = None
        bc= None

    mesh = Mesh(poly, c, alpha).computeNormals().lighting("plastic")

    if cmap:
        mesh.addElevationScalars().cmap(cmap)
    if bc:
        mesh.bc(bc)
    if texture:
        mesh.texture(texture)

    acts = [mesh]
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
        zbandsact = Mesh(zpoly, "k", alpha).lw(1).lighting('off')
        zbandsact._mapper.SetResolveCoincidentTopologyToPolygonOffset()
        acts.append(zbandsact)

    if showNan and len(todel):
        bb = mesh.GetBounds()
        if bb[4] <= 0 and bb[5] >= 0:
            zm = 0.0
        else:
            zm = (bb[4] + bb[5]) / 2
        nans = np.array(nans) + [0, 0, zm]
        nansact = shapes.Points(nans, r=2, c="red", alpha=alpha)
        nansact.GetProperty().RenderPointsAsSpheresOff()
        acts.append(nansact)

    if axes:
        axs = addons.Axes(mesh)
        acts.append(axs)
    asse = Assembly(acts)
    asse.name = "plotFxy"
    if isinstance(z, str):
        asse.name += " " + z
    return asse


def _plotFz(
    z,
    x=(-1, 1),
    y=(-1, 1),
    zlimits=(None, None),
    cmap="PiYG",
    alpha=1,
    lw=0.1,
    bins=(75, 75),
    axes=True,
):
    if isinstance(z, str):
        try:
            z = z.replace("np.", "")
            namespace = locals()
            code = "from math import*\ndef zfunc(x,y): return " + z
            exec(code, namespace)
            z = namespace["zfunc"]
        except:
            colors.printc("Syntax Error in complex plotFz()", c='r')
            return None

    ps = vtk.vtkPlaneSource()
    ps.SetResolution(bins[0], bins[1])
    ps.SetNormal([0, 0, 1])
    ps.Update()
    poly = ps.GetOutput()
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    arrImg = []
    for i in range(poly.GetNumberOfPoints()):
        px, py, _ = poly.GetPoint(i)
        xv = (px + 0.5) * dx + x[0]
        yv = (py + 0.5) * dy + y[0]
        try:
            zv = z(np.complex(xv), np.complex(yv))
        except:
            zv = 0
        poly.GetPoints().SetPoint(i, [xv, yv, np.real(zv)])
        arrImg.append(np.imag(zv))

    mesh = Mesh(poly, alpha).lighting("plastic")
    v = max(abs(np.min(arrImg)), abs(np.max(arrImg)))
    mesh.cmap(cmap, arrImg, vmin=-v, vmax=v)
    mesh.computeNormals().lw(lw)

    if zlimits[0]:
        mesh.cutWithPlane((0, 0, zlimits[0]), (0, 0, 1))
    if zlimits[1]:
        mesh.cutWithPlane((0, 0, zlimits[1]), (0, 0, -1))

    acts = [mesh]
    if axes:
        axs = addons.Axes(mesh, ztitle="Real part")
        acts.append(axs)
    asse = Assembly(acts)
    asse.name = "plotFz"
    if isinstance(z, str):
        asse.name += " " + z
    return asse


def _plotPolar(
    rphi,
    title="",
    tsize=0.1,
    lsize=0.05,
    r1=0,
    r2=1,
    c="blue",
    bc="k",
    alpha=1,
    ps=5,
    lw=3,
    deg=False,
    vmax=None,
    fill=False,
    spline=False,
    smooth=0,
    showDisc=True,
    nrays=8,
    showLines=True,
    showAngles=True,
):
    if len(rphi) == 2:
        rphi = np.stack((rphi[0], rphi[1]), axis=1)

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
    points = []
    for i in range(len(thetas)):
        t = thetas[i]
        r = (radii[i]) / vmax * r2 + r1
        ct, st = np.cos(t), np.sin(t)
        points.append([r * ct, r * st, 0])
    p0 = points[0]
    points.append(p0)

    r2e = r1 + r2
    lines = None
    if spline:
        lines = shapes.KSpline(points, closed=True)
        lines.c(c).lw(lw).alpha(alpha)
    elif lw:
        lines = shapes.Line(points)
        lines.c(c).lw(lw).alpha(alpha)

    points.pop()

    ptsact = None
    if ps:
        ptsact = shapes.Points(points, r=ps, c=c, alpha=alpha)

    filling = None
    if fill and lw:
        faces = []
        coords = [[0, 0, 0]] + lines.points().tolist()
        for i in range(1, lines.N()):
            faces.append([0, i, i + 1])
        filling = Mesh([coords, faces]).c(c).alpha(alpha)

    back = None
    back2 = None
    if showDisc:
        back = shapes.Disc(r1=r2e, r2=r2e * 1.01, c=bc, res=(1,360))
        back.z(-0.01).lighting('off').alpha(alpha)
        back2 = shapes.Disc(r1=r2e/2, r2=r2e/2 * 1.005, c=bc, res=(1,360))
        back2.z(-0.01).lighting('off').alpha(alpha)

    ti = None
    if title:
        ti = shapes.Text3D(title, (0, 0, 0), s=tsize, depth=0, justify="top-center")
        ti.pos(0, -r2e * 1.15, 0.01)

    rays = []
    if showDisc:
        rgap = 0.05
        for t in np.linspace(0, 2 * np.pi, num=nrays, endpoint=False):
            ct, st = np.cos(t), np.sin(t)
            if showLines:
                l = shapes.Line((0, 0, -0.01), (r2e * ct * 1.03, r2e * st * 1.03, -0.01))
                rays.append(l)
                ct2, st2 = np.cos(t+np.pi/nrays), np.sin(t+np.pi/nrays)
                lm = shapes.DashedLine((0, 0, -0.01),
                                       (r2e * ct2, r2e * st2, -0.01),
                                       spacing=0.25)
                rays.append(lm)
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
                a = shapes.Text3D(int(t * k), pos=(0, 0, 0), s=lsize, depth=0, justify=ju)
                a.pos(r2e * ct * (1 + rgap), r2e * st * (1 + rgap), -0.01)
                angles.append(a)

    mrg = merge(back, back2, angles, rays, ti)
    if mrg:
        mrg.color(bc).alpha(alpha).lighting('off')
    rh = Assembly([lines, ptsact, filling] + [mrg])
    rh.base = np.array([0, 0, 0])
    rh.top = np.array([0, 0, 1])
    rh.name = "plotPolar"
    return rh


def _plotSpheric(rfunc, normalize=True, res=33, scalarbar=True, c="grey", alpha=0.05, cmap="jet"):
    sg = shapes.Sphere(res=res, quads=True)
    sg.alpha(alpha).c(c).wireframe()

    cgpts = sg.points()
    r, theta, phi = utils.cart2spher(*cgpts.T)

    newr, inans = [], []
    for i in range(len(r)):
        try:
            ri = rfunc(theta[i], phi[i])
            if np.isnan(ri):
                inans.append(i)
                newr.append(1)
            else:
                newr.append(ri)
        except:
            inans.append(i)
            newr.append(1)

    newr = np.array(newr)
    if normalize:
        newr = newr / np.max(newr)
        newr[inans] = 1

    nanpts = []
    if len(inans):
        redpts = utils.spher2cart(newr[inans], theta[inans], phi[inans])
        nanpts.append(shapes.Points(redpts, r=4, c="r"))

    pts = utils.spher2cart(newr, theta, phi)

    ssurf = sg.clone().points(pts)
    if len(inans):
        ssurf.deletePoints(inans)

    ssurf.alpha(1).wireframe(0).lw(0.1)

    ssurf.cmap(cmap, newr)
    ssurf.computeNormals()

    if scalarbar:
        xm = np.max([np.max(pts[0]), 1])
        ym = np.max([np.abs(np.max(pts[1])), 1])
        ssurf.mapper().SetScalarRange(np.min(newr), np.max(newr))
        sb3d = ssurf.addScalarBar3D(sx=xm * 0.07, sy=ym, c='k').scalarbar
        sb3d.rotateX(90).pos(xm * 1.1, 0, -0.5)
    else:
        sb3d = None

    sg.pickable(False)
    asse = Assembly([ssurf, sg] + nanpts + [sb3d])
    asse.name = "plotSpheric"
    return asse

#########################################################################################
def _barplot(
    data,
    format=None,
    errors=False,
    aspect=4/3,
    xlim=None,
    ylim=(0,None),
    xtitle=" ",
    ytitle="counts",
    title="",
    titleSize=None,
    titleColor=None,
    logscale=False,
    fill=True,
    c="olivedrab",
    gap=0.02,
    alpha=1,
    outline=False,
    lw=2,
    lc="k",
    pad=0.05,
    axes={},
    bc="k",
):
    offs = 0  # z offset
    if len(data) == 4:
        counts, xlabs, cols, edges = data
    elif len(data) == 3:
        counts, xlabs, cols = data
        edges = np.array(range(len(counts)+1))+0.5
    elif len(data) == 2:
        counts, xlabs = data
        edges = np.array(range(len(counts)+1))+0.5
        cols = [c] * len(counts)
    else:
        m = "barplot error: data must be given as [counts, labels, colors, edges] not\n"
        colors.printc(m, data, c='r')
        colors.printc("     bin edges and colors are optional. Abort.", c='r')
        raise RuntimeError()
    counts = np.asarray(counts)
    edges  = np.asarray(edges)

    # sanity checks
    assert len(counts) == len(xlabs)
    assert len(counts) == len(cols)
    assert len(counts) == len(edges)-1

    if format is not None:  # reset to allow meaningful overlap
        xlim = format.xlim
        ylim = format.ylim
        aspect = format.aspect
        pad = format.pad
        axes = 0
        title = ""
        xtitle = ""
        ytitle = ""
        offs = format.zmax

    if logscale:
        counts = np.log10(counts + 1)
        if ytitle=='counts':
            ytitle='log_10 (counts+1)'

    x0, x1 = np.min(edges), np.max(edges)
    y0, y1 = 0, np.max(counts)
    binsize = edges[1] - edges[0]

    x0lim, x1lim = x0 - pad * (x1 - x0), x1 + pad * (x1 - x0)
    y0lim, y1lim = y0 - pad * (y1 - y0) / 100, y1 + pad * (y1 - y0)
    if errors:
        y1lim += np.sqrt(y1) / 2

    if y0lim == y1lim:  # in case y is constant
        y0lim = y0lim - (x1lim - x0lim) / 2
        y1lim = y1lim + (x1lim - x0lim) / 2
    elif x0lim == x1lim:  # in case x is constant
        x0lim = x0lim - (y1lim - y0lim) / 2
        x1lim = x1lim + (y1lim - y0lim) / 2

    if xlim is not None and xlim[0] is not None:
        x0lim = xlim[0]
    if xlim is not None and xlim[1] is not None:
        x1lim = xlim[1]
    if ylim is not None and ylim[0] is not None:
        y0lim = ylim[0]
    if ylim is not None and ylim[1] is not None:
        y1lim = ylim[1]

    dx = x1lim - x0lim
    dy = y1lim - y0lim
    if dx == 0 and dy == 0:  # in case x and y are all constant
        x0lim = x0lim - 1
        x1lim = x1lim + 1
        y0lim = y0lim - 1
        y1lim = y1lim + 1
        dx, dy = 1, 1

    yscale = dx / dy / aspect
    y0lim, y1lim = y0lim * yscale, y1lim * yscale

    if format is not None:
        x0lim = format._x0lim
        y0lim = format._y0lim
        x1lim = format._x1lim
        y1lim = format._y1lim
        yscale = format.yscale

    dx = x1lim - x0lim
    dy = y1lim - y0lim
    offs += np.sqrt(dx * dx + dy * dy) / 10000

    counts = counts * yscale
    centers = (edges[0:-1] + edges[1:]) / 2

    rs = []
    maxheigth = 0
    if fill:  #####################
        if outline:
            gap = 0

        for i in range(len(centers)):
            p0 = (edges[i] + gap * binsize, 0, 0)
            p1 = (edges[i + 1] - gap * binsize, counts[i], 0)
            r = shapes.Rectangle(p0, p1)
            r.origin(p0).PickableOff()
            maxheigth = max(maxheigth, p1[1])
            if c in colors.cmaps_names:
                col = colors.colorMap((p0[0]+p1[0])/2, c, edges[0], edges[-1])
            else:
                col = cols[i]
            r.color(col).alpha(alpha).lighting('off').z(offs)
            r.name = f'bar_{i}'
            rs.append(r)

    if outline or not fill:  #####################
        lns = [[edges[0], 0, 0]]
        for i in range(len(centers)):
            lns.append([edges[i], counts[i], 0])
            lns.append([edges[i + 1], counts[i], 0])
            maxheigth = max(maxheigth, counts[i])
        lns.append([edges[-1], 0, 0])
        outl = shapes.Line(lns, c=lc, alpha=alpha, lw=lw).z(offs)
        outl.name = f'bar_outline_{i}'
        rs.append(outl)

    bin_centers_pos = []
    for i in range(len(centers)):
        if counts[i]:
            bin_centers_pos.append([centers[i], counts[i], 0])

    if errors:  #####################
        for bcp in bin_centers_pos:
            x = bcp[0]
            f = bcp[1]
            err = np.sqrt(f / yscale) * yscale
            el = shapes.Line([x, f-err/2, 0], [x, f+err/2, 0], c=lc, alpha=alpha, lw=lw)
            el.z(offs * 1.9)
            rs.append(el)
        # print('errors', el.z())

    for a in rs:  #####################
        a.cutWithPlane([0, y0lim, 0], [0,  1, 0])
        a.cutWithPlane([0, y1lim, 0], [0, -1, 0])
        a.cutWithPlane([x0lim, 0, 0], [1,  0, 0])
        a.cutWithPlane([x1lim, 0, 0], [-1, 0, 0])
        a.lighting('off')

    if title:  #####################
        if titleColor is None:
            titleColor = bc

        if titleSize is None:
            titleSize = dx / 40.0
        tit = shapes.Text3D(
            title,
            s=titleSize,
            c=titleColor,
            depth=0,
            alpha=alpha,
            pos=((x0lim + x1lim) / 2, y1lim + (y1lim-y0lim) / 80, 0),
            justify="bottom-center",
        )
        tit.pickable(False).z(2.5 * offs)
        rs.append(tit)

    if axes == 1 or axes == True: #####################
        axes = {}
    if isinstance(axes, dict):
        ndiv = 6
        if "numberOfDivisions" in axes:
            ndiv = axes["numberOfDivisions"]
        tp, ts = utils.makeTicks(y0lim / yscale, y1lim / yscale, ndiv / aspect)
        ylabs = []
        for i in range(1, len(tp) - 1):
            ynew = utils.linInterpolate(tp[i], [0, 1], [y0lim, y1lim])
            ylabs.append([ynew, ts[i]])
        axes["yValuesAndLabels"] = ylabs
        _xlabs = []
        for i in range(len(centers)):
            _xlabs.append([centers[i], str(xlabs[i])])
        axes["xValuesAndLabels"] = _xlabs
        if "xtitle" not in axes: axes["xtitle"] = xtitle
        if "ytitle" not in axes: axes["ytitle"] = ytitle
        axes["xrange"] = (x0lim, x1lim)
        axes["yrange"] = (y0lim, y1lim)
        axes["zrange"] = (0, 0)
        axes["c"] = bc
        axs = addons.Axes(**axes)
        axs.name = "axes"
        asse = Plot(rs, axs)
        asse.axes = axs
        asse.SetOrigin(x0lim, y0lim, 0)
    else:
#        settings.xtitle = xtitle
#        settings.ytitle = ytitle
        asse = Plot(rs)

    asse.yscale = yscale
    asse.xlim = xlim
    asse.ylim = ylim
    asse.aspect = aspect
    asse.pad = pad
    asse.title = title
    asse.xtitle = xtitle
    asse.ytitle = ytitle
    asse._x0lim = x0lim
    asse._y0lim = y0lim
    asse._x1lim = x1lim
    asse._y1lim = y1lim
    asse.zmax = offs * 3  # z-order
    asse.bins = edges
    asse.centers = centers
    asse.freqs = counts / yscale
    asse.name = "BarPlot"
    return asse

#########################################################################################
def _histogram1D(
    data,
    format=None,
    bins=25,
    aspect=4/3,
    xlim=None,
    ylim=(0,None),
    errors=False,
    title="",
    xtitle=" ",
    ytitle="counts",
    titleSize=None,
    titleColor=None,
    density=False,
    logscale=False,
    fill=True,
    c="olivedrab",
    gap=0.02,
    alpha=1,
    outline=False,
    lw=2,
    lc="k",
    marker="",
    ms=None,
    mc=None,
    ma=None,
    pad=0.05,
    axes={},
    bc="k",
):
    # purge NaN from data
    validIds = np.all(np.logical_not(np.isnan(data)))
    data = data[validIds]
    offs = 0  # z offset

    if format is not None:  # reset to allow meaningful overlap
        xlim = format.xlim
        ylim = format.ylim
        aspect = format.aspect
        pad = format.pad
        bins = format.bins
        axes = 0
        title = ""
        xtitle = ""
        ytitle = ""
        offs = format.zmax

    fs, edges = np.histogram(data, bins=bins, range=xlim)
    # print('frequencies', fs)
    # print('edges', edges)
    if density:
        ntot = len(data.ravel())
        binsize = edges[1]-edges[0]
        fs = fs/(ntot*binsize)
        if ytitle=='counts':
            ytitle=f"counts/({ntot}~\dot~{utils.precision(binsize,3)})"
    elif logscale:
        fs = np.log10(fs + 1)
        if ytitle=='counts':
            ytitle='log_10 (counts+1)'

    x0, x1 = np.min(edges), np.max(edges)
    y0, y1 = 0, np.max(fs)
    binsize = edges[1] - edges[0]

    x0lim, x1lim = x0 - pad * (x1 - x0), x1 + pad * (x1 - x0)
    y0lim, y1lim = y0 - pad * (y1 - y0) / 100, y1 + pad * (y1 - y0)
    if errors:
        y1lim += np.sqrt(y1) / 2

    if y0lim == y1lim:  # in case y is constant
        y0lim = y0lim - (x1lim - x0lim) / 2
        y1lim = y1lim + (x1lim - x0lim) / 2
    elif x0lim == x1lim:  # in case x is constant
        x0lim = x0lim - (y1lim - y0lim) / 2
        x1lim = x1lim + (y1lim - y0lim) / 2

    if xlim is not None and xlim[0] is not None:
        x0lim = xlim[0]
    if xlim is not None and xlim[1] is not None:
        x1lim = xlim[1]
    if ylim is not None and ylim[0] is not None:
        y0lim = ylim[0]
    if ylim is not None and ylim[1] is not None:
        y1lim = ylim[1]

    dx = x1lim - x0lim
    dy = y1lim - y0lim
    if dx == 0 and dy == 0:  # in case x and y are all constant
        x0lim = x0lim - 1
        x1lim = x1lim + 1
        y0lim = y0lim - 1
        y1lim = y1lim + 1
        dx, dy = 1, 1

    yscale = dx / dy / aspect
    y0lim, y1lim = y0lim * yscale, y1lim * yscale

    if format is not None:
        x0lim = format._x0lim
        y0lim = format._y0lim
        x1lim = format._x1lim
        y1lim = format._y1lim
        yscale = format.yscale

    dx = x1lim - x0lim
    dy = y1lim - y0lim
    offs += np.sqrt(dx * dx + dy * dy) / 10000

    fs = fs * yscale

    if utils.isSequence(bins):
        myedges = np.array(bins)
        bins = len(bins) - 1
    else:
        myedges = edges

    rs = []
    maxheigth = 0
    if fill:  #####################
        if outline:
            gap = 0

        for i in range(bins):
            p0 = (myedges[i] + gap * binsize, 0, 0)
            p1 = (myedges[i + 1] - gap * binsize, fs[i], 0)
            r = shapes.Rectangle(p0, p1)
            r.origin(p0).PickableOff()
            maxheigth = max(maxheigth, p1[1])
            if c in colors.cmaps_names:
                col = colors.colorMap((p0[0]+p1[0])/2, c, myedges[0], myedges[-1])
            else:
                col = c
            r.color(col).alpha(alpha).lighting('off').z(offs)
            rs.append(r)
#            print('rectangles', r.z())

    if outline:  #####################
        lns = [[myedges[0], 0, 0]]
        for i in range(bins):
            lns.append([myedges[i], fs[i], 0])
            lns.append([myedges[i + 1], fs[i], 0])
            maxheigth = max(maxheigth, fs[i])
        lns.append([myedges[-1], 0, 0])
        outl = shapes.Line(lns, c=lc, alpha=alpha, lw=lw).z(offs)
        rs.append(outl)
        # print('histo outline', outl.z())

    bin_centers_pos = []
    for i in range(bins):
        x = (myedges[i] + myedges[i + 1]) / 2
        if fs[i]:
            bin_centers_pos.append([x, fs[i], 0])

    if marker:  #####################

        pts = shapes.Points(bin_centers_pos)
        if mc is None:
            mc = lc
        if ma is None:
            ma = alpha

        if utils.isSequence(ms):  ### variable point size
            mk = shapes.Marker(marker, s=1)
            msv = np.zeros_like(pts.points())
            msv[:, 0] = ms
            marked = shapes.Glyph(
                pts, glyphObj=mk, c=mc, orientationArray=msv, scaleByVectorSize=True
            )
        else:  ### fixed point size

            if ms is None:
                ms = dx / 100.0

            if utils.isSequence(mc):
                mk = shapes.Marker(marker, s=ms)
                msv = np.zeros_like(pts.points())
                msv[:, 0] = 1
                marked = shapes.Glyph(
                    pts, glyphObj=mk, c=mc, orientationArray=msv, scaleByVectorSize=True
                )
            else:
                mk = shapes.Marker(marker, s=ms)
                marked = shapes.Glyph(pts, glyphObj=mk, c=mc)

        marked.alpha(ma).z(offs * 2)
        # print('marker', marked.z())
        rs.append(marked)

    if errors:  #####################
        for bcp in bin_centers_pos:
            x = bcp[0]
            f = bcp[1]
            err = np.sqrt(f / yscale) * yscale
            el = shapes.Line([x, f-err/2, 0], [x, f+err/2, 0], c=lc, alpha=alpha, lw=lw)
            el.z(offs * 1.9)
            rs.append(el)
        # print('errors', el.z())

    for a in rs:  #####################
        a.cutWithPlane([0, y0lim, 0], [0, 1, 0])
        a.cutWithPlane([0, y1lim, 0], [0, -1, 0])
        a.cutWithPlane([x0lim, 0, 0], [1, 0, 0])
        a.cutWithPlane([x1lim, 0, 0], [-1, 0, 0])
        a.lighting('off').phong()

    if title:  #####################
        if titleColor is None:
            titleColor = bc

        if titleSize is None:
            titleSize = dx / 40.0
        tit = shapes.Text3D(
            title,
            s=titleSize,
            c=titleColor,
            depth=0,
            alpha=alpha,
            pos=((x0lim + x1lim) / 2, y1lim + (y1lim-y0lim) / 80, 0),
            justify="bottom-center",
        )
        tit.pickable(False).z(2.5 * offs)
        rs.append(tit)

    if axes == 1 or axes == True:
        axes = {}
    if isinstance(axes, dict):  #####################
        ndiv = 6
        if "numberOfDivisions" in axes.keys():
            ndiv = axes["numberOfDivisions"]
        tp, ts = utils.makeTicks(y0lim / yscale, y1lim / yscale, ndiv / aspect)
        labs = []
        for i in range(1, len(tp) - 1):
            ynew = utils.linInterpolate(tp[i], [0, 1], [y0lim, y1lim])
            labs.append([ynew, ts[i]])
        if "xtitle" not in axes:  axes["xtitle"] = xtitle
        if "ytitle" not in axes:  axes["ytitle"] = ytitle
        axes["yValuesAndLabels"] = labs
        axes["xrange"] = (x0lim, x1lim)
        axes["yrange"] = (y0lim, y1lim)
        axes["zrange"] = (0, 0)
        axes["c"] = bc
        axs = addons.Axes(**axes)
        axs.name = "axes"
        asse = Plot(rs, axs)
        asse.axes = axs
        asse.SetOrigin(x0lim, y0lim, 0)
    else:
#        settings.xtitle = xtitle
#        settings.ytitle = ytitle
        asse = Plot(rs)

    asse.yscale = yscale
    asse.xlim = xlim
    asse.ylim = ylim
    asse.aspect = aspect
    asse.pad = pad
    asse.title = title
    asse.xtitle = xtitle
    asse.ytitle = ytitle
    asse._x0lim = x0lim
    asse._y0lim = y0lim
    asse._x1lim = x1lim
    asse._y1lim = y1lim
    asse.zmax = offs * 3  # z-order
    asse.bins = edges
    asse.centers = (edges[0:-1] + edges[1:]) / 2
    asse.freqs = fs / yscale
    asse.name = "histogram1D"
    return asse

def _histogram2D(
    xvalues,
    yvalues=None,
    format=None,
    bins=25,
    aspect=1,
    xlim=None,
    ylim=None,
    weights=None,
    cmap="cividis",
    alpha=1,
    title="",
    xtitle="x",
    ytitle="y",
    ztitle="z",
    titleSize=None,
    titleColor=None,
    # logscale=False,
    lw=0,
    scalarbar=True,
    axes=True,
    bc="k",
):
    offs = 0  # z offset

    if format is not None:  # reset to allow meaningful overlap
        xlim = format.xlim
        ylim = format.ylim
        aspect = format.aspect
        bins = format.bins
        axes = 0
        title = ""
        xtitle = ""
        ytitle = ""
        ztitle = ""
        offs = format.zmax

    if yvalues is None:
        # assume [(x1,y1), (x2,y2) ...] format
        yvalues = xvalues[:, 1]
        xvalues = xvalues[:, 0]

    if isinstance(bins, int):
        bins = (bins, bins)
    H, xedges, yedges = np.histogram2d(xvalues, yvalues, weights=weights,
                                       bins=bins, range=(xlim, ylim))

    x0lim, x1lim = np.min(xedges), np.max(xedges)
    y0lim, y1lim = np.min(yedges), np.max(yedges)
    dx, dy = x1lim - x0lim, y1lim - y0lim

    if dx == 0 and dy == 0:  # in case x and y are all constant
        x0lim = x0lim - 1
        x1lim = x1lim + 1
        y0lim = y0lim - 1
        y1lim = y1lim + 1
        dx, dy = 1, 1

    yscale = dx / dy / aspect
    y0lim, y1lim = y0lim * yscale, y1lim * yscale

    acts = []

    #####################
    g = shapes.Grid(
        pos=[(x0lim + x1lim) / 2, (y0lim + y1lim) / 2, 0],
        sx=dx,
        sy=dy * yscale,
        resx=bins[0],
        resy=bins[1],
    )
    g.alpha(alpha).lw(lw).wireframe(0).flat().lighting('off')
    g.cmap(cmap, np.ravel(H.T), on='cells')
    g.SetOrigin(x0lim, y0lim, 0)
    if scalarbar:
        sc = g.addScalarBar3D(c=bc).scalarbar
        scy0, scy1 = sc.ybounds()
        sc_scale = (y1lim-y0lim)/(scy1-scy0)
        sc.scale(sc_scale)
        acts.append(sc)
    g.base = np.array([0, 0, 0])
    g.top = np.array([0, 0, 1])
    acts.append(g)

    if title:  #####################
        if titleColor is None:
            titleColor = bc

        if titleSize is None:
            titleSize = dx / 40.0
        tit = shapes.Text3D(
            title,
            s=titleSize,
            c=titleColor,
            depth=0,
            alpha=alpha,
            pos=((x0lim + x1lim) / 2, y1lim + (y1lim-y0lim) / 80, 0),
            justify="bottom-center",
        )
        tit.pickable(False).z(2.5 * offs)
        acts.append(tit)

    if axes == 1 or axes == True:  #####################
        axes = {"xyGridTransparent": True, "xyAlpha": 0}
    if isinstance(axes, dict):
        ndiv = 6
        if "numberOfDivisions" in axes.keys():
            ndiv = axes["numberOfDivisions"]
        tp, ts = utils.makeTicks(y0lim / yscale, y1lim / yscale, ndiv / aspect)
        labs = []
        for i in range(1, len(tp) - 1):
            ynew = utils.linInterpolate(tp[i], [0, 1], [y0lim, y1lim])
            labs.append([ynew, ts[i]])
        if "xtitle" not in axes:  axes["xtitle"] = xtitle
        if "ytitle" not in axes:  axes["ytitle"] = ytitle
        if "ztitle" not in axes:  axes["ztitle"] = ztitle
        axes["yValuesAndLabels"] = labs
        axes["xrange"] = (x0lim, x1lim)
        axes["yrange"] = (y0lim, y1lim)
        axes["zrange"] = (0, 0) # todo
        axes["c"] = bc
        axs = addons.Axes(**axes)
        axs.name = "axes"
        asse = Plot(acts, axs)
        asse.axes = axs
        asse.SetOrigin(x0lim, y0lim, 0)
    else:
#        settings.xtitle = xtitle
#        settings.ytitle = ytitle
#        settings.ytitle = ztitle
        asse = Plot(acts)

    asse.yscale = yscale
    asse.xlim = xlim
    asse.ylim = ylim
    asse.aspect = aspect
    asse.title = title
    asse.xtitle = xtitle
    asse.ytitle = ytitle
    asse._x0lim = x0lim
    asse._y0lim = y0lim
    asse._x1lim = x1lim
    asse._y1lim = y1lim
    asse.freqs = H
    asse.bins = (xedges, yedges)
    asse.zmax = offs * 3  # z-order
    asse.name = "histogram2D"
    return asse


def _histogramHexBin(
    xvalues,
    yvalues,
    xtitle="",
    ytitle="",
    ztitle="",
    bins=12,
    vrange=None,
    norm=1,
    fill=True,
    c=None,
    cmap="terrain_r",
    alpha=1,
):
#    if xtitle:
#        settings.xtitle = xtitle
#    if ytitle:
#        settings.ytitle = ytitle
#    if ztitle:
#        settings.ztitle = ztitle

    xmin, xmax = np.min(xvalues), np.max(xvalues)
    ymin, ymax = np.min(yvalues), np.max(yvalues)
    dx, dy = xmax - xmin, ymax - ymin

    if utils.isSequence(bins):
        n,m = bins
    else:
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

    # values = list(zip(xvalues, yvalues))
    values = np.stack((xvalues, yvalues), axis=1)
    zs = [[0.0]] * len(values)
    values = np.append(values, zs, axis=1)

    pointsPolydata.GetPoints().SetData(utils.numpy2vtk(values, dtype=float))
    cloud = Mesh(pointsPolydata)

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
            ids = cloud.closestPoint(q, radius=r, returnCellId=True)
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
            h = Mesh(tf.GetOutput(), c=col, alpha=alpha).flat()
            h.lighting('plastic')
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
    asse.base = np.array([0, 0, 0])
    asse.top = np.array([0, 0, 1])
    asse.name = "histogramHexBin"
    return asse


def _histogramPolar(
    values,
    weights=None,
    title="",
    tsize=0.1,
    bins=16,
    r1=0.25,
    r2=1,
    phigap=0.5,
    rgap=0.05,
    lpos=1,
    lsize=0.04,
    c='grey',
    bc="k",
    alpha=1,
    cmap=None,
    deg=False,
    vmin=None,
    vmax=None,
    labels=(),
    showDisc=True,
    nrays=8,
    showLines=True,
    showAngles=True,
    showErrors=False,
):
    k = 180 / np.pi
    if deg:
        values = np.array(values) / k
    else:
        values = np.array(values)

    vals = []
    for v in values:  # normalize range
        t = np.arctan2(np.sin(v), np.cos(v))
        if t < 0:
            t += 2 * np.pi
        vals.append(t+0.00001)

    histodata, edges = np.histogram(vals, weights=weights,
                                    bins=bins, range=(0, 2*np.pi))

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
        back = shapes.Disc(r1=r2e, r2=r2e * 1.01, c=bc, res=(1,360))
        back.z(-0.01)

    slices = []
    lines = []
    angles = []
    errbars = []

    for i, t in enumerate(thetas):
        r = histodata[i] / vmax * r2
        d = shapes.Disc((0, 0, 0), r1, r1+r, res=(1,360))
        delta = np.pi/bins - np.pi/2 - phigap/k
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
        d.alpha(alpha).lighting('off')
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

    labs=[]
    rays = []
    if showDisc:
        outerdisc = shapes.Disc(r1=r2e, r2=r2e * 1.01, c=bc, res=(1,360))
        outerdisc.z(-0.01)
        innerdisc = shapes.Disc(r1=r2e/2, r2=r2e/2 * 1.005, c=bc, res=(1, 360))
        innerdisc.z(-0.01)
        rays.append(outerdisc)
        rays.append(innerdisc)

        rgap = 0.05
        for t in np.linspace(0, 2 * np.pi, num=nrays, endpoint=False):
            ct, st = np.cos(t), np.sin(t)
            if showLines:
                l = shapes.Line((0, 0, -0.01), (r2e * ct * 1.03, r2e * st * 1.03, -0.01))
                rays.append(l)
                ct2, st2 = np.cos(t+np.pi/nrays), np.sin(t+np.pi/nrays)
                lm = shapes.DashedLine((0, 0, -0.01),
                                       (r2e * ct2, r2e * st2, -0.01),
                                       spacing=0.25)
                rays.append(lm)
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
                a = shapes.Text3D(int(t * k), pos=(0, 0, 0), s=lsize, depth=0, justify=ju)
                a.pos(r2e * ct * (1 + rgap), r2e * st * (1 + rgap), -0.01)
                angles.append(a)

    ti = None
    if title:
        ti = shapes.Text3D(title, (0, 0, 0), s=tsize, depth=0, justify="top-center")
        ti.pos(0, -r2e * 1.15, 0.01)

    for i,t in enumerate(thetas):
        if i < len(labels):
            lab = shapes.Text3D(labels[i], (0, 0, 0), #font="VTK",
                              s=lsize, depth=0, justify="center")
            lab.pos(r2e *np.cos(t) * (1 + rgap) * lpos / 2,
                    r2e *np.sin(t) * (1 + rgap) * lpos / 2, 0.01)
            labs.append(lab)

    mrg = merge(lines, angles, rays, ti, labs)
    if mrg:
        mrg.color(bc).lighting('off')

    rh = Plot(slices + errbars + [mrg])
    rh.freqs = histodata
    rh.bins = edges
    rh.base = np.array([0, 0, 0])
    rh.top = np.array([0, 0, 1])
    rh.name = "histogramPolar"
    return rh


def _histogramSpheric(
    thetavalues, phivalues, rmax=1.2, res=8, cmap="rainbow", lw=0.1, scalarbar=True,
):

    x, y, z = utils.spher2cart(np.ones_like(thetavalues) * 1.1, thetavalues, phivalues)
    ptsvals = np.c_[x, y, z]

    sg = shapes.Sphere(res=res, quads=True).shrink(0.999).computeNormals().lw(0.1)
    sgfaces = sg.faces()
    sgpts = sg.points()
    #    sgpts = np.vstack((sgpts, [0,0,0]))
    #    idx = sgpts.shape[0]-1
    #    newfaces = []
    #    for fc in sgfaces:
    #        f1,f2,f3,f4 = fc
    #        newfaces.append([idx,f1,f2, idx])
    #        newfaces.append([idx,f2,f3, idx])
    #        newfaces.append([idx,f3,f4, idx])
    #        newfaces.append([idx,f4,f1, idx])
    newsg = sg  # Mesh((sgpts, sgfaces)).computeNormals().phong()
    newsgpts = newsg.points()

    cntrs = sg.cellCenters()
    counts = np.zeros(len(cntrs))
    for p in ptsvals:
        cell = sg.closestPoint(p, returnCellId=True)
        counts[cell] += 1
    acounts = np.array(counts)
    counts *= (rmax - 1) / np.max(counts)

    for cell, cn in enumerate(counts):
        if not cn:
            continue
        fs = sgfaces[cell]
        pts = sgpts[fs]
        _, t1, p1 = utils.cart2spher(pts[:, 0], pts[:, 1], pts[:, 2])
        x, y, z = utils.spher2cart(1 + cn, t1, p1)
        newsgpts[fs] = np.c_[x, y, z]

    newsg.points(newsgpts)
    newsg.cmap(cmap, acounts, on='cells')

    if scalarbar:
        newsg.addScalarBar()
    newsg.name = "histogramSpheric"
    return newsg


def donut(
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
    showDisc=False,
):
    """
    Donut plot or pie chart.

    :param str title: plot title
    :param float tsize: title size
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

    |donut| |donut.py|_
    """
    fractions = np.array(fractions)
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
    labs = ()
    if len(labels):
        angles = np.concatenate([[0], angles])
        labs = [""] * 360
        for i in range(len(labels)):
            a = (angles[i + 1] + angles[i]) / 2
            j = int(a / np.pi * 180)
            labs[j] = labels[i]

    data = np.linspace(0, 2 * np.pi, 360, endpoint=False) + 0.005
    dn = _histogramPolar(
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
        showDisc=showDisc,
        showLines=0,
        showAngles=0,
        showErrors=0,
    )
    dn.name = "donut"
    return dn


def quiver(
    points,
    vectors,
    c="k",
    alpha=1,
    shaftLength=0.8,
    shaftWidth=0.05,
    headLength=0.25,
    headWidth=0.2,
    fill=True,
):
    """
    Quiver Plot, display `vectors` at `points` locations.

    Color can be specified as a colormap which maps the size of the arrows.

    :param float shaftLength: fractional shaft length
    :param float shaftWidth: fractional shaft width
    :param float headLength: fractional head length
    :param float headWidth: fractional head width
    :param bool fill: if False only generate the outline

    |quiver| |quiver.py|_
    """
    if isinstance(points, vedo.Points):
        points = points.points()
    else:
        points = np.array(points)
    vectors = np.array(vectors) / 2

    spts = points - vectors
    epts = points + vectors

    arrs2d = shapes.Arrows2D(
        spts,
        epts,
        c=c,
        shaftLength=shaftLength,
        shaftWidth=shaftWidth,
        headLength=headLength,
        headWidth=headWidth,
        fill=fill,
        alpha=alpha,
    )
    arrs2d.pickable(False)
    arrs2d.name = "quiver"
    return arrs2d


def violin(
    values,
    bins=10,
    vlim=None,
    x=0,
    width=3,
    spline=True,
    fill=True,
    c="violet",
    alpha=1,
    outline=True,
    centerline=True,
    lc="darkorchid",
    lw=3,
):
    """
    Violin style histogram.

    :param int bins: number of bins
    :param list vlim: input value limits. Crop values outside range.
    :param list x: x-position of the violin axis
    :param float width: width factor of the normalized distribution
    :param bool spline: spline points
    :param bool fill: fill violin with solid color
    :param bool outline: add the distribution outline
    :param bool centerline: add the vertical centerline at x
    :param lc: line color

    |histo_violin| |histo_violin.py|_
    """
    fs, edges = np.histogram(values, bins=bins, range=vlim)
    mine, maxe = np.min(edges), np.max(edges)
    fs = fs.astype(float) / len(values) * width

    rs = []

    if spline:
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
            rb = shapes.Ribbon(spl, spr, c=c, alpha=alpha).lighting('off')
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
                r.color(c).alpha(alpha).lighting('off')
                rs.append(r)

    if centerline:
        cl = shapes.Line([0, mine, 0.01], [0, maxe, 0.01], c=lc, alpha=alpha, lw=2).x(x)
        rs.append(cl)

    asse = Assembly(rs)
    asse.base = np.array([0, 0, 0])
    asse.top = np.array([0, 1, 0])
    asse.name = "violin"
    return asse


def whisker(data,
            s=0.25,
            c='k',
            lw=2,
            bc='blue',
            alpha=0.25,
            r=5,
            jitter=True,
            horizontal=False,
):
    """
    Generate a "whisker" bar from a 1-dimensional dataset.

    :param float s: size of the box
    :param c: color of the lines
    :param float lw: line width
    :param bc: color of the box
    :param float alpha: transparency of the box
    :param float r: point radius in pixels (use value 0 to disable)
    :param bool jitter: add some randomness to points to avoid overlap
    :param bool horizontal: set horizontal layout

    |whiskers| |whiskers.py|_
    """
    xvals = np.zeros_like(np.array(data))
    if jitter:
        xjit = np.random.randn(len(xvals))*s/9
        xjit = np.clip(xjit, -s/2.1, s/2.1)
        xvals += xjit

    dmean = np.mean(data)
    dq05 = np.quantile(data, 0.05)
    dq25 = np.quantile(data, 0.25)
    dq75 = np.quantile(data, 0.75)
    dq95 = np.quantile(data, 0.95)

    pts = None
    if r: pts = shapes.Points([xvals, data], c=c, r=r)

    rec = shapes.Rectangle([-s/2, dq25],[s/2, dq75], c=bc, alpha=alpha)
    rec.GetProperty().LightingOff()
    rl = shapes.Line([[-s/2, dq25],[s/2, dq25],[s/2, dq75],[-s/2, dq75]], closed=True)
    l1 = shapes.Line([0,dq05,0], [0,dq25,0], c=c, lw=lw)
    l2 = shapes.Line([0,dq75,0], [0,dq95,0], c=c, lw=lw)
    lm = shapes.Line([-s/2, dmean], [s/2, dmean])
    lns = merge(l1, l2, lm, rl)
    asse = Assembly([lns, rec, pts])
    if horizontal:
        asse.rotateZ(-90)
    asse.name = "Whisker"
    asse.info['mean'] = dmean
    asse.info['quantile_05'] = dq05
    asse.info['quantile_25'] = dq25
    asse.info['quantile_75'] = dq75
    asse.info['quantile_95'] = dq95
    return asse


def streamplot(X, Y, U, V, direction="both",
               maxPropagation=None, mode=1, lw=0.001, c=None, probes=()):
    """
    Generate a streamline plot of a vectorial field (U,V) defined at positions (X,Y).
    Returns a ``Mesh`` object.

    :param str direction: either "forward", "backward" or "both"
    :param float maxPropagation: maximum physical length of the streamline
    :param float lw: line width in absolute units
    :param int mode: vary line width

        - 0 - do not vary line width
        - 1 - vary line width by first vector component
        - 2 - vary line width vector magnitude
        - 3 - vary line width by absolute value of first vector component

    |plot_stream| |plot_stream.py|_
    """
    n = len(X)
    m = len(Y[0])
    if n != m:
        print("Limitation in streamplot(): only square grids are allowed.", n, m)
        raise RuntimeError()

    xmin, xmax = X[0][0], X[-1][-1]
    ymin, ymax = Y[0][0], Y[-1][-1]

    field = np.sqrt(U * U + V * V)

    vol = vedo.Volume(field, dims=(n, n, 1))

    uf = np.ravel(U, order="F")
    vf = np.ravel(V, order="F")
    vects = np.c_[uf, vf, np.zeros_like(uf)]
    vol.addPointArray(vects, "vects")

    if len(probes) == 0:
        probe = shapes.Grid(pos=((n-1)/2,(n-1)/2,0), sx=n-1, sy=n-1, resx=n-1, resy=n-1)
    else:
        if isinstance(probes, vedo.Points):
            probes = probes.points()
        else:
            probes = np.array(probes)
            if len(probes[0]) == 2:
                probes = np.c_[probes[:, 0], probes[:, 1], np.zeros(len(probes))]
        sv = [(n - 1) / (xmax - xmin), (n - 1) / (ymax - ymin), 1]
        probes = probes - [xmin, ymin, 0]
        probes = np.multiply(probes, sv)
        probe = vedo.Points(probes)

    stream = vedo.base.streamLines( vol.imagedata(),
                                    probe,
                                    tubes={"radius": lw, "varyRadius": mode,},
                                    lw=lw,
                                    maxPropagation=maxPropagation,
                                    direction=direction,
    )
    if c is not None:
        stream.color(c)
    else:
        stream.addScalarBar()
    stream.lighting('off')

    stream.scale([1 / (n - 1) * (xmax - xmin), 1 / (n - 1) * (ymax - ymin), 1])
    stream.shift(xmin, ymin)
    return stream


def matrix(M,
           title='Matrix',
           xtitle='',
           ytitle='',
           xlabels=[],
           ylabels=[],
           xrotation=0,
           cmap='Reds',
           vmin=None,
           vmax=None,
           precision=2,
           font='Theemim',
           scale=0,
           scalarbar=True,
           lc='white',
           lw=0,
           c='black',
           alpha=1,
    ):
    """
    Generate a matrix, or a 2D color-coded plot with bin labels.

    Returns an ``Assembly`` object.

    Parameters
    ----------
    M : list or numpy array
        the input array to visualize.
    title : str, optional
        title of the plot. The default is 'Matrix'.
    xtitle : str, optional
        title of the horizontal colmuns. The default is ''.
    ytitle : str, optional
        title of the vertical rows. The default is ''.
    xlabels : list, optional
        individual string labels for each column. Must be of length m. The default is [].
    ylabels : list, optional
        individual string labels for each row. Must be of length n. The default is [].
    xrotation : float, optional
        rotation of the horizontal labels. The default is 0.
    cmap : str, optional
        color map name. The default is 'Reds'.
    vmin : float, optional
        minimum value of the colormap range. The default is None.
    vmax : float, optional
        maximum value of the colormap range. The default is None.
    precision : int, optional
        number of digits for the matrix entries or bins. The default is 2.
    font : str, optional
        font name. The default is ''.
    scale : float, optional
        size of the numeric entries or bin values. The default is 0.
    scalarbar : bool, optional
        add a scalar bar to the right of the plot. The default is True.
    lc : str, optional
        color of the line separating the bins. The default is 'white'.
    lw : float, optional
        Width of the line separating the bins. The default is 0.
    c : str, optional
        text color. The default is 'k'.
    alpha : float, optional
        plot transparency. The default is 1.
    """
    M = np.asarray(M)
    n,m = M.shape
    gr = shapes.Grid(resx=m, resy=n, sx=m/(m+n)*2, sy=n/(m+n)*2, c=c, alpha=alpha)
    gr.wireframe(False).lc(lc).lw(lw)

    matr = np.flip( np.flip(M), axis=1).ravel(order='C')
    gr.cmap(cmap, matr, on='cells', vmin=vmin, vmax=vmax)
    sbar=None
    if scalarbar:
        gr.addScalarBar3D(titleFont=font, labelFont=font)
        sbar = gr.scalarbar
    labs=None
    if scale !=0:
        labs = gr.labels(cells=True, scale=scale/max(m,n),
                         precision=precision, font=font, justify='center', c=c)
        labs.z(0.001)
    t = None
    if title:
        if title == 'Matrix':
            title += ' '+str(n)+'x'+str(m)
        t = shapes.Text3D(title, font=font, s=0.04,
                          justify='bottom-center', c=c)
        t.shift(0, n/(m+n)*1.05)

    xlabs=None
    if len(xlabels)==m:
        xlabs=[]
        jus = 'top-center'
        if xrotation>44:
            jus = 'right-center'
        for i in range(m):
            xl = shapes.Text3D(xlabels[i], font=font, s=0.02,
                               justify=jus, c=c).rotateZ(xrotation)
            xl.shift((2*i-m+1)/(m+n), -n/(m+n)*1.05)
            xlabs.append(xl)

    ylabs=None
    if len(ylabels)==n:
        ylabs=[]
        for i in range(n):
            yl = shapes.Text3D(ylabels[i], font=font, s=.02,
                               justify='right-center', c=c)
            yl.shift(-m/(m+n)*1.05, (2*i-n+1)/(m+n))
            ylabs.append(yl)

    xt=None
    if xtitle:
        xt = shapes.Text3D(xtitle, font=font, s=0.035,
                           justify='top-center', c=c)
        xt.shift(0, -n/(m+n)*1.05)
        if xlabs is not None:
            y0,y1 = xlabs[0].ybounds()
            xt.shift(0, -(y1-y0)-0.55/(m+n))
    yt=None
    if ytitle:
        yt = shapes.Text3D(ytitle, font=font, s=0.035,
                           justify='bottom-center', c=c).rotateZ(90)
        yt.shift(-m/(m+n)*1.05, 0)
        if ylabs is not None:
            x0,x1 = ylabs[0].xbounds()
            yt.shift(-(x1-x0)-0.55/(m+n),0)
    asse = Assembly(gr, sbar, labs, t, xt, yt, xlabs, ylabs)
    asse.name = "Matrix"
    return asse


def cornerPlot(points, pos=1, s=0.2, title="", c="b", bg="k", lines=True, dots=True):
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
        points = np.stack((points[0], points[1]), axis=1)

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
    plot.SetPlotPoints(dots)

    if not lines:
        plot.PlotLinesOff()

    if isinstance(pos, str):
        spos = 2
        if "top" in pos:
            if "left" in pos: spos=1
            elif "right" in pos: spos=2
        elif "bottom" in pos:
            if "left" in pos: spos=3
            elif "right" in pos: spos=4
        pos = spos
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

    Use *vrange* to restrict the range of the histogram.

    :param int nmax: limit the sampling to this max nr of entries

    Use `pos` to assign its position:
        - 1, topleft,
        - 2, topright,
        - 3, bottomleft,
        - 4, bottomright,
        - (x, y), as fraction of the rendering window
    """
    if hasattr(values, '_data'):
        values = utils.vtk2numpy(values._data.GetPointData().GetScalars())

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

    plot = cornerPlot(pts, pos, s, title, c, bg, lines, dots)
    plot.SetNumberOfYLabels(2)
    plot.SetNumberOfXLabels(3)
    tprop = vtk.vtkTextProperty()
    tprop.SetColor(colors.getColor(bg))
    tprop.SetFontFamily(vtk.VTK_FONT_FILE)
    tprop.SetFontFile(utils.getFontPath(settings.defaultFont))
    tprop.SetOpacity(alpha)
    plot.SetAxisTitleTextProperty(tprop)
    plot.GetProperty().SetOpacity(alpha)
    plot.GetXAxisActor2D().SetLabelTextProperty(tprop)
    plot.GetXAxisActor2D().SetTitleTextProperty(tprop)
    plot.GetXAxisActor2D().SetFontFactor(0.55)
    plot.GetYAxisActor2D().SetLabelFactor(0.0)
    plot.GetYAxisActor2D().LabelVisibilityOff()
    return plot


class DirectedGraph(Assembly):
    """A graph consists of a collection of nodes (without postional information)
    and a collection of edges connecting pairs of nodes.
    The task is to determine the node positions only based on their connections.

    This class is derived from class ``Assembly``, and it assembles 4 Mesh objects
    representing the graph, the node labels, edge labels and edge arrows.

    :param c: color of the Graph
    :param int n: number of the initial set of nodes
    :param int,str layout: layout in ['2d', 'fast2d', 'clustering2d', 'circular',
                                      'circular3d', 'cone', 'force', 'tree']

    Each of these layouts has diferent available options.

    Options for layouts '2d', 'fast2d' and 'clustering2d':

        :param int seed: seed of the random number generator used to jitter point positions
        :param float restDistance: manually set the resting distance
        :param int maxNumberOfIterations: the maximum number of iterations to be used
        :param float zrange: expand 2d graph along z axis.

    Options for layouts 'circular', and 'circular3d':

        :param float radius: set the radius of the circles.
        :param float height: set the vertical (local z) distance between the circles
        :param float zrange: expand 2d graph along z axis.

    Options for layout 'cone':

        :param float compactness: ratio between the average width of a cone in the tree,
            and the height of the cone. The default setting is 0.75.

        :param bool compression: put children closer together, possibly allowing sub-trees to overlap.
            This is useful if the tree is actually the spanning tree of a graph.

        :param float spacing: space between layers of the tree

    Options for layout 'force':

        :param int seed: seed the random number generator used to jitter point positions
        :param list bounds: set the region in space in which to place the final graph
        :param int maxNumberOfIterations: the maximum number of iterations to be used
        :param bool threeDimensional: allow optimization in the 3rd dimension too
        :param bool randomInitialPoints: use random positions within the graph bounds as initial points

    Example:

        |lineage_graph| |lineage_graph.py|_

        |graph_network| |graph_network.py|_
    """
    def __init__(self, **kargs):
        vedo.base.BaseActor.__init__(self)

        self.nodes = []
        self.edges = []

        self._nodeLabels = []  # holds strings
        self._edgeLabels = []
        self.edgeOrientations = []
        self.edgeGlyphPosition = 0.6

        self.zrange = 0.0

        self.rotX = 0
        self.rotY = 0
        self.rotZ = 0

        self.arrowScale = 0.15
        self.nodeLabelScale = None
        self.nodeLabelJustify = "bottom-left"

        self.edgeLabelScale = None

        self.mdg = vtk.vtkMutableDirectedGraph()

        n = kargs.pop('n', 0)
        for i in range(n): self.addNode()

        self._c = kargs.pop('c', (0.3,0.3,0.3))

        self.gl = vtk.vtkGraphLayout()

        self.font = kargs.pop('font', '')

        s = kargs.pop('layout', '2d')
        if isinstance(s, int):
            ss = ['2d', 'fast2d', 'clustering2d', 'circular', 'circular3d',
                  'cone', 'force', 'tree']
            s = ss[s]
        self.layout = s

        if '2d' in s:
            if 'clustering' in s:
                self.strategy = vtk.vtkClustering2DLayoutStrategy()
            elif 'fast' in s:
                self.strategy = vtk.vtkFast2DLayoutStrategy()
            else:
                self.strategy = vtk.vtkSimple2DLayoutStrategy()
            self.rotX = 180
            opt = kargs.pop('restDistance', None)
            if opt is not None: self.strategy.SetRestDistance(opt)
            opt = kargs.pop('seed', None)
            if opt is not None: self.strategy.SetRandomSeed(opt)
            opt = kargs.pop('maxNumberOfIterations', None)
            if opt is not None: self.strategy.SetMaxNumberOfIterations(opt)
            self.zrange = kargs.pop('zrange', 0)

        elif 'circ' in s:
            if '3d' in s:
                self.strategy = vtk.vtkSimple3DCirclesStrategy()
                self.strategy.SetDirection(0,0,-1)
                self.strategy.SetAutoHeight(True)
                self.strategy.SetMethod(1)
                self.rotX = -90
                opt = kargs.pop('radius', None) # float
                if opt is not None:
                    self.strategy.SetMethod(0)
                    self.strategy.SetRadius(opt) # float
                opt = kargs.pop('height', None)
                if opt is not None:
                    self.strategy.SetAutoHeight(False)
                    self.strategy.SetHeight(opt) # float
            else:
                self.strategy = vtk.vtkCircularLayoutStrategy()
                self.zrange = kargs.pop('zrange', 0)

        elif 'cone' in s:
            self.strategy = vtk.vtkConeLayoutStrategy()
            self.rotX = 180
            opt = kargs.pop('compactness', None)
            if opt is not None: self.strategy.SetCompactness(opt)
            opt = kargs.pop('compression', None)
            if opt is not None: self.strategy.SetCompression(opt)
            opt = kargs.pop('spacing', None)
            if opt is not None: self.strategy.SetSpacing(opt)

        elif 'force' in s:
            self.strategy = vtk.vtkForceDirectedLayoutStrategy()
            opt = kargs.pop('seed', None)
            if opt is not None: self.strategy.SetRandomSeed(opt)
            opt = kargs.pop('bounds', None)
            if opt is not None:
                self.strategy.SetAutomaticBoundsComputation(False)
                self.strategy.SetGraphBounds(opt) # list
            opt = kargs.pop('maxNumberOfIterations', None)
            if opt is not None: self.strategy.SetMaxNumberOfIterations(opt) # int
            opt = kargs.pop('threeDimensional', True)
            if opt is not None: self.strategy.SetThreeDimensionalLayout(opt) # bool
            opt = kargs.pop('randomInitialPoints', None)
            if opt is not None: self.strategy.SetRandomInitialPoints(opt) # bool

        elif 'tree' in s:
            self.strategy = vtk.vtkSpanTreeLayoutStrategy()
            self.rotX = 180

        else:
            colors.printc("Cannot understand layout:", s, c='r')
            colors.printc("Available layouts:", c='r')
            colors.printc("[2d,fast2d,clustering2d,circular,circular3d,cone,force,tree]", c='r')
            raise RuntimeError()

        self.gl.SetLayoutStrategy(self.strategy)

        if len(kargs):
            colors.printc("Cannot understand options:", kargs, c='r')
        return


    def addNode(self, label="id"):
        """Add a new node to the Graph."""
        v = self.mdg.AddVertex() # vtk calls it vertex..
        self.nodes.append(v)
        if label == 'id': label=int(v)
        self._nodeLabels.append(str(label))
        return v

    def addEdge(self, v1, v2, label=""):
        """Add a new edge between to nodes.
        An extra node is created automatically if needed."""
        nv = len(self.nodes)
        if v1>=nv:
            for i in range(nv, v1+1):
                self.addNode()
        nv = len(self.nodes)
        if v2>=nv:
            for i in range(nv, v2+1):
                self.addNode()
        e = self.mdg.AddEdge(v1,v2)
        self.edges.append(e)
        self._edgeLabels.append(str(label))
        return e

    def addChild(self, v, nodeLabel="id", edgeLabel=""):
        """Add a new edge to a new node as its child.
        The extra node is created automatically if needed."""
        nv = len(self.nodes)
        if v>=nv:
            for i in range(nv, v+1):
                self.addNode()
        child = self.mdg.AddChild(v)
        self.edges.append((v,child))
        self.nodes.append(child)
        if nodeLabel == 'id': nodeLabel=int(child)
        self._nodeLabels.append(str(nodeLabel))
        self._edgeLabels.append(str(edgeLabel))
        return child

    def build(self):
        """
        Build the DirectedGraph(Assembly).
        Accessory objects are also created for labels and arrows.
        """
        self.gl.SetZRange(self.zrange)
        self.gl.SetInputData(self.mdg)
        self.gl.Update()

        graphToPolyData = vtk.vtkGraphToPolyData()
        graphToPolyData.EdgeGlyphOutputOn()
        graphToPolyData.SetEdgeGlyphPosition(self.edgeGlyphPosition)
        graphToPolyData.SetInputData(self.gl.GetOutput())
        graphToPolyData.Update()

        dgraph = Mesh(graphToPolyData.GetOutput(0))
        # dgraph.clean() # WRONG!!! dont uncomment
        dgraph.flat().color(self._c).lw(2)
        dgraph.name = "DirectedGraph"

        diagsz = self.diagonalSize()/1.42
        if not diagsz:
            return None

        dgraph.SetScale(1/diagsz)
        if self.rotX:
            dgraph.rotateX(self.rotX)
        if self.rotY:
            dgraph.rotateY(self.rotY)
        if self.rotZ:
            dgraph.rotateZ(self.rotZ)

        vecs = graphToPolyData.GetOutput(1).GetPointData().GetVectors()
        self.edgeOrientations = utils.vtk2numpy(vecs)

        # Use Glyph3D to repeat the glyph on all edges.
        arrows=None
        if self.arrowScale:
            arrowSource = vtk.vtkGlyphSource2D()
            arrowSource.SetGlyphTypeToEdgeArrow()
            arrowSource.SetScale(self.arrowScale)
            arrowSource.Update()
            arrowGlyph = vtk.vtkGlyph3D()
            arrowGlyph.SetInputData(0, graphToPolyData.GetOutput(1))
            arrowGlyph.SetInputData(1, arrowSource.GetOutput())
            arrowGlyph.Update()
            arrows = Mesh(arrowGlyph.GetOutput())
            arrows.SetScale(1/diagsz)
            arrows.lighting('off').color(self._c)
            if self.rotX:
                arrows.rotateX(self.rotX)
            if self.rotY:
                arrows.rotateY(self.rotY)
            if self.rotZ:
                arrows.rotateZ(self.rotZ)
            arrows.name = "DirectedGraphArrows"

        nodeLabels = dgraph.labels(self._nodeLabels,
                                    scale=self.nodeLabelScale,
                                    precision=0,
                                    font=self.font,
                                    justify=self.nodeLabelJustify,
                                    )
        nodeLabels.color(self._c).pickable(True)
        nodeLabels.name = "DirectedGraphNodeLabels"

        edgeLabels = dgraph.labels(self._edgeLabels,
                                    cells=True,
                                    scale=self.edgeLabelScale,
                                    precision=0,
                                    font=self.font,
                                    )
        edgeLabels.color(self._c).pickable(True)
        edgeLabels.name = "DirectedGraphEdgeLabels"

        Assembly.__init__(self, [dgraph,
                                 nodeLabels,
                                 edgeLabels,
                                 arrows])
        self.name = "DirectedGraphAssembly"
        return self
