#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Function- and field-based plotting helpers."""

from typing import Union
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

from .figure import Figure
from .charts import Histogram1D, Histogram2D, PlotBars, PlotXY

def plot(*args, **kwargs):
    """
    Draw a 2D line plot, or scatter plot, of variable x vs variable y.
    Input format can be either `[allx], [allx, ally] or [(x1,y1), (x2,y2), ...]`

    Use `like=...` if you want to use the same format of a previously
    created Figure (useful when superimposing Figures) to make sure
    they are compatible and comparable. If they are not compatible
    you will receive an error message.

    Arguments:
        xerrors : (bool)
            show error bars associated to each point in x
        yerrors : (bool)
            show error bars associated to each point in y
        lw : (int)
            width of the line connecting points in pixel units.
            Set it to 0 to remove the line.
        lc : (str)
            line color
        la : (float)
            line "alpha", opacity of the line
        dashed : (bool)
            draw a dashed line instead of a continuous line
        splined : (bool)
            spline the line joining the point as a countinous curve
        elw : (int)
            width of error bar lines in units of pixels
        ec : (color)
            color of error bar, by default the same as marker color
        error_band : (bool)
            represent errors on y as a filled error band.
            Use `ec` keyword to modify its color.
        marker : (str, int)
            use a marker for the data points
        ms : (float)
            marker size
        mc : (color)
            color of the marker
        ma : (float)
            opacity of the marker
        xlim : (list)
            set limits to the range for the x variable
        ylim : (list)
            set limits to the range for the y variable
        aspect : (float)
            Desired aspect ratio.
            If None, it is automatically calculated to get a reasonable aspect ratio.
            Scaling factor is saved in Figure.yscale
        padding : (float, list)
            keep a padding space from the axes (as a fraction of the axis size).
            This can be a list of four numbers.
        title : (str)
            title to appear on the top of the frame, like a header.
        xtitle : (str)
            title for the x-axis, can also be set using `axes=dict(xtitle="my x axis")`
        ytitle : (str)
            title for the y-axis, can also be set using `axes=dict(ytitle="my y axis")`
        ac : (str)
            axes color
        grid : (bool)
            show the background grid for the axes, can also be set using `axes=dict(xygrid=True)`
        ztolerance : (float)
            a tolerance factor to superimpose objects (along the z-axis).

    Example:
        ```python
        import numpy as np
        from vedo.pyplot import plot
        from vedo import settings
        settings.remember_last_figure_format = True #############
        x = np.linspace(0, 6.28, num=50)
        fig = plot(np.sin(x), 'r-')
        fig+= plot(np.cos(x), 'bo-') # no need to specify like=...
        fig.show().close()
        ```
        <img src="https://user-images.githubusercontent.com/32848391/74363882-c3638300-4dcb-11ea-8a78-eb492ad9711f.png" width="600">

    Examples:
        - [plot_errbars.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_errbars.py)
        - [plot_errband.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_errband.py)
        - [plot_pip.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_pip.py)

            ![](https://vedo.embl.es/images/pyplot/plot_pip.png)

        - [scatter1.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/scatter1.py)
        - [scatter2.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/scatter2.py)



    -------------------------------------------------------------------------
    .. note:: mode="bar"

    Creates a `PlotBars(Figure)` object.

    Input must be in format `[counts, labels, colors, edges]`.
    Either or both `edges` and `colors` are optional and can be omitted.

    Arguments:
        errors : (bool)
            show error bars
        logscale : (bool)
            use logscale on y-axis
        fill : (bool)
            fill bars with solid color `c`
        gap : (float)
            leave a small space btw bars
        radius : (float)
            border radius of the top of the histogram bar. Default value is 0.1.
        texture : (str)
            url or path to an image to be used as texture for the bin
        outline : (bool)
            show outline of the bins
        xtitle : (str)
            title for the x-axis, can also be set using `axes=dict(xtitle="my x axis")`
        ytitle : (str)
            title for the y-axis, can also be set using `axes=dict(ytitle="my y axis")`
        ac : (str)
            axes color
        padding : (float, list)
            keep a padding space from the axes (as a fraction of the axis size).
            This can be a list of four numbers.
        aspect : (float)
            the desired aspect ratio of the figure. Default is 4/3.
        grid : (bool)
            show the background grid for the axes, can also be set using `axes=dict(xygrid=True)`

    Examples:
        - [histo_1d_a.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_1d_a.py)
        - [histo_1d_b.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_1d_b.py)
        - [histo_1d_c.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_1d_c.py)
        - [histo_1d_d.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_1d_d.py)

        ![](https://vedo.embl.es/images/pyplot/histo_1D.png)


    ----------------------------------------------------------------------
    .. note:: 2D functions

    If input is an external function or a formula, draw the surface
    representing the function `f(x,y)`.

    Arguments:
        x : (float)
            x range of values
        y : (float)
            y range of values
        zlimits : (float)
            limit the z range of the independent variable
        zlevels : (int)
            will draw the specified number of z-levels contour lines
        show_nan : (bool)
            show where the function does not exist as red points
        bins : (list)
            number of bins in x and y

    Examples:
        - [plot_fxy1.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_fxy1.py)

            ![](https://vedo.embl.es/images/pyplot/plot_fxy.png)

        - [plot_fxy2.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_fxy2.py)


    --------------------------------------------------------------------
    .. note:: mode="complex"

    If `mode='complex'` draw the real value of the function and color map the imaginary part.

    Arguments:
        cmap : (str)
            diverging color map (white means `imag(z)=0`)
        lw : (float)
            line with of the binning
        bins : (list)
            binning in x and y

    Examples:
        - [plot_fxy.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_fxy.py)

            ![](https://user-images.githubusercontent.com/32848391/73392962-1709a300-42db-11ea-9278-30c9d6e5eeaa.png)


    --------------------------------------------------------------------
    .. note:: mode="polar"

    If `mode='polar'` input arrays are interpreted as a list of polar angles and radii.
    Build a polar (radar) plot by joining the set of points in polar coordinates.

    Arguments:
        title : (str)
            plot title
        tsize : (float)
            title size
        bins : (int)
            number of bins in phi
        r1 : (float)
            inner radius
        r2 : (float)
            outer radius
        lsize : (float)
            label size
        c : (color)
            color of the line
        ac : (color)
            color of the frame and labels
        alpha : (float)
            opacity of the frame
        ps : (int)
            point size in pixels, if ps=0 no point is drawn
        lw : (int)
            line width in pixels, if lw=0 no line is drawn
        deg : (bool)
            input array is in degrees
        vmax : (float)
            normalize radius to this maximum value
        fill : (bool)
            fill convex area with solid color
        splined : (bool)
            interpolate the set of input points
        show_disc : (bool)
            draw the outer ring axis
        nrays : (int)
            draw this number of axis rays (continuous and dashed)
        show_lines : (bool)
            draw lines to the origin
        show_angles : (bool)
            draw angle values

    Examples:
        - [histo_polar.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_polar.py)

            ![](https://user-images.githubusercontent.com/32848391/64992590-7fc82400-d8d4-11e9-9c10-795f4756a73f.png)


    --------------------------------------------------------------------
    .. note:: mode="spheric"

    If `mode='spheric'` input must be an external function rho(theta, phi).
    A surface is created in spherical coordinates.

    Return an `Figure(Assembly)` of 2 objects: the unit
    sphere (in wireframe representation) and the surface `rho(theta, phi)`.

    Arguments:
        rfunc : function
            handle to a user defined function `rho(theta, phi)`.
        normalize : (bool)
            scale surface to fit inside the unit sphere
        res : (int)
            grid resolution of the unit sphere
        scalarbar : (bool)
            add a 3D scalarbar to the plot for radius
        c : (color)
            color of the unit sphere
        alpha : (float)
            opacity of the unit sphere
        cmap : (str)
            color map for the surface

    Examples:
        - [plot_spheric.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_spheric.py)

            ![](https://vedo.embl.es/images/pyplot/plot_spheric.png)
    """
    mode = kwargs.pop("mode", "")
    if "spher" in mode:
        return _plot_spheric(args[0], **kwargs)

    if "bar" in mode:
        return PlotBars(args[0], **kwargs)

    if isinstance(args[0], str) or "function" in str(type(args[0])):
        if "complex" in mode:
            return _plot_fz(args[0], **kwargs)
        return _plot_fxy(args[0], **kwargs)

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

        symbs = [".", "o", "O", "0", "p", "*", "h", "D", "d", "v", "^", ">", "<", "s", "x", "a"]

        allcols = list(colors.colors.keys()) + list(colors.color_nicks.keys())
        for cc in allcols:
            if cc == "o":
                continue
            if cc in opts:
                opts = opts.replace(cc, "")
                kwargs["lc"] = cc
                kwargs["mc"] = cc
                break

        for ss in symbs:
            if ss in opts:
                opts = opts.replace(ss, "", 1)
                kwargs["marker"] = ss
                break

        opts.replace(" ", "")
        if opts:
            vedo.logger.error(f"in plot(), could not understand option(s): {opts}")

    if optidx == 1 or optidx is None:
        if utils.is_sequence(args[0][0]) and len(args[0][0]) > 1:
            # print('------------- case 1', 'plot([(x,y),..])')
            data = np.asarray(args[0])  # (x,y)
            x = np.asarray(data[:, 0])
            y = np.asarray(data[:, 1])

        elif len(args) == 1 or optidx == 1:
            # print('------------- case 2', 'plot(x)')
            if "pandas" in str(type(args[0])):
                if "ytitle" not in kwargs:
                    kwargs.update({"ytitle": args[0].name.replace("_", "_ ")})
            x = np.linspace(0, len(args[0]), num=len(args[0]))
            y = np.asarray(args[0]).ravel()

        elif utils.is_sequence(args[1]):
            # print('------------- case 3', 'plot(allx,ally)',str(type(args[0])))
            if "pandas" in str(type(args[0])):
                if "xtitle" not in kwargs:
                    kwargs.update({"xtitle": args[0].name.replace("_", "_ ")})
            if "pandas" in str(type(args[1])):
                if "ytitle" not in kwargs:
                    kwargs.update({"ytitle": args[1].name.replace("_", "_ ")})
            x = np.asarray(args[0]).ravel()
            y = np.asarray(args[1]).ravel()

        elif utils.is_sequence(args[0]) and utils.is_sequence(args[0][0]):
            # print('------------- case 4', 'plot([allx,ally])')
            x = np.asarray(args[0][0]).ravel()
            y = np.asarray(args[0][1]).ravel()

    elif optidx == 2:
        # print('------------- case 5', 'plot(x,y)')
        x = np.asarray(args[0]).ravel()
        y = np.asarray(args[1]).ravel()

    else:
        vedo.logger.error(f"plot(): Could not understand input arguments {args}")
        return None

    if "polar" in mode:
        return _plot_polar(np.c_[x, y], **kwargs)

    return PlotXY(np.c_[x, y], **kwargs)


def histogram(*args, **kwargs):
    """
    Histogramming for 1D and 2D data arrays.

    This is meant as a convenience function that creates the appropriate object
    based on the shape of the provided input data.

    Use keyword `like=...` if you want to use the same format of a previously
    created Figure (useful when superimposing Figures) to make sure
    they are compatible and comparable. If they are not compatible
    you will receive an error message.

    -------------------------------------------------------------------------
    .. note:: default mode, for 1D arrays

    Creates a `Histogram1D(Figure)` object.

    Arguments:
        weights : (list)
            An array of weights, of the same shape as `data`. Each value in `data`
            only contributes its associated weight towards the bin count (instead of 1).
        bins : (int)
            number of bins
        vrange : (list)
            restrict the range of the histogram
        density : (bool)
            normalize the area to 1 by dividing by the nr of entries and bin size
        logscale : (bool)
            use logscale on y-axis
        fill : (bool)
            fill bars with solid color `c`
        gap : (float)
            leave a small space btw bars
        radius : (float)
            border radius of the top of the histogram bar. Default value is 0.1.
        texture : (str)
            url or path to an image to be used as texture for the bin
        outline : (bool)
            show outline of the bins
        errors : (bool)
            show error bars
        xtitle : (str)
            title for the x-axis, can also be set using `axes=dict(xtitle="my x axis")`
        ytitle : (str)
            title for the y-axis, can also be set using `axes=dict(ytitle="my y axis")`
        padding : (float, list)
            keep a padding space from the axes (as a fraction of the axis size).
            This can be a list of four numbers.
        aspect : (float)
            the desired aspect ratio of the histogram. Default is 4/3.
        grid : (bool)
            show the background grid for the axes, can also be set using `axes=dict(xygrid=True)`
        ztolerance : (float)
            a tolerance factor to superimpose objects (along the z-axis).

    Examples:
        - [histo_1d_a.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_1d_a.py)
        - [histo_1d_b.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_1d_b.py)
        - [histo_1d_c.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_1d_c.py)
        - [histo_1d_d.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_1d_d.py)

        ![](https://vedo.embl.es/images/pyplot/histo_1D.png)


    -------------------------------------------------------------------------
    .. note:: default mode, for 2D arrays

    Input data formats `[(x1,x2,..), (y1,y2,..)] or [(x1,y1), (x2,y2),..]`
    are both valid.

    Arguments:
        bins : (list)
            binning as (nx, ny)
        weights : (list)
            array of weights to assign to each entry
        cmap : (str, lookuptable)
            color map name or look up table
        alpha : (float)
            opacity of the histogram
        gap : (float)
            separation between adjacent bins as a fraction for their size.
            Set gap=-1 to generate a quad surface.
        scalarbar : (bool)
            add a scalarbar to right of the histogram
        like : (Figure)
            grab and use the same format of the given Figure (for superimposing)
        xlim : (list)
            [x0, x1] range of interest. If left to None will automatically
            choose the minimum or the maximum of the data range.
            Data outside the range are completely ignored.
        ylim : (list)
            [y0, y1] range of interest. If left to None will automatically
            choose the minimum or the maximum of the data range.
            Data outside the range are completely ignored.
        aspect : (float)
            the desired aspect ratio of the figure.
        title : (str)
            title of the plot to appear on top.
            If left blank some statistics will be shown.
        xtitle : (str)
            x axis title
        ytitle : (str)
            y axis title
        ztitle : (str)
            title for the scalar bar
        ac : (str)
            axes color, additional keyword for Axes can also be added
            using e.g. `axes=dict(xygrid=True)`

    Examples:
        - [histo_2d_a.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_2d_a.py)
        - [histo_2d_b.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_2d_b.py)

        ![](https://vedo.embl.es/images/pyplot/histo_2D.png)


    -------------------------------------------------------------------------
    .. note:: mode="3d"

    If `mode='3d'`, build a 2D histogram as 3D bars from a list of x and y values.

    Arguments:
        xtitle : (str)
            x axis title
        bins : (int)
            nr of bins for the smaller range in x or y
        vrange : (list)
            range in x and y in format `[(xmin,xmax), (ymin,ymax)]`
        norm : (float)
            sets a scaling factor for the z axis (frequency axis)
        fill : (bool)
            draw solid hexagons
        cmap : (str)
            color map name for elevation
        gap : (float)
            keep a internal empty gap between bins [0,1]
        zscale : (float)
            rescale the (already normalized) zaxis for visual convenience

    Examples:
        - [histo_2d_b.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_2d_b.py)


    -------------------------------------------------------------------------
    .. note:: mode="hexbin"

    If `mode='hexbin'`, build a hexagonal histogram from a list of x and y values.

    Arguments:
        xtitle : (str)
            x axis title
        bins : (int)
            nr of bins for the smaller range in x or y
        vrange : (list)
            range in x and y in format `[(xmin,xmax), (ymin,ymax)]`
        norm : (float)
            sets a scaling factor for the z axis (frequency axis)
        fill : (bool)
            draw solid hexagons
        cmap : (str)
            color map name for elevation

    Examples:
        - [histo_hexagonal.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_hexagonal.py)

        ![](https://vedo.embl.es/images/pyplot/histo_hexagonal.png)


    -------------------------------------------------------------------------
    .. note:: mode="polar"

    If `mode='polar'` assume input is polar coordinate system (rho, theta):

    Arguments:
        weights : (list)
            Array of weights, of the same shape as the input.
            Each value only contributes its associated weight towards the bin count (instead of 1).
        title : (str)
            histogram title
        tsize : (float)
            title size
        bins : (int)
            number of bins in phi
        r1 : (float)
            inner radius
        r2 : (float)
            outer radius
        phigap : (float)
            gap angle btw 2 radial bars, in degrees
        rgap : (float)
            gap factor along radius of numeric angle labels
        lpos : (float)
            label gap factor along radius
        lsize : (float)
            label size
        c : (color)
            color of the histogram bars, can be a list of length `bins`
        bc : (color)
            color of the frame and labels
        alpha : (float)
            opacity of the frame
        cmap : (str)
            color map name
        deg : (bool)
            input array is in degrees
        vmin : (float)
            minimum value of the radial axis
        vmax : (float)
            maximum value of the radial axis
        labels : (list)
            list of labels, must be of length `bins`
        show_disc : (bool)
            show the outer ring axis
        nrays : (int)
            draw this number of axis rays (continuous and dashed)
        show_lines : (bool)
            show lines to the origin
        show_angles : (bool)
            show angular values
        show_errors : (bool)
            show error bars

    Examples:
        - [histo_polar.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_polar.py)

        ![](https://vedo.embl.es/images/pyplot/histo_polar.png)


    -------------------------------------------------------------------------
    .. note:: mode="spheric"

    If `mode='spheric'`, build a histogram from list of theta and phi values.

    Arguments:
        rmax : (float)
            maximum radial elevation of bin
        res : (int)
            sphere resolution
        cmap : (str)
            color map name
        lw : (int)
            line width of the bin edges

    Examples:
        - [histo_spheric.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_spheric.py)

        ![](https://vedo.embl.es/images/pyplot/histo_spheric.png)
    """
    mode = kwargs.pop("mode", "")
    if len(args) == 2:  # x, y

        if "spher" in mode:
            return _histogram_spheric(args[0], args[1], **kwargs)

        if "hex" in mode:
            return _histogram_hex_bin(args[0], args[1], **kwargs)

        if "3d" in mode.lower():
            return _histogram_quad_bin(args[0], args[1], **kwargs)

        return Histogram2D(args[0], args[1], **kwargs)

    elif len(args) == 1:

        if isinstance(args[0], vedo.Volume):
            data = args[0].pointdata[0]
        elif isinstance(args[0], vedo.Points):
            pd0 = args[0].pointdata[0]
            if pd0 is not None:
                data = pd0.ravel()
            else:
                data = args[0].celldata[0].ravel()
        else:
            try:
                if "pandas" in str(type(args[0])):
                    if "xtitle" not in kwargs:
                        kwargs.update({"xtitle": args[0].name.replace("_", "_ ")})
            except:
                pass
            data = np.asarray(args[0])

        if "spher" in mode:
            return _histogram_spheric(args[0][:, 0], args[0][:, 1], **kwargs)

        if data.ndim == 1:
            if "polar" in mode:
                return _histogram_polar(data, **kwargs)
            return Histogram1D(data, **kwargs)

        if "hex" in mode:
            return _histogram_hex_bin(args[0][:, 0], args[0][:, 1], **kwargs)

        if "3d" in mode.lower():
            return _histogram_quad_bin(args[0][:, 0], args[0][:, 1], **kwargs)

        return Histogram2D(args[0], **kwargs)

    vedo.logger.error(f"in histogram(): could not understand input {args[0]}")
    return None


def fit(
    points, deg=1, niter=0, nstd=3, xerrors=None, yerrors=None, vrange=None, res=250, lw=3, c="red4"
) -> "vedo.shapes.Line":
    """
    Polynomial fitting with parameter error and error bands calculation.
    Errors bars in both x and y are supported.

    Returns a `vedo.shapes.Line` object.

    Additional information about the fitting output can be accessed with:

    `fitd = fit(pts)`

    - `fitd.coefficients` will contain the coefficients of the polynomial fit
    - `fitd.coefficient_errors`, errors on the fitting coefficients
    - `fitd.monte_carlo_coefficients`, fitting coefficient set from MC generation
    - `fitd.covariance_matrix`, covariance matrix as a numpy array
    - `fitd.reduced_chi2`, reduced chi-square of the fitting
    - `fitd.ndof`, number of degrees of freedom
    - `fitd.data_sigma`, mean data dispersion from the central fit assuming `Chi2=1`
    - `fitd.error_lines`, a `vedo.shapes.Line` object for the upper and lower error band
    - `fitd.error_band`, the `vedo.mesh.Mesh` object representing the error band

    Errors on x and y can be specified. If left to `None` an estimate is made from
    the statistical spread of the dataset itself. Errors are always assumed gaussian.

    Arguments:
        deg : (int)
            degree of the polynomial to be fitted
        niter : (int)
            number of monte-carlo iterations to compute error bands.
            If set to 0, return the simple least-squares fit with naive error estimation
            on coefficients only. A reasonable non-zero value to set is about 500, in
            this case *error_lines*, *error_band* and the other class attributes are filled
        nstd : (float)
            nr. of standard deviation to use for error calculation
        xerrors : (list)
            array of the same length of points with the errors on x
        yerrors : (list)
            array of the same length of points with the errors on y
        vrange : (list)
            specify the domain range of the fitting line
            (only affects visualization, but can be used to extrapolate the fit
            outside the data range)
        res : (int)
            resolution of the output fitted line and error lines

    Examples:
        - [fit_polynomial1.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/fit_polynomial1.py)

        ![](https://vedo.embl.es/images/pyplot/fitPolynomial1.png)
    """
    if isinstance(points, vedo.pointcloud.Points):
        points = points.coordinates
    points = np.asarray(points)
    if len(points) == 2:  # assume user is passing [x,y]
        points = np.c_[points[0], points[1]]
    x = points[:, 0]
    y = points[:, 1]  # ignore z

    n = len(x)
    ndof = n - deg - 1
    if vrange is not None:
        x0, x1 = vrange
    else:
        x0, x1 = np.min(x), np.max(x)
        if xerrors is not None:
            x0 -= xerrors[0] / 2
            x1 += xerrors[-1] / 2

    tol = (x1 - x0) / 10000
    xr = np.linspace(x0, x1, res)

    # project x errs on y
    if xerrors is not None:
        xerrors = np.asarray(xerrors)
        if yerrors is not None:
            yerrors = np.asarray(yerrors)
            w = 1.0 / yerrors
            coeffs = np.polyfit(x, y, deg, w=w, rcond=None)
        else:
            coeffs = np.polyfit(x, y, deg, rcond=None)
        # update yerrors, 1 bootstrap iteration is enough
        p1d = np.poly1d(coeffs)
        der = (p1d(x + tol) - p1d(x)) / tol
        yerrors = np.sqrt(yerrors * yerrors + np.power(der * xerrors, 2))

    if yerrors is not None:
        yerrors = np.asarray(yerrors)
        w = 1.0 / yerrors
        coeffs, V = np.polyfit(x, y, deg, w=w, rcond=None, cov=True)
    else:
        w = 1
        coeffs, V = np.polyfit(x, y, deg, rcond=None, cov=True)

    p1d = np.poly1d(coeffs)
    theor = p1d(xr)
    fitl = shapes.Line(np.c_[xr, theor], lw=lw, c=c).z(tol * 2)
    fitl.coefficients = coeffs
    fitl.covariance_matrix = V
    residuals2_sum = np.sum(np.power(p1d(x) - y, 2)) / ndof
    sigma = np.sqrt(residuals2_sum)
    fitl.reduced_chi2 = np.sum(np.power((p1d(x) - y) * w, 2)) / ndof
    fitl.ndof = ndof
    fitl.data_sigma = sigma  # worked out from data using chi2=1 hypo
    fitl.name = "LinearPolynomialFit"

    if not niter:
        fitl.coefficient_errors = np.sqrt(np.diag(V))
        return fitl  ################################

    if yerrors is not None:
        sigma = yerrors
    else:
        w = None
        fitl.reduced_chi2 = 1

    Theors, all_coeffs = [], []
    for i in range(niter):
        noise = np.random.randn(n) * sigma
        coeffs = np.polyfit(x, y + noise, deg, w=w, rcond=None)
        all_coeffs.append(coeffs)
        P1d = np.poly1d(coeffs)
        Theor = P1d(xr)
        Theors.append(Theor)
    # all_coeffs = np.array(all_coeffs)
    fitl.monte_carlo_coefficients = np.array(all_coeffs)

    stds = np.std(Theors, axis=0)
    fitl.coefficient_errors = np.std(all_coeffs, axis=0)

    # check distributions on the fly
    # for i in range(deg+1):
    #     histogram(all_coeffs[:,i],title='par'+str(i)).show(new=1)
    # histogram(all_coeffs[:,0], all_coeffs[:,1],
    #           xtitle='param0', ytitle='param1',scalarbar=1).show(new=1)
    # histogram(all_coeffs[:,1], all_coeffs[:,2],
    #           xtitle='param1', ytitle='param2').show(new=1)
    # histogram(all_coeffs[:,0], all_coeffs[:,2],
    #           xtitle='param0', ytitle='param2').show(new=1)

    error_lines = []
    for i in [nstd, -nstd]:
        pp = np.c_[xr, theor + stds * i]
        el = shapes.Line(pp, lw=1, alpha=0.2, c="k").z(tol)
        error_lines.append(el)
        el.name = "ErrorLine for sigma=" + str(i)

    fitl.error_lines = error_lines
    l1 = error_lines[0].coordinates.tolist()
    cband = l1 + list(reversed(error_lines[1].coordinates.tolist())) + [l1[0]]
    fitl.error_band = shapes.Line(cband).triangulate().lw(0).c("k", 0.15)
    fitl.error_band.name = "PolynomialFitErrorBand"
    return fitl


def _plot_fxy(
    z,
    xlim=(0, 3),
    ylim=(0, 3),
    zlim=(None, None),
    show_nan=True,
    zlevels=10,
    c=None,
    bc="aqua",
    alpha=1,
    texture="",
    bins=(100, 100),
    axes=True,
):
    import warnings

    if c is not None:
        texture = None  # disable

    ps = vtki.new("PlaneSource")
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                zv = z(xv, yv)
                if np.isnan(zv) or np.isinf(zv) or np.iscomplex(zv):
                    zv = 0
                    todel.append(i)
                    nans.append([xv, yv, 0])
        except:
            zv = 0
            todel.append(i)
            nans.append([xv, yv, 0])
        poly.GetPoints().SetPoint(i, [xv, yv, zv])

    if todel:
        cellIds = vtki.vtkIdList()
        poly.BuildLinks()
        for i in todel:
            poly.GetPointCells(i, cellIds)
            for j in range(cellIds.GetNumberOfIds()):
                poly.DeleteCell(cellIds.GetId(j))  # flag cell
        poly.RemoveDeletedCells()
        cl = vtki.new("CleanPolyData")
        cl.SetInputData(poly)
        cl.Update()
        poly = cl.GetOutput()

    if not poly.GetNumberOfPoints():
        vedo.logger.error("function is not real in the domain")
        return None

    if zlim[0]:
        poly = Mesh(poly).cut_with_plane((0, 0, zlim[0]), (0, 0, 1)).dataset
    if zlim[1]:
        poly = Mesh(poly).cut_with_plane((0, 0, zlim[1]), (0, 0, -1)).dataset

    cmap = ""
    if c in colors.cmaps_names:
        cmap = c
        c = None
        bc = None

    mesh = Mesh(poly, c, alpha).compute_normals().lighting("plastic")

    if cmap:
        mesh.compute_elevation().cmap(cmap)
    if bc:
        mesh.bc(bc)
    if texture:
        mesh.texture(texture)

    acts = [mesh]
    if zlevels:
        elevation = vtki.new("ElevationFilter")
        elevation.SetInputData(poly)
        bounds = poly.GetBounds()
        elevation.SetLowPoint(0, 0, bounds[4])
        elevation.SetHighPoint(0, 0, bounds[5])
        elevation.Update()
        bcf = vtki.new("BandedPolyDataContourFilter")
        bcf.SetInputData(elevation.GetOutput())
        bcf.SetScalarModeToValue()
        bcf.GenerateContourEdgesOn()
        bcf.GenerateValues(zlevels, elevation.GetScalarRange())
        bcf.Update()
        zpoly = bcf.GetContourEdgesOutput()
        zbandsact = Mesh(zpoly, "k", alpha).lw(1).lighting("off")
        zbandsact.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        acts.append(zbandsact)

    if show_nan and todel:
        bb = mesh.bounds()
        if bb[4] <= 0 and bb[5] >= 0:
            zm = 0.0
        else:
            zm = (bb[4] + bb[5]) / 2
        nans = np.array(nans) + [0, 0, zm]
        nansact = Points(nans, r=2, c="red5", alpha=alpha)
        nansact.properties.RenderPointsAsSpheresOff()
        acts.append(nansact)

    if isinstance(axes, dict):
        axs = addons.Axes(mesh, **axes)
        acts.append(axs)
    elif axes:
        axs = addons.Axes(mesh)
        acts.append(axs)

    assem = Assembly(acts)
    assem.name = "PlotFxy"
    return assem


def _plot_fz(
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
    ps = vtki.new("PlaneSource")
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
            zv = z(complex(xv), complex(yv))
        except:
            zv = 0
        poly.GetPoints().SetPoint(i, [xv, yv, np.real(zv)])
        arrImg.append(np.imag(zv))

    mesh = Mesh(poly, alpha).lighting("plastic")
    v = max(abs(np.min(arrImg)), abs(np.max(arrImg)))
    mesh.cmap(cmap, arrImg, vmin=-v, vmax=v)
    mesh.compute_normals().lw(lw)

    if zlimits[0]:
        mesh.cut_with_plane((0, 0, zlimits[0]), (0, 0, 1))
    if zlimits[1]:
        mesh.cut_with_plane((0, 0, zlimits[1]), (0, 0, -1))

    acts = [mesh]
    if axes:
        axs = addons.Axes(mesh, ztitle="Real part")
        acts.append(axs)
    asse = Assembly(acts)
    asse.name = "PlotFz"
    if isinstance(z, str):
        asse.name += " " + z
    return asse


def _plot_polar(
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
    splined=False,
    nrays=8,
    show_disc=True,
    show_lines=True,
    show_angles=True,
):
    if len(rphi) == 2:
        rphi = np.stack((rphi[0], rphi[1]), axis=1)

    rphi = np.array(rphi, dtype=float)
    thetas = rphi[:, 0]
    radii = rphi[:, 1]

    k = 180 / np.pi
    if deg:
        thetas = np.array(thetas, dtype=float) / k

    vals = []
    for v in thetas:  # normalize range
        t = np.arctan2(np.sin(v), np.cos(v))
        if t < 0:
            t += 2 * np.pi
        vals.append(t)
    thetas = np.array(vals, dtype=float)

    if vmax is None:
        vmax = np.max(radii)

    angles = []
    points = []
    for t, r in zip(thetas, radii):
        r = r / vmax * r2 + r1
        ct, st = np.cos(t), np.sin(t)
        points.append([r * ct, r * st, 0])
    p0 = points[0]
    points.append(p0)

    r2e = r1 + r2
    lines = None
    if splined:
        lines = shapes.KSpline(points, closed=True)
        lines.c(c).lw(lw).alpha(alpha)
    elif lw:
        lines = shapes.Line(points)
        lines.c(c).lw(lw).alpha(alpha)

    points.pop()

    ptsact = None
    if ps:
        ptsact = Points(points, r=ps, c=c, alpha=alpha)

    filling = None
    if fill and lw:
        faces = []
        coords = [[0, 0, 0]] + lines.coordinates.tolist()
        for i in range(1, lines.npoints):
            faces.append([0, i, i + 1])
        filling = Mesh([coords, faces]).c(c).alpha(alpha)

    back = None
    back2 = None
    if show_disc:
        back = shapes.Disc(r1=r2e, r2=r2e * 1.01, c=bc, res=(1, 360))
        back.z(-0.01).lighting("off").alpha(alpha)
        back2 = shapes.Disc(r1=r2e / 2, r2=r2e / 2 * 1.005, c=bc, res=(1, 360))
        back2.z(-0.01).lighting("off").alpha(alpha)

    ti = None
    if title:
        ti = shapes.Text3D(title, (0, 0, 0), s=tsize, depth=0, justify="top-center")
        ti.pos(0, -r2e * 1.15, 0.01)

    rays = []
    if show_disc:
        rgap = 0.05
        for t in np.linspace(0, 2 * np.pi, num=nrays, endpoint=False):
            ct, st = np.cos(t), np.sin(t)
            if show_lines:
                l = shapes.Line((0, 0, -0.01), (r2e * ct * 1.03, r2e * st * 1.03, -0.01))
                rays.append(l)
                ct2, st2 = np.cos(t + np.pi / nrays), np.sin(t + np.pi / nrays)
                lm = shapes.DashedLine((0, 0, -0.01), (r2e * ct2, r2e * st2, -0.01), spacing=0.25)
                rays.append(lm)
            elif show_angles:  # just the ticks
                l = shapes.Line(
                    (r2e * ct * 0.98, r2e * st * 0.98, -0.01),
                    (r2e * ct * 1.03, r2e * st * 1.03, -0.01),
                )
            if show_angles:
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
        mrg.color(bc).alpha(alpha).lighting("off")
    rh = Assembly([lines, ptsact, filling] + [mrg])
    rh.name = "PlotPolar"
    return rh


def _plot_spheric(rfunc, normalize=True, res=33, scalarbar=True, c="grey", alpha=0.05, cmap="jet"):
    sg = shapes.Sphere(res=res, quads=True)
    sg.alpha(alpha).c(c).wireframe()

    cgpts = sg.coordinates
    r, theta, phi = cart2spher(*cgpts.T)

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

    newr = np.array(newr, dtype=float)
    if normalize:
        newr = newr / np.max(newr)
        newr[inans] = 1

    nanpts = []
    if inans:
        redpts = spher2cart(newr[inans], theta[inans], phi[inans]).T
        nanpts.append(Points(redpts, r=4, c="r"))

    pts = spher2cart(newr, theta, phi).T
    ssurf = sg.clone()
    ssurf.coordinates = pts
    if inans:
        ssurf.delete_cells_by_point_index(inans)

    ssurf.alpha(1).wireframe(0).lw(0.1)

    ssurf.cmap(cmap, newr)
    ssurf.compute_normals()

    if scalarbar:
        xm = np.max([np.max(pts[0]), 1])
        ym = np.max([np.abs(np.max(pts[1])), 1])
        ssurf.mapper.SetScalarRange(np.min(newr), np.max(newr))
        sb3d = ssurf.add_scalarbar3d(size=(xm * 0.07, ym), c="k").scalarbar
        sb3d.rotate_x(90).pos(xm * 1.1, 0, -0.5)
    else:
        sb3d = None

    sg.pickable(False)
    asse = Assembly([ssurf, sg] + nanpts + [sb3d])
    asse.name = "PlotSpheric"
    return asse


def _histogram_quad_bin(x, y, **kwargs):
    # generate a histogram with 3D bars
    #
    histo = Histogram2D(x, y, **kwargs)

    gap = kwargs.pop("gap", 0)
    zscale = kwargs.pop("zscale", 1)
    cmap = kwargs.pop("cmap", "Blues_r")

    gr = histo.objects[2]
    d = gr.diagonal_size()
    tol = d / 1_000_000  # tolerance
    if gap >= 0:
        gr.shrink(1 - gap - tol)
    gr.map_cells_to_points()

    faces = np.array(gr.cells)
    s = 1 / histo.entries * len(faces) * zscale
    zvals = gr.pointdata["Scalars"] * s

    pts1 = gr.coordinates
    pts2 = np.copy(pts1)
    pts2[:, 2] = zvals + tol
    newpts = np.vstack([pts1, pts2])
    newzvals = np.hstack([zvals, zvals]) / s

    n = pts1.shape[0]
    newfaces = []
    for f in faces:
        f0, f1, f2, f3 = f
        f0n, f1n, f2n, f3n = f + n
        newfaces.extend(
            [
                [f0, f1, f2, f3],
                [f0n, f1n, f2n, f3n],
                [f0, f1, f1n, f0n],
                [f1, f2, f2n, f1n],
                [f2, f3, f3n, f2n],
                [f3, f0, f0n, f3n],
            ]
        )

    msh = Mesh([newpts, newfaces]).pickable(False)
    msh.cmap(cmap, newzvals, name="Frequency")
    msh.lw(1).lighting("ambient")

    histo.objects[2] = msh
    histo.actor.RemovePart(gr.actor)
    histo.actor.AddPart(msh.actor)
    histo.objects.append(msh)
    return histo


def _histogram_hex_bin(
    xvalues, yvalues, bins=12, norm=1, fill=True, c=None, cmap="terrain_r", alpha=1
) -> "Assembly":
    xmin, xmax = np.min(xvalues), np.max(xvalues)
    ymin, ymax = np.min(yvalues), np.max(yvalues)
    dx, dy = xmax - xmin, ymax - ymin

    if utils.is_sequence(bins):
        n, m = bins
    else:
        if xmax - xmin < ymax - ymin:
            n = bins
            m = np.rint(dy / dx * n / 1.2 + 0.5).astype(int)
        else:
            m = bins
            n = np.rint(dx / dy * m * 1.2 + 0.5).astype(int)

    values = np.stack((xvalues, yvalues), axis=1)
    zs = [[0.0]] * len(values)
    values = np.append(values, zs, axis=1)
    cloud = vedo.Points(values)

    col = None
    if c is not None:
        col = colors.get_color(c)

    hexs, binmax = [], 0
    ki, kj = 1.33, 1.12
    r = 0.47 / n * 1.2 * dx
    for i in range(n + 3):
        for j in range(m + 2):
            cyl = vtki.new("CylinderSource")
            cyl.SetResolution(6)
            cyl.CappingOn()
            cyl.SetRadius(0.5)
            cyl.SetHeight(0.1)
            cyl.Update()
            t = vtki.vtkTransform()
            if not i % 2:
                p = (i / ki, j / kj, 0)
            else:
                p = (i / ki, j / kj + 0.45, 0)
            q = (p[0] / n * 1.2 * dx + xmin, p[1] / m * dy + ymin, 0)
            ne = len(cloud.closest_point(q, radius=r))
            if fill:
                t.Translate(p[0], p[1], ne / 2)
                t.Scale(1, 1, ne * 10)
            else:
                t.Translate(p[0], p[1], ne)
            t.RotateX(90)  # put it along Z
            tf = vtki.new("TransformPolyDataFilter")
            tf.SetInputData(cyl.GetOutput())
            tf.SetTransform(t)
            tf.Update()
            if c is None:
                col = i
            h = Mesh(tf.GetOutput(), c=col, alpha=alpha).flat()
            h.lighting("plastic")
            h.actor.PickableOff()
            hexs.append(h)
            if ne > binmax:
                binmax = ne

    if cmap is not None:
        for h in hexs:
            z = h.bounds()[5]
            col = colors.color_map(z, cmap, 0, binmax)
            h.color(col)

    asse = Assembly(hexs)
    asse.scale([1.2 / n * dx, 1 / m * dy, norm / binmax * (dx + dy) / 4])
    asse.pos([xmin, ymin, 0])
    asse.name = "HistogramHexBin"
    return asse


def _histogram_polar(
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
    c="grey",
    bc="k",
    alpha=1,
    cmap=None,
    deg=False,
    vmin=None,
    vmax=None,
    labels=(),
    show_disc=True,
    nrays=8,
    show_lines=True,
    show_angles=True,
    show_errors=False,
):
    k = 180 / np.pi
    if deg:
        values = np.array(values, dtype=float) / k
    else:
        values = np.array(values, dtype=float)

    vals = []
    for v in values:  # normalize range
        t = np.arctan2(np.sin(v), np.cos(v))
        if t < 0:
            t += 2 * np.pi
        vals.append(t + 0.00001)

    histodata, edges = np.histogram(vals, weights=weights, bins=bins, range=(0, 2 * np.pi))

    thetas = []
    for i in range(bins):
        thetas.append((edges[i] + edges[i + 1]) / 2)

    if vmin is None:
        vmin = np.min(histodata)
    if vmax is None:
        vmax = np.max(histodata)

    errors = np.sqrt(histodata)
    r2e = r1 + r2
    if show_errors:
        r2e += np.max(errors) / vmax * 1.5

    back = None
    if show_disc:
        back = shapes.Disc(r1=r2e, r2=r2e * 1.01, c=bc, res=(1, 360))
        back.z(-0.01)

    slices = []
    lines = []
    angles = []
    errbars = []

    for i, t in enumerate(thetas):
        r = histodata[i] / vmax * r2
        d = shapes.Disc((0, 0, 0), r1, r1 + r, res=(1, 360))
        delta = np.pi / bins - np.pi / 2 - phigap / k
        d.cut_with_plane(normal=(np.cos(t + delta), np.sin(t + delta), 0))
        d.cut_with_plane(normal=(np.cos(t - delta), np.sin(t - delta), 0))
        if cmap is not None:
            cslice = colors.color_map(histodata[i], cmap, vmin, vmax)
            d.color(cslice)
        else:
            if c is None:
                d.color(i)
            elif utils.is_sequence(c) and len(c) == bins:
                d.color(c[i])
            else:
                d.color(c)
        d.alpha(alpha).lighting("off")
        slices.append(d)

        ct, st = np.cos(t), np.sin(t)

        if show_errors:
            err = np.sqrt(histodata[i]) / vmax * r2
            errl = shapes.Line(
                ((r1 + r - err) * ct, (r1 + r - err) * st, 0.01),
                ((r1 + r + err) * ct, (r1 + r + err) * st, 0.01),
            )
            errl.alpha(alpha).lw(3).color(bc)
            errbars.append(errl)

    labs = []
    rays = []
    if show_disc:
        outerdisc = shapes.Disc(r1=r2e, r2=r2e * 1.01, c=bc, res=(1, 360))
        outerdisc.z(-0.01)
        innerdisc = shapes.Disc(r1=r2e / 2, r2=r2e / 2 * 1.005, c=bc, res=(1, 360))
        innerdisc.z(-0.01)
        rays.append(outerdisc)
        rays.append(innerdisc)

        rgap = 0.05
        for t in np.linspace(0, 2 * np.pi, num=nrays, endpoint=False):
            ct, st = np.cos(t), np.sin(t)
            if show_lines:
                l = shapes.Line((0, 0, -0.01), (r2e * ct * 1.03, r2e * st * 1.03, -0.01))
                rays.append(l)
                ct2, st2 = np.cos(t + np.pi / nrays), np.sin(t + np.pi / nrays)
                lm = shapes.DashedLine((0, 0, -0.01), (r2e * ct2, r2e * st2, -0.01), spacing=0.25)
                rays.append(lm)
            elif show_angles:  # just the ticks
                l = shapes.Line(
                    (r2e * ct * 0.98, r2e * st * 0.98, -0.01),
                    (r2e * ct * 1.03, r2e * st * 1.03, -0.01),
                )
            if show_angles:
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

    for i, t in enumerate(thetas):
        if i < len(labels):
            lab = shapes.Text3D(
                labels[i], (0, 0, 0), s=lsize, depth=0, justify="center"  # font="VTK",
            )
            lab.pos(
                r2e * np.cos(t) * (1 + rgap) * lpos / 2,
                r2e * np.sin(t) * (1 + rgap) * lpos / 2,
                0.01,
            )
            labs.append(lab)

    mrg = merge(lines, angles, rays, ti, labs)
    if mrg:
        mrg.color(bc).lighting("off")

    acts = slices + errbars + [mrg]
    asse = Assembly(acts)
    asse.frequencies = histodata
    asse.bins = edges
    asse.name = "HistogramPolar"
    return asse


def _histogram_spheric(thetavalues, phivalues, rmax=1.2, res=8, cmap="rainbow", gap=0.1):

    x, y, z = spher2cart(np.ones_like(thetavalues) * 1.1, thetavalues, phivalues)
    ptsvals = np.c_[x, y, z]

    sg = shapes.Sphere(res=res, quads=True).shrink(1 - gap)
    sgfaces = sg.cells
    sgpts = sg.coordinates

    cntrs = sg.cell_centers().coordinates
    counts = np.zeros(len(cntrs))
    for p in ptsvals:
        cell = sg.closest_point(p, return_cell_id=True)
        counts[cell] += 1
    acounts = np.array(counts, dtype=float)
    counts *= (rmax - 1) / np.max(counts)

    for cell, cn in enumerate(counts):
        if not cn:
            continue
        fs = sgfaces[cell]
        pts = sgpts[fs]
        _, t1, p1 = cart2spher(pts[:, 0], pts[:, 1], pts[:, 2])
        x, y, z = spher2cart(1 + cn, t1, p1)
        sgpts[fs] = np.c_[x, y, z]

    sg.coordinates = sgpts
    sg.cmap(cmap, acounts, on="cells")
    vals = sg.celldata["Scalars"]

    faces = sg.cells
    points = sg.coordinates.tolist() + [[0.0, 0.0, 0.0]]
    lp = len(points) - 1
    newfaces = []
    newvals = []
    for i, f in enumerate(faces):
        p0, p1, p2, p3 = f
        newfaces.append(f)
        newfaces.append([p0, lp, p1])
        newfaces.append([p1, lp, p2])
        newfaces.append([p2, lp, p3])
        newfaces.append([p3, lp, p0])
        for _ in range(5):
            newvals.append(vals[i])

    newsg = Mesh([points, newfaces]).cmap(cmap, newvals, on="cells")
    newsg.compute_normals().flat()
    newsg.name = "HistogramSpheric"
    return newsg


def streamplot(
    X, Y, U, V, direction="both", max_propagation=None, lw=2, cmap="viridis", probes=()
) -> Union["vedo.shapes.Lines", None]:
    """
    Generate a streamline plot of a vectorial field (U,V) defined at positions (X,Y).
    Returns a `Mesh` object.

    Arguments:
        direction : (str)
            either "forward", "backward" or "both"
        max_propagation : (float)
            maximum physical length of the streamline
        lw : (float)
            line width in absolute units

    Examples:
        - [plot_stream.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_stream.py)

            ![](https://vedo.embl.es/images/pyplot/plot_stream.png)
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
    vol.pointdata["StreamPlotField"] = vects

    if len(probes) == 0:
        probe = shapes.Grid(pos=((n - 1) / 2, (n - 1) / 2, 0), s=(n - 1, n - 1), res=(n - 1, n - 1))
    else:
        if isinstance(probes, vedo.Points):
            probes = probes.coordinates
        else:
            probes = np.array(probes, dtype=float)
            if len(probes[0]) == 2:
                probes = np.c_[probes[:, 0], probes[:, 1], np.zeros(len(probes))]
        sv = [(n - 1) / (xmax - xmin), (n - 1) / (ymax - ymin), 1]
        probes = probes - [xmin, ymin, 0]
        probes = np.multiply(probes, sv)
        probe = vedo.Points(probes)

    stream = vol.compute_streamlines(probe, direction=direction, max_propagation=max_propagation)
    if stream:
        stream.lw(lw).cmap(cmap).lighting("off")
        stream.scale([1 / (n - 1) * (xmax - xmin), 1 / (n - 1) * (ymax - ymin), 1])
        stream.shift(xmin, ymin)
    return stream



__all__ = ["plot", "histogram", "fit", "streamplot"]
