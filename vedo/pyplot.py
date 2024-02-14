#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import settings
from vedo.transformations import cart2spher, spher2cart
from vedo import addons
from vedo import colors
from vedo import utils
from vedo import shapes
from vedo.pointcloud import merge
from vedo.mesh import Mesh
from vedo.assembly import Assembly

__docformat__ = "google"

__doc__ = """
Advanced plotting functionalities.

![](https://vedo.embl.es/images/pyplot/fitPolynomial2.png)
"""

__all__ = [
    "Figure",
    "Histogram1D",
    "Histogram2D",
    "PlotXY",
    "PlotBars",
    "plot",
    "histogram",
    "fit",
    "donut",
    "violin",
    "whisker",
    "streamplot",
    "matrix",
    "DirectedGraph",
]


##########################################################################
class LabelData:
    """Helper internal class to hold label information."""

    def __init__(self):
        """Helper internal class to hold label information."""
        self.text   = "dataset"
        self.tcolor = "black"
        self.marker = "s"
        self.mcolor = "black"


##########################################################################
class Figure(Assembly):
    """Format class for figures."""

    def __init__(self, xlim, ylim, aspect=4 / 3, padding=(0.05, 0.05, 0.05, 0.05), **kwargs):
        """
        Create an empty formatted figure for plotting.

        Arguments:
            xlim : (list)
                range of the x-axis as [x0, x1]
            ylim : (list)
                range of the y-axis as [y0, y1]
            aspect : (float, str)
                the desired aspect ratio of the histogram. Default is 4/3.
                Use `aspect="equal"` to force the same units in x and y.
            padding : (float, list)
                keep a padding space from the axes (as a fraction of the axis size).
                This can be a list of four numbers.
            xtitle : (str)
                title for the x-axis, can also be set using `axes=dict(xtitle="my x axis")`
            ytitle : (str)
                title for the y-axis, can also be set using `axes=dict(ytitle="my y axis")`
            grid : (bool)
                show the background grid for the axes, can also be set using `axes=dict(xygrid=True)`
            axes : (dict)
                an extra dictionary of options for the `vedo.addons.Axes` object
        """

        self.verbose = True  # printing to stdout on every mouse click

        self.xlim = np.asarray(xlim)
        self.ylim = np.asarray(ylim)
        self.aspect = aspect
        self.padding = padding
        if not utils.is_sequence(self.padding):
            self.padding = [self.padding, self.padding, self.padding, self.padding]

        self.force_scaling_types = (
            shapes.Glyph,
            shapes.Line,
            shapes.Rectangle,
            shapes.DashedLine,
            shapes.Tube,
            shapes.Ribbon,
            shapes.GeoCircle,
            shapes.Arc,
            shapes.Grid,
            # shapes.Arrows, # todo
            # shapes.Arrows2D, # todo
            shapes.Brace,  # todo
        )

        options = dict(kwargs)

        self.title  = options.pop("title", "")
        self.xtitle = options.pop("xtitle", " ")
        self.ytitle = options.pop("ytitle", " ")
        number_of_divisions = 6

        self.legend = None
        self.labels = []
        self.label = options.pop("label", None)
        if self.label:
            self.labels = [self.label]

        self.axopts = options.pop("axes", {})
        if isinstance(self.axopts, (bool, int, float)):
            if self.axopts:
                self.axopts = {}
        if self.axopts or isinstance(self.axopts, dict):
            number_of_divisions = self.axopts.pop("number_of_divisions", number_of_divisions)

            self.axopts["xtitle"] = self.xtitle
            self.axopts["ytitle"] = self.ytitle

            if "xygrid" not in self.axopts:  ## modify the default
                self.axopts["xygrid"] = options.pop("grid", False)

            if "xygrid_transparent" not in self.axopts:  ## modify the default
                self.axopts["xygrid_transparent"] = True

            if "xtitle_position" not in self.axopts:  ## modify the default
                self.axopts["xtitle_position"] = 0.5
                self.axopts["xtitle_justify"] = "top-center"

            if "ytitle_position" not in self.axopts:  ## modify the default
                self.axopts["ytitle_position"] = 0.5
                self.axopts["ytitle_justify"] = "bottom-center"

            if self.label:
                if "c" in self.axopts:
                    self.label.tcolor = self.axopts["c"]

        x0, x1 = self.xlim
        y0, y1 = self.ylim
        dx = x1 - x0
        dy = y1 - y0
        x0lim, x1lim = (x0 - self.padding[0] * dx, x1 + self.padding[1] * dx)
        y0lim, y1lim = (y0 - self.padding[2] * dy, y1 + self.padding[3] * dy)
        dy = y1lim - y0lim

        self.axes = None
        if xlim[0] >= xlim[1] or ylim[0] >= ylim[1]:
            vedo.logger.warning(f"Null range for Figure {self.title}... returning an empty Assembly.")
            super().__init__()
            self.yscale = 0
            return

        if aspect == "equal":
            self.aspect = dx / dy  # so that yscale becomes 1

        self.yscale = dx / dy / self.aspect

        y0lim *= self.yscale
        y1lim *= self.yscale

        self.x0lim = x0lim
        self.x1lim = x1lim
        self.y0lim = y0lim
        self.y1lim = y1lim

        self.ztolerance = options.pop("ztolerance", None)
        if self.ztolerance is None:
            self.ztolerance = dx / 5000

        ############## create axes
        if self.axopts:
            axes_opts = self.axopts
            if self.axopts is True or self.axopts == 1:
                axes_opts = {}

            tp, ts = utils.make_ticks(y0lim / self.yscale, 
                                      y1lim / self.yscale, number_of_divisions)
            labs = []
            for i in range(1, len(tp) - 1):
                ynew = utils.lin_interpolate(tp[i], [0, 1], [y0lim, y1lim])
                labs.append([ynew, ts[i]])

            if self.title:
                axes_opts["htitle"] = self.title
            axes_opts["y_values_and_labels"] = labs
            axes_opts["xrange"] = (x0lim, x1lim)
            axes_opts["yrange"] = (y0lim, y1lim)
            axes_opts["zrange"] = (0, 0)
            axes_opts["y_use_bounds"] = True

            if "c" not in axes_opts and "ac" in options:
                axes_opts["c"] = options["ac"]

            self.axes = addons.Axes(**axes_opts)

        super().__init__([self.axes])
        self.name = "Figure"

        vedo.last_figure = self if settings.remember_last_figure_format else None


    ##################################################################
    def _repr_html_(self):
        """
        HTML representation of the Figure object for Jupyter Notebooks.

        Returns:
            HTML text with the image and some properties.
        """
        import io
        import base64
        from PIL import Image

        library_name = "vedo.pyplot.Figure"
        help_url = "https://vedo.embl.es/docs/vedo/pyplot.html#Figure"

        arr = self.thumbnail(zoom=1.1)

        im = Image.fromarray(arr)
        buffered = io.BytesIO()
        im.save(buffered, format="PNG", quality=100)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = "data:image/png;base64," + encoded
        image = f"<img src='{url}'></img>"

        bounds = "<br/>".join(
            [
                vedo.utils.precision(min_x, 4) + " ... " + vedo.utils.precision(max_x, 4)
                for min_x, max_x in zip(self.bounds()[::2], self.bounds()[1::2])
            ]
        )

        help_text = ""
        if self.name:
            help_text += f"<b> {self.name}: &nbsp&nbsp</b>"
        help_text += '<b><a href="' + help_url + '" target="_blank">' + library_name + "</a></b>"
        if self.filename:
            dots = ""
            if len(self.filename) > 30:
                dots = "..."
            help_text += f"<br/><code><i>({dots}{self.filename[-30:]})</i></code>"

        all = [
            "<table>",
            "<tr>",
            "<td>",
            image,
            "</td>",
            "<td style='text-align: center; vertical-align: center;'><br/>",
            help_text,
            "<table>",
            "<tr><td><b> nr. of parts </b></td><td>" + str(self.GetNumberOfPaths()) + "</td></tr>",
            "<tr><td><b> position </b></td><td>" + str(self.GetPosition()) + "</td></tr>",
            "<tr><td><b> x-limits </b></td><td>" + utils.precision(self.xlim, 4) + "</td></tr>",
            "<tr><td><b> y-limits </b></td><td>" + utils.precision(self.ylim, 4) + "</td></tr>",
            "<tr><td><b> world bounds </b> <br/> (x/y/z) </td><td>" + str(bounds) + "</td></tr>",
            "</table>",
            "</table>",
        ]
        return "\n".join(all)

    def __add__(self, *obj):
        # just to avoid confusion, supersede Assembly.__add__
        return self.__iadd__(*obj)

    def __iadd__(self, *obj):
        if len(obj) == 1 and isinstance(obj[0], Figure):
            return self._check_unpack_and_insert(obj[0])

        obj = utils.flatten(obj)
        return self.insert(*obj)

    def _check_unpack_and_insert(self, fig):

        if fig.label:
            self.labels.append(fig.label)

        if abs(self.yscale - fig.yscale) > 0.0001:

            colors.printc(":bomb:ERROR: adding incompatible Figure. Y-scales are different:", c='r', invert=True)
            colors.printc("  first  figure:", self.yscale, c='r')
            colors.printc("  second figure:", fig.yscale, c='r')

            colors.printc("One or more of these parameters can be the cause:", c="r")
            if list(self.xlim) != list(fig.xlim):
                colors.printc("xlim --------------------------------------------\n",
                              " first  figure:", self.xlim, "\n",
                              " second figure:", fig.xlim, c='r')
            if list(self.ylim) != list(fig.ylim):
                colors.printc("ylim --------------------------------------------\n",
                              " first  figure:", self.ylim, "\n",
                              " second figure:", fig.ylim, c='r')
            if list(self.padding) != list(fig.padding):
                colors.printc("padding -----------------------------------------\n",
                              " first  figure:", self.padding,
                              " second figure:", fig.padding, c='r')
            if self.aspect != fig.aspect:
                colors.printc("aspect ------------------------------------------\n",
                              " first  figure:", self.aspect, "\n",
                              " second figure:", fig.aspect, c='r')

            colors.printc("\n:idea: Consider using fig2 = histogram(..., like=fig1)", c="r")
            colors.printc(" Or fig += histogram(..., like=fig)\n", c="r")
            return self

        offset = self.zbounds()[1] + self.ztolerance

        for ele in fig.unpack():
            if "Axes" in ele.name:
                continue
            ele.z(offset)
            self.insert(ele, rescale=False)

        return self

    def insert(self, *objs, rescale=True, as3d=True, adjusted=False, cut=True):
        """
        Insert objects into a Figure.

        The recommended syntax is to use "+=", which calls `insert()` under the hood.
        If a whole Figure is added with "+=", it is unpacked and its objects are added
        one by one.

        Arguments:
            rescale : (bool)
                rescale the y axis position while inserting the object.
            as3d : (bool)
                if True keep the aspect ratio of the 3d object, otherwise stretch it in y.
            adjusted : (bool)
                adjust the scaling according to the shortest axis
            cut : (bool)
                cut off the parts of the object which go beyond the axes frame.
        """
        for a in objs:

            if a in self.objects:
                # should not add twice the same object in plot
                continue

            if isinstance(a, vedo.Points):  # hacky way to identify Points
                if a.ncells == a.npoints:
                    poly = a.dataset
                    if poly.GetNumberOfPolys() == 0 and poly.GetNumberOfLines() == 0:
                        as3d = False
                        rescale = True

            if isinstance(a, (shapes.Arrow, shapes.Arrow2D)):
                # discard input Arrow and substitute it with a brand new one
                # (because scaling would fatally distort the shape)

                py = a.base[1]
                a.top[1] = (a.top[1] - py) * self.yscale + py
                b = shapes.Arrow2D(a.base, a.top, s=a.s, fill=a.fill).z(a.z())

                prop = a.properties
                prop.LightingOff()
                b.actor.SetProperty(prop)
                b.properties = prop
                b.y(py * self.yscale)
                a = b

            # elif isinstance(a, shapes.Rectangle) and a.radius is not None:
            #     # discard input Rectangle and substitute it with a brand new one
            #     # (because scaling would fatally distort the shape of the corners)
            #     py = a.corner1[1]
            #     rx1,ry1,rz1 = a.corner1
            #     rx2,ry2,rz2 = a.corner2
            #     ry2 = (ry2-py) * self.yscale + py
            #     b = shapes.Rectangle([rx1,0,rz1], [rx2,ry2,rz2], radius=a.radius).z(a.z())
            #     b.SetProperty(a.properties)
            #     b.y(py / self.yscale)
            #     a = b

            else:

                if rescale:

                    if not isinstance(a, Figure):

                        if as3d and not isinstance(a, self.force_scaling_types):
                            if adjusted:
                                scl = np.min([1, self.yscale])
                            else:
                                scl = self.yscale

                            a.scale(scl)

                        else:
                            a.scale([1, self.yscale, 1])

                    # shift it in y
                    a.y(a.y() * self.yscale)

            if cut:
                try:
                    bx0, bx1, by0, by1, _, _ = a.bounds()
                    if self.y0lim > by0:
                        a.cut_with_plane([0, self.y0lim, 0], [0, 1, 0])
                    if self.y1lim < by1:
                        a.cut_with_plane([0, self.y1lim, 0], [0, -1, 0])
                    if self.x0lim > bx0:
                        a.cut_with_plane([self.x0lim, 0, 0], [1, 0, 0])
                    if self.x1lim < bx1:
                        a.cut_with_plane([self.x1lim, 0, 0], [-1, 0, 0])
                except:
                    # print("insert(): cannot cut", [a])
                    pass

            self.AddPart(a.actor)
            self.objects.append(a)

        return self

    def add_label(self, text, c=None, marker="", mc="black"):
        """
        Manually add en entry label to the legend.

        Arguments:
            text : (str)
                text string for the label.
            c : (str)
                color of the text
            marker : (str), Mesh
                a marker char or a Mesh object to be used as marker
            mc : (str)
                color for the marker
        """
        newlabel = LabelData()
        newlabel.text = text.replace("\n", " ")
        newlabel.tcolor = c
        newlabel.marker = marker
        newlabel.mcolor = mc
        self.labels.append(newlabel)
        return self

    def add_legend(
        self,
        pos="top-right",
        relative=True,
        font=None,
        s=1,
        c=None,
        vspace=1.75,
        padding=0.1,
        radius=0,
        alpha=1,
        bc="k7",
        lw=1,
        lc="k4",
        z=0,
    ):
        """
        Add existing labels to form a legend box.
        Labels have been previously filled with eg: `plot(..., label="text")`

        Arguments:
            pos : (str, list)
                A string or 2D coordinates. The default is "top-right".
            relative : (bool)
                control whether `pos` is absolute or relative, e.i. normalized
                to the x and y ranges so that x and y in `pos=[x,y]` should be
                both in the range [0,1].
                This flag is ignored if a string despcriptor is passed.
                Default is True.
            font : (str, int)
                font name or number.
                Check [available fonts here](https://vedo.embl.es/fonts).
            s : (float)
                global size of the legend
            c : (str)
                color of the text
            vspace : (float)
                vertical spacing of lines
            padding : (float)
                padding of the box as a fraction of the text size
            radius : (float)
                border radius of the box
            alpha : (float)
                opacity of the box. Values below 1 may cause poor rendering
                because of antialiasing.
                Use alpha = 0 to remove the box.
            bc : (str)
                box color
            lw : (int)
                border line width of the box in pixel units
            lc : (int)
                border line color of the box
            z : (float)
                set the zorder as z position (useful to avoid overlap)
        """
        sx = self.x1lim - self.x0lim
        s = s * sx / 55  # so that input can be about 1

        ds = 0
        texts = []
        mks = []
        for i, t in enumerate(self.labels):
            label = self.labels[i]
            t = label.text

            if label.tcolor is not None:
                c = label.tcolor

            tx = vedo.shapes.Text3D(t, s=s, c=c, justify="center-left", font=font)
            y0, y1 = tx.ybounds()
            ds = max(y1 - y0, ds)
            texts.append(tx)

            mk = label.marker
            if isinstance(mk, vedo.Points):
                mk = mk.clone(deep=False).lighting("off")
                cm = mk.center_of_mass()
                ty0, ty1 = tx.ybounds()
                oby0, oby1 = mk.ybounds()
                mk.shift(-cm)
                mk.SetOrigin(cm)
                mk.scale((ty1 - ty0) / (oby1 - oby0))
                mk.scale([1.1, 1.1, 0.01])
            elif mk == "-":
                mk = vedo.shapes.Marker(mk, s=s * 2)
                mk.color(label.mcolor)
            else:
                mk = vedo.shapes.Marker(mk, s=s)
                mk.color(label.mcolor)
            mks.append(mk)

        for i, tx in enumerate(texts):
            tx.shift(0, -(i + 0) * ds * vspace)

        for i, mk in enumerate(mks):
            mk.shift(-ds * 1.75, -(i + 0) * ds * vspace, 0)

        acts = texts + mks

        aleg = Assembly(acts)  # .show(axes=1).close()
        x0, x1, y0, y1, _, _ = aleg.GetBounds()

        if alpha:
            dx = x1 - x0
            dy = y1 - y0

            if not utils.is_sequence(padding):
                padding = [padding] * 4
            padding = min(padding)
            padding = min(padding * dx, padding * dy)
            if len(self.labels) == 1:
                padding *= 4
            x0 -= padding
            x1 += padding
            y0 -= padding
            y1 += padding

            box = shapes.Rectangle([x0, y0], [x1, y1], radius=radius, c=bc, alpha=alpha)
            box.shift(0, 0, -dy / 100).pickable(False)
            if lc:
                box.lc(lc).lw(lw)
            aleg.AddPart(box.actor)
            aleg.objects.append(box)

        xlim = self.xlim
        ylim = self.ylim
        if isinstance(pos, str):
            px, py = 0, 0
            rx, ry = (xlim[1] + xlim[0]) / 2, (ylim[1] + ylim[0]) / 2
            shx, shy = 0, 0
            if "top" in pos:
                if "cent" in pos:
                    px, py = rx, ylim[1]
                    shx, shy = (x0 + x1) / 2, y1
                elif "left" in pos:
                    px, py = xlim[0], ylim[1]
                    shx, shy = x0, y1
                else:  # "right"
                    px, py = xlim[1], ylim[1]
                    shx, shy = x1, y1
            elif "bot" in pos:
                if "left" in pos:
                    px, py = xlim[0], ylim[0]
                    shx, shy = x0, y0
                elif "right" in pos:
                    px, py = xlim[1], ylim[0]
                    shx, shy = x1, y0
                else:  # "cent"
                    px, py = rx, ylim[0]
                    shx, shy = (x0 + x1) / 2, y0
            elif "cent" in pos:
                if "left" in pos:
                    px, py = xlim[0], ry
                    shx, shy = x0, (y0 + y1) / 2
                elif "right" in pos:
                    px, py = xlim[1], ry
                    shx, shy = x1, (y0 + y1) / 2
            else:
                vedo.logger.error(f"in add_legend(), cannot understand {pos}")
                raise RuntimeError

        else:

            if relative:
                rx, ry = pos[0], pos[1]
                px = (xlim[1] - xlim[0]) * rx + xlim[0]
                py = (ylim[1] - ylim[0]) * ry + ylim[0]
                z *= xlim[1] - xlim[0]
            else:
                px, py = pos[0], pos[1]
            shx, shy = x0, y1

        zpos = aleg.pos()[2]
        aleg.pos(px - shx, py * self.yscale - shy, zpos + sx / 50 + z)

        self.insert(aleg, rescale=False, cut=False)
        self.legend = aleg
        aleg.name = "Legend"
        return self


#########################################################################################
class Histogram1D(Figure):
    "1D histogramming."

    def __init__(
        self,
        data,
        weights=None,
        bins=None,
        errors=False,
        density=False,
        logscale=False,
        max_entries=None,
        fill=True,
        radius=0.075,
        c="olivedrab",
        gap=0.0,
        alpha=1,
        outline=False,
        lw=2,
        lc="k",
        texture="",
        marker="",
        ms=None,
        mc=None,
        ma=None,
        # Figure and axes options:
        like=None,
        xlim=None,
        ylim=(0, None),
        aspect=4 / 3,
        padding=(0.0, 0.0, 0.0, 0.05),
        title="",
        xtitle=" ",
        ytitle=" ",
        ac="k",
        grid=False,
        ztolerance=None,
        label="",
        **fig_kwargs,
    ):
        """
        Creates a `Histogram1D(Figure)` object.

        Arguments:
            weights : (list)
                An array of weights, of the same shape as `data`. Each value in `data`
                only contributes its associated weight towards the bin count (instead of 1).
            bins : (int)
                number of bins
            density : (bool)
                normalize the area to 1 by dividing by the nr of entries and bin size
            logscale : (bool)
                use logscale on y-axis
            max_entries : (int)
                if `data` is larger than `max_entries`, a random sample of `max_entries` is used
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
            padding : (float), list
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
        """

        if max_entries and data.shape[0] > max_entries:
            data = np.random.choice(data, int(max_entries))

        # purge NaN from data
        valid_ids = np.all(np.logical_not(np.isnan(data)))
        data = np.asarray(data[valid_ids]).ravel()

        # if data.dtype is integer try to center bins by default
        if like is None and bins is None and np.issubdtype(data.dtype, np.integer):
            if xlim is None and ylim == (0, None):
                x1, x0 = data.max(), data.min()
                if 0 < x1 - x0 <= 100:
                    bins = x1 - x0 + 1
                    xlim = (x0 - 0.5, x1 + 0.5)

        if like is None and vedo.last_figure is not None:
            if xlim is None and ylim == (0, None):
                like = vedo.last_figure

        if like is not None:
            xlim = like.xlim
            ylim = like.ylim
            aspect = like.aspect
            padding = like.padding
            if bins is None:
                bins = like.bins
        if bins is None:
            bins = 20

        if utils.is_sequence(xlim):
            # deal with user passing eg [x0, None]
            _x0, _x1 = xlim
            if _x0 is None:
                _x0 = data.min()
            if _x1 is None:
                _x1 = data.max()
            xlim = [_x0, _x1]

        fs, edges = np.histogram(data, bins=bins, weights=weights, range=xlim)
        binsize = edges[1] - edges[0]
        ntot = data.shape[0]

        fig_kwargs["title"] = title
        fig_kwargs["xtitle"] = xtitle
        fig_kwargs["ytitle"] = ytitle
        fig_kwargs["ac"] = ac
        fig_kwargs["ztolerance"] = ztolerance
        fig_kwargs["grid"] = grid

        unscaled_errors = np.sqrt(fs)
        if density:
            scaled_errors = unscaled_errors / (ntot * binsize)
            fs = fs / (ntot * binsize)
            if ytitle == " ":
                ytitle = f"counts / ({ntot} x {utils.precision(binsize,3)})"
                fig_kwargs["ytitle"] = ytitle
        elif logscale:
            se_up = np.log10(fs + unscaled_errors / 2 + 1)
            se_dw = np.log10(fs - unscaled_errors / 2 + 1)
            scaled_errors = np.c_[se_up, se_dw]
            fs = np.log10(fs + 1)
            if ytitle == " ":
                ytitle = "log_10 (counts+1)"
                fig_kwargs["ytitle"] = ytitle

        x0, x1 = np.min(edges), np.max(edges)
        y0, y1 = ylim[0], np.max(fs)

        _errors = []
        if errors:
            if density:
                y1 += max(scaled_errors) / 2
                _errors = scaled_errors
            elif logscale:
                y1 = max(scaled_errors[:, 0])
                _errors = scaled_errors
            else:
                y1 += max(unscaled_errors) / 2
                _errors = unscaled_errors

        if like is None:
            ylim = list(ylim)
            if xlim is None:
                xlim = [x0, x1]
            if ylim[1] is None:
                ylim[1] = y1
            if ylim[0] != 0:
                ylim[0] = y0

        self.title = title
        self.xtitle = xtitle
        self.ytitle = ytitle
        self.entries = ntot
        self.frequencies = fs
        self.errors = _errors
        self.edges = edges
        self.centers = (edges[0:-1] + edges[1:]) / 2
        self.mean = data.mean()
        self.mode = self.centers[np.argmax(fs)]
        self.std = data.std()
        self.bins = edges  # internally used by "like"

        ############################### stats legend as htitle
        addstats = False
        if not title:
            if "axes" not in fig_kwargs:
                addstats = True
                axes_opts = {}
                fig_kwargs["axes"] = axes_opts
            elif fig_kwargs["axes"] is False:
                pass
            else:
                axes_opts = fig_kwargs["axes"]
                if "htitle" not in axes_opts:
                    addstats = True

        if addstats:
            htitle = f"Entries:~~{int(self.entries)}  "
            htitle += f"Mean:~~{utils.precision(self.mean, 4)}  "
            htitle += f"STD:~~{utils.precision(self.std, 4)}  "

            axes_opts["htitle"] = htitle
            axes_opts["htitle_justify"] = "bottom-left"
            axes_opts["htitle_size"] = 0.016
            # axes_opts["htitle_offset"] = [-0.49, 0.01, 0]

        if mc is None:
            mc = lc
        if ma is None:
            ma = alpha

        if label:
            nlab = LabelData()
            nlab.text = label
            nlab.tcolor = ac
            nlab.marker = marker
            nlab.mcolor = mc
            if not marker:
                nlab.marker = "s"
                nlab.mcolor = c
            fig_kwargs["label"] = nlab

        ############################################### Figure init
        super().__init__(xlim, ylim, aspect, padding, **fig_kwargs)

        if not self.yscale:
            return

        if utils.is_sequence(bins):
            myedges = np.array(bins)
            bins = len(bins) - 1
        else:
            myedges = edges

        bin_centers = []
        for i in range(bins):
            x = (myedges[i] + myedges[i + 1]) / 2
            bin_centers.append([x, fs[i], 0])

        rs = []
        maxheigth = 0
        if not fill and not outline and not errors and not marker:
            outline = True  # otherwise it's empty..

        if fill:  #####################
            if outline:
                gap = 0

            for i in range(bins):
                F = fs[i]
                if not F:
                    continue
                p0 = (myedges[i] + gap * binsize, 0, 0)
                p1 = (myedges[i + 1] - gap * binsize, F, 0)

                if radius:
                    if gap:
                        rds = np.array([0, 0, radius, radius])
                    else:
                        rd1 = 0 if i < bins - 1 and fs[i + 1] >= F else radius / 2
                        rd2 = 0 if i > 0 and fs[i - 1] >= F else radius / 2
                        rds = np.array([0, 0, rd1, rd2])
                    p1_yscaled = [p1[0], p1[1] * self.yscale, 0]
                    r = shapes.Rectangle(p0, p1_yscaled, radius=rds * binsize, res=6)
                    r.scale([1, 1 / self.yscale, 1])
                    r.radius = None  # so it doesnt get recreated and rescaled by insert()
                else:
                    r = shapes.Rectangle(p0, p1)

                if texture:
                    r.texture(texture)
                    c = "w"

                r.actor.PickableOff()
                maxheigth = max(maxheigth, p1[1])
                if c in colors.cmaps_names:
                    col = colors.color_map((p0[0] + p1[0]) / 2, c, myedges[0], myedges[-1])
                else:
                    col = c
                r.color(col).alpha(alpha).lighting("off")
                r.z(self.ztolerance)
                rs.append(r)

        if outline:  #####################
            lns = [[myedges[0], 0, 0]]
            for i in range(bins):
                lns.append([myedges[i], fs[i], 0])
                lns.append([myedges[i + 1], fs[i], 0])
                maxheigth = max(maxheigth, fs[i])
            lns.append([myedges[-1], 0, 0])
            outl = shapes.Line(lns, c=lc, alpha=alpha, lw=lw)
            outl.z(self.ztolerance * 2)
            rs.append(outl)

        if errors:  #####################
            for i in range(bins):
                x = self.centers[i]
                f = fs[i]
                if not f:
                    continue
                err = _errors[i]
                if utils.is_sequence(err):
                    el = shapes.Line([x, err[0], 0], [x, err[1], 0], c=lc, alpha=alpha, lw=lw)
                else:
                    el = shapes.Line(
                        [x, f - err / 2, 0], [x, f + err / 2, 0], c=lc, alpha=alpha, lw=lw
                    )
                el.z(self.ztolerance * 3)
                rs.append(el)

        if marker:  #####################

            # remove empty bins (we dont want a marker there)
            bin_centers = np.array(bin_centers)
            bin_centers = bin_centers[bin_centers[:, 1] > 0]

            if utils.is_sequence(ms):  ### variable point size
                mk = shapes.Marker(marker, s=1)
                mk.scale([1, 1 / self.yscale, 1])
                msv = np.zeros_like(bin_centers)
                msv[:, 0] = ms
                marked = shapes.Glyph(
                    bin_centers, mk, c=mc, orientation_array=msv, scale_by_vector_size=True
                )
            else:  ### fixed point size

                if ms is None:
                    ms = (xlim[1] - xlim[0]) / 100.0
                else:
                    ms = (xlim[1] - xlim[0]) / 100.0 * ms

                if utils.is_sequence(mc):
                    mk = shapes.Marker(marker, s=ms)
                    mk.scale([1, 1 / self.yscale, 1])
                    msv = np.zeros_like(bin_centers)
                    msv[:, 0] = 1
                    marked = shapes.Glyph(
                        bin_centers, mk, c=mc, orientation_array=msv, scale_by_vector_size=True
                    )
                else:
                    mk = shapes.Marker(marker, s=ms)
                    mk.scale([1, 1 / self.yscale, 1])
                    marked = shapes.Glyph(bin_centers, mk, c=mc)

            marked.alpha(ma)
            marked.z(self.ztolerance * 4)
            rs.append(marked)

        self.insert(*rs, as3d=False)
        self.name = "Histogram1D"

    def print(self, **kwargs):
        """Print infos about this histogram"""
        txt = (
            f"{self.name}  {self.title}\n"
            f"    xtitle  = '{self.xtitle}'\n"
            f"    ytitle  = '{self.ytitle}'\n"
            f"    entries = {self.entries}\n"
            f"    mean    = {self.mean}\n"
            f"    std     = {self.std}"
        )
        colors.printc(txt, **kwargs)


#########################################################################################
class Histogram2D(Figure):
    """2D histogramming."""

    def __init__(
        self,
        xvalues,
        yvalues=None,
        bins=25,
        weights=None,
        cmap="cividis",
        alpha=1,
        gap=0,
        scalarbar=True,
        # Figure and axes options:
        like=None,
        xlim=None,
        ylim=(None, None),
        zlim=(None, None),
        aspect=1,
        title="",
        xtitle=" ",
        ytitle=" ",
        ztitle="",
        ac="k",
        **fig_kwargs,
    ):
        """
        Input data formats `[(x1,x2,..), (y1,y2,..)] or [(x1,y1), (x2,y2),..]`
        are both valid.

        Use keyword `like=...` if you want to use the same format of a previously
        created Figure (useful when superimposing Figures) to make sure
        they are compatible and comparable. If they are not compatible
        you will receive an error message.

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
                separation between adjacent bins as a fraction for their size
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
        """
        xvalues = np.asarray(xvalues)
        if yvalues is None:
            # assume [(x1,y1), (x2,y2) ...] format
            yvalues = xvalues[:, 1]
            xvalues = xvalues[:, 0]
        else:
            yvalues = np.asarray(yvalues)

        padding = [0, 0, 0, 0]

        if like is None and vedo.last_figure is not None:
            if xlim is None and ylim == (None, None) and zlim == (None, None):
                like = vedo.last_figure

        if like is not None:
            xlim = like.xlim
            ylim = like.ylim
            aspect = like.aspect
            padding = like.padding
            if bins is None:
                bins = like.bins
        if bins is None:
            bins = 20

        if isinstance(bins, int):
            bins = (bins, bins)

        if utils.is_sequence(xlim):
            # deal with user passing eg [x0, None]
            _x0, _x1 = xlim
            if _x0 is None:
                _x0 = xvalues.min()
            if _x1 is None:
                _x1 = xvalues.max()
            xlim = [_x0, _x1]

        if utils.is_sequence(ylim):
            # deal with user passing eg [x0, None]
            _y0, _y1 = ylim
            if _y0 is None:
                _y0 = yvalues.min()
            if _y1 is None:
                _y1 = yvalues.max()
            ylim = [_y0, _y1]

        H, xedges, yedges = np.histogram2d(
            xvalues, yvalues, weights=weights, bins=bins, range=(xlim, ylim)
        )

        xlim = np.min(xedges), np.max(xedges)
        ylim = np.min(yedges), np.max(yedges)
        dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]

        fig_kwargs["title"] = title
        fig_kwargs["xtitle"] = xtitle
        fig_kwargs["ytitle"] = ytitle
        fig_kwargs["ac"] = ac

        self.entries = len(xvalues)
        self.frequencies = H
        self.edges = (xedges, yedges)
        self.mean = (xvalues.mean(), yvalues.mean())
        self.std = (xvalues.std(), yvalues.std())
        self.bins = bins  # internally used by "like"

        ############################### stats legend as htitle
        addstats = False
        if not title:
            if "axes" not in fig_kwargs:
                addstats = True
                axes_opts = {}
                fig_kwargs["axes"] = axes_opts
            elif fig_kwargs["axes"] is False:
                pass
            else:
                axes_opts = fig_kwargs["axes"]
                if "htitle" not in fig_kwargs["axes"]:
                    addstats = True

        if addstats:
            htitle = f"Entries:~~{int(self.entries)}  "
            htitle += f"Mean:~~{utils.precision(self.mean, 3)}  "
            htitle += f"STD:~~{utils.precision(self.std, 3)}  "
            axes_opts["htitle"] = htitle
            axes_opts["htitle_justify"] = "bottom-left"
            axes_opts["htitle_size"] = 0.0175

        ############################################### Figure init
        super().__init__(xlim, ylim, aspect, padding, **fig_kwargs)

        if self.yscale:
            ##################### the grid
            acts = []
            g = shapes.Grid(
                pos=[(xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2, 0], s=(dx, dy), res=bins[:2]
            )
            g.alpha(alpha).lw(0).wireframe(False).flat().lighting("off")
            g.cmap(cmap, np.ravel(H.T), on="cells", vmin=zlim[0], vmax=zlim[1])
            if gap:
                g.shrink(abs(1 - gap))

            if scalarbar:
                sc = g.add_scalarbar3d(ztitle, c=ac).scalarbar

                # print(" g.GetBounds()[0]", g.bounds()[:2])
                # print("sc.GetBounds()[0]",sc.GetBounds()[:2])
                delta = sc.GetBounds()[0] - g.bounds()[1]

                sc_size = sc.GetBounds()[1] - sc.GetBounds()[0]

                sc.SetOrigin(sc.GetBounds()[0], 0, 0)
                sc.scale([self.yscale, 1, 1])  ## prescale trick
                sc.shift(-delta + 0.25*sc_size*self.yscale)

                acts.append(sc)
            acts.append(g)

            self.insert(*acts, as3d=False)
            self.name = "Histogram2D"


#########################################################################################
class PlotBars(Figure):
    """Creates a `PlotBars(Figure)` object."""

    def __init__(
        self,
        data,
        errors=False,
        logscale=False,
        fill=True,
        gap=0.02,
        radius=0.05,
        c="olivedrab",
        alpha=1,
        texture="",
        outline=False,
        lw=2,
        lc="k",
        # Figure and axes options:
        like=None,
        xlim=(None, None),
        ylim=(0, None),
        aspect=4 / 3,
        padding=(0.025, 0.025, 0, 0.05),
        #
        title="",
        xtitle=" ",
        ytitle=" ",
        ac="k",
        grid=False,
        ztolerance=None,
        **fig_kwargs,
    ):
        """
        Input must be in format `[counts, labels, colors, edges]`.
        Either or both `edges` and `colors` are optional and can be omitted.

        Use keyword `like=...` if you want to use the same format of a previously
        created Figure (useful when superimposing Figures) to make sure
        they are compatible and comparable. If they are not compatible
        you will receive an error message.

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
            - [plot_bars.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_bars.py)

               ![](https://vedo.embl.es/images/pyplot/plot_bars.png)
        """
        ndata = len(data)
        if ndata == 4:
            counts, xlabs, cols, edges = data
        elif ndata == 3:
            counts, xlabs, cols = data
            edges = np.array(range(len(counts) + 1)) + 0.5
        elif ndata == 2:
            counts, xlabs = data
            edges = np.array(range(len(counts) + 1)) + 0.5
            cols = [c] * len(counts)
        else:
            m = "barplot error: data must be given as [counts, labels, colors, edges] not\n"
            vedo.logger.error(m + f" {data}\n     bin edges and colors are optional.")
            raise RuntimeError()

        # sanity checks
        assert len(counts) == len(xlabs)
        assert len(counts) == len(cols)
        assert len(counts) == len(edges) - 1

        counts = np.asarray(counts)
        edges = np.asarray(edges)

        if logscale:
            counts = np.log10(counts + 1)
            if ytitle == " ":
                ytitle = "log_10 (counts+1)"

        if like is None and vedo.last_figure is not None:
            if xlim == (None, None) and ylim == (0, None):
                like = vedo.last_figure

        if like is not None:
            xlim = like.xlim
            ylim = like.ylim
            aspect = like.aspect
            padding = like.padding

        if utils.is_sequence(xlim):
            # deal with user passing eg [x0, None]
            _x0, _x1 = xlim
            if _x0 is None:
                _x0 = np.min(edges)
            if _x1 is None:
                _x1 = np.max(edges)
            xlim = [_x0, _x1]

        x0, x1 = np.min(edges), np.max(edges)
        y0, y1 = ylim[0], np.max(counts)

        if like is None:
            ylim = list(ylim)
            if xlim is None:
                xlim = [x0, x1]
            if ylim[1] is None:
                ylim[1] = y1
            if ylim[0] != 0:
                ylim[0] = y0

        fig_kwargs["title"] = title
        fig_kwargs["xtitle"] = xtitle
        fig_kwargs["ytitle"] = ytitle
        fig_kwargs["ac"] = ac
        fig_kwargs["ztolerance"] = ztolerance
        fig_kwargs["grid"] = grid

        centers = (edges[0:-1] + edges[1:]) / 2
        binsizes = (centers - edges[0:-1]) * 2

        if "axes" not in fig_kwargs:
            fig_kwargs["axes"] = {}

        _xlabs = []
        for center, xlb in zip(centers, xlabs):
            _xlabs.append([center, str(xlb)])
        fig_kwargs["axes"]["x_values_and_labels"] = _xlabs

        ############################################### Figure
        self.statslegend = ""
        self.edges = edges
        self.centers = centers
        self.bins = edges  # internal used by "like"
        super().__init__(xlim, ylim, aspect, padding, **fig_kwargs)
        if not self.yscale:
            return

        rs = []
        maxheigth = 0
        if fill:  #####################
            if outline:
                gap = 0

            for i in range(len(centers)):
                binsize = binsizes[i]
                p0 = (edges[i] + gap * binsize, 0, 0)
                p1 = (edges[i + 1] - gap * binsize, counts[i], 0)

                if radius:
                    rds = np.array([0, 0, radius, radius])
                    p1_yscaled = [p1[0], p1[1] * self.yscale, 0]
                    r = shapes.Rectangle(p0, p1_yscaled, radius=rds * binsize, res=6)
                    r.scale([1, 1 / self.yscale, 1])
                    r.radius = None  # so it doesnt get recreated and rescaled by insert()
                else:
                    r = shapes.Rectangle(p0, p1)

                if texture:
                    r.texture(texture)
                    c = "w"

                r.actor.PickableOff()
                maxheigth = max(maxheigth, p1[1])
                if c in colors.cmaps_names:
                    col = colors.color_map((p0[0] + p1[0]) / 2, c, edges[0], edges[-1])
                else:
                    col = cols[i]
                r.color(col).alpha(alpha).lighting("off")
                r.name = f"bar_{i}"
                r.z(self.ztolerance)
                rs.append(r)

        elif outline:  #####################
            lns = [[edges[0], 0, 0]]
            for i in range(len(centers)):
                lns.append([edges[i], counts[i], 0])
                lns.append([edges[i + 1], counts[i], 0])
                maxheigth = max(maxheigth, counts[i])
            lns.append([edges[-1], 0, 0])
            outl = shapes.Line(lns, c=lc, alpha=alpha, lw=lw).z(self.ztolerance)
            outl.name = f"bar_outline_{i}"
            rs.append(outl)

        if errors:  #####################
            for x, f in centers:
                err = np.sqrt(f)
                el = shapes.Line([x, f - err / 2, 0], [x, f + err / 2, 0], c=lc, alpha=alpha, lw=lw)
                el.z(self.ztolerance * 2)
                rs.append(el)

        self.insert(*rs, as3d=False)
        self.name = "PlotBars"


#########################################################################################
class PlotXY(Figure):
    """Creates a `PlotXY(Figure)` object."""

    def __init__(
        self,
        #
        data,
        xerrors=None,
        yerrors=None,
        #
        lw=2,
        lc=None,
        la=1,
        dashed=False,
        splined=False,
        #
        elw=2,  # error line width
        ec=None,  # error line or band color
        error_band=False,  # errors in x are ignored
        #
        marker="",
        ms=None,
        mc=None,
        ma=None,
        # Figure and axes options:
        like=None,
        xlim=None,
        ylim=(None, None),
        aspect=4 / 3,
        padding=0.05,
        #
        title="",
        xtitle=" ",
        ytitle=" ",
        ac="k",
        grid=True,
        ztolerance=None,
        label="",
        **fig_kwargs,
    ):
        """
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
            aspect : (float, str)
                Desired aspect ratio.
                Use `aspect="equal"` to force the same units in x and y.
                Scaling factor is saved in Figure.yscale.
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
            x = np.arange(0, np.pi, 0.1)
            fig = plot(x, np.sin(2*x), 'r0-', aspect='equal')
            fig+= plot(x, np.cos(2*x), 'blue4 o-', like=fig)
            fig.show().close()
            ```
            ![](https://vedo.embl.es/images/feats/plotxy.png)

        Examples:
            - [plot_errbars.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_errbars.py)
            - [plot_errband.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_errband.py)
            - [plot_pip.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_pip.py)

                ![](https://vedo.embl.es/images/pyplot/plot_pip.png)

            - [scatter1.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/scatter1.py)
            - [scatter2.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/scatter2.py)

                ![](https://vedo.embl.es/images/pyplot/scatter2.png)
        """
        line = False
        if lw > 0:
            line = True
        if marker == "" and not line and not splined:
            marker = "o"

        if like is None and vedo.last_figure is not None:
            if xlim is None and ylim == (None, None):
                like = vedo.last_figure

        if like is not None:
            xlim = like.xlim
            ylim = like.ylim
            aspect = like.aspect
            padding = like.padding

        if utils.is_sequence(xlim):
            # deal with user passing eg [x0, None]
            _x0, _x1 = xlim
            if _x0 is None:
                _x0 = data.min()
            if _x1 is None:
                _x1 = data.max()
            xlim = [_x0, _x1]

        # purge NaN from data
        validIds = np.all(np.logical_not(np.isnan(data)))
        data = np.array(data[validIds])[0]

        fig_kwargs["title"] = title
        fig_kwargs["xtitle"] = xtitle
        fig_kwargs["ytitle"] = ytitle
        fig_kwargs["ac"] = ac
        fig_kwargs["ztolerance"] = ztolerance
        fig_kwargs["grid"] = grid

        x0, y0 = np.min(data, axis=0)
        x1, y1 = np.max(data, axis=0)
        if xerrors is not None and not error_band:
            x0 = min(data[:, 0] - xerrors)
            x1 = max(data[:, 0] + xerrors)
        if yerrors is not None:
            y0 = min(data[:, 1] - yerrors)
            y1 = max(data[:, 1] + yerrors)

        if like is None:
            if xlim is None:
                xlim = (None, None)
            xlim = list(xlim)
            if xlim[0] is None:
                xlim[0] = x0
            if xlim[1] is None:
                xlim[1] = x1
            ylim = list(ylim)
            if ylim[0] is None:
                ylim[0] = y0
            if ylim[1] is None:
                ylim[1] = y1

        self.entries = len(data)
        self.mean = data.mean()
        self.std = data.std()
        
        self.ztolerance = 0
        
        ######### the PlotXY marker
        # fall back solutions logic for colors
        if "c" in fig_kwargs:
            if mc is None:
                mc = fig_kwargs["c"]
            if lc is None:
                lc = fig_kwargs["c"]
            if ec is None:
                ec = fig_kwargs["c"]
        if lc is None:
            lc = "k"
        if mc is None:
            mc = lc
        if ma is None:
            ma = la
        if ec is None:
            if mc is None:
                ec = lc
            else:
                ec = mc

        if label:
            nlab = LabelData()
            nlab.text = label
            nlab.tcolor = ac
            nlab.marker = marker
            if line and marker == "":
                nlab.marker = "-"
            nlab.mcolor = mc
            fig_kwargs["label"] = nlab

        ############################################### Figure init
        super().__init__(xlim, ylim, aspect, padding, **fig_kwargs)

        if not self.yscale:
            return

        acts = []

        ######### the PlotXY Line or Spline
        if dashed:
            l = shapes.DashedLine(data, c=lc, alpha=la, lw=lw)
            acts.append(l)
        elif splined:
            l = shapes.KSpline(data).lw(lw).c(lc).alpha(la)
            acts.append(l)
        elif line:
            l = shapes.Line(data, c=lc, alpha=la).lw(lw)
            acts.append(l)

        if marker:

            pts = np.c_[data, np.zeros(len(data))]

            if utils.is_sequence(ms):
                ### variable point size
                mk = shapes.Marker(marker, s=1)
                mk.scale([1, 1 / self.yscale, 1])
                msv = np.zeros_like(pts)
                msv[:, 0] = ms
                marked = shapes.Glyph(
                    pts, mk, c=mc, orientation_array=msv, scale_by_vector_size=True
                )
            else:
                ### fixed point size
                if ms is None:
                    ms = (xlim[1] - xlim[0]) / 100.0

                if utils.is_sequence(mc):
                    fig_kwargs["marker_color"] = None  # for labels
                    mk = shapes.Marker(marker, s=ms)
                    mk.scale([1, 1 / self.yscale, 1])
                    msv = np.zeros_like(pts)
                    msv[:, 0] = 1
                    marked = shapes.Glyph(
                        pts, mk, c=mc, orientation_array=msv, scale_by_vector_size=True
                    )
                else:
                    mk = shapes.Marker(marker, s=ms)
                    mk.scale([1, 1 / self.yscale, 1])
                    marked = shapes.Glyph(pts, mk, c=mc)

            marked.name = "Marker"
            marked.alpha(ma)
            marked.z(3 * self.ztolerance)
            acts.append(marked)

        ######### the PlotXY marker errors
        ztol = self.ztolerance

        if error_band:
            yerrors = np.abs(yerrors)
            du = np.array(data)
            dd = np.array(data)
            du[:, 1] += yerrors
            dd[:, 1] -= yerrors
            if splined:
                res = len(data) * 20
                band1 = shapes.KSpline(du, res=res)
                band2 = shapes.KSpline(dd, res=res)
                band = shapes.Ribbon(band1, band2, res=(res, 2))
            else:
                dd = list(reversed(dd.tolist()))
                band = shapes.Line(du.tolist() + dd, closed=True)
                band.triangulate().lw(0)
            if ec is None:
                band.c(lc)
            else:
                band.c(ec)
            band.lighting("off").alpha(la).z(ztol / 20)
            acts.append(band)

        else:

            ## xerrors
            if xerrors is not None:
                if len(xerrors) == len(data):
                    errs = []
                    for i, val in enumerate(data):
                        xval, yval = val
                        xerr = xerrors[i] / 2
                        el = shapes.Line((xval - xerr, yval, ztol), (xval + xerr, yval, ztol))
                        el.lw(elw)
                        errs.append(el)
                    mxerrs = merge(errs).c(ec).lw(lw).alpha(ma).z(2 * ztol)
                    acts.append(mxerrs)
                else:
                    vedo.logger.error("in PlotXY(xerrors=...): mismatch in array length")

            ## yerrors
            if yerrors is not None:
                if len(yerrors) == len(data):
                    errs = []
                    for i, val in enumerate(data):
                        xval, yval = val
                        yerr = yerrors[i]
                        el = shapes.Line((xval, yval - yerr, ztol), (xval, yval + yerr, ztol))
                        el.lw(elw)
                        errs.append(el)
                    myerrs = merge(errs).c(ec).lw(lw).alpha(ma).z(2 * ztol)
                    acts.append(myerrs)
                else:
                    vedo.logger.error("in PlotXY(yerrors=...): mismatch in array length")

        self.insert(*acts, as3d=False)
        self.name = "PlotXY"


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
        - [histo_2d_b.py](https://github.com/marcomusy/vedo/tree/master/examples/examples/pyplot/histo_2d_b.py)


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
):
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
        points = points.vertices
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
    all_coeffs = np.array(all_coeffs)
    fitl.monte_carlo_coefficients = all_coeffs

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
    l1 = error_lines[0].vertices.tolist()
    cband = l1 + list(reversed(error_lines[1].vertices.tolist())) + [l1[0]]
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
        nansact = shapes.Points(nans, r=2, c="red5", alpha=alpha)
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
        ptsact = shapes.Points(points, r=ps, c=c, alpha=alpha)

    filling = None
    if fill and lw:
        faces = []
        coords = [[0, 0, 0]] + lines.vertices.tolist()
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

    cgpts = sg.vertices
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
        nanpts.append(shapes.Points(redpts, r=4, c="r"))

    pts = spher2cart(newr, theta, phi).T
    ssurf = sg.clone()
    ssurf.vertices = pts
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

    pts1 = gr.vertices
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
    histo.RemovePart(gr.actor)
    histo.AddPart(msh.actor)
    histo.objects.append(msh)
    return histo


def _histogram_hex_bin(
    xvalues, yvalues, bins=12, norm=1, fill=True, c=None, cmap="terrain_r", alpha=1
):
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
            show_lines = False
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
    sgpts = sg.vertices

    cntrs = sg.cell_centers
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

    sg.vertices = sgpts
    sg.cmap(cmap, acounts, on="cells")
    vals = sg.celldata["Scalars"]

    faces = sg.cells
    points = sg.vertices.tolist() + [[0.0, 0.0, 0.0]]
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
    show_disc=False,
):
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
    labs = ()
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
):
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
        - [histo_violin.py](https://github.com/marcomusy/vedo/tree/master/examples/examples/pyplot/histo_violin.py)

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


def whisker(data, s=0.25, c="k", lw=2, bc="blue", alpha=0.25, r=5, jitter=True, horizontal=False):
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
        - [whiskers.py](https://github.com/marcomusy/vedo/tree/master/examples/examples/pyplot/whiskers.py)

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
        pts = shapes.Points(np.array([xvals, data]).T, c=c, r=r)

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


def streamplot(
    X, Y, U, V, direction="both", max_propagation=None, lw=2, cmap="viridis", probes=()
):
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
        - [plot_stream.py](https://github.com/marcomusy/vedo/tree/master/examples/examples/pyplot/plot_stream.py)

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
            probes = probes.vertices
        else:
            probes = np.array(probes, dtype=float)
            if len(probes[0]) == 2:
                probes = np.c_[probes[:, 0], probes[:, 1], np.zeros(len(probes))]
        sv = [(n - 1) / (xmax - xmin), (n - 1) / (ymax - ymin), 1]
        probes = probes - [xmin, ymin, 0]
        probes = np.multiply(probes, sv)
        probe = vedo.Points(probes)

    stream = vol.compute_streamlines(probe, direction=direction, max_propagation=max_propagation)
    stream.lw(lw).cmap(cmap).lighting("off")
    stream.scale([1 / (n - 1) * (xmax - xmin), 1 / (n - 1) * (ymax - ymin), 1])
    stream.shift(xmin, ymin)
    return stream


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
):
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
        - [np_matrix.py](https://github.com/marcomusy/vedo/tree/master/examples/examples/pyplot/np_matrix.py)

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


class DirectedGraph(Assembly):
    """
    Support for Directed Graphs.
    """

    def __init__(self, **kargs):
        """
        A graph consists of a collection of nodes (without postional information)
        and a collection of edges connecting pairs of nodes.
        The task is to determine the node positions only based on their connections.

        This class is derived from class `Assembly`, and it assembles 4 Mesh objects
        representing the graph, the node labels, edge labels and edge arrows.

        Arguments:
            c : (color)
                Color of the Graph
            n : (int)
                number of the initial set of nodes
            layout : (int, str)
                layout in
                `['2d', 'fast2d', 'clustering2d', 'circular', 'circular3d', 'cone', 'force', 'tree']`.
                Each of these layouts has different available options.

        ---------------------------------------------------------------
        .. note:: Options for layouts '2d', 'fast2d' and 'clustering2d'

        Arguments:
            seed : (int)
                seed of the random number generator used to jitter point positions
            rest_distance : (float)
                manually set the resting distance
            nmax : (int)
                the maximum number of iterations to be used
            zrange : (list)
                expand 2d graph along z axis.

        ---------------------------------------------------------------
        .. note:: Options for layouts 'circular', and 'circular3d':

        Arguments:
            radius : (float)
                set the radius of the circles
            height : (float)
                set the vertical (local z) distance between the circles
            zrange : (float)
                expand 2d graph along z axis

        ---------------------------------------------------------------
        .. note:: Options for layout 'cone'

        Arguments:
            compactness : (float)
                ratio between the average width of a cone in the tree,
                and the height of the cone.
            compression : (bool)
                put children closer together, possibly allowing sub-trees to overlap.
                This is useful if the tree is actually the spanning tree of a graph.
            spacing : (float)
                space between layers of the tree

        ---------------------------------------------------------------
        .. note:: Options for layout 'force'

        Arguments:
            seed : (int)
                seed the random number generator used to jitter point positions
            bounds : (list)
                set the region in space in which to place the final graph
            nmax : (int)
                the maximum number of iterations to be used
            three_dimensional : (bool)
                allow optimization in the 3rd dimension too
            random_initial_points : (bool)
                use random positions within the graph bounds as initial points

        Examples:
            - [lineage_graph.py](https://github.com/marcomusy/vedo/tree/master/examples/examples/pyplot/lineage_graph.py)

                ![](https://vedo.embl.es/images/pyplot/graph_lineage.png)

            - [graph_network.py](https://github.com/marcomusy/vedo/tree/master/examples/examples/pyplot/graph_network.py)

                ![](https://vedo.embl.es/images/pyplot/graph_network.png)
        """

        super().__init__()

        self.nodes = []
        self.edges = []

        self._node_labels = []  # holds strings
        self._edge_labels = []
        self.edge_orientations = []
        self.edge_glyph_position = 0.6

        self.zrange = 0.0

        self.rotX = 0
        self.rotY = 0
        self.rotZ = 0

        self.arrow_scale = 0.15
        self.node_label_scale = None
        self.node_label_justify = "bottom-left"

        self.edge_label_scale = None

        self.mdg = vtki.new("MutableDirectedGraph")

        n = kargs.pop("n", 0)
        for _ in range(n):
            self.add_node()

        self._c = kargs.pop("c", (0.3, 0.3, 0.3))

        self.gl = vtki.new("GraphLayout")

        self.font = kargs.pop("font", "")

        s = kargs.pop("layout", "2d")
        if isinstance(s, int):
            ss = ["2d", "fast2d", "clustering2d", "circular", "circular3d", "cone", "force", "tree"]
            s = ss[s]
        self.layout = s

        if "2d" in s:
            if "clustering" in s:
                self.strategy = vtki.new("Clustering2DLayoutStrategy")
            elif "fast" in s:
                self.strategy = vtki.new("Fast2DLayoutStrategy")
            else:
                self.strategy = vtki.new("Simple2DLayoutStrategy")
            self.rotX = 180
            opt = kargs.pop("rest_distance", None)
            if opt is not None:
                self.strategy.SetRestDistance(opt)
            opt = kargs.pop("seed", None)
            if opt is not None:
                self.strategy.SetRandomSeed(opt)
            opt = kargs.pop("nmax", None)
            if opt is not None:
                self.strategy.SetMaxNumberOfIterations(opt)
            self.zrange = kargs.pop("zrange", 0)

        elif "circ" in s:
            if "3d" in s:
                self.strategy = vtki.new("Simple3DCirclesStrategy")
                self.strategy.SetDirection(0, 0, -1)
                self.strategy.SetAutoHeight(True)
                self.strategy.SetMethod(1)
                self.rotX = -90
                opt = kargs.pop("radius", None)  # float
                if opt is not None:
                    self.strategy.SetMethod(0)
                    self.strategy.SetRadius(opt)  # float
                opt = kargs.pop("height", None)
                if opt is not None:
                    self.strategy.SetAutoHeight(False)
                    self.strategy.SetHeight(opt)  # float
            else:
                self.strategy = vtki.new("CircularLayoutStrategy")
                self.zrange = kargs.pop("zrange", 0)

        elif "cone" in s:
            self.strategy = vtki.new("ConeLayoutStrategy")
            self.rotX = 180
            opt = kargs.pop("compactness", None)
            if opt is not None:
                self.strategy.SetCompactness(opt)
            opt = kargs.pop("compression", None)
            if opt is not None:
                self.strategy.SetCompression(opt)
            opt = kargs.pop("spacing", None)
            if opt is not None:
                self.strategy.SetSpacing(opt)

        elif "force" in s:
            self.strategy = vtki.new("ForceDirectedLayoutStrategy")
            opt = kargs.pop("seed", None)
            if opt is not None:
                self.strategy.SetRandomSeed(opt)
            opt = kargs.pop("bounds", None)
            if opt is not None:
                self.strategy.SetAutomaticBoundsComputation(False)
                self.strategy.SetGraphBounds(opt)  # list
            opt = kargs.pop("nmax", None)
            if opt is not None:
                self.strategy.SetMaxNumberOfIterations(opt)  # int
            opt = kargs.pop("three_dimensional", True)
            if opt is not None:
                self.strategy.SetThreeDimensionalLayout(opt)  # bool
            opt = kargs.pop("random_initial_points", None)
            if opt is not None:
                self.strategy.SetRandomInitialPoints(opt)  # bool

        elif "tree" in s:
            self.strategy = vtki.new("SpanTreeLayoutStrategy")
            self.rotX = 180

        else:
            vedo.logger.error(f"Cannot understand layout {s}. Available layouts:")
            vedo.logger.error("[2d,fast2d,clustering2d,circular,circular3d,cone,force,tree]")
            raise RuntimeError()

        self.gl.SetLayoutStrategy(self.strategy)

        if len(kargs) > 0:
            vedo.logger.error(f"Cannot understand options: {kargs}")

    def add_node(self, label="id"):
        """Add a new node to the `Graph`."""
        v = self.mdg.AddVertex()  # vtk calls it vertex..
        self.nodes.append(v)
        if label == "id":
            label = int(v)
        self._node_labels.append(str(label))
        return v

    def add_edge(self, v1, v2, label=""):
        """Add a new edge between to nodes.
        An extra node is created automatically if needed."""
        nv = len(self.nodes)
        if v1 >= nv:
            for _ in range(nv, v1 + 1):
                self.add_node()
        nv = len(self.nodes)
        if v2 >= nv:
            for _ in range(nv, v2 + 1):
                self.add_node()
        e = self.mdg.AddEdge(v1, v2)
        self.edges.append(e)
        self._edge_labels.append(str(label))
        return e

    def add_child(self, v, node_label="id", edge_label=""):
        """Add a new edge to a new node as its child.
        The extra node is created automatically if needed."""
        nv = len(self.nodes)
        if v >= nv:
            for _ in range(nv, v + 1):
                self.add_node()
        child = self.mdg.AddChild(v)
        self.edges.append((v, child))
        self.nodes.append(child)
        if node_label == "id":
            node_label = int(child)
        self._node_labels.append(str(node_label))
        self._edge_labels.append(str(edge_label))
        return child

    def build(self):
        """
        Build the `DirectedGraph(Assembly)`.
        Accessory objects are also created for labels and arrows.
        """
        self.gl.SetZRange(self.zrange)
        self.gl.SetInputData(self.mdg)
        self.gl.Update()

        gr2poly = vtki.new("GraphToPolyData")
        gr2poly.EdgeGlyphOutputOn()
        gr2poly.SetEdgeGlyphPosition(self.edge_glyph_position)
        gr2poly.SetInputData(self.gl.GetOutput())
        gr2poly.Update()

        dgraph = Mesh(gr2poly.GetOutput(0))
        # dgraph.clean() # WRONG!!! dont uncomment
        dgraph.flat().color(self._c).lw(2)
        dgraph.name = "DirectedGraph"

        diagsz = self.diagonal_size() / 1.42
        if not diagsz:
            return None

        dgraph.scale(1 / diagsz)
        if self.rotX:
            dgraph.rotate_x(self.rotX)
        if self.rotY:
            dgraph.rotate_y(self.rotY)
        if self.rotZ:
            dgraph.rotate_z(self.rotZ)

        vecs = gr2poly.GetOutput(1).GetPointData().GetVectors()
        self.edge_orientations = utils.vtk2numpy(vecs)

        # Use Glyph3D to repeat the glyph on all edges.
        arrows = None
        if self.arrow_scale:
            arrow_source = vtki.new("GlyphSource2D")
            arrow_source.SetGlyphTypeToEdgeArrow()
            arrow_source.SetScale(self.arrow_scale)
            arrow_source.Update()
            arrow_glyph = vtki.vtkGlyph3D()
            arrow_glyph.SetInputData(0, gr2poly.GetOutput(1))
            arrow_glyph.SetInputData(1, arrow_source.GetOutput())
            arrow_glyph.Update()
            arrows = Mesh(arrow_glyph.GetOutput())
            arrows.scale(1 / diagsz)
            arrows.lighting("off").color(self._c)
            if self.rotX:
                arrows.rotate_x(self.rotX)
            if self.rotY:
                arrows.rotate_y(self.rotY)
            if self.rotZ:
                arrows.rotate_z(self.rotZ)
            arrows.name = "DirectedGraphArrows"

        node_labels = dgraph.labels(
            self._node_labels,
            scale=self.node_label_scale,
            precision=0,
            font=self.font,
            justify=self.node_label_justify,
        )
        node_labels.color(self._c).pickable(True)
        node_labels.name = "DirectedGraphNodeLabels"

        edge_labels = dgraph.labels(
            self._edge_labels, on="cells", scale=self.edge_label_scale, precision=0, font=self.font
        )
        edge_labels.color(self._c).pickable(True)
        edge_labels.name = "DirectedGraphEdgeLabels"

        super().__init__([dgraph, node_labels, edge_labels, arrows])
        self.name = "DirectedGraphAssembly"
        return self
