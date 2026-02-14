#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chart classes for pyplot."""

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

from .figure import LabelData, Figure

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
        aspect=1.333,
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
        data = np.asarray(data).ravel()
        data = data[np.logical_not(np.isnan(data))]

        # if data.dtype is integer try to center bins by default
        if like is None and bins is None and np.issubdtype(data.dtype, np.integer):
            if xlim is None and ylim == (0, None):
                x1, x0 = data.max(), data.min()
                if 0 < x1 - x0 <= 100:
                    bins = x1 - x0 + 1
                    xlim = (x0 - 0.5, x1 + 0.5)

        if like is None and vedo.current_last_figure() is not None:
            if xlim is None and ylim == (0, None):
                like = vedo.current_last_figure()

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

    def print(self, **kwargs) -> None:
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

        if like is None and vedo.current_last_figure() is not None:
            if xlim is None and ylim == (None, None) and zlim == (None, None):
                like = vedo.current_last_figure()

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

                # print(" g.actor.GetBounds()[0]", g.actor.GetBounds()[:2])
                # print("sc.actor.GetBounds()[0]",sc.actor.GetBounds()[:2])
                delta = sc.actor.GetBounds()[0] - g.actor.GetBounds()[1]

                sc_size = sc.actor.GetBounds()[1] - sc.actor.GetBounds()[0]

                sc.actor.SetOrigin(sc.actor.GetBounds()[0], 0, 0)
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
        aspect=1.333,
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
            vedo.logger.error(f"{m}{data}\n     bin edges and colors are optional.")
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

        if like is None and vedo.current_last_figure() is not None:
            if xlim == (None, None) and ylim == (0, None):
                like = vedo.current_last_figure()

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
        aspect=1.333,
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

        if like is None and vedo.current_last_figure() is not None:
            if xlim is None and ylim == (None, None):
                like = vedo.current_last_figure()

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
        data = data[~np.isnan(data).any(axis=1), :]

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



__all__ = ["Histogram1D", "Histogram2D", "PlotBars", "PlotXY"]
