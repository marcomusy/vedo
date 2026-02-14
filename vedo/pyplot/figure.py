#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure primitives for pyplot."""

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

    def __init__(self, xlim, ylim, aspect=1.333, padding=(0.05, 0.05, 0.05, 0.05), **kwargs):
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

        vedo.set_last_figure(self if settings.remember_last_figure_format else None)


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

        _all = [
            "<table>",
            "<tr>",
            "<td>",
            image,
            "</td>",
            "<td style='text-align: center; vertical-align: center;'><br/>",
            help_text,
            "<table>",
            "<tr><td><b> nr. of parts </b></td><td>" + str(self.actor.GetNumberOfPaths()) + "</td></tr>",
            "<tr><td><b> position </b></td><td>" + str(self.actor.GetPosition()) + "</td></tr>",
            "<tr><td><b> x-limits </b></td><td>" + utils.precision(self.xlim, 4) + "</td></tr>",
            "<tr><td><b> y-limits </b></td><td>" + utils.precision(self.ylim, 4) + "</td></tr>",
            "<tr><td><b> world bounds </b> <br/> (x/y/z) </td><td>" + str(bounds) + "</td></tr>",
            "</table>",
            "</table>",
        ]
        return "\n".join(_all)

    def __add__(self, *obj):
        # just to avoid confusion, supersede Assembly.__add__
        return self.__iadd__(*obj)

    def __iadd__(self, *obj):
        if len(obj) == 1 and isinstance(obj[0], Figure):
            return self._check_unpack_and_insert(obj[0])

        obj = utils.flatten(obj)
        return self.insert(*obj)

    def _check_unpack_and_insert(self, fig: "Figure") -> Self:

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

    def insert(self, *objs, rescale=True, as3d=True, adjusted=False, cut=True) -> Self:
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

            self.actor.AddPart(a.actor)
            self.objects.append(a)

        return self

    def add_label(self, text: str, c=None, marker="", mc="black") -> Self:
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
        alpha=0.8,
        bc="k7",
        lw=1,
        lc="k4",
        z=0,
    ) -> Self:
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
        x0, x1, y0, y1, _, _ = aleg.actor.GetBounds()

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
            aleg.actor.AddPart(box.actor)
            aleg.objects.append(box)

        xlim = self.xlim
        ylim = self.ylim
        if isinstance(pos, str):
            px, py = 0.0, 0.0
            rx, ry = (xlim[1] + xlim[0]) / 2, (ylim[1] + ylim[0]) / 2
            shx, shy = 0.0, 0.0
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

__all__ = ["LabelData", "Figure"]
