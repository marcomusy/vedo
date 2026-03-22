#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Volume slicing and ray-casting application plotters."""

import os

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo.colors import color_map, get_color
from vedo.utils import is_sequence, lin_interpolate, mag, precision
from vedo.plotter import Plotter
from vedo.pointcloud import fit_plane, Points
from vedo.shapes import Line, Ribbon, Spline, Text2D
from vedo.pyplot import CornerHistogram, histogram
from vedo.addons import SliderWidget

class Slicer3DPlotter(Plotter):
    """
    Generate a rendering window with slicing planes for the input Volume.
    """

    _slice_names = ("XSlice", "YSlice", "ZSlice")

    def __init__(
        self,
        volume: vedo.Volume,
        cmaps=("gist_ncar_r", "hot_r", "bone", "bone_r", "jet", "Spectral_r"),
        clamp=True,
        use_slider3d=False,
        show_histo=True,
        show_icon=True,
        draggable=False,
        at=0,
        **kwargs,
    ):
        """
        Generate a rendering window with slicing planes for the input Volume.

        Arguments:
            cmaps : (list)
                list of color maps names to cycle when clicking button
            clamp : (bool)
                clamp scalar range to reduce the effect of tails in color mapping
            use_slider3d : (bool)
                show sliders attached along the axes
            show_histo : (bool)
                show histogram on bottom left
            show_icon : (bool)
                show a small 3D rendering icon of the volume
            draggable : (bool)
                make the 3D icon draggable
            at : (int)
                subwindow number to plot to
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        Examples:
            - [slicer1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slicer1.py)

            <img src="https://vedo.embl.es/images/volumetric/slicer1.jpg" width="500">
        """
        ################################
        super().__init__(**kwargs)
        self.at(at)
        ################################

        cx, cy, cz, ch = "dr", "dg", "db", (0.3, 0.3, 0.3)
        if np.sum(self.renderer.GetBackground()) < 1.5:
            cx, cy, cz = "lr", "lg", "lb"
            ch = (0.8, 0.8, 0.8)

        if len(self.renderers) > 1:
            # 2d sliders do not work with multiple renderers
            use_slider3d = True

        self._slice_ambient = 0.7
        self._slice_diffuse = 0.3
        self._show_histo = show_histo
        self._show_icon = show_icon
        self._draggable_icon = draggable
        self._use_slider3d = use_slider3d
        self._clamp = clamp
        self._histogram_bg = ch
        self._slider_colors = (cx, cy, cz)
        self._cmaps = tuple(cmaps)
        self._inset_widget = None
        self._inset_marker = None
        self._box = None
        self._dims = (0, 0, 0)
        self._scalar_range = (0, 1)
        self._histogram_data = None
        self.cmap_slicer = cmaps[0]
        self.current_i = None
        self.current_j = None
        self.current_k = None

        self.xslice = None
        self.yslice = None
        self.zslice = None
        self.histogram = None

        if not use_slider3d:
            self.xslider = self.add_slider(
                self._slider_function_x,
                0,
                1,
                title="",
                title_size=0.5,
                pos=[(0.8, 0.12), (0.95, 0.12)],
                show_value=False,
                c=cx,
            )
            self.yslider = self.add_slider(
                self._slider_function_y,
                0,
                1,
                title="",
                title_size=0.5,
                pos=[(0.8, 0.08), (0.95, 0.08)],
                show_value=False,
                c=cy,
            )
            self.zslider = self.add_slider(
                self._slider_function_z,
                0,
                1,
                title="",
                title_size=0.6,
                value=0,
                pos=[(0.8, 0.04), (0.95, 0.04)],
                show_value=False,
                c=cz,
            )

        else:  # 3d sliders attached to the axes bounds
            self.xslider = self.add_slider3d(
                self._slider_function_x,
                pos1=(0, 0, 0),
                pos2=(0, 0, 0),
                xmin=0,
                xmax=1,
                t=1,
                c=cx,
                show_value=False,
            )
            self.yslider = self.add_slider3d(
                self._slider_function_y,
                pos1=(0, 0, 0),
                pos2=(0, 0, 0),
                xmin=0,
                xmax=1,
                t=1,
                c=cy,
                show_value=False,
            )
            self.zslider = self.add_slider3d(
                self._slider_function_z,
                pos1=(0, 0, 0),
                pos2=(0, 0, 0),
                xmin=0,
                xmax=1,
                value=0,
                t=1,
                c=cz,
                show_value=False,
            )

        if len(cmaps) > 1:
            self._cmap_button = self.add_button(
                self._button_func,
                states=cmaps,
                c=["k9"] * len(cmaps),
                bc=["k1"] * len(cmaps),  # colors of states
                size=16,
                bold=True,
            )
            if self._cmap_button:
                self._cmap_button.pos([0.04, 0.01], "bottom-left")
        else:
            self._cmap_button = None

        self.set_volume(volume, reset_slices=True, reset_camera=False, render=False)

    def _make_box(self, volume):
        box = volume.box().alpha(0.2)
        box.name = "VolumeBox"
        return box

    def _make_inset_axes(self, box):
        return vedo.addons.Axes(
            box,
            xtitle=" ",
            ytitle=" ",
            ztitle=" ",
            yzgrid=False,
            xlabel_size=0,
            ylabel_size=0,
            zlabel_size=0,
            tip_size=0.08,
            axes_linewidth=3,
            xline_color="dr",
            yline_color="dg",
            zline_color="db",
        )

    def _compute_scalar_range(self, volume):
        data = volume.pointdata[0]
        rmin, rmax = volume.scalar_range()
        if self._clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata + 1)
            meanlog = np.sum(np.multiply(edg[:-1], logdata)) / np.sum(logdata)
            rmax = min(rmax, meanlog + (meanlog - rmin) * 0.9)
            rmin = max(rmin, meanlog - (rmax - meanlog) * 0.9)
        return data, (rmin, rmax)

    def _make_histogram(self):
        if not self._show_histo or self._histogram_data is None:
            return None
        return histogram(
            self._histogram_data,
            bins=20,
            logscale=True,
            c=self.cmap_slicer,
            bg=self._histogram_bg,
            alpha=1,
            axes=dict(text_scale=2),
        ).clone2d(pos=[-0.925, -0.88], size=0.4)

    def _update_histogram(self):
        if self.histogram is not None:
            self.remove(self.histogram)
            self.histogram = None
        self.histogram = self._make_histogram()
        if self.histogram is not None:
            self.add(self.histogram)

    def _make_slice(self, axis, index):
        rmin, rmax = self._scalar_range
        slicer = getattr(self.volume, f"{axis}slice")(index).lighting(
            "", self._slice_ambient, self._slice_diffuse, 0
        )
        slicer.name = f"{axis.upper()}Slice"
        slicer.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
        return slicer

    def _update_slice(self, axis, index, render=True):
        dims = self._dims
        dim = {"x": dims[0], "y": dims[1], "z": dims[2]}[axis]
        name = f"{axis.upper()}Slice"
        self.remove(name)
        new_slice = None
        if 0 < index < dim:
            new_slice = self._make_slice(axis, index)
            self.add(new_slice)
        setattr(self, f"{axis}slice", new_slice)
        if render:
            self.render()

    def _slider_function_x(self, _widget, _event):
        i = int(self.xslider.value)
        if i == self.current_i:
            return
        self.current_i = i
        self._update_slice("x", i)

    def _slider_function_y(self, _widget, _event):
        j = int(self.yslider.value)
        if j == self.current_j:
            return
        self.current_j = j
        self._update_slice("y", j)

    def _slider_function_z(self, _widget, _event):
        k = int(self.zslider.value)
        if k == self.current_k:
            return
        self.current_k = k
        self._update_slice("z", k)

    def _button_func(self, _obj, _evtname):
        self._cmap_button.switch()
        self.cmap_slicer = self._cmap_button.status()
        for axis in "xyz":
            msh = getattr(self, f"{axis}slice")
            if msh is not None:
                rmin, rmax = self._scalar_range
                msh.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
        self._update_histogram()
        self.render()

    def _update_inset(self):
        if not self._show_icon or not self.interactor:
            return
        self._inset_marker = vedo.Assembly(self.volume, self._make_inset_axes(self._box))
        if self._inset_widget is None:
            self._inset_widget = self.add_inset(
                self._inset_marker,
                pos=(0.9, 0.9),
                size=0.15,
                c="w",
                draggable=self._draggable_icon,
            )
        elif self._inset_widget:
            self._inset_widget.SetOrientationMarker(self._inset_marker.actor)

    def _update_slider_positions(self):
        if not self._use_slider3d:
            return
        bs = self._box.bounds()
        repx = self.xslider.representation
        repx.GetPoint1Coordinate().SetValue((bs[1], bs[2], bs[4]))
        repx.GetPoint2Coordinate().SetValue((bs[0], bs[2], bs[4]))
        repx.SetSliderWidth(0.03 * self._box.diagonal_size() / mag(self._box.xbounds()) * 0.6)
        repx.SetTubeWidth(0.01 * self._box.diagonal_size() / mag(self._box.xbounds()) * 0.6)
        repx.SetSliderLength(0.04 * self._box.diagonal_size() / mag(self._box.xbounds()) * 0.6)

        repy = self.yslider.representation
        repy.GetPoint1Coordinate().SetValue((bs[1], bs[3], bs[4]))
        repy.GetPoint2Coordinate().SetValue((bs[1], bs[2], bs[4]))
        repy.SetSliderWidth(0.03 * self._box.diagonal_size() / mag(self._box.ybounds()) * 0.6)
        repy.SetTubeWidth(0.01 * self._box.diagonal_size() / mag(self._box.ybounds()) * 0.6)
        repy.SetSliderLength(0.04 * self._box.diagonal_size() / mag(self._box.ybounds()) * 0.6)

        repz = self.zslider.representation
        repz.GetPoint1Coordinate().SetValue((bs[0], bs[2], bs[5]))
        repz.GetPoint2Coordinate().SetValue((bs[0], bs[2], bs[4]))
        repz.SetSliderWidth(0.03 * self._box.diagonal_size() / mag(self._box.zbounds()) * 0.6)
        repz.SetTubeWidth(0.01 * self._box.diagonal_size() / mag(self._box.zbounds()) * 0.6)
        repz.SetSliderLength(0.04 * self._box.diagonal_size() / mag(self._box.zbounds()) * 0.6)

    def _update_slider_ranges(self):
        dims = self._dims
        self.xslider.range = (0, dims[0])
        self.yslider.range = (0, dims[1])
        self.zslider.range = (0, dims[2])
        self.xslider.value = 0 if self.current_i is None else self.current_i
        self.yslider.value = 0 if self.current_j is None else self.current_j
        self.zslider.value = 0 if self.current_k is None else self.current_k
        self._update_slider_positions()

    def set_volume(
        self,
        volume: vedo.Volume,
        reset_slices=True,
        reset_camera=False,
        render=True,
    ) -> Slicer3DPlotter:
        """
        Replace the input volume while preserving the existing plotter window.

        Arguments:
            volume : (Volume)
                the new input volume to visualize.
            reset_slices : (bool)
                if True reset slices to their default positions, otherwise preserve them when possible.
            reset_camera : (bool)
                if True reset the camera after swapping the volume.
            render : (bool)
                if True render the scene after the update.
        """
        if self._box is not None:
            self.remove(self._box)
        self.remove(*self._slice_names)

        self.volume = volume
        self._box = self._make_box(volume)
        self.add(self._box)
        self._dims = volume.dimensions()

        data, self._scalar_range = self._compute_scalar_range(volume)
        self._histogram_data = None
        if self._show_histo and data.ndim == 1:
            n = (self._dims[0] - 1) * (self._dims[1] - 1) * (self._dims[2] - 1)
            n = min(1_000_000, n)
            self._histogram_data = np.random.choice(data, n)

        if reset_slices:
            self.current_i = None
            self.current_j = None
            self.current_k = int(self._dims[2] / 2)
        else:
            if self.current_i is not None:
                self.current_i = min(self.current_i, self._dims[0] - 1)
            if self.current_j is not None:
                self.current_j = min(self.current_j, self._dims[1] - 1)
            if self.current_k is None:
                self.current_k = int(self._dims[2] / 2)
            else:
                self.current_k = min(self.current_k, self._dims[2] - 1)

        self._update_slider_ranges()
        self._update_histogram()
        self._update_inset()

        self.xslice = None
        self.yslice = None
        self.zslice = None
        if self.current_i is not None:
            self._update_slice("x", self.current_i, render=False)
        if self.current_j is not None:
            self._update_slice("y", self.current_j, render=False)
        if self.current_k is not None:
            self._update_slice("z", self.current_k, render=False)

        if reset_camera:
            self.reset_camera()
        if render:
            self.render()
        return self


####################################################################################
class Slicer3DTwinPlotter(Plotter):
    """
    Create a window with two side-by-side 3D slicers for two Volumes.

    Arguments:
        vol1 : (Volume)
            the first Volume object to be isosurfaced.
        vol2 : (Volume)
            the second Volume object to be isosurfaced.
        clamp : (bool)
            clamp scalar range to reduce the effect of tails in color mapping
        **kwargs : (dict)
            keyword arguments to pass to a `vedo.plotter.Plotter` instance.

    Example:
        ```python
        from vedo import *
        from vedo.applications import Slicer3DTwinPlotter

        vol1 = Volume(dataurl + "embryo.slc")
        vol2 = Volume(dataurl + "embryo.slc")

        plt = Slicer3DTwinPlotter(
            vol1, vol2,
            shape=(1, 2),
            sharecam=True,
            bg="white",
            bg2="lightblue",
        )

        plt.at(0).add(Text2D("Volume 1", pos="top-center"))
        plt.at(1).add(Text2D("Volume 2", pos="top-center"))

        plt.show(viewup='z')
        plt.at(0).reset_camera()
        plt.interactive().close()
        ```

        <img src="https://vedo.embl.es/images/volumetric/slicer3dtwin.png" width="650">
    """

    def __init__(self, vol1: vedo.Volume, vol2: vedo.Volume, clamp=True, **kwargs):

        super().__init__(**kwargs)

        cmap = "gist_ncar_r"
        cx, cy, cz = "dr", "dg", "db"  # slider colors
        ambient, diffuse = 0.7, 0.3  # lighting params

        self.at(0)
        box1 = vol1.box().alpha(0.1)
        box2 = vol2.box().alpha(0.1)
        self.add(box1)

        self.at(1).add(box2)
        self.add_inset(vol2, pos=(0.85, 0.15), size=0.15, c="white", draggable=0)

        dims = vol1.dimensions()
        data = vol1.pointdata[0]
        rmin, rmax = vol1.scalar_range()
        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata + 1)
            meanlog = np.sum(np.multiply(edg[:-1], logdata)) / np.sum(logdata)
            rmax = min(rmax, meanlog + (meanlog - rmin) * 0.9)
            rmin = max(rmin, meanlog - (rmax - meanlog) * 0.9)

        def slider_function_x(_widget, _event):
            i = int(self.xslider.value)
            msh1 = vol1.xslice(i).lighting("", ambient, diffuse, 0)
            msh1.cmap(cmap, vmin=rmin, vmax=rmax)
            msh1.name = "XSlice"
            self.at(0).remove("XSlice")  # removes the old one
            msh2 = vol2.xslice(i).lighting("", ambient, diffuse, 0)
            msh2.cmap(cmap, vmin=rmin, vmax=rmax)
            msh2.name = "XSlice"
            self.at(1).remove("XSlice")
            if 0 < i < dims[0]:
                self.at(0).add(msh1)
                self.at(1).add(msh2)

        def slider_function_y(_widget, _event):
            i = int(self.yslider.value)
            msh1 = vol1.yslice(i).lighting("", ambient, diffuse, 0)
            msh1.cmap(cmap, vmin=rmin, vmax=rmax)
            msh1.name = "YSlice"
            self.at(0).remove("YSlice")
            msh2 = vol2.yslice(i).lighting("", ambient, diffuse, 0)
            msh2.cmap(cmap, vmin=rmin, vmax=rmax)
            msh2.name = "YSlice"
            self.at(1).remove("YSlice")
            if 0 < i < dims[1]:
                self.at(0).add(msh1)
                self.at(1).add(msh2)

        def slider_function_z(_widget, _event):
            i = int(self.zslider.value)
            msh1 = vol1.zslice(i).lighting("", ambient, diffuse, 0)
            msh1.cmap(cmap, vmin=rmin, vmax=rmax)
            msh1.name = "ZSlice"
            self.at(0).remove("ZSlice")
            msh2 = vol2.zslice(i).lighting("", ambient, diffuse, 0)
            msh2.cmap(cmap, vmin=rmin, vmax=rmax)
            msh2.name = "ZSlice"
            self.at(1).remove("ZSlice")
            if 0 < i < dims[2]:
                self.at(0).add(msh1)
                self.at(1).add(msh2)

        self.at(0)
        bs = box1.bounds()
        self.xslider = self.add_slider3d(
            slider_function_x,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[1], bs[2], bs[4]),
            xmin=0,
            xmax=dims[0],
            t=box1.diagonal_size() / mag(box1.xbounds()) * 0.6,
            c=cx,
            show_value=False,
        )
        self.yslider = self.add_slider3d(
            slider_function_y,
            pos1=(bs[1], bs[2], bs[4]),
            pos2=(bs[1], bs[3], bs[4]),
            xmin=0,
            xmax=dims[1],
            t=box1.diagonal_size() / mag(box1.ybounds()) * 0.6,
            c=cy,
            show_value=False,
        )
        self.zslider = self.add_slider3d(
            slider_function_z,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[0], bs[2], bs[5]),
            xmin=0,
            xmax=dims[2],
            value=int(dims[2] / 2),
            t=box1.diagonal_size() / mag(box1.zbounds()) * 0.6,
            c=cz,
            show_value=False,
        )

        #################
        hist = CornerHistogram(data, s=0.2, bins=25, logscale=True, c="k")
        self.add(hist)
        slider_function_z(0, 0)  ## init call


########################################################################################
class Slicer2DPlotter(Plotter):
    """
    A single slice of a Volume which always faces the camera,
    but at the same time can be oriented arbitrarily in space.
    """

    def __init__(self, vol: vedo.Volume, levels=(None, None), histo_color="red4", **kwargs):
        """
        A single slice of a Volume which always faces the camera,
        but at the same time can be oriented arbitrarily in space.

        Arguments:
            vol : (Volume)
                the Volume object to be isosurfaced.
            levels : (list)
                window and color levels
            histo_color : (color)
                histogram color, use `None` to disable it
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        <img src="https://vedo.embl.es/images/volumetric/read_volume3.jpg" width="500">
        """

        if "shape" not in kwargs:
            custom_shape = [  # define here the 2 rendering rectangle spaces
                dict(bottomleft=(0.0, 0.0), topright=(1, 1), bg="k9"),  # the full window
                dict(bottomleft=(0.8, 0.8), topright=(1, 1), bg="k8", bg2="lb"),
            ]
            kwargs["shape"] = custom_shape

        if "interactive" not in kwargs:
            kwargs["interactive"] = True

        super().__init__(**kwargs)

        self.user_mode("image")
        self.add_callback("KeyPress", self.on_key_press)

        orig_volume = vol.clone(deep=False)
        self.volume = vol

        self.volume.actor = vtki.new("ImageSlice")

        self.volume.properties = self.volume.actor.GetProperty()
        self.volume.properties.SetInterpolationTypeToLinear()

        self.volume.mapper = vtki.new("ImageResliceMapper")
        self.volume.mapper.SetInputData(self.volume.dataset)
        self.volume.mapper.SliceFacesCameraOn()
        self.volume.mapper.SliceAtFocalPointOn()
        self.volume.mapper.SetAutoAdjustImageQuality(False)
        self.volume.mapper.BorderOff()

        # no argument will grab the existing cmap in vol (or use build_lut())
        self.lut = None
        self.cmap()

        if levels[0] and levels[1]:
            self.lighting(window=levels[0], level=levels[1])

        self.usage_txt = (
            "H                  :rightarrow Toggle this banner on/off\n"
            "Left click & drag  :rightarrow Modify luminosity and contrast\n"
            "SHIFT-Left click   :rightarrow Slice image obliquely\n"
            "SHIFT-Middle click :rightarrow Slice image perpendicularly\n"
            "SHIFT-R            :rightarrow Fly to closest cartesian view\n"
            "SHIFT-U            :rightarrow Toggle parallel projection"
        )

        self.usage = Text2D(
            self.usage_txt, font="Calco", pos="top-left", s=0.8, bg="yellow", alpha=0.25
        )

        hist = None
        if histo_color is not None:
            data = self.volume.pointdata[0]
            arr = data
            if data.ndim == 1:
                # try to reduce the number of values to histogram
                dims = self.volume.dimensions()
                n = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1)
                n = min(1_000_000, n)
                arr = np.random.choice(self.volume.pointdata[0], n)
                hist = vedo.pyplot.histogram(
                    arr,
                    bins=12,
                    logscale=True,
                    c=histo_color,
                    ytitle="log_10 (counts)",
                    axes=dict(text_scale=1.9),
                ).clone2d(pos="bottom-left", size=0.4)

        axes = kwargs.pop("axes", 7)
        axe = None
        if axes == 7:
            axe = vedo.addons.RulerAxes(
                orig_volume, xtitle="x - ", ytitle="y - ", ztitle="z - "
            )

        box = orig_volume.box().alpha(0.25)

        volume_axes_inset = vedo.addons.Axes(
            box,
            yzgrid=False,
            xlabel_size=0,
            ylabel_size=0,
            zlabel_size=0,
            tip_size=0.08,
            axes_linewidth=3,
            xline_color="dr",
            yline_color="dg",
            zline_color="db",
            xtitle_color="dr",
            ytitle_color="dg",
            ztitle_color="db",
            xtitle_size=0.1,
            ytitle_size=0.1,
            ztitle_size=0.1,
            title_font="VictorMono",
        )

        self.at(0).add(self.volume, box, axe, self.usage, hist)
        self.at(1).add(orig_volume, volume_axes_inset)
        self.at(0)  # set focus at renderer 0

    ####################################################################
    def on_key_press(self, evt):
        """Handle keyboard events"""
        if evt.keypress == "q":
            self.break_interaction()
        elif evt.keypress.lower() == "h":
            t = self.usage
            if len(t.text()) > 50:
                self.usage.text("Press H to show help")
            else:
                self.usage.text(self.usage_txt)
            self.render()

    def cmap(self, lut=None, fix_scalar_range=False) -> Slicer2DPlotter:
        """
        Assign a LUT (Look Up Table) to colorize the slice, leave it `None`
        to reuse an existing Volume color map.
        Use "bw" for automatic black and white.
        """
        if lut is None and self.lut:
            self.volume.properties.SetLookupTable(self.lut)
        elif isinstance(lut, vtki.vtkLookupTable):
            self.volume.properties.SetLookupTable(lut)
        elif lut == "bw":
            self.volume.properties.SetLookupTable(None)
        self.volume.properties.SetUseLookupTableScalarRange(fix_scalar_range)
        return self

    def alpha(self, value: float) -> Slicer2DPlotter:
        """Set opacity to the slice"""
        self.volume.properties.SetOpacity(value)
        return self

    def auto_adjust_quality(self, value=True) -> Slicer2DPlotter:
        """Automatically reduce the rendering quality for greater speed when interacting"""
        self.volume.mapper.SetAutoAdjustImageQuality(value)
        return self

    def slab(self, thickness=0, mode=0, sample_factor=2) -> Slicer2DPlotter:
        """
        Make a thick slice (slab).

        Arguments:
            thickness : (float)
                set the slab thickness, for thick slicing
            mode : (int)
                The slab type:
                    0 = min
                    1 = max
                    2 = mean
                    3 = sum
            sample_factor : (float)
                Set the number of slab samples to use as a factor of the number of input slices
                within the slab thickness. The default value is 2, but 1 will increase speed
                with very little loss of quality.
        """
        self.volume.mapper.SetSlabThickness(thickness)
        self.volume.mapper.SetSlabType(mode)
        self.volume.mapper.SetSlabSampleFactor(sample_factor)
        return self

    def face_camera(self, value=True) -> Slicer2DPlotter:
        """Make the slice always face the camera or not."""
        self.volume.mapper.SetSliceFacesCameraOn(value)
        return self

    def jump_to_nearest_slice(self, value=True) -> Slicer2DPlotter:
        """
        This causes the slicing to occur at the closest slice to the focal point,
        instead of the default behavior where a new slice is interpolated between
        the original slices.
        Nothing happens if the plane is oblique to the original slices.
        """
        self.volume.mapper.SetJumpToNearestSlice(value)
        return self

    def fill_background(self, value=True) -> Slicer2DPlotter:
        """
        Instead of rendering only to the image border,
        render out to the viewport boundary with the background color.
        The background color will be the lowest color on the lookup
        table that is being used for the image.
        """
        self.volume.mapper.SetBackground(value)
        return self

    def lighting(self, window, level, ambient=1.0, diffuse=0.0) -> Slicer2DPlotter:
        """Assign the values for window and color level."""
        self.volume.properties.SetColorWindow(window)
        self.volume.properties.SetColorLevel(level)
        self.volume.properties.SetAmbient(ambient)
        self.volume.properties.SetDiffuse(diffuse)
        return self


########################################################################
class RayCastPlotter(Plotter):
    """
    Generate Volume rendering using ray casting.
    """

    def __init__(self, volume, **kwargs):
        """
        Generate a window for Volume rendering using ray casting.

        Arguments:
            volume : (Volume)
                the Volume object to be isosurfaced.
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        Returns:
            `vedo.Plotter` object.

        Examples:
            - [app_raycaster.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/app_raycaster.py)

            ![](https://vedo.embl.es/images/advanced/app_raycaster.gif)
        """

        super().__init__(**kwargs)

        self.alphaslider0 = 0.33
        self.alphaslider1 = 0.66
        self.alphaslider2 = 1
        self.color_scalarbar = None

        self.properties = volume.properties

        if volume.dimensions()[2] < 3:
            vedo.logger.error("RayCastPlotter: not enough z slices.")
            raise RuntimeError

        smin, smax = volume.scalar_range()
        x0alpha = smin + (smax - smin) * 0.25
        x1alpha = smin + (smax - smin) * 0.5
        x2alpha = smin + (smax - smin) * 1.0

        ############################## color map slider
        # Create transfer mapping scalar value to color
        cmaps = [
            "rainbow", "rainbow_r",
            "viridis", "viridis_r",
            "bone", "bone_r",
            "hot", "hot_r",
            "plasma", "plasma_r",
            "gist_earth", "gist_earth_r",
            "coolwarm", "coolwarm_r",
            "tab10_r",
        ]
        cols_cmaps = []
        for cm in cmaps:
            cols = color_map(range(0, 21), cm, 0, 20)  # sample 20 colors
            cols_cmaps.append(cols)
        Ncols = len(cmaps)
        csl = "k9"
        if sum(get_color(self.background())) > 1.5:
            csl = "k1"

        def slider_cmap(widget=None, _event=""):
            if widget:
                k = int(widget.value)
                volume.cmap(cmaps[k])
            self.remove(self.color_scalarbar)
            self.color_scalarbar = vedo.addons.ScalarBar(
                volume,
                horizontal=True,
                pos=[(0.8, 0), (0.97, 0.1)],
                font_size=0
            )
            self.add(self.color_scalarbar)

        w1 = self.add_slider(
            slider_cmap,
            0, Ncols - 1,
            value=0,
            show_value=False,
            c=csl,
            pos=[(0.8, 0.05), (0.965, 0.05)],
        )
        w1.representation.SetTitleHeight(0.018)

        ############################## alpha sliders
        # Create transfer mapping scalar value to opacity transfer function
        def setOTF():
            otf = self.properties.GetScalarOpacity()
            otf.RemoveAllPoints()
            otf.AddPoint(smin, 0.0)
            otf.AddPoint(smin + (smax - smin) * 0.1, 0.0)
            otf.AddPoint(x0alpha, self.alphaslider0)
            otf.AddPoint(x1alpha, self.alphaslider1)
            otf.AddPoint(x2alpha, self.alphaslider2)
            slider_cmap()

        setOTF()  ################

        def sliderA0(widget, _event):
            self.alphaslider0 = widget.value
            setOTF()

        self.add_slider(
            sliderA0,
            0, 1,
            value=self.alphaslider0,
            pos=[(0.84, 0.1), (0.84, 0.26)],
            c=csl,
            show_value=0,
        )

        def sliderA1(widget, _event):
            self.alphaslider1 = widget.value
            setOTF()

        self.add_slider(
            sliderA1,
            0, 1,
            value=self.alphaslider1,
            pos=[(0.89, 0.1), (0.89, 0.26)],
            c=csl,
            show_value=0,
        )

        def sliderA2(widget, _event):
            self.alphaslider2 = widget.value
            setOTF()

        w2 = self.add_slider(
            sliderA2,
            0, 1,
            value=self.alphaslider2,
            pos=[(0.96, 0.1), (0.96, 0.26)],
            c=csl,
            show_value=0,
            title="Opacity Levels",
        )
        w2.GetRepresentation().SetTitleHeight(0.015)

        # add a button
        def button_func_mode(_obj, _ename):
            s = volume.mode()
            snew = (s + 1) % 2
            volume.mode(snew)
            bum.switch()

        bum = self.add_button(
            button_func_mode,
            pos=(0.89, 0.31),
            states=["  composite   ", "max projection"],
            c=[ "k3", "k6"],
            bc=["k6", "k3"],  # colors of states
            font="Calco",
            size=18,
            bold=0,
            italic=False,
        )
        bum.frame(color="k6")
        bum.status(volume.mode())

        slider_cmap() ############# init call to create scalarbar

        # add histogram of scalar
        plot = CornerHistogram(
            volume,
            bins=25,
            logscale=1,
            c='k5',
            bg='k5',
            pos=(0.78, 0.065),
            lines=True,
            dots=False,
            nmax=3.1415e06,  # subsample otherwise is too slow
        )

        plot.GetPosition2Coordinate().SetValue(0.197, 0.20, 0)
        plot.GetXAxisActor2D().SetFontFactor(0.7)
        plot.GetProperty().SetOpacity(0.5)
        self.add([plot, volume])


#####################################################################################
