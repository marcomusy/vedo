#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isosurface and object browser applications."""

import os
from typing import Union

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

class IsosurfaceBrowser(Plotter):
    """
    Generate a Volume isosurfacing controlled by a slider.
    """

    def __init__(
        self,
        volume: vedo.Volume,
        isovalue=None,
        scalar_range=(),
        c=None,
        alpha=1,
        lego=False,
        res=50,
        use_gpu=False,
        precompute=False,
        cmap="hot",
        delayed=False,
        sliderpos=4,
        **kwargs,
    ) -> None:
        """
        Generate a `vedo.Plotter` for Volume isosurfacing using a slider.

        Arguments:
            volume : (Volume)
                the Volume object to be isosurfaced.
            isovalues : (float, list)
                isosurface value(s) to be displayed.
            scalar_range : (list)
                scalar range to be used.
            c : str, (list)
                color(s) of the isosurface(s).
            alpha : (float, list)
                opacity of the isosurface(s).
            lego : (bool)
                if True generate a lego plot instead of a surface.
            res : (int)
                resolution of the isosurface.
            use_gpu : (bool)
                use GPU acceleration.
            precompute : (bool)
                precompute the isosurfaces (so slider browsing will be smoother).
            cmap : (str)
                color map name to be used.
            delayed : (bool)
                delay the slider update on mouse release.
            sliderpos : (int)
                position of the slider.
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        Examples:
            - [app_isobrowser.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/app_isobrowser.py)

                ![](https://vedo.embl.es/images/advanced/app_isobrowser.gif)
        """

        super().__init__(**kwargs)

        self.slider = None

        ### GPU ################################
        if use_gpu and hasattr(volume.properties, "GetIsoSurfaceValues"):

            if len(scalar_range) == 2:
                scrange = scalar_range
            else:
                scrange = volume.scalar_range()
            delta = scrange[1] - scrange[0]
            if not delta:
                return

            if isovalue is None:
                isovalue = delta / 3.0 + scrange[0]

            ### isovalue slider callback
            def slider_isovalue(widget, _event):
                value = widget.GetRepresentation().GetValue()
                isovals.SetValue(0, value)

            isovals = volume.properties.GetIsoSurfaceValues()
            isovals.SetValue(0, isovalue)
            self.add(volume.mode(5).alpha(alpha).cmap(c))

            self.slider = self.add_slider(
                slider_isovalue,
                scrange[0] + 0.02 * delta,
                scrange[1] - 0.02 * delta,
                value=isovalue,
                pos=sliderpos,
                title="scalar value",
                show_value=True,
                delayed=delayed,
            )

        ### CPU ################################
        else:

            self._prev_value = 1e30

            scrange = volume.scalar_range()
            delta = scrange[1] - scrange[0]
            if not delta:
                return

            if lego:
                res = int(res / 2)  # because lego is much slower
                slidertitle = ""
            else:
                slidertitle = "scalar value"

            allowed_vals = np.linspace(scrange[0], scrange[1], num=res)

            bacts = {}  # cache the meshes so we dont need to recompute
            if precompute:
                delayed = False  # no need to delay the slider in this case

                for value in allowed_vals:
                    value_name = precision(value, 2)
                    if lego:
                        mesh = volume.legosurface(vmin=value)
                        if mesh.ncells:
                            mesh.cmap(cmap, vmin=scrange[0], vmax=scrange[1], on="cells")
                    else:
                        mesh = volume.isosurface(value).color(c).alpha(alpha)
                    bacts.update({value_name: mesh})  # store it

            ### isovalue slider callback
            def slider_isovalue(widget, _event):

                prevact = self.vol_actors[0]
                if isinstance(widget, float):
                    value = widget
                else:
                    value = widget.GetRepresentation().GetValue()

                # snap to the closest
                idx = (np.abs(allowed_vals - value)).argmin()
                value = allowed_vals[idx]

                if abs(value - self._prev_value) / delta < 0.001:
                    return
                self._prev_value = value

                value_name = precision(value, 2)
                if value_name in bacts:  # reusing the already existing mesh
                    # print('reusing')
                    mesh = bacts[value_name]
                else:  # else generate it
                    # print('generating', value)
                    if lego:
                        mesh = volume.legosurface(vmin=value)
                        if mesh.ncells:
                            mesh.cmap(cmap, vmin=scrange[0], vmax=scrange[1], on="cells")
                    else:
                        mesh = volume.isosurface(value).color(c).alpha(alpha)
                    bacts.update({value_name: mesh})  # store it

                self.remove(prevact).add(mesh)
                self.vol_actors[0] = mesh

            ################################################

            if isovalue is None:
                isovalue = delta / 3.0 + scrange[0]

            self.vol_actors = [None]
            slider_isovalue(isovalue, "")  # init call
            if lego:
                if self.vol_actors[0]:
                    self.vol_actors[0].add_scalarbar(pos=(0.8, 0.12))

            self.slider = self.add_slider(
                slider_isovalue,
                scrange[0] + 0.02 * delta,
                scrange[1] - 0.02 * delta,
                value=isovalue,
                pos=sliderpos,
                title=slidertitle,
                show_value=True,
                delayed=delayed,
            )


##############################################################################
class Browser(Plotter):
    """Browse a series of vedo objects by using a simple slider."""

    def __init__(
        self,
        objects=(),
        sliderpos=((0.50, 0.07), (0.95, 0.07)),
        c=None,  # slider color
        slider_title="",
        font="Calco",  # slider font
        resetcam=False,  # resetcam while using the slider
        **kwargs,
    ):
        """
        Browse a series of vedo objects by using a simple slider.

        The input object can be a list of objects or a list of lists of objects.

        Arguments:
            objects : (list)
                list of objects to be browsed.
            sliderpos : (list)
                position of the slider.
            c : (str)
                color of the slider.
            slider_title : (str)
                title of the slider.
            font : (str)
                font of the slider.
            resetcam : (bool)
                resetcam while using the slider.
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        Examples:
            ```python
            from vedo import load, dataurl
            from vedo.applications import Browser
            meshes = load(dataurl+'timecourse1d.npy') # python list of Meshes
            plt = Browser(meshes, bg='k')             # vedo.Plotter
            plt.show(interactive=False, zoom='tight') # show the meshes
            plt.play(dt=50)                           # delay in milliseconds
            plt.close()
            ```

        - [morphomatics_tube.py](https://github.com/marcomusy/vedo/tree/master/examples/other/morphomatics_tube.py)
        """
        kwargs.pop("N", 1)
        kwargs.pop("shape", [])
        kwargs.pop("axes", 1)
        super().__init__(**kwargs)

        if isinstance(objects, str):
            objects = vedo.file_io.load(objects)

        self += objects

        if len(objects)>0 and is_sequence(objects[0]):
            nobs = len(objects[0])
            for ob in objects:
                n = len(ob)
                msg = f"in Browser lists must have the same length but found {n} and {nobs}"
                assert len(ob) == nobs, msg
        else:
            nobs = len(objects)
            if nobs:
                objects = [objects]

        self.slider = None
        self.timer_callback_id = None
        self._oldk = None

        # define the slider func ##########################
        def slider_function(_widget=None, _event=None):

            k = int(self.slider.value)

            if k == self._oldk:
                return  # no change
            self._oldk = k

            n = len(objects)
            m = len(objects[0])
            for i in range(n):
                for j in range(m):
                    ak = objects[i][j]
                    try:
                        if j == k:
                            ak.on()
                            akon = ak
                        else:
                            ak.off()
                    except AttributeError:
                        pass

            try:
                tx = str(k)
                if slider_title:
                    tx = slider_title + " " + tx
                elif n == 1 and akon.filename:
                    tx = akon.filename.split("/")[-1]
                    tx = tx.split("\\")[-1]  # windows os
                elif akon.name:
                    tx = ak.name + " " + tx
            except:
                pass
            self.slider.title = tx

            if resetcam:
                self.reset_camera()
            self.render()

        ##################################################

        self.slider_function = slider_function
        self.slider = self.add_slider(
            slider_function,
            0.5,
            nobs - 0.5,
            pos=sliderpos,
            font=font,
            c=c,
            show_value=False,
        )
        self.slider.GetRepresentation().SetTitleHeight(0.020)
        slider_function()  # init call

    def play(self, dt=100):
        """Start playing the slides at a given speed."""
        self.timer_callback_id = self.add_callback("timer", self.slider_function)
        self.timer_callback("start", dt=dt)
        self.interactive()


#############################################################################################
