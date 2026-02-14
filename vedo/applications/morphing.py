#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Morphing application plotters."""

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

class MorphPlotter(Plotter):
    """
    A Plotter with 3 renderers to show the source, target and warped meshes.

    Examples:
        - [warp4b.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp4b.py)

            ![](https://vedo.embl.es/images/advanced/warp4b.jpg)
    """

    def __init__(self, source, target, **kwargs):

        vedo.settings.enable_default_keyboard_callbacks = False
        vedo.settings.enable_default_mouse_callbacks = False

        kwargs.update({"N": 3})
        kwargs.update({"sharecam": 0})

        super().__init__(**kwargs)

        self.n_intermediates = 10  # number of intermediate shapes to generate
        self.intermediates = []
        self.auto_warping = True   # automatically warp the source mesh on click
        self.automatic_picking_distance = 0.075
        self.automatic_picking_use_source = False # use source mesh to pick points
        self.cmap_name = "coolwarm"
        self.output_filename = "morphed.vtk"
        self.nbins = 25

        self.source = source.pickable(True)
        self.target = target.pickable(False)
        self.dottedln = None
        self.clicked = []
        self.sources = []
        self.targets = []
        self.warped = None
        self.source_labels = None
        self.target_labels = None
        self.msg0 = Text2D(
            "Pick a point on the surface",
            pos="bottom-center", c='white', bg="blue4", alpha=1, font="Calco")
        self.msg1 = Text2D(pos="bottom-center", c='white', bg="blue4", alpha=1, font="Calco")
        self.instructions = Text2D(s=0.7, bg="blue4", alpha=0.1, font="Calco")
        self.instructions.text(
            "  Morphological alignment of 3D surfaces\n"
            "Pick a point on the source surface, then\n"
            "pick the corresponding point on the target \n"
            "Pick at least 4 point pairs. Press:\n"
            "- d to delete the last landmark pair\n"
            "- C to clear all landmarks\n"
            "- f to add a pair of fixed landmarks\n"
            "- a to auto-pick additional landmarks\n"
            "- z to compute and show the residuals\n"
            "- g to generate intermediate shapes\n"
            "- Ctrl+s to save the morphed mesh\n"
            "- q to quit and proceed"
        )
        self.output_text = Text2D("Morphing Output", font="Calco")
        self.at(0).add_renderer_frame()
        self.add(source, self.msg0, self.instructions).reset_camera()
        self.at(1).add_renderer_frame()
        self.add(Text2D(f"Target: {target.filename[-35:]}", bg="blue4", alpha=0.1, font="Calco"))
        self.add(self.msg1, target)
        cam1 = self.camera  # save camera at 1
        self.at(2).background("k9")
        self.camera = cam1  # use the same camera of renderer1

        self.add(target, self.output_text)
        self.add_renderer_frame()

        self.callid1 = self.add_callback("KeyPress", self.on_keypress)
        self.callid2 = self.add_callback("LeftButtonPress", self.on_click)
        self._interactive = True

    ################################################
    def update(self):
        """Update the rendering window"""
        source_pts = Points(self.sources).color("purple5").ps(12)
        target_pts = Points(self.targets).color("purple5").ps(12)
        source_pts.name = "source_pts"
        target_pts.name = "target_pts"
        self.source_labels = source_pts.labels2d("id", c="purple3")
        self.target_labels = target_pts.labels2d("id", c="purple3")
        self.source_labels.name = "source_pts"
        self.target_labels.name = "target_pts"
        self.at(0).remove("source_pts").add(source_pts, self.source_labels)
        self.at(1).remove("target_pts").add(target_pts, self.target_labels)
        self.render()

        if self.auto_warping:
            if len(self.sources) == len(self.targets) and len(self.sources) > 3:
                self.warped = self.source.clone().warp(self.sources, self.targets)
                self.warped.name = "warped"
                self.at(2).remove("warped").add(self.warped)
                self.render()

    def on_click(self, evt):
        """Handle mouse click events"""
        if evt.object == self.source:
            self.sources.append(evt.picked3d)
            self.source.pickable(False)
            self.target.pickable(True)
            self.msg0.text("--->")
            self.msg1.text("now pick a target point")
            self.update()
        elif evt.object == self.target:
            self.targets.append(evt.picked3d)
            self.source.pickable(True)
            self.target.pickable(False)
            self.msg0.text("now pick a source point")
            self.msg1.text("<---")
            self.update()

    def on_keypress(self, evt):
        """Handle keyboard events"""
        if evt.keypress == "C":
            self.sources.clear()
            self.targets.clear()
            self.at(0).remove("source_pts")
            self.at(1).remove("target_pts")
            self.at(2).remove("warped")
            self.msg0.text("CLEARED! Pick a point here")
            self.msg1.text("")
            self.source.pickable(True)
            self.target.pickable(False)
            self.update()

        if evt.keypress == "w":
            rep = (self.warped.properties.GetRepresentation() == 1)
            self.warped.wireframe(not rep)
            self.render()

        if evt.keypress == "d":
            if len(self.sources) == len(self.targets) + 1:
                self.sources.pop()
            elif len(self.targets) == len(self.sources) + 1:
                self.targets.pop()
            elif len(self.sources) == 0 or len(self.targets) == 0:
                return
            n = min(len(self.sources), len(self.targets))
            self.sources = self.sources[:n-1]
            self.targets = self.targets[:n-1]
            self.msg0.text("Last point deleted! Pick a point here")
            self.msg1.text("")
            self.source.pickable(True)
            self.target.pickable(False)
            self.update()

        if evt.keypress == "f":
            # add a pair of fixed landmarks, a point that does not change
            if evt.object == self.source and len(self.sources) == len(self.targets):
                self.sources.append(evt.picked3d)
                self.targets.append(evt.picked3d)
                self.update()

        if evt.keypress == "m":
            if len(self.sources) == len(self.targets) and len(self.sources) > 3:
                self.warped = self.source.clone().warp(self.sources, self.targets)
                self.warped.name = "warped"
                self.output_text.text("Morphed output:")
                self.at(2).remove("warped").add(self.warped).render()

        if evt.keypress == 'g':  ##------- generate intermediate shapes
            if not self.warped:
                vedo.printc("Morph the source mesh first.", c="r")
                return
            if len(self.sources) != len(self.targets) or len(self.sources) < 4:
                vedo.printc("Pick at least 4 pairs of points.", c="r")
                return
            self.output_text.text("Generating intermediate shapes...")
            self.output_text.c("white").background("red4")
            self.arrow_starts = np.array(self.sources)
            self.arrow_stops = np.array(self.targets)
            self.dottedln = vedo.Lines(self.arrow_starts, self.arrow_stops, res=self.n_intermediates)
            self.dottedln.name = "warped"
            self.dottedln.c("blue3").ps(5).alpha(0.5)
            self.at(2).add(self.dottedln).render()
            self.intermediates = []
            allpts = self.dottedln.vertices
            allpts = allpts.reshape(len(self.arrow_starts), self.n_intermediates + 1, 3)
            for i in range(self.n_intermediates):
                pi = allpts[:, i, :]
                m_nterp = self.source.clone().warp(self.arrow_starts, pi)
                m_nterp.name = f"morphed"
                m_nterp.c("blue4", 0.05)
                self.intermediates.append(m_nterp)
            self.output_text.text(f"Morphed output + Interpolation ({self.n_intermediates} shapes):")
            self.output_text.c("k").background(None)
            self.at(2).add(self.intermediates).render()

        if evt.keypress == "a":
            # auto-pick points on the target surface
            if not self.warped:
                vedo.printc("At least 4 points are needed.", c="r")
                return
            if len(self.sources) > len(self.targets):
                self.sources.pop()

            if self.automatic_picking_use_source:
                pts = self.warped.clone().subsample(self.automatic_picking_distance)
                TI = self.warped.transform.compute_inverse()
                # d = self.target.diagonal_size()
                # r = d * self.automatic_picking_distance
                for p in pts.coordinates:
                    # pp = vedo.utils.closest(p, self.targets)[1]
                    # if vedo.mag(pp - p) < r:
                    #     continue
                    q = self.target.closest_point(p)
                    self.sources.append(TI(p))
                    self.targets.append(q)
            else:
                pts = self.target.clone().subsample(self.automatic_picking_distance)
                d = self.target.diagonal_size()
                r = d * self.automatic_picking_distance
                TI = self.warped.transform.compute_inverse()
                for p in pts.coordinates:
                    pp = vedo.utils.closest(p, self.targets)[1]
                    if vedo.mag(pp - p) < r:
                        continue
                    q = self.warped.closest_point(p)
                    self.sources.append(TI(q))
                    self.targets.append(p)

            self.source.pickable(True)
            self.target.pickable(False)
            self.update()

        if evt.keypress == "z" or evt.keypress == "a":
            dists = self.warped.distance_to(self.target, signed=True)
            v = np.std(dists) * 2
            self.warped.cmap(self.cmap_name, dists, vmin=-v, vmax=+v)

            h = vedo.pyplot.histogram(
                dists,
                bins=self.nbins,
                title=" ",
                xtitle=f"STD = {v/2:.2f}",
                ytitle="",
                c=self.cmap_name,
                xlim=(-v, v),
                aspect=16/9,
                axes=dict(
                    number_of_divisions=5,
                    text_scale=2,
                    xtitle_offset=0.075,
                    xlabel_justify="top-center",
                ),
            )

            # try to fit a gaussian to the histogram
            def gauss(x, A, B, sigma):
                return A + B * np.exp(-x**2 / (2 * sigma**2))
            try:
                from scipy.optimize import curve_fit
                inits = [0, len(dists)/self.nbins*2.5, v/2]
                popt, _ = curve_fit(gauss, xdata=h.centers, ydata=h.frequencies, p0=inits)
                x = np.linspace(-v, v, 300)
                h += vedo.pyplot.plot(x, gauss(x, *popt), like=h, lw=1, lc="k2")
                h["Axes"]["xtitle"].text(f":sigma = {abs(popt[2]):.3f}", font="VictorMono")
            except:
                pass

            h = h.clone2d(pos="bottom-left", size=0.575)
            h.name = "warped"
            self.at(2).add(h)
            self.render()
        
        if evt.keypress == "Ctrl+s":
            # write the warped mesh to file along with the transformation
            if self.warped:
                m =  self.warped.clone()
                m.pointdata.remove("Scalars")
                m.pointdata.remove("Distance")
                m.write(str(self.output_filename))
                matfile = str(self.output_filename).split(".")[0] + ".mat"
                m.transform.write(matfile)
                print(f"Warped mesh saved to: {self.output_filename}\n with transformation: {matfile}")

        if evt.keypress == "q":
            self.break_interaction()

########################################################################################
class MorphByLandmarkPlotter(Plotter):

    def __init__(self, msh, init_pts=(), show_labels=True, **kwargs):
        """
        A Plotter to morph a mesh by moving points on it.
        This application allows you to morph a mesh by moving points on it.
        It is useful in shape analysis, morphometrics, and mesh deformation.
        The initial points are used as source points, and the points are moved to the target positions.

        Arguments:
            msh : (Mesh)
                the mesh to morph
            init_pts : (list of tuples)
                initial points to use for morphing
            show_labels : (bool)
                show point labels on the mesh
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.
        
        Example:
            ```python
            from vedo import Mesh, MorphByLandmarkPlotter
            msh = Mesh(dataurl + "bunny.ply")
            pts = msh.clone().subsample(0.1) # subsample points
            plt = MorphByLandmarkPlotter(msh, pts.coordinates, axes=1)
            plt.morphing_on_change = True  # morph the mesh when points change
            plt.show()
            ```
        """
        vedo.settings.enable_default_keyboard_callbacks = False
        vedo.settings.enable_default_mouse_callbacks = False

        super().__init__(**kwargs)
        self.parallel_projection(True)

        self.step = 10
        self.point_size = 20
        self.morphing_mode = "2d"  # "2d", "3d"
        self.morphing_on_change = True  # morph the mesh when points change
        self.recompute_normals = False
        self.move_keys = ["Left", "Right", "Up", "Down", "UP", "DOWN", "n", "N"]
        self.output_filename = "morphed.vtk"

        self.mesh = msh.clone().rename("morphable_mesh").pickable(1)
        self.mesh.compute_normals(cells=False)
        self.selected_pid = None
        self.highlighted = None

        self.sources = list(init_pts)
        self.targets = list(init_pts)
        self._sources = np.array(self.sources, dtype=float)
        self.vhighlighted = vedo.Point(r=self.point_size + 5, c="red6", alpha=0.5)
        self.vhighlighted.off().pickable(0)

        self.current_pid = None
        self.current_pt = None
        self.id_callback1 = self.add_callback("key press", self._fkey)
        self.id_callback2 = self.add_callback("mouse click", self._on_mouse_click)

        self.vtargets = Points(self.targets, r=self.point_size, c="blue6", alpha=0.5)
        self.vtargets.name = "vtargets"

        self.vlines = vedo.Lines(self.sources, self.targets, c="k4")
        self.vlines.name = "vlines"

        self.vsources = Points(self.sources, r=self.point_size - 1)
        self.vsources.c("red6", alpha=0.5).pickable(0)
        self.vsources.name = "vsources"

        self.instructions = Text2D(
            "Click on a point to select it.\n"
            "Use the arrow keys to move the selected point.\n"
            "Press n or N to move the point along its normal.\n"
            "Press o to add a new point pair.\n"
            "Press +/- to change the step size.\n"
            "Press m to morph the mesh.\n"
            "Press r to reset the camera.\n"
            "Press Ctrl+s to save the morphed mesh.\n"
            "Press q to quit.",
            pos="top-left",
            bg="blue4",
            alpha=0.1,
            font="Calco",
            s=0.7,
        )

        self.status = Text2D(
            "No point selected.",
            pos="bottom-left",
            font="Calco",
            bg="yellow5",
            alpha=0.2,
            s=0.8,
        )

        self.labels = None
        if show_labels:
            self.labels = self.vtargets.labels2d("id")
            self.labels.name = "labels"
        self.add(
            self.mesh,
            self.labels,
            self.vsources,
            self.vtargets,
            self.vlines,
            self.vhighlighted,
            self.instructions,
            self.status,
        )

    def do_morph(self):
        """
        Perform the morphing operation on the mesh.
        This method morphs the mesh using the provided source and target points.
        """
        self.sources = list(self.vsources.coordinates)
        self.targets = list(self.vtargets.coordinates)

        self.mesh.warp(self.sources, self.targets, mode=self.morphing_mode)

        self.sources = list(self.targets)

        self.vsources = Points(self.sources, r=self.point_size - 1)
        self.vsources.c("red6", alpha=0.5).pickable(0)
        self.vsources.name = "vsources"

        self.vtargets = Points(self.targets, r=self.point_size)
        self.vtargets.c("blue6", alpha=0.5).pickable(0)
        self.vtargets.name = "vtargets"

        self.vlines = vedo.Lines(self.vsources, self.targets, c="k4").pickable(0)
        self.vlines.name = "vlines"
        self.remove("vtargets", "vlines")
        self.add(self.vtargets, self.vlines).render()

    def _on_mouse_click(self, evt):
        if evt.picked3d is None:
            return
        self.current_pid = self.vtargets.closest_point(evt.picked3d, return_point_id=True)
        if self.current_pid < 0:
            self.status.text("No point exists. Press o to place a point.")
            self.render()
            return
        self.current_pt = self.targets[self.current_pid]
        self.status.text(
            f"Selected point ID {self.current_pid}" 
            f" coordinates: {precision(self.current_pt, 3)}"
        )
        self.vhighlighted.on()
        self.vhighlighted.coordinates = [self.current_pt]
        self.render()

    def _fkey(self, evt):
        k = evt.keypress
        # print(f"Key pressed: {k}")

        if k == "q":
            self.break_interaction()

        elif k == "r":
            self.reset_camera().render()

        elif k == "o":
            if evt.object.name == "morphable_mesh":
                self.sources.append(evt.picked3d)
                if len(self._sources) == 0:
                    self._sources = np.array([evt.picked3d], dtype=float)
                else:
                    self._sources = np.append(self._sources, [evt.picked3d], axis=0)
                self.targets.append(evt.picked3d)
                self.status.text(
                    f"Added point {precision(evt.picked3d, 3)}. "
                    f" Total number of points: {len(self._sources)}"
                )

                self.vsources = Points(self._sources, r=self.point_size - 1)
                self.vsources.c("red6", alpha=0.5).pickable(0)
                self.vsources.name = "vsources"

                self.vtargets = Points(self.targets, r=self.point_size)
                self.vtargets.c("blue6", alpha=0.5).pickable(0)
                self.vtargets.name = "vtargets"

                self.vlines = vedo.Lines(self._sources, self.targets, c="k5").pickable(0)
                self.vlines.name = "vlines"

                self.remove("vsources", "vtargets", "vlines")
                self.add(self.vsources, self.vtargets, self.vlines).render()

        elif k == "c":  # clear all points
            self.sources.clear()
            self.targets.clear()
            self._sources = np.array(self.sources, dtype=float)
            self.vsources = Points(self._sources, r=self.point_size - 1)
            self.vsources.c("red6", alpha=0.5).pickable(0)
            self.vsources.name = "vsources"
            self.vtargets = Points(self.targets, r=self.point_size)
            self.vtargets.c("blue6", alpha=0.5).pickable(0)
            self.vtargets.name = "vtargets"
            self.vlines = vedo.Lines(self._sources, self.targets, c="k4").pickable(0)
            self.vlines.name = "vlines"
            self.remove("vsources", "vtargets", "vlines", "labels")
            self.add(self.vsources, self.vtargets, self.vlines).render()

        elif k == "d":  # delete the last point pair
            if len(self.sources) == 0 or len(self.targets) == 0:
                self.status.text("No points to delete.")
                return
            if len(self.sources) == len(self.targets) + 1:
                self.sources.pop()
            elif len(self.targets) == len(self.sources) + 1:
                self.targets.pop()

            self._sources = np.array(self.sources, dtype=float)
            self.vsources = Points(self._sources, r=self.point_size - 1)
            self.vsources.c("red6", alpha=0.5).pickable(0)
            self.vsources.name = "vsources"
            self.vtargets = Points(self.targets, r=self.point_size)
            self.vtargets.c("blue6", alpha=0.5).pickable(0)
            self.vtargets.name = "vtargets"
            self.vlines = vedo.Lines(self._sources, self.targets, c="k4").pickable(0)
            self.vlines.name = "vlines"
            self.remove("vsources", "vtargets", "vlines", "labels")
            self.add(self.vsources, self.vtargets, self.vlines).render()

        elif k in self.move_keys:  # use arrows to move the picked point
            if self.current_pid is None:
                self.status.text("No point picked. Click to pick a point first.")
                return

            pid = self.current_pid

            std_keys = ["Left", "Right", "Up", "Down", "UP", "DOWN", "n", "N"]
            i = self.move_keys.index(k)
            k = std_keys[i]  # normalize the key to the standard keys

            if   k == "Left":  self.targets[pid] += [-self.step, 0, 0]
            elif k == "Right": self.targets[pid] += [self.step, 0, 0]
            elif k == "Up":    self.targets[pid] += [0, self.step, 0]
            elif k == "Down":  self.targets[pid] += [0, -self.step, 0]
            elif k == "UP":    self.targets[pid] += [0, 0, self.step]
            elif k == "DOWN":  self.targets[pid] += [0, 0, -self.step]
            elif k == "n" or k == "N":
                # move along the normal of the picked point
                if self.recompute_normals:
                    self.mesh.compute_normals(cells=False)
                mid = self.mesh.closest_point(self.current_pt, return_point_id=True)
                normals = self.mesh.point_normals.copy()
                if k == "n":
                    self.targets[pid] += normals[mid] * self.step
                else:
                    self.targets[pid] -= normals[mid] * self.step

            self.status.text(
                f"Moved point ID {self.current_pid}" 
                f" {precision(self.targets[pid], 3)} {k} by {self.step} units"
            )
            self.vtargets = Points(self.targets, r=self.point_size)
            self.vtargets.c("blue6", alpha=0.5).pickable(0)
            self.vtargets.name = "vtargets"
            self.vlines = vedo.Lines(self._sources, self.targets, c="k4").pickable(0)
            self.vlines.name = "vlines"
            self.remove("vlines", "vtargets").add(self.vlines, self.vtargets)
            if self.morphing_on_change:
                self.do_morph()
            else:
                self.render()

        elif k == "m":
            self.do_morph()

        elif k == "l" and evt.object:  # toggle lines visibility
            ev = evt.object.properties.GetEdgeVisibility()
            evt.object.properties.SetEdgeVisibility(not ev)
            self.render()

        elif k in ["plus", "equal", "minus", "underscore"]:  # change step size
            if k in ["plus", "equal"]:
                self.step += self.step / 10
            elif k in ["minus", "underscore"]:
                self.step -= self.step / 10
            self.step = max(0, self.step)  # ensure step is at least 0
            self.status.text(f"Step size changed to {self.step:.2f} units.")
            self.render()

        elif k == "Ctrl+s":  # write the morphed mesh to file
            if self.mesh:
                # self.mesh.compute_normals(cells=False)
                self.mesh.write(self.output_filename)
                self.status.text(f"Morphed mesh saved to {self.output_filename}")
                self.render()

########################################################################################
class MorphBySplinesPlotter(Plotter):

    def __init__(self, meshes, **kwargs):
        """
        A Plotter to morph a mesh by moving points on it using splines.
        It is useful in shape analysis, morphometrics, and mesh deformation.

        Arguments:
            meshes : (list of Mesh)
                the mesh set to be matched
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.
        
        Example:
            ```python
            from vedo import Mesh, dataurl
            from vedo.applications import MorphBySplinesPlotter

            m0 = Mesh(dataurl+"250.vtk").subdivide().c("o8", 0.5)
            mm = Mesh(dataurl+"270.vtk").subdivide().c("k5", 0.5).x(800)
            m1 = Mesh(dataurl+"290.vtk").subdivide().c("g8", 0.5).x(2000)

            plt = MorphBySplinesPlotter([m0, mm, m1], size=(2050, 1250), axes=14)
            plt.n_intermediates = 5  # number of intermediate shapes to generate
            plt.show().close()
            ```
        """
        vedo.settings.enable_default_keyboard_callbacks = False
        vedo.settings.enable_default_mouse_callbacks = False
        
        super().__init__(**kwargs)

        self.parallel_projection(True)
        self.add_callback("key press", self._func)

        self.n_intermediates = 10  # number of intermediate shapes to generate
        self.mode = "2d"  # "2d", "3d" morphing mode

        self.splinetools = []
        self.intermediates = []
        self.meshes = meshes
        self.instructions = Text2D(
            "     --- Morphing by Splines ---\n"
            "Ctrl+l to load the splines from file.\n"
            "Ctrl+a to add a new spline.\n"
            "Ctrl+f to fit the splines to the mesh.\n"
            "Ctrl+z to remove the last spline.\n"
            "Ctrl+c to clear all splines.\n"
            "Ctrl+g to generate intermediate shapes.\n"
            "Ctrl+s to save the splines.\n"
            "Ctrl+q to quit.",
            pos="top-left",
            bg="blue4",
            alpha=0.1,
            font="Calco",
            s=0.75,
        )
        self.status = Text2D(
            "Use Ctrl+a to add a new spline, or Ctrl+l to load splines from file.",
            pos="bottom-left",
            font="Calco",
            bg="yellow5",
            alpha=0.5,
            s=1.1,
        )
        b = meshes[0].bounds()
        self.offset = [0, (b[3] - b[2])/2, 0]
        self.output_prefix = "morphed_"
        self.output_suffix = ".vtk"
        self.output_spline_filename = f"{self.output_prefix}splines.npy"
        self.output_offset = 0  # used to generate an offset for the output filenames
        self.output_offset_factor = 1
        self.add(meshes, self.status, self.instructions)


    def _func(self, evt):

        n = len(self.meshes)

        if evt.keypress in ["q", "Ctrl+q"]:
            self.break_interaction()

        elif evt.keypress == "Ctrl+a":
            if len(self.splinetools) > 0:
                init_nodes = self.splinetools[-1].nodes() + self.offset
            else:
                b = self.meshes[0].bounds()
                init_nodes = vedo.Line(
                    [b[0], b[2], (b[4] + b[5]) / 2],
                    [b[1], b[3], (b[4] + b[5]) / 2], res=n).coordinates
            st = self.add_spline_tool(
                init_nodes,
                lc=len(self.splinetools),
                lw=3,
                can_add_nodes=False,
            )
            self.splinetools.append(st)
            self.status.text(f"Added spline.")
            self.render()

        elif evt.keypress == "Ctrl+f":
            for st in self.splinetools:
                nodes = st.nodes()
                for i in range(len(nodes)):
                    nodes[i] = self.meshes[i].closest_point(nodes[i])
                st.set_nodes(nodes)
            self.status.text("Snapped spline points to meshes.")
            self.render()

        elif evt.keypress == "Ctrl+s":
            all_nodes = []
            for st in self.splinetools:
                all_nodes.append(st.nodes())
            all_nodes = np.array(all_nodes)
            fname = f"{self.output_prefix}splines.npy"
            np.save(fname, all_nodes)
            for i, mi in enumerate(self.intermediates):
                mi.write(f"{self.output_prefix}{self.output_offset+i*int(self.output_offset_factor)}{self.output_suffix}")
            self.status.text(
                f"Saved {len(self.splinetools)} splines to {fname}.\n"
                f"Saved {len(self.intermediates)} intermediate meshes saved as {self.output_prefix}*{self.output_suffix}")
            self.render()

        elif evt.keypress == "Ctrl+l":
            if not os.path.exists(f"{self.output_prefix}splines.npy"):
                self.status.text(f"File {self.output_prefix}splines.npy does not exist.")
                self.render()
                return
            all_nodes = np.load(f"{self.output_prefix}splines.npy")
            for st in self.splinetools:
                st.off()
            self.splinetools.clear()
            for nodes in all_nodes:
                st = self.add_spline_tool(
                    nodes,
                    lc=len(self.splinetools),
                    lw=3,
                    can_add_nodes=False,
                )
                self.splinetools.append(st)
            self.status.text(
                f"Loaded {len(self.splinetools)} splines from {self.output_prefix}splines.npy")
            self.render()

        elif evt.keypress == "Ctrl+z":
            if len(self.splinetools) > 0:
                self.splinetools[-1].off()
                self.splinetools.pop()
                self.status.text(f"Removed last spline.")
                self.render()
        
        elif evt.keypress == "Ctrl+c":
            self.remove(self.intermediates)
            self.intermediates.clear()
            self.status.text("Cleared all intermediate shapes.")
            self.render()

        elif evt.keypress == "Ctrl+g":
            all_pts = []
            for st in self.splinetools:
                nodes = st.nodes()
                coords = Spline(nodes, res=self.n_intermediates+1).coordinates
                all_pts.append(coords)
            all_pts = np.array(all_pts)
            self.remove(self.intermediates)
            self.intermediates.clear()
            if len(all_pts) == 0:
                self.status.text("No splines defined. Use Ctrl+a to add a spline.")
                self.render()
                return
            warped0 = self.meshes[0].clone().wireframe().lighting("off").c("k4").alpha(1/n/5)
            for i in range(1, self.n_intermediates+1):
                sources = all_pts[:, i - 1]
                targets = all_pts[:, i]
                warped0.warp(sources, targets, mode=self.mode)
                self.intermediates.append(warped0.clone())
            self.status.text(f"Generated {len(self.intermediates)} intermediate shapes.")
            self.add(self.intermediates).render()


########################################################################################
