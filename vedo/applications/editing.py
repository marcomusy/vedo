#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Interactive editing and drawing applications."""

import os

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo.colors import color_map, get_color
from vedo.utils import is_sequence, lin_interpolate, mag, precision
from vedo.plotter import Plotter
from vedo.plotter.modes import MousePan
from vedo.pointcloud import fit_plane, Points
from vedo.shapes import Line, Ribbon, Spline, Text2D
from vedo.pyplot import CornerHistogram, histogram
from vedo.addons import SliderWidget

class FreeHandCutPlotter(Plotter):
    """A tool to edit meshes interactively."""

    # thanks to Jakub Kaminski for the original version of this script
    def __init__(
        self,
        mesh: vedo.Mesh | vedo.Points,
        splined=True,
        font="Bongas",
        alpha=0.9,
        lw=4,
        lc="red5",
        pc="red4",
        c="green3",
        tc="k9",
        tol=0.008,
        **options,
    ):
        """
        A `vedo.Plotter` derived class which edits polygonal meshes interactively.

        Can also be invoked from command line with:

        ```bash
        vedo --edit https://vedo.embl.es/examples/data/porsche.ply
        ```

        Usage:
            - Left-click and hold to rotate
            - Right-click and move to draw line
            - Second right-click to stop drawing
            - Press "c" to clear points
            -       "z/Z" to cut mesh (Z inverts inside-out the selection area)
            -       "L" to keep only the largest connected surface
            -       "s" to save mesh to file (tag `_edited` is appended to filename)
            -       "u" to undo last action
            -       "h" for help, "i" for info

        Arguments:
            mesh : (Mesh, Points)
                The input Mesh or pointcloud.
            splined : (bool)
                join points with a spline or a simple line.
            font : (str)
                Font name for the instructions.
            alpha : (float)
                transparency of the instruction message panel.
            lw : (str)
                selection line width.
            lc : (str)
                selection line color.
            pc : (str)
                selection points color.
            c : (str)
                background color of instructions.
            tc : (str)
                text color of instructions.
            tol : (int)
                tolerance of the point proximity.
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.

        Examples:
            - [cut_freehand.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/cut_freehand.py)

                ![](https://vedo.embl.es/images/basic/cutFreeHand.gif)
        """

        if not isinstance(mesh, Points):
            vedo.logger.error("FreeHandCutPlotter input must be Points or Mesh")
            raise RuntimeError()

        super().__init__(**options)

        self.mesh = mesh
        self.mesh_prev = mesh
        self.splined = splined
        self.linecolor = lc
        self.linewidth = lw
        self.pointcolor = pc
        self.color = c
        self.alpha = alpha

        self.msg = "Right-click and move to draw line\n"
        self.msg += "Second right-click to stop drawing\n"
        self.msg += "Press L to extract largest surface\n"
        self.msg += "        z/Z to cut mesh (s to save)\n"
        self.msg += "        c to clear points, u to undo"
        self.txt2d = Text2D(self.msg, pos="top-left", font=font, s=0.9)
        self.txt2d.c(tc).background(c, alpha).frame()

        self.idkeypress = self.add_callback("KeyPress", self._on_keypress)
        self.idrightclck = self.add_callback("RightButton", self._on_right_click)
        self.idmousemove = self.add_callback("MouseMove", self._on_mouse_move)
        self.drawmode = False
        self.tol = tol  # tolerance of point distance
        self.cpoints = []
        self.points = None
        self.spline = None
        self.jline = None
        self.topline = None
        self.top_pts = []

    def init(self, init_points):
        """Set an initial number of points to define a region"""
        if isinstance(init_points, Points):
            self.cpoints = init_points.coordinates
        else:
            self.cpoints = np.array(init_points)
        self.points = Points(self.cpoints, r=self.linewidth).c(self.pointcolor).pickable(0)
        if self.splined:
            self.spline = Spline(self.cpoints, res=len(self.cpoints) * 4)
        else:
            self.spline = Line(self.cpoints)
        self.spline.lw(self.linewidth).c(self.linecolor).pickable(False)
        self.jline = Line(self.cpoints[0], self.cpoints[-1], lw=1, c=self.linecolor).pickable(0)
        self.add([self.points, self.spline, self.jline]).render()
        return self

    def _on_right_click(self, _evt):
        self.drawmode = not self.drawmode  # toggle mode
        if self.drawmode:
            self.txt2d.background(self.linecolor, self.alpha)
        else:
            self.txt2d.background(self.color, self.alpha)
            if len(self.cpoints) > 2:
                self.remove([self.spline, self.jline])
                if self.splined:  # show the spline closed
                    self.spline = Spline(self.cpoints, closed=True, res=len(self.cpoints) * 4)
                else:
                    self.spline = Line(self.cpoints, closed=True)
                self.spline.lw(self.linewidth).c(self.linecolor).pickable(False)
                self.add(self.spline)
        self.render()

    def _on_mouse_move(self, evt):
        if self.drawmode:
            cpt = self.compute_world_coordinate(evt.picked2d)  # make this 2d-screen point 3d
            if self.cpoints and mag(cpt - self.cpoints[-1]) < self.mesh.diagonal_size() * self.tol:
                return  # new point is too close to the last one. skip
            self.cpoints.append(cpt)
            if len(self.cpoints) > 2:
                self.remove([self.points, self.spline, self.jline, self.topline])
                self.points = Points(self.cpoints, r=self.linewidth).c(self.pointcolor).pickable(0)
                if self.splined:
                    self.spline = Spline(self.cpoints, res=len(self.cpoints) * 4)  # not closed here
                else:
                    self.spline = Line(self.cpoints)

                if evt.actor:
                    self.top_pts.append(evt.picked3d)
                    self.topline = Points(self.top_pts, r=self.linewidth)
                    self.topline.c(self.linecolor).pickable(False)

                self.spline.lw(self.linewidth).c(self.linecolor).pickable(False)
                self.txt2d.background(self.linecolor)
                self.jline = Line(self.cpoints[0], self.cpoints[-1], lw=1, c=self.linecolor).pickable(0)
                self.add([self.points, self.spline, self.jline, self.topline]).render()

    def _on_keypress(self, evt):
        if evt.keypress.lower() == "z" and self.spline:  # Cut mesh with a ribbon-like surface
            inv = False
            if evt.keypress == "Z":
                inv = True
            self.txt2d.background("red8").text("  ... working ...  ")
            self.render()
            self.mesh_prev = self.mesh.clone()
            tol = self.mesh.diagonal_size() / 2  # size of ribbon (not shown)
            pts = self.spline.coordinates
            n = fit_plane(pts, signed=True).normal  # compute normal vector to points
            rb = Ribbon(pts - tol * n, pts + tol * n, closed=True)
            self.mesh.cut_with_mesh(rb, invert=inv)  # CUT
            self.txt2d.text(self.msg)  # put back original message
            if self.drawmode:
                self._on_right_click(evt)  # toggle mode to normal
            else:
                self.txt2d.background(self.color, self.alpha)
            self.remove([self.spline, self.points, self.jline, self.topline]).render()
            self.cpoints, self.points, self.spline = [], None, None
            self.top_pts, self.topline = [], None

        elif evt.keypress == "L":
            self.txt2d.background("red8")
            self.txt2d.text(" ... removing smaller ... \n ... parts of the mesh ... ")
            self.render()
            self.remove(self.mesh)
            self.mesh_prev = self.mesh
            mcut = self.mesh.extract_largest_region()
            mcut.filename = self.mesh.filename  # copy over various properties
            mcut.name = self.mesh.name
            mcut.scalarbar = self.mesh.scalarbar
            mcut.info = self.mesh.info
            self.mesh = mcut                            # discard old mesh by overwriting it
            self.txt2d.text(self.msg).background(self.color)   # put back original message
            self.add(mcut).render()

        elif evt.keypress == "u":  # Undo last action
            if self.drawmode:
                self._on_right_click(evt)  # toggle mode to normal
            else:
                self.txt2d.background(self.color, self.alpha)
            self.remove([self.mesh, self.spline, self.jline, self.points, self.topline])
            self.mesh = self.mesh_prev
            self.cpoints, self.points, self.spline = [], None, None
            self.top_pts, self.topline = [], None
            self.add(self.mesh).render()

        elif evt.keypress in ("c", "Delete"):
            # clear all points
            self.remove([self.spline, self.points, self.jline, self.topline]).render()
            self.cpoints, self.points, self.spline = [], None, None
            self.top_pts, self.topline = [], None

        elif evt.keypress == "r":  # reset camera and axes
            self.remove(self.axes_instances[0])
            self.axes_instances[0] = None
            self.add_global_axes(axtype=1, c=None)
            self.renderer.ResetCamera()
            self.render()

        elif evt.keypress == "s":
            if self.mesh.filename:
                fname = os.path.basename(self.mesh.filename)
                fname, extension = os.path.splitext(fname)
                fname = fname.replace("_edited", "")
                fname = f"{fname}_edited{extension}"
            else:
                fname = "mesh_edited.vtk"
            self.write(fname)

    def write(self, filename="mesh_edited.vtk") -> "FreeHandCutPlotter":
        """Save the resulting mesh to file"""
        self.mesh.write(filename)
        vedo.logger.info(f"mesh saved to file {filename}")
        return self

    def start(self, *args, **kwargs) -> "FreeHandCutPlotter":
        """Start window interaction (with mouse and keyboard)"""
        acts = [self.txt2d, self.mesh, self.points, self.spline, self.jline]
        self.show(acts + list(args), **kwargs)
        return self


########################################################################
class SplinePlotter(Plotter):
    """
    Interactive drawing of splined curves on meshes.
    """

    def __init__(self, obj, init_points=(), closed=False, splined=True, mode="auto", **kwargs):
        """
        Create an interactive application that allows the user to click points and
        retrieve the coordinates of such points and optionally a spline or line
        (open or closed).
        Input object can be a image file name or a 3D mesh.

        Arguments:
            obj : (Mesh, str)
                The input object can be a image file name or a 3D mesh.
            init_points : (list)
                Set an initial number of points to define a region.
            closed : (bool)
                Close the spline or line.
            splined : (bool)
                Join points with a spline or a simple line.
            mode : (str)
                Set the mode of interaction.
            **kwargs : (dict)
                keyword arguments to pass to a `vedo.plotter.Plotter` instance.
        """
        super().__init__(**kwargs)

        self.verbose = True
        self.splined = splined
        self.resolution = None  # spline resolution (None = automatic)
        self.closed = closed
        self.lcolor = "yellow4"
        self.lwidth = 3
        self.pcolor = "purple5"
        self.psize = 10

        self.cpoints = list(init_points)
        self.vpoints = None
        self.line = None

        if isinstance(obj, str):
            self.object = vedo.file_io.load(obj)
        else:
            self.object = obj

        self.mode = mode
        if self.mode == "auto":
            if isinstance(self.object, vedo.Image):
                self.mode = "image"
                self.parallel_projection(True)
            else:
                self.mode = "TrackballCamera"

        t = (
            "Click to add a point\n"
            "Right-click to remove it\n"
            "Drag mouse to change contrast\n"
            "Press c to clear points\n"
            "Press q to continue"
        )
        self.instructions = Text2D(t, pos="bottom-left", c="white", bg="green", font="Calco")

        self += [self.object, self.instructions]

        self.callid1 = self.add_callback("KeyPress", self._key_press)
        self.callid2 = self.add_callback("LeftButtonPress", self._on_left_click)
        self.callid3 = self.add_callback("RightButtonPress", self._on_right_click)


    def points(self, newpts=None) -> "SplinePlotter" | np.ndarray:
        """Retrieve the 3D coordinates of the clicked points"""
        if newpts is not None:
            self.cpoints = newpts
            self.update()
            return self
        return np.array(self.cpoints)

    def _on_left_click(self, evt):
        if not evt.actor:
            return
        if evt.actor.name == "points":
            # remove clicked point if clicked twice
            pid = self.vpoints.closest_point(evt.picked3d, return_point_id=True)
            self.cpoints.pop(pid)
            self.update()
            return
        p = evt.picked3d
        self.cpoints.append(p)
        self.update()
        if self.verbose:
            vedo.colors.printc("Added point:", precision(p, 4), c="g")

    def _on_right_click(self, evt):
        if evt.actor and len(self.cpoints) > 0:
            self.cpoints.pop()  # pop removes from the list the last pt
            self.update()
            if self.verbose:
                vedo.colors.printc("Deleted last point", c="r")

    def update(self):
        """Update the plot with the new points"""
        self.remove(self.line, self.vpoints)  # remove old points and spline
        self.vpoints = Points(self.cpoints).ps(self.psize).c(self.pcolor)
        self.vpoints.name = "points"
        self.vpoints.pickable(True)  # to allow toggle
        minnr = 1
        if self.splined:
            minnr = 2
        if self.lwidth and len(self.cpoints) > minnr:
            if self.splined:
                try:
                    self.line = Spline(self.cpoints, closed=self.closed, res=self.resolution)
                except ValueError:
                    # if clicking too close splining might fail
                    self.cpoints.pop()
                    return
            else:
                self.line = Line(self.cpoints, closed=self.closed)
            self.line.c(self.lcolor).lw(self.lwidth).pickable(False)
            self.add(self.vpoints, self.line)
        else:
            self.add(self.vpoints)

    def _key_press(self, evt):
        if evt.keypress == "c":
            self.cpoints = []
            self.remove(self.line, self.vpoints).render()
            if self.verbose:
                vedo.colors.printc("==== Cleared all points ====", c="r", invert=True)

    def start(self) -> "SplinePlotter":
        """Start the interaction"""
        self.update()
        self.show(self.object, self.instructions, mode=self.mode)
        return self

########################################################################
class ImageEditor(Plotter):
    """A simple image editor for applying various filters to images."""

    def __init__(self, filename, **kwargs):
        """Initialize the Image Editor with a given image file."""

        if "shape" not in kwargs:
            kwargs["shape"] = (1, 3)
        if "size" not in kwargs:
            kwargs["size"] = (1800, 600)
        if "title" not in kwargs:
            kwargs["title"] = "vedo - Image Editor"

        vedo.settings.enable_default_keyboard_callbacks = False

        super().__init__(**kwargs)

        self.cutfilter = 0.035  # value for the high cutoff frequency in the filter

        self.filename = filename
        self.image0 = vedo.Image(filename)
        self.image2 = self.image0.clone()
        self.image0.pickable(False)

        self.header = Text2D(
            f"File name:\n {filename[-35:]}\nSize: {self.image2.shape}",
            c="w",
            bg="k2",
            alpha=1,
            font="Calco",
            s=1.2,
        )
        self.instructions = Text2D(
            "Press -----------------------\n"
            "Ctrl+e to enhance\n"
            "Ctrl+m to median\n"
            "Ctrl+f to flip horizontally\n"
            "Ctrl+v to flip vertically\n"
            "Ctrl+t to rotate\n"
            "Ctrl+b to binarize\n"
            "Ctrl+i to invert\n"
            "Ctrl+g to smooth\n"
            "Ctrl+h to print info\n"
            "Ctrl+o to increase luminosity\n"
            "Ctrl+l to decrease luminosity\n"
            "Shift+O to increase contrast\n"
            "Shift+L to decrease contrast\n"
            "Ctrl+d to denoise\n"
            "Ctrl+r to reset\n"
            "Ctrl+s to save image\n"
            "Ctrl+c to open color picker\n"
            "q to quit\n"
            "-----------------------------",
            pos=(0.01, 0.8),
            c="w",
            bg="b4",
            alpha=1,
            font="Calco",
            s=0.8,
        )
        self.status = Text2D(
            pos="bottom-center", c="w", bg="r6", alpha=0.8, font="Calco"
        )

        self.pan = MousePan(enable_rotate=False)

        self.add_callback("key", self.key_func)

        self.at(0).add(self.header, self.instructions, self.status)
        self.at(1).add(vedo.Text2D("Input",  font="Calco", c="k5", bg="y5"), self.image0)
        self.at(2).add(vedo.Text2D("Output", font="Calco", c="k5", bg="y5"), self.image2)
        self.user_mode(self.pan).show()
        self.interactor.RemoveObservers("CharEvent")

    def key_func(self, evt):
        """Handle key events for image editing."""
        if evt.keypress in ["q", "Ctrl+q", "Ctrl+w"]:
            self.close()
            return

        elif evt.keypress == "Ctrl+e":
            self.image2.enhance()
            self.status.text("Image enhanced")

        elif evt.keypress == "Ctrl+m":
            self.image2.median()
            self.status.text("Image median filtered")

        elif evt.keypress == "Ctrl+f":
            self.image2.mirror()
            self.status.text("Image flipped horizontally")

        elif evt.keypress == "Ctrl+v":
            self.image2.flip()
            self.status.text("Image flipped vertically")

        elif evt.keypress == "Ctrl+t":
            self.image2.rotate(90)
            self.status.text("Image rotated by 90 degrees")

        elif evt.keypress == "Ctrl+b":
            self.image2.binarize()
            self.status.text("Image binarized")

        elif evt.keypress == "Ctrl+i":
            self.image2.invert()
            self.status.text("Image inverted")

        elif evt.keypress == "Ctrl+g":
            self.image2.smooth(sigma=1)
            self.status.text("Image smoothed")

        elif evt.keypress == "Ctrl+h":
            print(self.image2)
            return

        elif evt.keypress == "Ctrl+o":  # change luminosity
            narray = self.image2.tonumpy(raw=True)
            narray[:] = (narray*1.1).clip(0, 255).astype(np.uint8)
            self.image2.modified()
            self.status.text("Increased luminosity")
        elif evt.keypress == "Ctrl+l":  # change luminosity
            narray = self.image2.tonumpy(raw=True)
            narray[:] = (narray*0.9).clip(0, 255).astype(np.uint8)
            self.image2.modified()
            self.status.text("Decreased luminosity")

        elif evt.keypress == "O":  # change contrast
            narray = self.image2.tonumpy(raw=True)
            m = np.median(narray)
            narray[:] = ((narray - m) * 1.1 + m).clip(0, 255).astype(np.uint8)
            self.image2.modified()
            self.status.text("Increased contrast")
        elif evt.keypress == "L":
            narray = self.image2.tonumpy(raw=True)
            m = np.median(narray)
            narray[:] = ((narray - m) * 0.9 + m).clip(0, 255).astype(np.uint8)
            self.image2.modified()
            self.status.text("Decreased contrast")

        elif evt.keypress == "Ctrl+d":  # denoise
            self.image2.filterpass(highcutoff=self.cutfilter)
            self.status.text("Denoised image")

        elif evt.keypress == "Ctrl+r":
            self.image2 = self.image0.clone()
            self.status.text("Image reset to original")

        elif evt.keypress == "Ctrl+s":  # save image
            basename = os.path.basename(self.filename)
            sp = basename.split(".")
            name = f"{sp[0]}_edited.{sp[-1]}"
            self.image2.filename = name
            self.image2.write(name)
            self.status.text(f"Image saved as:\n{name}")
            self.render()
            return

        elif evt.keypress == "Ctrl+c":  # open color picker
            self.color_picker(evt.picked2d, verbose=True)
            return

        else:
            self.status.text("")
            self.render()
            return

        self.at(2).remove("Image").add(self.image2).render()

    def start(self):
        """Start the interactive image editor."""
        self.at(1).reset_camera(0.01).interactive()
        return self
    

########################################################################
