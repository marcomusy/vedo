#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Animation and timeline helper applications."""

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

class Animation(Plotter):
    """
    A `Plotter` derived class that allows to animate simultaneously various objects
    by specifying event times and durations of different visual effects.

    Arguments:
        total_duration : (float)
            expand or shrink the total duration of video to this value
        time_resolution : (float)
            in seconds, save a frame at this rate
        show_progressbar : (bool)
            whether to show a progress bar or not
        video_filename : (str)
            output file name of the video
        video_fps : (int)
            desired value of the nr of frames per second

    .. warning:: this is still very experimental at the moment.
    """

    def __init__(
        self,
        total_duration=None,
        time_resolution=0.02,
        show_progressbar=True,
        video_filename="animation.mp4",
        video_fps=12,
    ):
        super().__init__()
        self.resetcam = True

        self.events = []
        self.time_resolution = time_resolution
        self.total_duration = total_duration
        self.show_progressbar = show_progressbar
        self.video_filename = video_filename
        self.video_fps = video_fps
        self.bookingMode = True
        self._inputvalues = []
        self._performers = []
        self._lastT = None
        self._lastDuration = None
        self._lastActs = None
        self.eps = 0.00001

    def _parse(self, objs, t, duration):
        if t is None:
            if self._lastT:
                t = self._lastT
            else:
                t = 0.0
        if duration is None:
            if self._lastDuration:
                duration = self._lastDuration
            else:
                duration = 0.0
        if objs is None:
            if self._lastActs:
                objs = self._lastActs
            else:
                vedo.logger.error("Need to specify actors!")
                raise RuntimeError

        objs2 = objs

        if is_sequence(objs):
            objs2 = objs
        else:
            objs2 = [objs]

        # quantize time steps and duration
        t = int(t / self.time_resolution + 0.5) * self.time_resolution
        nsteps = int(duration / self.time_resolution + 0.5)
        duration = nsteps * self.time_resolution

        rng = np.linspace(t, t + duration, nsteps + 1)

        self._lastT = t
        self._lastDuration = duration
        self._lastActs = objs2

        for a in objs2:
            if a not in self.objects:
                self.objects.append(a)

        return objs2, t, duration, rng

    def switch_on(self, acts=None, t=None):
        """Switch on the input list of meshes."""
        return self.fade_in(acts, t, 0)

    def switch_off(self, acts=None, t=None):
        """Switch off the input list of meshes."""
        return self.fade_out(acts, t, 0)

    def fade_in(self, acts=None, t=None, duration=None):
        """Gradually switch on the input list of meshes by increasing opacity."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = lin_interpolate(tt, [t, t + duration], [0, 1])
                self.events.append((tt, self.fade_in, acts, alpha))
        else:
            for a in self._performers:
                if hasattr(a, "alpha"):
                    if a.alpha() >= self._inputvalues:
                        continue
                    a.alpha(self._inputvalues)
        return self

    def fade_out(self, acts=None, t=None, duration=None):
        """Gradually switch off the input list of meshes by increasing transparency."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = lin_interpolate(tt, [t, t + duration], [1, 0])
                self.events.append((tt, self.fade_out, acts, alpha))
        else:
            for a in self._performers:
                if a.alpha() <= self._inputvalues:
                    continue
                a.alpha(self._inputvalues)
        return self

    def change_alpha_between(self, alpha1, alpha2, acts=None, t=None, duration=None):
        """Gradually change transparency for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                alpha = lin_interpolate(tt, [t, t + duration], [alpha1, alpha2])
                self.events.append((tt, self.fade_out, acts, alpha))
        else:
            for a in self._performers:
                a.alpha(self._inputvalues)
        return self

    def change_color(self, c, acts=None, t=None, duration=None):
        """Gradually change color for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            col2 = get_color(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    col1 = a.color()
                    r = lin_interpolate(tt, [t, t + duration], [col1[0], col2[0]])
                    g = lin_interpolate(tt, [t, t + duration], [col1[1], col2[1]])
                    b = lin_interpolate(tt, [t, t + duration], [col1[2], col2[2]])
                    inputvalues.append((r, g, b))
                self.events.append((tt, self.change_color, acts, inputvalues))
        else:
            for i, a in enumerate(self._performers):
                a.color(self._inputvalues[i])
        return self

    def change_backcolor(self, c, acts=None, t=None, duration=None):
        """Gradually change backface color for the input list of meshes.
        An initial backface color should be set in advance."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            col2 = get_color(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    if a.GetBackfaceProperty():
                        col1 = a.backColor()
                        r = lin_interpolate(tt, [t, t + duration], [col1[0], col2[0]])
                        g = lin_interpolate(tt, [t, t + duration], [col1[1], col2[1]])
                        b = lin_interpolate(tt, [t, t + duration], [col1[2], col2[2]])
                        inputvalues.append((r, g, b))
                    else:
                        inputvalues.append(None)
                self.events.append((tt, self.change_backcolor, acts, inputvalues))
        else:
            for i, a in enumerate(self._performers):
                a.backColor(self._inputvalues[i])
        return self

    def change_to_wireframe(self, acts=None, t=None):
        """Switch representation to wireframe for the input list of meshes at time `t`."""
        if self.bookingMode:
            acts, t, _, _ = self._parse(acts, t, None)
            self.events.append((t, self.change_to_wireframe, acts, True))
        else:
            for a in self._performers:
                a.wireframe(self._inputvalues)
        return self

    def change_to_surface(self, acts=None, t=None):
        """Switch representation to surface for the input list of meshes at time `t`."""
        if self.bookingMode:
            acts, t, _, _ = self._parse(acts, t, None)
            self.events.append((t, self.change_to_surface, acts, False))
        else:
            for a in self._performers:
                a.wireframe(self._inputvalues)
        return self

    def change_line_width(self, lw, acts=None, t=None, duration=None):
        """Gradually change line width of the mesh edges for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    newlw = lin_interpolate(tt, [t, t + duration], [a.lw(), lw])
                    inputvalues.append(newlw)
                self.events.append((tt, self.change_line_width, acts, inputvalues))
        else:
            for i, a in enumerate(self._performers):
                a.lw(self._inputvalues[i])
        return self

    def change_line_color(self, c, acts=None, t=None, duration=None):
        """Gradually change line color of the mesh edges for the input list of meshes."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            col2 = get_color(c)
            for tt in rng:
                inputvalues = []
                for a in acts:
                    col1 = a.linecolor()
                    r = lin_interpolate(tt, [t, t + duration], [col1[0], col2[0]])
                    g = lin_interpolate(tt, [t, t + duration], [col1[1], col2[1]])
                    b = lin_interpolate(tt, [t, t + duration], [col1[2], col2[2]])
                    inputvalues.append((r, g, b))
                self.events.append((tt, self.change_line_color, acts, inputvalues))
        else:
            for i, a in enumerate(self._performers):
                a.linecolor(self._inputvalues[i])
        return self

    def change_lighting(self, style, acts=None, t=None, duration=None):
        """Gradually change the lighting style for the input list of meshes.

        Allowed styles are: [metallic, plastic, shiny, glossy, default].
        """
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)

            c = (1,1,0.99)
            if   style=='metallic': pars = [0.1, 0.3, 1.0, 10, c]
            elif style=='plastic' : pars = [0.3, 0.4, 0.3,  5, c]
            elif style=='shiny'   : pars = [0.2, 0.6, 0.8, 50, c]
            elif style=='glossy'  : pars = [0.1, 0.7, 0.9, 90, c]
            elif style=='default' : pars = [0.1, 1.0, 0.05, 5, c]
            else:
                vedo.logger.error(f"Unknown lighting style {style}")

            for tt in rng:
                inputvalues = []
                for a in acts:
                    pr = a.properties
                    aa = pr.GetAmbient()
                    ad = pr.GetDiffuse()
                    asp = pr.GetSpecular()
                    aspp = pr.GetSpecularPower()
                    naa = lin_interpolate(tt, [t, t + duration], [aa, pars[0]])
                    nad = lin_interpolate(tt, [t, t + duration], [ad, pars[1]])
                    nasp = lin_interpolate(tt, [t, t + duration], [asp, pars[2]])
                    naspp = lin_interpolate(tt, [t, t + duration], [aspp, pars[3]])
                    inputvalues.append((naa, nad, nasp, naspp))
                self.events.append((tt, self.change_lighting, acts, inputvalues))
        else:
            for i, a in enumerate(self._performers):
                pr = a.properties
                vals = self._inputvalues[i]
                pr.SetAmbient(vals[0])
                pr.SetDiffuse(vals[1])
                pr.SetSpecular(vals[2])
                pr.SetSpecularPower(vals[3])
        return self

    def move(self, act=None, pt=(0, 0, 0), t=None, duration=None, style="linear"):
        """Smoothly change the position of a specific object to a new point in space."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                vedo.logger.error("in move(), can move only one object.")
            cpos = acts[0].pos()
            pt = np.array(pt)
            dv = (pt - cpos) / len(rng)
            for j, tt in enumerate(rng):
                i = j + 1
                if "quad" in style:
                    x = i / len(rng)
                    y = x * x
                    self.events.append((tt, self.move, acts, cpos + dv * i * y))
                else:
                    self.events.append((tt, self.move, acts, cpos + dv * i))
        else:
            self._performers[0].pos(self._inputvalues)
        return self

    def rotate(self, act=None, axis=(1, 0, 0), angle=0, t=None, duration=None):
        """Smoothly rotate a specific object by a specified angle and axis."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                vedo.logger.error("in rotate(), can move only one object.")
            for tt in rng:
                ang = angle / len(rng)
                self.events.append((tt, self.rotate, acts, (axis, ang)))
        else:
            ax = self._inputvalues[0]
            if ax == "x":
                self._performers[0].rotate_x(self._inputvalues[1])
            elif ax == "y":
                self._performers[0].rotate_y(self._inputvalues[1])
            elif ax == "z":
                self._performers[0].rotate_z(self._inputvalues[1])
        return self

    def scale(self, acts=None, factor=1, t=None, duration=None):
        """Smoothly scale a specific object to a specified scale factor."""
        if self.bookingMode:
            acts, t, duration, rng = self._parse(acts, t, duration)
            for tt in rng:
                fac = lin_interpolate(tt, [t, t + duration], [1, factor])
                self.events.append((tt, self.scale, acts, fac))
        else:
            for a in self._performers:
                a.scale(self._inputvalues)
        return self

    def mesh_erode(self, act=None, corner=6, t=None, duration=None):
        """Erode a mesh by removing cells that are close to one of the 8 corners
        of the bounding box.
        """
        if self.bookingMode:
            acts, t, duration, rng = self._parse(act, t, duration)
            if len(acts) != 1:
                vedo.logger.error("in meshErode(), can erode only one object.")
            diag = acts[0].diagonal_size()
            x0, x1, y0, y1, z0, z1 = acts[0].GetBounds()
            corners = [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ]
            pcl = acts[0].closest_point(corners[corner])
            dmin = np.linalg.norm(pcl - corners[corner])
            for tt in rng:
                d = lin_interpolate(tt, [t, t + duration], [dmin, diag * 1.01])
                if d > 0:
                    ids = acts[0].closest_point(corners[corner], radius=d, return_point_id=True)
                    if len(ids) <= acts[0].npoints:
                        self.events.append((tt, self.mesh_erode, acts, ids))
        return self

    def play(self):
        """Play the internal list of events and save a video."""

        self.events = sorted(self.events, key=lambda x: x[0])
        self.bookingMode = False

        if self.show_progressbar:
            pb = vedo.ProgressBar(0, len(self.events), c="g")

        if self.total_duration is None:
            self.total_duration = self.events[-1][0] - self.events[0][0]

        if self.video_filename:
            vd = vedo.Video(self.video_filename, fps=self.video_fps, duration=self.total_duration)

        ttlast = 0
        for e in self.events:

            tt, action, self._performers, self._inputvalues = e
            action(0, 0)

            dt = tt - ttlast
            if dt > self.eps:
                self.show(interactive=False, resetcam=self.resetcam)
                if self.video_filename:
                    vd.add_frame()

                if dt > self.time_resolution + self.eps:
                    if self.video_filename:
                        vd.pause(dt)

            ttlast = tt

            if self.show_progressbar:
                pb.print("t=" + str(int(tt * 100) / 100) + "s,  " + action.__name__)

        self.show(interactive=False, resetcam=self.resetcam)
        if self.video_filename:
            vd.add_frame()
            vd.close()

        self.show(interactive=True, resetcam=self.resetcam)
        self.bookingMode = True


########################################################################
class AnimationPlayer(vedo.Plotter):
    """
    A Plotter with play/pause, step forward/backward and slider functionalties.
    Useful for inspecting time series.

    The user has the responsibility to update all actors in the callback function.

    Arguments:
        func :  (Callable)
            a function that passes an integer as input and updates the scene
        irange : (tuple)
            the range of the integer input representing the time series index
        dt : (float)
            the time interval between two calls to `func` in milliseconds
        loop : (bool)
            whether to loop the animation
        c : (list, str)
            the color of the play/pause button
        bc : (list)
            the background color of the play/pause button and the slider
        button_size : (int)
            the size of the play/pause buttons
        button_pos : (float, float)
            the position of the play/pause buttons as a fraction of the window size
        button_gap : (float)
            the gap between the buttons
        slider_length : (float)
            the length of the slider as a fraction of the window size
        slider_pos : (float, float)
            the position of the slider as a fraction of the window size
        kwargs: (dict)
            keyword arguments to be passed to `Plotter`

    Examples:
        - [aspring2_player.py](https://vedo.embl.es/images/simulations/spring_player.gif)
    """

    # Original class contributed by @mikaeltulldahl (Mikael Tulldahl)

    PLAY_SYMBOL        = "    \u23F5   "
    PAUSE_SYMBOL       = "   \u23F8   "
    ONE_BACK_SYMBOL    = " \u29CF"
    ONE_FORWARD_SYMBOL = "\u29D0 "

    def __init__(
        self,
        func,
        irange: tuple,
        dt: float = 1.0,
        loop: bool = True,
        c=("white", "white"),
        bc=("green3", "red4"),
        button_size=25,
        button_pos=(0.5, 0.04),
        button_gap=0.055,
        slider_length=0.5,
        slider_pos=(0.5, 0.055),
        **kwargs,
    ):
        super().__init__(**kwargs)

        min_value, max_value = np.array(irange).astype(int)
        button_pos = np.array(button_pos)
        slider_pos = np.array(slider_pos)

        self._func = func

        self.value = min_value - 1
        self.min_value = min_value
        self.max_value = max_value
        self.dt = max(dt, 1)
        self.is_playing = False
        self._loop = loop

        self.timer_callback_id = self.add_callback(
            "timer", self._handle_timer, enable_picking=False
        )
        self.timer_id = None

        self.play_pause_button = self.add_button(
            self.toggle,
            pos=button_pos,  # x,y fraction from bottom left corner
            states=[self.PLAY_SYMBOL, self.PAUSE_SYMBOL],
            font="Kanopus",
            size=button_size,
            bc=bc,
        )
        self.button_oneback = self.add_button(
            self.onebackward,
            pos=(-button_gap, 0) + button_pos,
            states=[self.ONE_BACK_SYMBOL],
            font="Kanopus",
            size=button_size,
            c=c,
            bc=bc,
        )
        self.button_oneforward = self.add_button(
            self.oneforward,
            pos=(button_gap, 0) + button_pos,
            states=[self.ONE_FORWARD_SYMBOL],
            font="Kanopus",
            size=button_size,
            bc=bc,
        )
        d = (1 - slider_length) / 2
        self.slider: SliderWidget = self.add_slider(
            self._slider_callback,
            self.min_value,
            self.max_value - 1,
            value=self.min_value,
            pos=[(d - 0.5, 0) + slider_pos, (0.5 - d, 0) + slider_pos],
            show_value=False,
            c=bc[0],
            alpha=1,
        )

    def pause(self) -> None:
        """Pause the animation."""
        self.is_playing = False
        if self.timer_id is not None:
            self.timer_callback("destroy", self.timer_id)
            self.timer_id = None
        self.play_pause_button.status(self.PLAY_SYMBOL)

    def resume(self) -> None:
        """Resume the animation."""
        if self.timer_id is not None:
            self.timer_callback("destroy", self.timer_id)
        self.timer_id = self.timer_callback("create", dt=int(self.dt))
        self.is_playing = True
        self.play_pause_button.status(self.PAUSE_SYMBOL)

    def toggle(self, _obj, _evt) -> None:
        """Toggle between play and pause."""
        if not self.is_playing:
            self.resume()
        else:
            self.pause()

    def oneforward(self, _obj, _evt) -> None:
        """Advance the animation by one frame."""
        self.pause()
        self.set_frame(self.value + 1)

    def onebackward(self, _obj, _evt) -> None:
        """Go back one frame in the animation."""
        self.pause()
        self.set_frame(self.value - 1)

    def set_frame(self, value: int) -> None:
        """Set the current value of the animation."""
        if self._loop:
            if value < self.min_value:
                value = self.max_value - 1
            elif value >= self.max_value:
                value = self.min_value
        else:
            if value < self.min_value:
                self.pause()
                value = self.min_value
            elif value >= self.max_value - 1:
                value = self.max_value - 1
                self.pause()

        if self.value != value:
            self.value = value
            self.slider.value = value
            self._func(value)

    def _slider_callback(self, widget: SliderWidget, _: str) -> None:
        self.pause()
        self.set_frame(int(round(widget.value)))

    def _handle_timer(self, _evt=None) -> None:
        self.set_frame(self.value + 1)

    def stop(self) -> "AnimationPlayer":
        """
        Stop the animation timers, remove buttons and slider.
        Behave like a normal `Plotter` after this.
        """
        # stop timer
        if self.timer_id is not None:
            self.timer_callback("destroy", self.timer_id)
            self.timer_id = None

        # remove callbacks
        self.remove_callback(self.timer_callback_id)

        # remove buttons
        self.slider.off()
        self.renderer.RemoveActor(self.play_pause_button.actor)
        self.renderer.RemoveActor(self.button_oneback.actor)
        self.renderer.RemoveActor(self.button_oneforward.actor)
        return self


########################################################################
class Clock(vedo.Assembly):
    """Create a clock with current time or user provided time."""

    def __init__(self, h=None, m=None, s=None, font="Quikhand", title="", c="k"):
        """
        Create a clock with current time or user provided time.

        Arguments:
            h : (int)
                hours in range [0,23]
            m : (int)
                minutes in range [0,59]
            s : (int)
                seconds in range [0,59]
            font : (str)
                font type
            title : (str)
                some extra text to show on the clock
            c : (str)
                color of the numbers

        Example:
            ```python
            import time
            from vedo import show
            from vedo.applications import Clock
            clock = Clock()
            plt = show(clock, interactive=False)
            for i in range(10):
                time.sleep(1)
                clock.update()
                plt.render()
            plt.close()
            ```
            ![](https://vedo.embl.es/images/feats/clock.png)
        """
        import time

        self.elapsed = 0
        self._start = time.time()

        wd = ""
        if h is None and m is None:
            t = time.localtime()
            h = t.tm_hour
            m = t.tm_min
            s = t.tm_sec
            if not title:
                d = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                wd = f"{d[t.tm_wday]} {t.tm_mday}/{t.tm_mon}/{t.tm_year} "

        h = int(h) % 24
        m = int(m) % 60
        t = (h * 60 + m) / 12 / 60

        alpha = 2 * np.pi * t + np.pi / 2
        beta = 12 * 2 * np.pi * t + np.pi / 2

        x1, y1 = np.cos(alpha), np.sin(alpha)
        x2, y2 = np.cos(beta), np.sin(beta)
        if s is not None:
            s = int(s) % 60
            gamma = s * 2 * np.pi / 60 + np.pi / 2
            x3, y3 = np.cos(gamma), np.sin(gamma)

        ore = Line([0, 0], [x1, y1], lw=14, c="red4").scale(0.5).mirror()
        minu = Line([0, 0], [x2, y2], lw=7, c="blue3").scale(0.75).mirror()
        secs = None
        if s is not None:
            secs = Line([0, 0], [x3, y3], lw=1, c="k").scale(0.95).mirror()
            secs.z(0.003)
        back1 = vedo.shapes.Circle(res=180, c="k5")
        back2 = vedo.shapes.Circle(res=12).mirror().scale(0.84).rotate_z(-360 / 12)
        labels = back2.labels(range(1, 13), justify="center", font=font, c=c, scale=0.14)
        txt = vedo.shapes.Text3D(wd + title, font="VictorMono", justify="top-center", s=0.07, c=c)
        txt.pos(0, -0.25, 0.001)
        labels.z(0.001)
        minu.z(0.002)
        super().__init__([back1, labels, ore, minu, secs, txt])
        self.name = "Clock"

    def update(self, h=None, m=None, s=None) -> "Clock":
        """Update clock with current or user time."""
        import time
        parts = self.unpack()
        self.elapsed = time.time() - self._start

        if h is None and m is None:
            t = time.localtime()
            h = t.tm_hour
            m = t.tm_min
            s = t.tm_sec

        h = int(h) % 24
        m = int(m) % 60
        t = (h * 60 + m) / 12 / 60

        alpha = 2 * np.pi * t + np.pi / 2
        beta = 12 * 2 * np.pi * t + np.pi / 2

        x1, y1 = np.cos(alpha), np.sin(alpha)
        x2, y2 = np.cos(beta), np.sin(beta)
        if s is not None:
            s = int(s) % 60
            gamma = s * 2 * np.pi / 60 + np.pi / 2
            x3, y3 = np.cos(gamma), np.sin(gamma)

        pts2 = parts[2].coordinates
        pts2[1] = [-x1 * 0.5, y1 * 0.5, 0.001]
        parts[2].coordinates = pts2

        pts3 = parts[3].coordinates
        pts3[1] = [-x2 * 0.75, y2 * 0.75, 0.002]
        parts[3].coordinates = pts3

        if s is not None:
            pts4 = parts[4].coordinates
            pts4[1] = [-x3 * 0.95, y3 * 0.95, 0.003]
            parts[4].coordinates = pts4

        return self
