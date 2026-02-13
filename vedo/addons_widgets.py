#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive widget classes extracted from vedo.addons."""

import vedo
import vedo.vtkclasses as vtki
from typing_extensions import Self

from vedo import utils


class ButtonWidget:
    """
    Create a button widget.
    """

    def __init__(
        self,
        function,
        states=(),
        c=("white"),
        bc=("green4"),
        alpha=1.0,
        font="Calco",
        size=100,
        plotter=None,
    ):
        self.widget = vtki.new("ButtonWidget")
        self.name = "ButtonWidget"

        self.function = function
        self.states = states
        self.colors = c
        self.background_colors = bc
        self.plotter = plotter
        self.size = size

        assert len(states) == len(c), "states and colors must have the same length"
        assert len(states) == len(bc), "states and background colors must have the same length"

        self.interactor = None
        if plotter is not None:
            self.interactor = plotter.interactor
            self.widget.SetInteractor(plotter.interactor)
        else:
            plt = vedo.current_plotter()
            if plt:
                self.interactor = plt.interactor
                self.widget.SetInteractor(self.interactor)

        self.representation = vtki.new("TexturedButtonRepresentation2D")
        self.representation.SetNumberOfStates(len(states))
        for i, state in enumerate(states):

            if isinstance(state, vedo.Image):
                state = state.dataset

            elif isinstance(state, str):
                txt = state
                tp = vtki.vtkTextProperty()
                tp.BoldOff()
                tp.FrameOff()
                col = c[i]
                tp.SetColor(vedo.get_color(col))
                tp.ShadowOff()
                tp.ItalicOff()
                col = bc[i]
                tp.SetBackgroundColor(vedo.get_color(col))
                tp.SetBackgroundOpacity(alpha)
                tp.UseTightBoundingBoxOff()
                width, height = 100 * len(txt), 1000

                fpath = vedo.utils.get_font_path(font)
                tp.SetFontFamily(vtki.VTK_FONT_FILE)
                tp.SetFontFile(fpath)

                tr = vtki.new("TextRenderer")
                fs = tr.GetConstrainedFontSize(txt, tp, width, height, 500)
                tp.SetFontSize(fs)

                img = vtki.vtkImageData()
                tr.RenderString(tp, txt, img, [width, height], 500)
                state = img

            self.representation.SetButtonTexture(i, state)

        self.widget.SetRepresentation(self.representation)
        self.widget.AddObserver("StateChangedEvent", function)

    def __del__(self):
        self.widget.Off()
        self.widget.SetInteractor(None)
        self.widget.SetRepresentation(None)
        self.representation = None
        self.interactor = None
        self.function = None
        self.states = ()
        self.widget = None
        self.plotter = None

    def pos(self, pos):
        """Set the position of the button widget."""
        assert len(pos) == 2, "pos must be a 2D position"
        if not self.plotter:
            vedo.logger.warning("ButtonWidget: pos() can only be used if a Plotter is provided")
            return self
        coords = vtki.vtkCoordinate()
        coords.SetCoordinateSystemToNormalizedDisplay()
        coords.SetValue(pos[0], pos[1])
        sz = self.size
        ren = self.plotter.renderer
        p = coords.GetComputedDisplayValue(ren)
        bds = [0, 0, 0, 0, 0, 0]
        bds[0] = p[0] - sz
        bds[1] = bds[0] + sz
        bds[2] = p[1] - sz
        bds[3] = bds[2] + sz
        self.representation.SetPlaceFactor(1)
        self.representation.PlaceWidget(bds)
        return self

    def enable(self):
        """Enable the button widget."""
        self.widget.On()
        return self

    def disable(self):
        """Disable the button widget."""
        self.widget.Off()
        return self

    def next_state(self):
        """Change to the next state."""
        self.representation.NextState()
        return self

    @property
    def state(self):
        """Return the current state."""
        return self.representation.GetState()

    @state.setter
    def state(self, i):
        """Set the current state."""
        self.representation.SetState(i)


class Button(vedo.shapes.Text2D):
    """
    Build a Button object to be shown in the rendering window.
    """

    def __init__(
        self,
        fnc=None,
        states=("Button"),
        c=("white"),
        bc=("green4"),
        pos=(0.7, 0.1),
        size=24,
        font="Courier",
        bold=True,
        italic=False,
        alpha=1,
        angle=0,
    ):
        super().__init__()

        self.name = "Button"
        self.status_idx = 0
        self.spacer = " "
        self.states = states

        if not utils.is_sequence(c):
            c = [c]
        self.colors = c

        if not utils.is_sequence(bc):
            bc = [bc]
        self.bcolors = bc

        assert len(c) == len(bc), "in Button color number mismatch!"

        self.function = fnc
        self.function_id = None

        self.status(0)

        if font == "courier":
            font = font.capitalize()
        self.font(font).bold(bold).italic(italic)

        self.alpha(alpha).angle(angle)
        self.size(size / 20)
        self.pos(pos, "center")
        self.pickable(1)

    def status(self, s=None) -> "Button":
        """Set/Get the status of the button."""
        if s is None:
            return self.states[self.status_idx]

        if isinstance(s, str):
            s = self.states.index(s)
        self.status_idx = s
        self.text(self.spacer + self.states[s] + self.spacer)
        s = s % len(self.bcolors)
        self.color(self.colors[s])
        self.background(self.bcolors[s])
        return self

    def switch(self) -> "Button":
        """Change/cycle button status to the next defined status in states list."""
        self.status_idx = (self.status_idx + 1) % len(self.states)
        self.status(self.status_idx)
        return self


class DrawingWidget:
    """
    3D widget for tracing on planar props.
    """

    def __init__(self, obj, c="green5", lw=4, closed=False, snap_to_image=False):
        self.widget = vtki.new("ImageTracerWidget")
        self.name = "DrawingWidget"

        self.line = None
        self.line_properties = self.widget.GetLineProperty()
        self.line_properties.SetColor(vedo.get_color(c))
        self.line_properties.SetLineWidth(lw)
        self.callback_id = None
        self.event_name = "EndInteractionEvent"

        plt = vedo.current_plotter()
        if plt:
            self.widget.SetInteractor(plt.interactor)
            if plt.renderer:
                self.widget.SetDefaultRenderer(plt.renderer)

        try:
            self.widget.SetViewProp(obj.actor)
        except AttributeError:
            self.widget.SetViewProp(obj)

        if closed:
            closing_radius = 1e10
            self.widget.SetAutoClose(1)
            self.widget.SetCaptureRadius(closing_radius)

        self.widget.SetProjectToPlane(0)
        self.widget.SetProjectionNormal(2)
        self.widget.SetProjectionPosition(0)
        self.widget.SetSnapToImage(snap_to_image)

    def callback(self, widget, _event_name) -> None:
        """Callback function for the widget."""
        path = vtki.vtkPolyData()
        widget.GetPath(path)
        self.line = vedo.shapes.Line(path, c=self.line_properties.GetColor())

    def add_observer(self, event, func, priority=1) -> int:
        """Add an observer to the widget."""
        event = utils.get_vtk_name_event(event)
        cid = self.widget.AddObserver(event, func, priority)
        return cid

    @property
    def interactor(self):
        """Return the interactor for the widget."""
        return self.widget.GetInteractor()

    @interactor.setter
    def interactor(self, value):
        """Set the interactor for the widget."""
        self.widget.SetInteractor(value)

    @property
    def renderer(self):
        """Return the renderer for the widget."""
        return self.widget.GetDefaultRenderer()

    @renderer.setter
    def renderer(self, value):
        """Set the renderer for the widget."""
        self.widget.SetDefaultRenderer(value)

    def on(self) -> Self:
        """Activate/Enable the widget."""
        self.widget.On()
        ev_name = vedo.utils.get_vtk_name_event(self.event_name)
        self.callback_id = self.widget.AddObserver(ev_name, self.callback, 1000)
        return self

    def off(self) -> None:
        """Disactivate/Disable the widget."""
        self.widget.Off()
        self.widget.RemoveObserver(self.callback_id)

    def freeze(self, value=True) -> Self:
        """Freeze the widget by disabling interaction."""
        self.widget.SetInteraction(not value)
        return self

    def remove(self) -> None:
        """Remove the widget."""
        self.widget.Off()
        self.widget.RemoveObserver(self.callback_id)
        self.widget.SetInteractor(None)
        self.line = None
        self.line_properties = None
        self.callback_id = None
        self.widget = None
