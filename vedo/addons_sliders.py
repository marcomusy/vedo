#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Slider widgets extracted from vedo.addons."""

import os
from typing import Union
from typing_extensions import Self
import numpy as np

import vedo
import vedo.vtkclasses as vtki

from vedo import settings
from vedo import utils
from vedo.colors import get_color
class SliderWidget(vtki.vtkSliderWidget):
    """Helper class for `vtkSliderWidget`"""

    def __init__(self):
        super().__init__()
        self.previous_value = None
        self.name = "SliderWidget"

    @property
    def interactor(self):
        """Return the interactor for the slider."""
        return self.GetInteractor()

    @interactor.setter
    def interactor(self, iren):
        """Set the interactor for the slider."""
        self.SetInteractor(iren)

    @property
    def representation(self):
        """Return the representation of the slider."""
        return self.GetRepresentation()

    @property
    def value(self):
        """Return the value of the slider."""
        val = self.GetRepresentation().GetValue()
        # self.previous_value = val
        return val

    @value.setter
    def value(self, val):
        """Set the value of the slider."""
        self.GetRepresentation().SetValue(val)

    @property
    def renderer(self):
        """Return the renderer for the slider."""
        return self.GetCurrentRenderer()

    @renderer.setter
    def renderer(self, ren):
        """Set the renderer for the slider."""
        self.SetCurrentRenderer(ren)

    @property
    def title(self):
        """Return the title of the slider."""
        self.GetRepresentation().GetTitleText()

    @title.setter
    def title(self, txt):
        """Set the title of the slider."""
        self.GetRepresentation().SetTitleText(str(txt))

    @property
    def range(self):
        """Return the range of the slider."""
        xmin = self.GetRepresentation().GetMinimumValue()
        xmax = self.GetRepresentation().GetMaximumValue()
        return [xmin, xmax]

    @range.setter
    def range(self, vals):
        """Set the range of the slider."""
        if vals[0] is not None:
            self.GetRepresentation().SetMinimumValue(vals[0])
        if vals[1] is not None:
            self.GetRepresentation().SetMaximumValue(vals[1])

    def on(self) -> Self:
        """Activate/Enable the widget"""
        self.EnabledOn()
        return self

    def off(self) -> Self:
        """Disactivate/Disable the widget"""
        self.EnabledOff()
        return self

    def toggle(self) -> Self:
        """Toggle the widget"""
        self.SetEnabled(not self.GetEnabled())
        return self
    
    def is_enabled(self) -> bool:
        """Check if the widget is enabled."""
        return bool(self.GetEnabled())

    def add_observer(self, event, func, priority=1) -> int:
        """Add an observer to the widget."""
        event = utils.get_vtk_name_event(event)
        cid = self.AddObserver(event, func, priority)
        return cid

    def render(self):
        """Render the widget."""
        self.Render()
        return self

#####################################################################
class Slider2D(SliderWidget):
    """
    Add a slider which can call an external custom function.
    """

    def __init__(
        self,
        sliderfunc,
        xmin,
        xmax,
        value=None,
        pos=4,
        title="",
        font="Calco",
        title_size=1,
        c="k",
        alpha=1,
        show_value=True,
        delayed=False,
        **options,
    ):
        """
        Add a slider which can call an external custom function.
        Set any value as float to increase the number of significant digits above the slider.

        Use `play()` to start an animation between the current slider value and the last value.

        Arguments:
            sliderfunc : (function)
                external function to be called by the widget
            xmin : (float)
                lower value of the slider
            xmax : (float)
                upper value
            value : (float)
                current value
            pos : (list, str)
                position corner number: horizontal [1-5] or vertical [11-15]
                it can also be specified by corners coordinates [(x1,y1), (x2,y2)]
                and also by a string descriptor (eg. "bottom-left")
            title : (str)
                title text
            font : (str)
                title font face. Check [available fonts here](https://vedo.embl.es/fonts).
            title_size : (float)
                title text scale [1.0]
            show_value : (bool)
                if True current value is shown
            delayed : (bool)
                if True the callback is delayed until when the mouse button is released
            alpha : (float)
                opacity of the scalar bar texts
            slider_length : (float)
                slider length
            slider_width : (float)
                slider width
            end_cap_length : (float)
                length of the end cap
            end_cap_width : (float)
                width of the end cap
            tube_width : (float)
                width of the tube
            title_height : (float)
                height of the title
            tformat : (str)
                format of the title

        Examples:
            - [sliders1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders1.py)
            - [sliders2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders2.py)

            ![](https://user-images.githubusercontent.com/32848391/50738848-be033480-11d8-11e9-9b1a-c13105423a79.jpg)
        """
        slider_length = options.pop("slider_length",  0.015)
        slider_width  = options.pop("slider_width",   0.025)
        end_cap_length= options.pop("end_cap_length", 0.0015)
        end_cap_width = options.pop("end_cap_width",  0.0125)
        tube_width    = options.pop("tube_width",     0.0075)
        title_height  = options.pop("title_height",   0.025)
        tformat       = options.pop("tformat",        None)

        if options:
            vedo.logger.warning(f"in Slider2D unknown option(s): {options}")

        c = get_color(c)

        if value is None or value < xmin:
            value = xmin

        slider_rep = vtki.new("SliderRepresentation2D")
        slider_rep.SetMinimumValue(xmin)
        slider_rep.SetMaximumValue(xmax)
        slider_rep.SetValue(value)
        slider_rep.SetSliderLength(slider_length)
        slider_rep.SetSliderWidth(slider_width)
        slider_rep.SetEndCapLength(end_cap_length)
        slider_rep.SetEndCapWidth(end_cap_width)
        slider_rep.SetTubeWidth(tube_width)
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()

        if isinstance(pos, str):
            if "top" in pos:
                if "left" in pos:
                    if "vert" in pos:
                        pos = 11
                    else:
                        pos = 1
                elif "right" in pos:
                    if "vert" in pos:
                        pos = 12
                    else:
                        pos = 2
            elif "bott" in pos:
                if "left" in pos:
                    if "vert" in pos:
                        pos = 13
                    else:
                        pos = 3
                elif "right" in pos:
                    if "vert" in pos:
                        if "span" in pos:
                            pos = 15
                        else:
                            pos = 14
                    else:
                        pos = 4
                elif "span" in pos:
                    pos = 5

        if utils.is_sequence(pos):
            slider_rep.GetPoint1Coordinate().SetValue(pos[0][0], pos[0][1])
            slider_rep.GetPoint2Coordinate().SetValue(pos[1][0], pos[1][1])
        elif pos == 1:  # top-left horizontal
            slider_rep.GetPoint1Coordinate().SetValue(0.04, 0.93)
            slider_rep.GetPoint2Coordinate().SetValue(0.45, 0.93)
        elif pos == 2:
            slider_rep.GetPoint1Coordinate().SetValue(0.55, 0.93)
            slider_rep.GetPoint2Coordinate().SetValue(0.95, 0.93)
        elif pos == 3:
            slider_rep.GetPoint1Coordinate().SetValue(0.05, 0.06)
            slider_rep.GetPoint2Coordinate().SetValue(0.45, 0.06)
        elif pos == 4:  # bottom-right
            slider_rep.GetPoint1Coordinate().SetValue(0.55, 0.06)
            slider_rep.GetPoint2Coordinate().SetValue(0.95, 0.06)
        elif pos == 5:  # bottom span horizontal
            slider_rep.GetPoint1Coordinate().SetValue(0.04, 0.06)
            slider_rep.GetPoint2Coordinate().SetValue(0.95, 0.06)
        elif pos == 11:  # top-left vertical
            slider_rep.GetPoint1Coordinate().SetValue(0.065, 0.54)
            slider_rep.GetPoint2Coordinate().SetValue(0.065, 0.9)
        elif pos == 12:
            slider_rep.GetPoint1Coordinate().SetValue(0.94, 0.54)
            slider_rep.GetPoint2Coordinate().SetValue(0.94, 0.9)
        elif pos == 13:
            slider_rep.GetPoint1Coordinate().SetValue(0.065, 0.1)
            slider_rep.GetPoint2Coordinate().SetValue(0.065, 0.54)
        elif pos == 14:  # bottom-right vertical
            slider_rep.GetPoint1Coordinate().SetValue(0.94, 0.1)
            slider_rep.GetPoint2Coordinate().SetValue(0.94, 0.54)
        elif pos == 15:  # right margin vertical
            slider_rep.GetPoint1Coordinate().SetValue(0.95, 0.1)
            slider_rep.GetPoint2Coordinate().SetValue(0.95, 0.9)
        else:  # bottom-right
            slider_rep.GetPoint1Coordinate().SetValue(0.55, 0.06)
            slider_rep.GetPoint2Coordinate().SetValue(0.95, 0.06)

        if show_value:
            if tformat is None:
                if isinstance(xmin, int) and isinstance(xmax, int) and isinstance(value, int):
                    tformat = "%0.0f"
                else:
                    tformat = "%0.2f"
            if utils.vtk_version_at_least(6,0):
                # replace the default format of '%0.2g' to '{:0.2f}' to show more significant digits
                tformat = tformat.replace("%", "{:").replace("f", "f}").replace("g", "g}")

            slider_rep.SetLabelFormat(tformat)  
            slider_rep.GetLabelProperty().SetShadow(0)
            slider_rep.GetLabelProperty().SetBold(0)
            slider_rep.GetLabelProperty().SetOpacity(alpha)
            slider_rep.GetLabelProperty().SetColor(c)
            if isinstance(pos, int) and pos > 10:
                slider_rep.GetLabelProperty().SetOrientation(90)
        else:
            slider_rep.ShowSliderLabelOff()
        slider_rep.GetTubeProperty().SetColor(c)
        slider_rep.GetTubeProperty().SetOpacity(0.75)
        slider_rep.GetSliderProperty().SetColor(c)
        slider_rep.GetSelectedProperty().SetColor(np.sqrt(np.array(c)))
        slider_rep.GetCapProperty().SetColor(c)

        slider_rep.SetTitleHeight(title_height * title_size)
        slider_rep.GetTitleProperty().SetShadow(0)
        slider_rep.GetTitleProperty().SetColor(c)
        slider_rep.GetTitleProperty().SetOpacity(alpha)
        slider_rep.GetTitleProperty().SetBold(0)
        if font.lower() == "courier":
            slider_rep.GetTitleProperty().SetFontFamilyToCourier()
        elif font.lower() == "times":
            slider_rep.GetTitleProperty().SetFontFamilyToTimes()
        elif font.lower() == "arial":
            slider_rep.GetTitleProperty().SetFontFamilyToArial()
        else:
            if font == "":
                font = utils.get_font_path(settings.default_font)
            else:
                font = utils.get_font_path(font)
            slider_rep.GetTitleProperty().SetFontFamily(vtki.VTK_FONT_FILE)
            slider_rep.GetLabelProperty().SetFontFamily(vtki.VTK_FONT_FILE)
            slider_rep.GetTitleProperty().SetFontFile(font)
            slider_rep.GetLabelProperty().SetFontFile(font)

        if title:
            slider_rep.SetTitleText(title)
            if not utils.is_sequence(pos):
                if isinstance(pos, int) and pos > 10:
                    slider_rep.GetTitleProperty().SetOrientation(90)
            else:
                if abs(pos[0][0] - pos[1][0]) < 0.1:
                    slider_rep.GetTitleProperty().SetOrientation(90)

        super().__init__()
        self.name = "Slider2D"

        self.SetAnimationModeToJump()
        self.SetRepresentation(slider_rep)
        if delayed:
            self.AddObserver("EndInteractionEvent", sliderfunc)
        else:
            self.AddObserver("InteractionEvent", sliderfunc)

    def color(self, c):
        c = get_color(c)
        self.GetRepresentation().GetTubeProperty().SetColor(c)
        self.GetRepresentation().GetSliderProperty().SetColor(c)
        self.GetRepresentation().GetCapProperty().SetColor(c)
        self.GetRepresentation().GetLabelProperty().SetColor(c)
        self.GetRepresentation().GetTitleProperty().SetColor(c)


#####################################################################
class Slider3D(SliderWidget):
    """
    Add a 3D slider which can call an external custom function.
    """

    def __init__(
        self,
        sliderfunc,
        pos1,
        pos2,
        xmin,
        xmax,
        value=None,
        s=0.03,
        t=1,
        title="",
        rotation=0,
        c=None,
        show_value=True,
    ):
        """
        Add a 3D slider which can call an external custom function.

        Arguments:
            sliderfunc : (function)
                external function to be called by the widget
            pos1 : (list)
                first position 3D coordinates
            pos2 : (list)
                second position 3D coordinates
            xmin : (float)
                lower value
            xmax : (float)
                upper value
            value : (float)
                initial value
            s : (float)
                label scaling factor
            t : (float)
                tube scaling factor
            title : (str)
                title text
            c : (color)
                slider color
            rotation : (float)
                title rotation around slider axis
            show_value : (bool)
                if True current value is shown on top of the slider

        Examples:
            - [sliders3d.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders3d.py)
        """
        c = get_color(c)

        if value is None or value < xmin:
            value = xmin

        slider_rep = vtki.new("SliderRepresentation3D")
        slider_rep.SetMinimumValue(xmin)
        slider_rep.SetMaximumValue(xmax)
        slider_rep.SetValue(value)

        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToWorld()
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToWorld()
        slider_rep.GetPoint1Coordinate().SetValue(pos2)
        slider_rep.GetPoint2Coordinate().SetValue(pos1)

        # slider_rep.SetPoint1InWorldCoordinates(pos2[0], pos2[1], pos2[2])
        # slider_rep.SetPoint2InWorldCoordinates(pos1[0], pos1[1], pos1[2])

        slider_rep.SetSliderWidth(0.03 * t)
        slider_rep.SetTubeWidth(0.01 * t)
        slider_rep.SetSliderLength(0.04 * t)
        slider_rep.SetSliderShapeToCylinder()
        slider_rep.GetSelectedProperty().SetColor(np.sqrt(np.array(c)))
        slider_rep.GetSliderProperty().SetColor(np.array(c) / 1.5)
        slider_rep.GetCapProperty().SetOpacity(0)
        slider_rep.SetRotation(rotation)

        if not show_value:
            slider_rep.ShowSliderLabelOff()

        slider_rep.SetTitleText(title)
        slider_rep.SetTitleHeight(s * t)
        slider_rep.SetLabelHeight(s * t * 0.85)

        slider_rep.GetTubeProperty().SetColor(c)

        super().__init__()
        self.name = "Slider3D"

        self.SetRepresentation(slider_rep)
        self.SetAnimationModeToJump()
        self.AddObserver("InteractionEvent", sliderfunc)


