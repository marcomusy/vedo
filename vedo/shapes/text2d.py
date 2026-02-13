#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D text overlay class."""

import os
from weakref import ref as weak_ref_to
import numpy as np

import vedo
import vedo.vtkclasses as vtki

from vedo import settings, utils
from vedo.colors import get_color
from vedo.shapes.text_utils import _reps

class Text2D:
    """
    Create a 2D text object.
    """
    def __init__(
        self,
        txt="",
        pos="top-left",
        s=1.0,
        bg=None,
        font="",
        justify="",
        bold=False,
        italic=False,
        c=None,
        alpha=0.5,
    ) -> None:
        """
        Create a 2D text object.

        All properties of the text, and the text itself, can be changed after creation
        (which is especially useful in loops).

        Arguments:
            pos : (str)
                text is placed in one of the 8 positions:
                - bottom-left
                - bottom-right
                - top-left
                - top-right
                - bottom-middle
                - middle-right
                - middle-left
                - top-middle

                If a pair (x,y) is passed as input the 2D text is place at that
                position in the coordinate system of the 2D screen (with the
                origin sitting at the bottom left).

            s : (float)
                size of text
            bg : (color)
                background color
            alpha : (float)
                background opacity
            justify : (str)
                text justification

            font : (str)
                built-in available fonts are:
                - Antares
                - Arial
                - Bongas
                - Calco
                - Comae
                - ComicMono
                - Courier
                - Glasgo
                - Kanopus
                - LogoType
                - Normografo
                - Quikhand
                - SmartCouric
                - Theemim
                - Times
                - VictorMono
                - More fonts at: https://vedo.embl.es/fonts/

                A path to a `.otf` or `.ttf` font-file can also be supplied as input.

        Examples:
            - [fonts.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/fonts.py)
            - [caption.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/caption.py)
            - [colorcubes.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/colorcubes.py)

                ![](https://vedo.embl.es/images/basic/colorcubes.png)
        """
        self.name = "Text2D"
        self.rendered_at = set()

        self.filename = ""
        self.time = 0
        self.info = {}

        if isinstance(settings.default_font, int):
            lfonts = list(settings.font_parameters.keys())
            font = settings.default_font % len(lfonts)
            self.fontname = lfonts[font]
        else:
            self.fontname = settings.default_font
        
        self.mapper = vtki.new("TextMapper")

        self.properties = self.mapper.GetTextProperty()

        self.actor = vedo.visual.Actor2D() # vtki.vtkActor2D()
        self.actor.SetMapper(self.mapper)
        
        self.actor.retrieve_object = weak_ref_to(self)

        self.actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()

        # automatic black or white
        if c is None:
            c = (0.1, 0.1, 0.1)
            plt = vedo.current_plotter()
            if plt and plt.renderer:
                if plt.renderer.GetGradientBackground():
                    bgcol = plt.renderer.GetBackground2()
                else:
                    bgcol = plt.renderer.GetBackground()
                c = (0.9, 0.9, 0.9)
                if np.sum(bgcol) > 1.5:
                    c = (0.1, 0.1, 0.1)

        self.font(font).color(c).background(bg, alpha).bold(bold).italic(italic)
        self.pos(pos, justify).size(s).text(txt).line_spacing(1.2).line_offset(5)
        self.actor.PickableOff()

    def pos(self, pos="top-left", justify=""):
        """
        Set position of the text to draw. Keyword `pos` can be a string
        or 2D coordinates in the range [0,1], being (0,0) the bottom left corner.
        """
        ajustify = "top-left"  # autojustify
        if isinstance(pos, str):  # corners
            ajustify = pos
            if "top" in pos:
                if "left" in pos:
                    pos = (0.008, 0.994)
                elif "right" in pos:
                    pos = (0.994, 0.994)
                elif "mid" in pos or "cent" in pos:
                    pos = (0.5, 0.994)
            elif "bottom" in pos:
                if "left" in pos:
                    pos = (0.008, 0.008)
                elif "right" in pos:
                    pos = (0.994, 0.008)
                elif "mid" in pos or "cent" in pos:
                    pos = (0.5, 0.008)
            elif "mid" in pos or "cent" in pos:
                if "left" in pos:
                    pos = (0.008, 0.5)
                elif "right" in pos:
                    pos = (0.994, 0.5)
                else:
                    pos = (0.5, 0.5)

            else:
                vedo.logger.warning(f"cannot understand text position {pos}")
                pos = (0.008, 0.994)
                ajustify = "top-left"

        elif len(pos) != 2:
            vedo.logger.error("pos must be of length 2 or integer value or string")
            raise RuntimeError()

        if not justify:
            justify = ajustify

        self.properties.SetJustificationToLeft()
        if "top" in justify:
            self.properties.SetVerticalJustificationToTop()
        if "bottom" in justify:
            self.properties.SetVerticalJustificationToBottom()
        if "cent" in justify or "mid" in justify:
            self.properties.SetJustificationToCentered()
        if "left" in justify:
            self.properties.SetJustificationToLeft()
        if "right" in justify:
            self.properties.SetJustificationToRight()

        self.actor.SetPosition(pos)
        return self

    def text(self, txt=None):
        """Set/get the input text string."""
        if txt is None:
            return self.mapper.GetInput()

        if ":" in txt:
            for r in _reps:
                txt = txt.replace(r[0], r[1])
        else:
            txt = str(txt)

        self.mapper.SetInput(txt)
        return self

    def size(self, s):
        """Set the font size."""
        self.properties.SetFontSize(int(s * 22.5))
        return self

    def angle(self, value: float):
        """Orientation angle in degrees"""
        self.properties.SetOrientation(value)
        return self

    def line_spacing(self, value: float):
        """Set the extra spacing between lines
        expressed as a text height multiplicative factor."""
        self.properties.SetLineSpacing(value)
        return self

    def line_offset(self, value: float):
        """Set/Get the vertical offset (measured in pixels)."""
        self.properties.SetLineOffset(value)
        return self

    def bold(self, value=True):
        """Set bold face"""
        self.properties.SetBold(value)
        return self

    def italic(self, value=True):
        """Set italic face"""
        self.properties.SetItalic(value)
        return self

    def shadow(self, offset=(1, -1)):
        """Text shadowing. Set to `None` to disable it."""
        if offset is None:
            self.properties.ShadowOff()
        else:
            self.properties.ShadowOn()
            self.properties.SetShadowOffset(offset)
        return self

    def color(self, c=None):
        """Set the text color"""
        if c is None:
            return get_color(self.properties.GetColor())
        self.properties.SetColor(get_color(c))
        return self

    def c(self, color=None):
        """Set the text color"""
        if color is None:
            return get_color(self.properties.GetColor())
        return self.color(color)

    def alpha(self, value: float):
        """Set the text opacity"""
        self.properties.SetBackgroundOpacity(value)
        return self

    def background(self, color="k9", alpha=1.0):
        """Text background. Set to `None` to disable it."""
        bg = get_color(color)
        if color is None:
            self.properties.SetBackgroundOpacity(0)
        else:
            self.properties.SetBackgroundColor(bg)
            if alpha:
                self.properties.SetBackgroundOpacity(alpha)
        return self

    def frame(self, color="k1", lw=2):
        """Border color and width"""
        if color is None:
            self.properties.FrameOff()
        else:
            c = get_color(color)
            self.properties.FrameOn()
            self.properties.SetFrameColor(c)
            self.properties.SetFrameWidth(lw)
        return self

    def font(self, font: str):
        """Text font face"""
        if isinstance(font, int):
            lfonts = list(settings.font_parameters.keys())
            n = font % len(lfonts)
            font = lfonts[n]
            self.fontname = font

        if not font:  # use default font
            font = self.fontname
            fpath = os.path.join(vedo.fonts_path, font + ".ttf")
        elif font.startswith("https"):  # user passed URL link, make it a path
            fpath = vedo.file_io.download(font, verbose=False, force=False)
        elif font.endswith(".ttf"):  # user passing a local path to font file
            fpath = font
        else:  # user passing name of preset font
            fpath = os.path.join(vedo.fonts_path, font + ".ttf")

        if   font == "Courier": self.properties.SetFontFamilyToCourier()
        elif font == "Times":   self.properties.SetFontFamilyToTimes()
        elif font == "Arial":   self.properties.SetFontFamilyToArial()
        else:
            fpath = utils.get_font_path(font)
            self.properties.SetFontFamily(vtki.VTK_FONT_FILE)
            self.properties.SetFontFile(fpath)
        self.fontname = font  # io.tonumpy() uses it

        return self

    def on(self):
        """Make text visible"""
        self.actor.SetVisibility(True)
        return self

    def off(self):
        """Make text invisible"""
        self.actor.SetVisibility(False)
        return self

    def toggle(self):
        """Toggle text visibility"""
        self.actor.SetVisibility(not self.actor.GetVisibility())
        return self

    def pickable(self, value=True):
        """Set the pickable state of the text"""
        self.actor.SetPickable(value)
        return self

    def add_observer(self, event, func, priority=1) -> int:
        """Add an observer to the widget."""
        event = utils.get_vtk_name_event(event)
        cid = self.actor.AddObserver(event, func, priority)
        return cid
