#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI actor helpers extracted from vedo.addons."""

from typing import Union
from typing_extensions import Self
import numpy as np

import vedo
import vedo.vtkclasses as vtki

from vedo import settings
from vedo import utils
from vedo.colors import get_color
class Flagpost(vtki.vtkFlagpoleLabel):
    """
    Create a flag post style element to describe an object.
    """

    def __init__(
        self,
        txt="",
        base=(0, 0, 0),
        top=(0, 0, 1),
        s=1,
        c="k9",
        bc="k1",
        alpha=1,
        lw=0,
        font="Calco",
        justify="center-left",
        vspacing=1,
    ):
        """
        Create a flag post style element to describe an object.

        Arguments:
            txt : (str)
                Text to display. The default is the filename or the object name.
            base : (list)
                position of the flag anchor point.
            top : (list)
                a 3D displacement or offset.
            s : (float)
                size of the text to be shown
            c : (list)
                color of text and line
            bc : (list)
                color of the flag background
            alpha : (float)
                opacity of text and box.
            lw : (int)
                line with of box frame. The default is 0.
            font : (str)
                font name. Use a monospace font for better rendering. The default is "Calco".
                Type `vedo -r fonts` for a font demo.
                Check [available fonts here](https://vedo.embl.es/fonts).
            justify : (str)
                internal text justification. The default is "center-left".
            vspacing : (float)
                vertical spacing between lines.

        Examples:
            - [flag_labels2.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels2.py)

            ![](https://vedo.embl.es/images/other/flag_labels2.png)
        """

        super().__init__()

        self.name = "Flagpost"

        base = utils.make3d(base)
        top = utils.make3d(top)

        self.SetBasePosition(*base)
        self.SetTopPosition(*top)

        self.SetFlagSize(s)
        self.SetInput(txt)
        self.PickableOff()

        self.GetProperty().LightingOff()
        self.GetProperty().SetLineWidth(lw + 1)

        prop = self.GetTextProperty()
        if bc is not None:
            prop.SetBackgroundColor(get_color(bc))

        prop.SetOpacity(alpha)
        prop.SetBackgroundOpacity(alpha)
        if bc is not None and len(bc) == 4:
            prop.SetBackgroundRGBA(alpha)

        c = get_color(c)
        prop.SetColor(c)
        self.GetProperty().SetColor(c)

        prop.SetFrame(bool(lw))
        prop.SetFrameWidth(lw)
        prop.SetFrameColor(prop.GetColor())

        prop.SetFontFamily(vtki.VTK_FONT_FILE)
        fl = utils.get_font_path(font)
        prop.SetFontFile(fl)
        prop.ShadowOff()
        prop.BoldOff()
        prop.SetOpacity(alpha)
        prop.SetJustificationToLeft()
        if "top" in justify:
            prop.SetVerticalJustificationToTop()
        if "bottom" in justify:
            prop.SetVerticalJustificationToBottom()
        if "cent" in justify:
            prop.SetVerticalJustificationToCentered()
            prop.SetJustificationToCentered()
        if "left" in justify:
            prop.SetJustificationToLeft()
        if "right" in justify:
            prop.SetJustificationToRight()
        prop.SetLineSpacing(vspacing * 1.2)
        self.SetUseBounds(False)

    def text(self, value: str) -> Self:
        """Set the text of the flagpost."""
        self.SetInput(value)
        return self

    def on(self) -> Self:
        """Show the flagpost."""
        self.VisibilityOn()
        return self

    def off(self) -> Self:
        """Hide the flagpost."""
        self.VisibilityOff()
        return self

    def toggle(self) -> Self:
        """Toggle the visibility of the flagpost."""
        self.SetVisibility(not self.GetVisibility())
        return self

    def use_bounds(self, value=True) -> Self:
        """Set the flagpost to keep bounds into account."""
        self.SetUseBounds(value)
        return self

    def color(self, c) -> Self:
        """Set the color of the flagpost."""
        c = get_color(c)
        self.GetTextProperty().SetColor(c)
        self.GetProperty().SetColor(c)
        return self

    def pos(self, p) -> Self:
        """Set the position of the flagpost."""
        p = np.asarray(p)
        self.top = self.top - self.base + p
        self.base = p
        return self

    @property
    def base(self) -> np.ndarray:
        """Return the base position of the flagpost."""
        return np.array(self.GetBasePosition())

    @base.setter
    def base(self, value):
        """Set the base position of the flagpost."""
        self.SetBasePosition(*value)

    @property
    def top(self) -> np.ndarray:
        """Return the top position of the flagpost."""
        return np.array(self.GetTopPosition())

    @top.setter
    def top(self, value):
        """Set the top position of the flagpost."""
        self.SetTopPosition(*value)


###########################################################################################
class LegendBox(vtki.vtkLegendBoxActor):
    """
    Create a 2D legend box.
    """

    def __init__(
        self,
        entries=(),
        nmax=12,
        c=None,
        font="",
        width=0.18,
        height=None,
        padding=2,
        bg="k8",
        alpha=0.25,
        pos="top-right",
        markers=None,
    ):
        """
        Create a 2D legend box for the list of specified objects.

        Arguments:
            nmax : (int)
                max number of legend entries
            c : (color)
                text color, leave as None to pick the mesh color automatically
            font : (str)
                Check [available fonts here](https://vedo.embl.es/fonts)
            width : (float)
                width of the box as fraction of the window width
            height : (float)
                height of the box as fraction of the window height
            padding : (int)
                padding space in units of pixels
            bg : (color)
                background color of the box
            alpha: (float)
                opacity of the box
            pos : (str, list)
                position of the box, can be either a string or a (x,y) screen position in range [0,1]

        Examples:
            - [legendbox.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/legendbox.py)
            - [flag_labels1.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels1.py)
            - [flag_labels2.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels2.py)

                ![](https://vedo.embl.es/images/other/flag_labels.png)
        """
        super().__init__()

        self.name = "LegendBox"
        self.entries = entries[:nmax]
        self.properties = self.GetEntryTextProperty()

        n = 0
        texts = []
        for e in self.entries:
            ename = e.name
            if "legend" in e.info.keys():
                if not e.info["legend"]:
                    ename = ""
                else:
                    ename = str(e.info["legend"])
            if ename:
                n += 1
            texts.append(ename)
        self.SetNumberOfEntries(n)

        if not n:
            return

        self.ScalarVisibilityOff()
        self.PickableOff()
        self.SetPadding(padding)

        self.properties.ShadowOff()
        self.properties.BoldOff()

        # self.properties.SetJustificationToLeft() # no effect
        # self.properties.SetVerticalJustificationToTop()

        if not font:
            font = settings.default_font
        self.font(font)

        n = 0
        for i in range(len(self.entries)):
            ti = texts[i]
            if not ti:
                continue
            e = entries[i]
            if c is None:
                col = e.properties.GetColor()
                if col == (1, 1, 1):
                    col = (0.2, 0.2, 0.2)
            else:
                col = get_color(c)
            if markers is None:  # default
                poly = e.dataset
            else:
                marker = markers[i] if utils.is_sequence(markers) else markers
                if isinstance(marker, Points):
                    poly = marker.clone(deep=False).normalize().shift(0, 1, 0).dataset
                else:  # assume string marker
                    poly = vedo.shapes.Marker(marker, s=1).shift(0, 1, 0).dataset

            self.SetEntry(n, poly, ti, col)
            n += 1

        self.SetWidth(width)
        if height is None:
            self.SetHeight(width / 3.0 * n)
        else:
            self.SetHeight(height)

        self.pos(pos)

        if alpha:
            self.UseBackgroundOn()
            self.SetBackgroundColor(get_color(bg))
            self.SetBackgroundOpacity(alpha)
        else:
            self.UseBackgroundOff()
        self.LockBorderOn()

    @property
    def width(self):
        """Return the width of the legend box."""
        return self.GetWidth()

    @property
    def height(self):
        """Return the height of the legend box."""
        return self.GetHeight()

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
    def pos(self, pos):
        """Set the position of the legend box."""
        sx, sy = 1 - self.GetWidth(), 1 - self.GetHeight()
        if pos == 1 or ("top" in pos and "left" in pos):
            self.GetPositionCoordinate().SetValue(0, sy)
        elif pos == 2 or ("top" in pos and "right" in pos):
            self.GetPositionCoordinate().SetValue(sx, sy)
        elif pos == 3 or ("bottom" in pos and "left" in pos):
            self.GetPositionCoordinate().SetValue(0, 0)
        elif pos == 4 or ("bottom" in pos and "right" in pos):
            self.GetPositionCoordinate().SetValue(sx, 0)
        elif "cent" in pos and "right" in pos:
            self.GetPositionCoordinate().SetValue(sx, sy - 0.25)
        elif "cent" in pos and "left" in pos:
            self.GetPositionCoordinate().SetValue(0, sy - 0.25)
        elif "cent" in pos and "bottom" in pos:
            self.GetPositionCoordinate().SetValue(sx - 0.25, 0)
        elif "cent" in pos and "top" in pos:
            self.GetPositionCoordinate().SetValue(sx - 0.25, sy)
        elif utils.is_sequence(pos):
            self.GetPositionCoordinate().SetValue(pos[0], pos[1])
        else:
            vedo.logger.error("LegendBox: pos must be in range [1-4] or a [x,y] list")

        return self


#####################################################################
