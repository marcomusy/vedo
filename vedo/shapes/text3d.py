#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D text mesh class."""

from typing import Union, Any
from weakref import ref as weak_ref_to
import numpy as np

import vedo
import vedo.vtkclasses as vtki

from vedo import settings, utils
from vedo.mesh import Mesh
from vedo.shapes.text_utils import _reps, _get_font_letter

class Text3D(Mesh):
    """
    Generate a 3D polygonal Mesh to represent a text string.
    """

    def __init__(
        self,
        txt,
        pos=(0, 0, 0),
        s=1.0,
        font="",
        hspacing=1.15,
        vspacing=2.15,
        depth=0.0,
        italic=False,
        justify="bottom-left",
        literal=False,
        c=None,
        alpha=1.0,
    ) -> None:
        """
        Generate a 3D polygonal `Mesh` representing a text string.

        Can render strings like `3.7 10^9` or `H_2 O` with subscripts and superscripts.
        Most Latex symbols are also supported.

        Symbols `~ ^ _` are reserved modifiers:
        - use ~ to add a short space, 1/4 of the default empty space,
        - use ^ and _ to start up/sub scripting, a space terminates their effect.

        Monospaced fonts are: `Calco, ComicMono, Glasgo, SmartCouric, VictorMono, Justino`.

        More fonts at: https://vedo.embl.es/fonts/

        Arguments:
            pos : (list)
                position coordinates in 3D space
            s : (float)
                vertical size of the text (as scaling factor)
            depth : (float)
                text thickness (along z)
            italic : (bool), float
                italic font type (can be a signed float too)
            justify : (str)
                text justification as centering of the bounding box
                (bottom-left, bottom-right, top-left, top-right, centered)
            font : (str, int)
                some of the available 3D-polygonized fonts are:
                Bongas, Calco, Comae, ComicMono, Kanopus, Glasgo, Ubuntu,
                LogoType, Normografo, Quikhand, SmartCouric, Theemim, VictorMono, VTK,
                Capsmall, Cartoons123, Vega, Justino, Spears, Meson.

                Check for more at https://vedo.embl.es/fonts/

                Or type in your terminal `vedo --run fonts`.

                Default is Normografo, which can be changed using `settings.default_font`.

            hspacing : (float)
                horizontal spacing of the font
            vspacing : (float)
                vertical spacing of the font for multiple lines text
            literal : (bool)
                if set to True will ignore modifiers like _ or ^

        Examples:
            - [markpoint.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/markpoint.py)
            - [fonts.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/fonts.py)
            - [caption.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/caption.py)

            ![](https://vedo.embl.es/images/pyplot/fonts3d.png)

        .. note:: Type `vedo -r fonts` for a demo.
        """
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)

        if c is None:  # automatic black or white
            pli = vedo.current_plotter()
            if pli and pli.renderer:
                c = (0.9, 0.9, 0.9)
                if pli.renderer.GetGradientBackground():
                    bgcol = pli.renderer.GetBackground2()
                else:
                    bgcol = pli.renderer.GetBackground()
                if np.sum(bgcol) > 1.5:
                    c = (0.1, 0.1, 0.1)
            else:
                c = (0.6, 0.6, 0.6)

        tpoly = self._get_text3d_poly(
            txt, s, font, hspacing, vspacing, depth, italic, justify, literal
        )

        super().__init__(tpoly, c, alpha)

        self.pos(pos)
        self.lighting("off")

        self.actor.PickableOff()
        self.actor.DragableOff()
        self.init_scale = s
        self.name = "Text3D"
        self.txt = txt
        self.justify = justify

    def text(
        self,
        txt=None,
        s=1,
        font="",
        hspacing=1.15,
        vspacing=2.15,
        depth=0,
        italic=False,
        justify="",
        literal=False,
    ) -> "Text3D":
        """
        Update the text and some of its properties.

        Check [available fonts here](https://vedo.embl.es/fonts).
        """
        if txt is None:
            return self.txt
        if not justify:
            justify = self.justify

        poly = self._get_text3d_poly(
            txt, self.init_scale * s, font, hspacing, vspacing,
            depth, italic, justify, literal
        )

        # apply the current transformation to the new polydata
        tf = vtki.new("TransformPolyDataFilter")
        tf.SetInputData(poly)
        tf.SetTransform(self.transform.T)
        tf.Update()
        tpoly = tf.GetOutput()

        self._update(tpoly)
        self.txt = txt
        return self

    @staticmethod
    def _get_text3d_poly(
        txt,
        s=1,
        font="",
        hspacing=1.15,
        vspacing=2.15,
        depth=0,
        italic=False,
        justify="bottom-left",
        literal=False,
    ) -> vtki.vtkPolyData:
        if not font:
            font = settings.default_font

        txt = str(txt)

        if font == "VTK":  #######################################
            vtt = vtki.new("VectorText")
            vtt.SetText(txt)
            vtt.Update()
            tpoly = vtt.GetOutput()

        else:  ###################################################

            stxt = set(txt)  # check here if null or only spaces
            if not txt or (len(stxt) == 1 and " " in stxt):
                return vtki.vtkPolyData()

            if italic is True:
                italic = 1

            if isinstance(font, int):
                lfonts = list(settings.font_parameters.keys())
                font = font % len(lfonts)
                font = lfonts[font]

            if font not in settings.font_parameters.keys():
                fpars = settings.font_parameters["Normografo"]
            else:
                fpars = settings.font_parameters[font]

            # ad hoc adjustments
            mono = fpars["mono"]
            lspacing = fpars["lspacing"]
            hspacing *= fpars["hspacing"]
            fscale = fpars["fscale"]
            dotsep = fpars["dotsep"]

            # replacements
            if ":" in txt:
                for r in _reps:
                    txt = txt.replace(r[0], r[1])

            if not literal:
                reps2 = [
                    (r"\_", "┭"),  # trick to protect ~ _ and ^ chars
                    (r"\^", "┮"),  #
                    (r"\~", "┯"),  #
                    ("**", "^"),   # order matters
                    ("e+0", dotsep + "10^"),
                    ("e-0", dotsep + "10^-"),
                    ("E+0", dotsep + "10^"),
                    ("E-0", dotsep + "10^-"),
                    ("e+", dotsep + "10^"),
                    ("e-", dotsep + "10^-"),
                    ("E+", dotsep + "10^"),
                    ("E-", dotsep + "10^-"),
                ]
                for r in reps2:
                    txt = txt.replace(r[0], r[1])

            xmax, ymax, yshift, scale = 0.0, 0.0, 0.0, 1.0
            save_xmax = 0.0

            notfounds = set()
            polyletters = []
            ntxt = len(txt)
            for i, t in enumerate(txt):
                ##########
                if t == "┭":
                    t = "_"
                elif t == "┮":
                    t = "^"
                elif t == "┯":
                    t = "~"
                elif t == "^" and not literal:
                    if yshift < 0:
                        xmax = save_xmax
                    yshift = 0.9 * fscale
                    scale = 0.5
                    continue
                elif t == "_" and not literal:
                    if yshift > 0:
                        xmax = save_xmax
                    yshift = -0.3 * fscale
                    scale = 0.5
                    continue
                elif (t in (" ", "\\n")) and yshift:
                    yshift = 0.0
                    scale = 1.0
                    save_xmax = xmax
                    if t == " ":
                        continue
                elif t == "~" and not literal:
                    if i < ntxt - 1 and txt[i + 1] == "_":
                        continue
                    xmax += hspacing * scale * fscale / 4
                    continue

                ############
                if t == " ":
                    xmax += hspacing * scale * fscale

                elif t == "\n":
                    xmax = 0.0
                    save_xmax = 0.0
                    ymax -= vspacing

                else:
                    poly = _get_font_letter(font, t)
                    if not poly:
                        notfounds.add(t)
                        xmax += hspacing * scale * fscale
                        continue

                    if poly.GetNumberOfPoints() == 0:
                        continue

                    tr = vtki.vtkTransform()
                    tr.Translate(xmax, ymax + yshift, 0)
                    pscale = scale * fscale / 1000
                    tr.Scale(pscale, pscale, pscale)
                    if italic:
                        tr.Concatenate([1, italic * 0.15, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
                    tf = vtki.new("TransformPolyDataFilter")
                    tf.SetInputData(poly)
                    tf.SetTransform(tr)
                    tf.Update()
                    poly = tf.GetOutput()
                    polyletters.append(poly)

                    bx = poly.GetBounds()
                    if mono:
                        xmax += hspacing * scale * fscale
                    else:
                        xmax += bx[1] - bx[0] + hspacing * scale * fscale * lspacing
                    if yshift == 0:
                        save_xmax = xmax

            if len(polyletters) == 1:
                tpoly = polyletters[0]
            else:
                polyapp = vtki.new("AppendPolyData")
                for polyd in polyletters:
                    polyapp.AddInputData(polyd)
                polyapp.Update()
                tpoly = polyapp.GetOutput()

            if notfounds:
                wmsg = f"unavailable characters in font name '{font}': {notfounds}."
                wmsg += '\nType "vedo -r fonts" for a demo.'
                vedo.logger.warning(wmsg)

        bb = tpoly.GetBounds()
        dx, dy = (bb[1] - bb[0]) / 2 * s, (bb[3] - bb[2]) / 2 * s
        shift = -np.array([(bb[1] + bb[0]), (bb[3] + bb[2]), (bb[5] + bb[4])]) * s /2
        if "bottom" in justify: shift += np.array([  0, dy, 0.])
        if "top"    in justify: shift += np.array([  0,-dy, 0.])
        if "left"   in justify: shift += np.array([ dx,  0, 0.])
        if "right"  in justify: shift += np.array([-dx,  0, 0.])

        if tpoly.GetNumberOfPoints():
            t = vtki.vtkTransform()
            t.PostMultiply()
            t.Scale(s, s, s)
            t.Translate(shift)
            tf = vtki.new("TransformPolyDataFilter")
            tf.SetInputData(tpoly)
            tf.SetTransform(t)
            tf.Update()
            tpoly = tf.GetOutput()

            if depth:
                extrude = vtki.new("LinearExtrusionFilter")
                extrude.SetInputData(tpoly)
                extrude.SetExtrusionTypeToVectorExtrusion()
                extrude.SetVector(0, 0, 1)
                extrude.SetScaleFactor(depth * dy)
                extrude.Update()
                tpoly = extrude.GetOutput()

        return tpoly


