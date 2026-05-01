#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""LaTeX rendering helper class."""

import os

from vedo import settings
from vedo.colors import get_color
from vedo.grids.image import Image


class Latex(Image):
    """
    Render Latex text and formulas.
    """

    def __init__(
        self,
        formula,
        pos=(0, 0, 0),
        s=1.0,
        bg=None,
        res=150,
        usetex=False,
        c="k",
        alpha=1.0,
    ) -> None:
        """
        Render Latex text and formulas.

        Args:
            formula (str):
                latex text string
            pos (list):
                position coordinates in space
            s (float):
                scale factor
            bg (color):
                background color box
            res (int):
                dpi resolution
            usetex (bool):
                use latex compiler of matplotlib if available
            c (color):
                text color
            alpha (float):
                opacity of the image

        You can access the latex formula in `Latex.formula`.

        Examples:
            - [latex.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/latex.py)

            ![](https://vedo.embl.es/images/pyplot/latex.png)
        """
        from tempfile import NamedTemporaryFile
        import matplotlib.pyplot as plt_matplib

        def build_img_plt(formula, tfile):
            plt_matplib.rc("text", usetex=usetex)
            formula1 = "$" + formula + "$"
            fig = plt_matplib.figure()
            plt_matplib.axis("off")
            col = get_color(c)
            if bg:
                bx = dict(boxstyle="square", ec=col, fc=get_color(bg))
            else:
                bx = None
            plt_matplib.text(
                0.5,
                0.5,
                formula1,
                size=res,
                color=col,
                alpha=alpha,
                ha="center",
                va="center",
                bbox=bx,
            )
            plt_matplib.savefig(
                tfile, format="png", transparent=True, bbox_inches="tight", pad_inches=0
            )
            plt_matplib.close(fig)

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)

        tmp_file = NamedTemporaryFile(suffix=".png", delete=False)
        tmp_name = tmp_file.name
        tmp_file.close()

        build_img_plt(formula, tmp_name)

        super().__init__(tmp_name, channels=4)
        os.unlink(tmp_name)

        self.alpha(alpha)
        self.scale([0.25 / res * s, 0.25 / res * s, 0.25 / res * s])
        self.pos(pos)
        self.name = "Latex"
        self.formula = formula
