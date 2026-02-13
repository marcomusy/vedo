#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LaTeX rendering helper class."""

import vedo
import vedo.vtkclasses as vtki

from vedo import settings
from vedo.colors import get_color
from vedo.image import Image

class Latex(Image):
    """
    Render Latex text and formulas.
    """

    def __init__(self, formula, pos=(0, 0, 0), s=1.0, bg=None, res=150, usetex=False, c="k", alpha=1.0) -> None:
        """
        Render Latex text and formulas.

        Arguments:
            formula : (str)
                latex text string
            pos : (list)
                position coordinates in space
            bg : (color)
                background color box
            res : (int)
                dpi resolution
            usetex : (bool)
                use latex compiler of matplotlib if available

        You can access the latex formula in `Latex.formula`.

        Examples:
            - [latex.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/latex.py)

            ![](https://vedo.embl.es/images/pyplot/latex.png)
        """
        from tempfile import NamedTemporaryFile
        import matplotlib.pyplot as mpltib

        def build_img_plt(formula, tfile):

            mpltib.rc("text", usetex=usetex)

            formula1 = "$" + formula + "$"
            mpltib.axis("off")
            col = get_color(c)
            if bg:
                bx = dict(boxstyle="square", ec=col, fc=get_color(bg))
            else:
                bx = None
            mpltib.text(
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
            mpltib.savefig(
                tfile, format="png", transparent=True, bbox_inches="tight", pad_inches=0
            )
            mpltib.close()

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)

        tmp_file = NamedTemporaryFile(delete=True)
        tmp_file.name = tmp_file.name + ".png"

        build_img_plt(formula, tmp_file.name)

        super().__init__(tmp_file.name, channels=4)
        self.alpha(alpha)
        self.scale([0.25 / res * s, 0.25 / res * s, 0.25 / res * s])
        self.pos(pos)
        self.name = "Latex"
        self.formula = formula

        # except:
        #     printc("Error in Latex()\n", formula, c="r")
        #     printc(" latex or dvipng not installed?", c="r")
        #     printc(" Try: usetex=False", c="r")
        #     printc(" Try: sudo apt install dvipng", c="r")

