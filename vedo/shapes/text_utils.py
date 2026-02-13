#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared text helpers and font loading utilities."""

import os
from functools import lru_cache
import numpy as np

import vedo

from vedo import settings, utils

_reps = (
    (":nabla", "∇"),
    (":inf", "∞"),
    (":rightarrow", "→"),
    (":leftarrow", "←"),
    (":partial", "∂"),
    (":sqrt", "√"),
    (":approx", "≈"),
    (":neq", "≠"),
    (":leq", "≤"),
    (":geq", "≥"),
    (":foreach", "∀"),
    (":permille", "‰"),
    (":euro", "€"),
    (":dot", "·"),
    (":int", "∫"),
    (":pm", "±"),
    (":times", "×"),
    (":Gamma", "Γ"),
    (":Delta", "Δ"),
    (":Theta", "Θ"),
    (":Lambda", "Λ"),
    (":Pi", "Π"),
    (":Sigma", "Σ"),
    (":Phi", "Φ"),
    (":Chi", "X"),
    (":Xi", "Ξ"),
    (":Psi", "Ψ"),
    (":Omega", "Ω"),
    (":alpha", "α"),
    (":beta", "β"),
    (":gamma", "γ"),
    (":delta", "δ"),
    (":epsilon", "ε"),
    (":zeta", "ζ"),
    (":eta", "η"),
    (":theta", "θ"),
    (":kappa", "κ"),
    (":lambda", "λ"),
    (":mu", "μ"),
    (":lowerxi", "ξ"),
    (":nu", "ν"),
    (":pi", "π"),
    (":rho", "ρ"),
    (":sigma", "σ"),
    (":tau", "τ"),
    (":varphi", "φ"),
    (":phi", "φ"),
    (":chi", "χ"),
    (":psi", "ψ"),
    (":omega", "ω"),
    (":circ", "°"),
    (":onehalf", "½"),
    (":onefourth", "¼"),
    (":threefourths", "¾"),
    (":^1", "¹"),
    (":^2", "²"),
    (":^3", "³"),
    (":,", "~"),
)


@lru_cache(None)
def _load_font(font) -> np.ndarray:
    if utils.is_number(font):
        font = list(settings.font_parameters.keys())[int(font)]

    if font.endswith(".npz"):
        fontfile = font
        font = os.path.basename(font).split(".")[0]
    elif font.startswith("https"):
        try:
            fontfile = vedo.file_io.download(font, verbose=False, force=False)
            font = os.path.basename(font).split(".")[0]
        except Exception:
            vedo.logger.warning(f"font {font} not found")
            font = settings.default_font
            fontfile = os.path.join(vedo.fonts_path, font + ".npz")
    else:
        font = font[:1].upper() + font[1:]
        fontfile = os.path.join(vedo.fonts_path, font + ".npz")

        if font not in settings.font_parameters.keys():
            vedo.logger.warning(
                f"Unknown font: {font}\n"
                f"Available 3D fonts are: "
                f"{list(settings.font_parameters.keys())}\n"
                f"Using font Normografo instead."
            )
            font = "Normografo"
            fontfile = os.path.join(vedo.fonts_path, font + ".npz")

        if not settings.font_parameters[font]["islocal"]:
            font = "https://vedo.embl.es/fonts/" + font + ".npz"
            try:
                fontfile = vedo.file_io.download(font, verbose=False, force=False)
                font = os.path.basename(font).split(".")[0]
            except Exception:
                vedo.logger.warning(f"font {font} not found")
                font = settings.default_font
                fontfile = os.path.join(vedo.fonts_path, font + ".npz")

    try:
        font_meshes = np.load(fontfile, allow_pickle=True)["font"][0]
    except Exception:
        vedo.logger.warning(f"font name {font} not found.")
        raise RuntimeError

    return font_meshes


@lru_cache(None)
def _get_font_letter(font, letter):
    font_meshes = _load_font(font)
    try:
        pts, faces = font_meshes[letter]
        return utils.buildPolyData(pts.astype(float), faces)
    except KeyError:
        return None
