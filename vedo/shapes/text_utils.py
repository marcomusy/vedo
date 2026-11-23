#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Shared text helpers and font loading utilities."""

import os
from functools import lru_cache

import vedo

from vedo import settings, utils
from vedo.file_io.numpy_io import NumpyIntegrityError, UnsafePickleError
from ._font_checksums import OFFICIAL_FONT_SHA256
from ._font_io import read_font

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


_LOCAL_FALLBACK_FONT = "Normografo"


def _font_path(font: str) -> str:
    return os.path.join(vedo.fonts_path, font + ".npz")


def _local_fallback_font() -> tuple[str, str]:
    font = settings.default_font
    if utils.is_number(font):
        font = list(settings.font_parameters.keys())[int(font)]

    if isinstance(font, str):
        font = font[:1].upper() + font[1:]
        fontfile = _font_path(font)
        if os.path.isfile(fontfile):
            return font, fontfile

    return _LOCAL_FALLBACK_FONT, _font_path(_LOCAL_FALLBACK_FONT)


def _load_local_fallback_font(missing_font: str) -> dict:
    font, fontfile = _local_fallback_font()
    try:
        font_meshes = read_font(fontfile)
    except (UnsafePickleError, NumpyIntegrityError):
        raise
    except Exception as e:
        vedo.logger.warning(f"font name {font} not found.")
        raise RuntimeError from e

    vedo.logger.warning(f"font name {missing_font} not found, using {font} instead.")
    return font_meshes


@lru_cache(None)
def _load_font(font) -> dict:
    if utils.is_number(font):
        font = list(settings.font_parameters.keys())[int(font)]

    expected_sha256 = None
    if font.startswith("https://"):
        # Only canonical official URLs get a packaged checksum. An arbitrary
        # URL or local file named like an official font does not acquire trust.
        font_name = font.rsplit("/", 1)[-1].removesuffix(".npz")
        if font == f"https://vedo.embl.es/fonts/{font_name}.npz":
            expected_sha256 = OFFICIAL_FONT_SHA256.get(font_name)
        try:
            fontfile = vedo.file_io.download(font, verbose=False, force=False)
            font = os.path.basename(font).split(".")[0]
        except Exception:
            vedo.logger.warning(f"font {font} not found")
            return _load_local_fallback_font(os.path.basename(font).split(".")[0])
    elif font.endswith(".npz"):
        fontfile = font
        font = os.path.basename(font).split(".")[0]
    else:
        font = font[:1].upper() + font[1:]
        fontfile = _font_path(font)

        if font not in settings.font_parameters.keys():
            vedo.logger.warning(
                f"Unknown font: {font}\n"
                f"Available 3D fonts are: "
                f"{list(settings.font_parameters.keys())}\n"
                f"Using font Normografo instead."
            )
            font = _LOCAL_FALLBACK_FONT
            fontfile = _font_path(font)

        if not settings.font_parameters[font]["islocal"]:
            expected_sha256 = OFFICIAL_FONT_SHA256.get(font)
            font = "https://vedo.embl.es/fonts/" + font + ".npz"
            try:
                fontfile = vedo.file_io.download(font, verbose=False, force=False)
                font = os.path.basename(font).split(".")[0]
            except Exception:
                vedo.logger.warning(f"font {font} not found")
                return _load_local_fallback_font(os.path.basename(font).split(".")[0])

    try:
        font_meshes = read_font(fontfile, expected_sha256=expected_sha256)
    except (UnsafePickleError, NumpyIntegrityError):
        # Do not hide a security rejection behind a successful fallback font.
        raise
    except Exception:
        vedo.logger.warning(f"font name {font} not found.")
        return _load_local_fallback_font(font)

    return font_meshes


@lru_cache(None)
def _get_font_letter(font, letter):
    font_meshes = _load_font(font)
    try:
        pts, faces = font_meshes[letter]
        return utils.buildPolyData(pts.astype(float), faces)
    except KeyError:
        return None
