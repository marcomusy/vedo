from __future__ import annotations

import numpy as np
import pytest

import vedo


def _make_vol(shape=(8, 8, 8)) -> vedo.Volume:
    rng = np.random.default_rng(0)
    return vedo.Volume(rng.uniform(0, 1, shape))


# ── Volume.shift ──────────────────────────────────────────────────────────────

def test_volume_shift_scalars() -> None:
    vol = _make_vol()
    orig = vol.origin().copy()
    vol.shift(1, 2, 3)
    assert np.allclose(vol.origin(), orig + [1, 2, 3], atol=1e-5)


def test_volume_shift_list() -> None:
    vol = _make_vol()
    orig = vol.origin().copy()
    vol.shift([4, 5, 6])
    assert np.allclose(vol.origin(), orig + [4, 5, 6], atol=1e-5)


def test_volume_shift_returns_self() -> None:
    vol = _make_vol()
    result = vol.shift(1, 0, 0)
    assert result is vol


def test_volume_shift_zero() -> None:
    vol = _make_vol()
    orig = vol.origin().copy()
    vol.shift(0, 0, 0)
    assert np.allclose(vol.origin(), orig)


# ── Volume.pos / origin ───────────────────────────────────────────────────────

def test_volume_pos_set_get() -> None:
    vol = _make_vol()
    vol.pos([3, 4, 5])
    assert np.allclose(vol.pos(), [3, 4, 5], atol=1e-5)


def test_volume_pos_returns_self() -> None:
    vol = _make_vol()
    result = vol.pos([1, 2, 3])
    assert result is vol


# ── Volume.color ──────────────────────────────────────────────────────────────

def test_volume_color_single_string() -> None:
    vol = _make_vol()
    vol.color("red")  # should not raise


def test_volume_color_list_of_colors() -> None:
    vol = _make_vol()
    vol.color(["red", "white", "blue"])  # should not raise


def test_volume_color_colormap_string() -> None:
    vol = _make_vol()
    vol.color("viridis")  # should not raise


def test_volume_color_matplotlib_cmap() -> None:
    import matplotlib.cm as cm
    vol = _make_vol()
    vol.color(cm.viridis)  # previously crashed with alpha.append on None


def test_volume_color_matplotlib_lsc() -> None:
    import matplotlib.cm as cm
    vol = _make_vol()
    vol.color(cm.hot)  # resampled path


# ── Volume.alpha ──────────────────────────────────────────────────────────────

def test_volume_alpha_list() -> None:
    vol = _make_vol()
    vol.alpha([0.0, 0.5, 1.0])  # should not raise


def test_volume_alpha_scalar() -> None:
    vol = _make_vol()
    vol.alpha(0.5)  # should not raise
