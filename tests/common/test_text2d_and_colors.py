from __future__ import annotations

import numpy as np
import pytest

from vedo import Text2D
import vedo.colors as colors_module
from vedo.colors import get_color, color_map


# ── Text2D.alpha ──────────────────────────────────────────────────────────────

def test_text2d_alpha_sets_text_opacity() -> None:
    t = Text2D("hello")
    t.alpha(0.4)
    assert np.isclose(t.properties.GetOpacity(), 0.4)


def test_text2d_alpha_does_not_touch_background() -> None:
    t = Text2D("hello")
    t.background("white", alpha=1.0)
    t.alpha(0.5)
    # background opacity should remain 1.0, only text opacity changed
    assert np.isclose(t.properties.GetBackgroundOpacity(), 1.0)
    assert np.isclose(t.properties.GetOpacity(), 0.5)


def test_text2d_alpha_returns_self() -> None:
    t = Text2D("hello")
    result = t.alpha(0.7)
    assert result is t


# ── Text2D.background ─────────────────────────────────────────────────────────

def test_text2d_background_sets_color() -> None:
    t = Text2D("hi")
    t.background("blue")
    bg = t.properties.GetBackgroundColor()
    expected = get_color("blue")
    assert np.allclose(bg, expected, atol=0.01)


def test_text2d_background_none_disables_it() -> None:
    t = Text2D("hi")
    t.background("red")
    t.background(None)
    assert np.isclose(t.properties.GetBackgroundOpacity(), 0.0)


def test_text2d_background_none_does_not_crash() -> None:
    t = Text2D("hi")
    t.background(None)  # must not raise (was calling get_color(None) before check)


def test_text2d_background_alpha() -> None:
    t = Text2D("hi")
    t.background("green", alpha=0.3)
    assert np.isclose(t.properties.GetBackgroundOpacity(), 0.3)


def test_text2d_background_auto_does_not_crash() -> None:
    t = Text2D("hi")
    t.background("auto", alpha=0.3)
    assert np.isclose(t.properties.GetBackgroundOpacity(), 0.3)


def test_text2d_background_returns_self() -> None:
    t = Text2D("hi")
    result = t.background("white")
    assert result is t


# ── colors.get_color ──────────────────────────────────────────────────────────

def test_get_color_by_name() -> None:
    r, g, b = get_color("red")
    assert r > 0.8
    assert r > g
    assert r > b


def test_get_color_hex() -> None:
    c = get_color("#ff0000")
    assert np.allclose(c, [1, 0, 0], atol=0.01)


def test_get_color_rgb_tuple() -> None:
    c = get_color((0.2, 0.4, 0.6))
    assert np.allclose(c, [0.2, 0.4, 0.6], atol=0.01)


def test_get_color_integer_index() -> None:
    c = get_color(0)
    assert len(c) == 3
    assert all(0 <= v <= 1 for v in c)


def test_get_color_white_black() -> None:
    assert np.allclose(get_color("white"), [1, 1, 1], atol=0.01)
    assert np.allclose(get_color("black"), [0, 0, 0], atol=0.01)


def test_get_color_uses_colorexists_fallback(monkeypatch) -> None:
    color_name = "customvtkblue"

    class DummyNamedColors:
        def ColorExists(self, name: str) -> bool:
            return name == color_name

        def GetColor(self, name: str, rgba) -> None:
            rgba[:] = [0, 0, 255, 255]

    colors_module._get_color_from_string.cache_clear()
    monkeypatch.setattr(colors_module, "_named_colors", DummyNamedColors())
    assert np.allclose(get_color(color_name), [0, 0, 1], atol=0.01)
    colors_module._get_color_from_string.cache_clear()


# ── colors.color_map ──────────────────────────────────────────────────────────

def test_color_map_scalar() -> None:
    c = color_map(0.5, "jet", vmin=0, vmax=1)
    assert len(c) == 3
    assert all(0 <= v <= 1 for v in c)


def test_color_map_boundary_values() -> None:
    c0 = color_map(0.0, "viridis", vmin=0, vmax=1)
    c1 = color_map(1.0, "viridis", vmin=0, vmax=1)
    assert len(c0) == 3
    assert len(c1) == 3
    assert not np.allclose(c0, c1)


def test_color_map_array_input() -> None:
    vals = np.linspace(0, 1, 10)
    result = color_map(vals, "hot", vmin=0, vmax=1)
    assert result.shape == (10, 3)
    assert np.all(result >= 0) and np.all(result <= 1)
