#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

import vedo
from vedo.pyplot.functions import fit, histogram, plot, streamplot


# ---------------------------------------------------------------------------
# plot() — input forms
# ---------------------------------------------------------------------------


def test_plot_xy():
    x = np.linspace(0, 2 * np.pi, 40)
    fig = plot(x, np.sin(x))
    assert fig is not None


def test_plot_y_only():
    fig = plot(np.cos(np.linspace(0, np.pi, 30)))
    assert fig is not None


def test_plot_list_of_tuples():
    fig = plot([(i, i**2) for i in range(10)])
    assert fig is not None


def test_plot_nested_xy():
    x = np.arange(10, dtype=float)
    fig = plot([x, x**2])
    assert fig is not None


def test_plot_matplotlib_line():
    x = np.linspace(0, 1, 20)
    fig = plot(x, np.sin(x), "r-")
    assert fig is not None


def test_plot_matplotlib_dashed():
    x = np.linspace(0, 1, 20)
    fig = plot(x, np.cos(x), "b--")
    assert fig is not None


def test_plot_matplotlib_scatter():
    x = np.linspace(0, 1, 20)
    fig = plot(x, x**2, "go")
    assert fig is not None


def test_plot_opts_space_stripped():
    # regression: opts.replace(" ", "") result was discarded; spaces never removed
    x = np.linspace(0, 1, 20)
    fig = plot(x, x, "r -")
    assert fig is not None


# ---------------------------------------------------------------------------
# plot() — modes
# ---------------------------------------------------------------------------


def test_plot_mode_bar():
    fig = plot([[3, 5, 2, 7], ["a", "b", "c", "d"]], mode="bar")
    assert fig is not None


def test_plot_mode_bar_errors():
    fig = plot([[3, 5, 2, 7], ["a", "b", "c", "d"]], mode="bar", errors=True)
    assert fig is not None


def test_plot_mode_polar_two_args():
    angles = np.linspace(0, 2 * np.pi, 30)
    radii = np.abs(np.sin(angles))
    fig = plot(angles, radii, mode="polar")
    assert fig is not None


def test_plot_mode_polar_two_column_array():
    # regression: len(rphi)==2 was wrong for (N,2) arrays
    data = np.c_[np.linspace(0, 2 * np.pi, 20), np.ones(20)]
    fig = plot(data, mode="polar")
    assert fig is not None


def test_plot_mode_polar_stack_form():
    # (2, N) stacked input
    angles = np.linspace(0, 2 * np.pi, 20)
    radii = np.ones(20)
    fig = plot(np.stack([angles, radii]), mode="polar")
    assert fig is not None


def test_plot_fxy_callable():
    fig = plot(lambda x, y: np.sin(x) * np.cos(y), bins=(10, 10))
    assert fig is not None


def test_plot_fxy_zlim_zero():
    # regression: if zlim[0]: was False for 0.0, skipping a valid cut plane
    fig = plot(
        lambda x, y: x + y - 3,
        xlim=(0, 3),
        ylim=(0, 3),
        zlim=(0.0, 2.0),
        bins=(10, 10),
    )
    assert fig is not None


def test_plot_fxy_zlim_negative():
    fig = plot(
        lambda x, y: x + y - 1,
        xlim=(0, 2),
        ylim=(0, 2),
        zlim=(-0.5, 1.5),
        bins=(10, 10),
    )
    assert fig is not None


def test_plot_complex_mode():
    fig = plot(lambda x, y: complex(x, y), mode="complex", bins=(10, 10))
    assert fig is not None


def test_plot_complex_all_real():
    # regression: v=0 caused degenerate colormap (vmin=vmax=0)
    fig = plot(lambda x, y: complex(x, 0), mode="complex", bins=(10, 10))
    assert fig is not None


def test_plot_complex_zlimits_zero():
    # regression: if zlimits[0]: was False for 0.0
    fig = plot(
        lambda x, y: complex(x, y),
        mode="complex",
        bins=(10, 10),
        zlimits=(0.0, 0.5),
    )
    assert fig is not None


def test_plot_spheric():
    fig = plot(lambda theta, phi: 1.0 + 0.3 * np.sin(theta), mode="spheric", res=8)
    assert fig is not None


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_fit_linear():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 20)
    y = 2 * x + 1 + rng.normal(0, 0.1, 20)
    result = fit(np.c_[x, y], deg=1)
    assert result is not None
    assert len(result.coefficients) == 2
    assert result.ndof == 18


def test_fit_quadratic():
    x = np.linspace(-2, 2, 30)
    y = x**2
    result = fit(np.c_[x, y], deg=2)
    assert result is not None
    assert len(result.coefficients) == 3


def test_fit_vrange():
    x = np.linspace(0, 4, 20)
    result = fit(np.c_[x, x], deg=1, vrange=(-1.0, 5.0))
    assert result is not None


def test_fit_with_yerrors():
    x = np.linspace(0, 4, 20)
    y = x + 0.5
    result = fit(np.c_[x, y], deg=1, yerrors=np.ones_like(x) * 0.1)
    assert result is not None


def test_fit_xerrors_without_yerrors():
    # regression: TypeError — yerrors*yerrors when yerrors is None
    x = np.linspace(0, 4, 20)
    y = x + 0.5
    result = fit(np.c_[x, y], deg=1, xerrors=np.ones_like(x) * 0.05)
    assert result is not None


def test_fit_both_errors():
    x = np.linspace(0, 4, 20)
    y = x + 0.5
    result = fit(
        np.c_[x, y],
        deg=1,
        xerrors=np.ones_like(x) * 0.05,
        yerrors=np.ones_like(x) * 0.1,
    )
    assert result is not None


def test_fit_monte_carlo():
    x = np.linspace(0, 4, 20)
    y = 2 * x
    result = fit(np.c_[x, y], deg=1, niter=50)
    assert result is not None
    assert result.monte_carlo_coefficients.shape == (50, 2)
    assert result.error_band is not None


def test_fit_insufficient_points_returns_none():
    # regression: ndof<=0 crashed with division by zero instead of returning None
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    result = fit(np.c_[x, y], deg=2)
    assert result is None


def test_fit_from_points_object():
    x = np.linspace(0, 3, 15)
    y = x * 1.5
    pts = vedo.Points(np.c_[x, y, np.zeros_like(x)])
    result = fit(pts, deg=1)
    assert result is not None


def test_fit_two_row_input():
    x = np.linspace(0, 5, 15)
    y = x * 2
    result = fit([x, y], deg=1)
    assert result is not None


# ---------------------------------------------------------------------------
# histogram()
# ---------------------------------------------------------------------------


def test_histogram_1d():
    data = np.random.default_rng(1).normal(0, 1, 200)
    h = histogram(data)
    assert h is not None


def test_histogram_1d_bins():
    data = np.random.default_rng(2).normal(0, 1, 100)
    h = histogram(data, bins=20)
    assert h is not None


def test_histogram_1d_polar():
    data = np.random.default_rng(3).uniform(0, 2 * np.pi, 100)
    h = histogram(data, mode="polar")
    assert h is not None


def test_histogram_2d_xy_args():
    rng = np.random.default_rng(4)
    h = histogram(rng.normal(size=100), rng.normal(size=100))
    assert h is not None


def test_histogram_2d_array_input():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(100, 2))
    h = histogram(data)
    assert h is not None


def test_histogram_hexbin():
    rng = np.random.default_rng(6)
    h = histogram(rng.uniform(-1, 1, 80), rng.uniform(-1, 1, 80), mode="hexbin")
    assert h is not None


def test_histogram_hexbin_identical_x():
    # regression: dx==0 caused division by zero
    h = histogram(np.ones(50), np.random.default_rng(7).uniform(-1, 1, 50), mode="hexbin")
    assert h is not None


def test_histogram_hexbin_identical_y():
    # regression: dy==0 caused division by zero
    h = histogram(np.random.default_rng(8).uniform(-1, 1, 50), np.ones(50), mode="hexbin")
    assert h is not None


def test_histogram_3d():
    rng = np.random.default_rng(9)
    h = histogram(rng.normal(size=80), rng.normal(size=80), mode="3d")
    assert h is not None


def test_histogram_spheric():
    rng = np.random.default_rng(10)
    theta = rng.uniform(0, np.pi, 100)
    phi = rng.uniform(0, 2 * np.pi, 100)
    h = histogram(theta, phi, mode="spheric")
    assert h is not None


def test_histogram_spheric_empty():
    # regression: np.max(counts)==0 caused division by zero on empty sphere
    h = histogram(np.array([0.0]), np.array([0.0]), mode="spheric")
    assert h is not None


def test_histogram_from_volume():
    vol = vedo.Volume(np.random.default_rng(11).uniform(0, 1, (8, 8, 8)))
    h = histogram(vol)
    assert h is not None


def test_histogram_from_points_with_data():
    # regression: celldata[0] was called without None-check when pointdata[0] is None
    pts = vedo.Points(np.random.default_rng(12).uniform(0, 1, (40, 3)))
    pts.pointdata["vals"] = np.ones(40)
    h = histogram(pts)
    assert h is not None


def test_histogram_no_args_returns_none():
    # histogram() with more than 2 positional args logs an error and returns None
    h = histogram(np.zeros(5), np.zeros(5), np.zeros(5))
    assert h is None


# ---------------------------------------------------------------------------
# streamplot()
# ---------------------------------------------------------------------------


def test_streamplot_basic():
    n = 15
    x, y = np.meshgrid(np.linspace(-2, 2, n), np.linspace(-2, 2, n))
    stream = streamplot(x, y, -y, x)
    assert stream is not None


def test_streamplot_forward():
    n = 12
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    stream = streamplot(x, y, np.ones_like(x), np.zeros_like(y), direction="forward")
    assert stream is not None


def test_streamplot_non_square_raises():
    # non-square grid must raise RuntimeError
    x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 15))
    with pytest.raises(RuntimeError):
        streamplot(x, y, np.ones_like(x), np.zeros_like(y))
