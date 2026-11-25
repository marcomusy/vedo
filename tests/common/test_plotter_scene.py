#!/usr/bin/env python3
from __future__ import annotations

"""Regression checks for Plotter scene object lookup."""

import numpy as np
import pytest

import vedo.vtkclasses as vtki
from vedo import Arrows, Cube, Plotter, Sphere, Text2D


def test_plotter_scene_object_lookup() -> None:
    plt = Plotter(offscreen=True)

    pts = np.column_stack(
        [np.linspace(0, 1, 5), np.sqrt(np.linspace(0, 1, 5)), np.zeros(5)]
    )
    arrows = Arrows(pts[:-1], pts[1:])

    sphere = Sphere()
    del sphere.actor.retrieve_object

    plt.add(sphere, arrows)

    pickable_meshes = plt.get_meshes()
    all_meshes = plt.get_meshes(include_non_pickables=True)
    plt.close()

    assert sphere in pickable_meshes
    assert arrows not in pickable_meshes
    assert sphere in all_meshes
    assert arrows in all_meshes


def test_close_before_interactor_initialization(monkeypatch) -> None:
    plt = Plotter()
    interactor = plt.interactor
    assert not interactor.GetInitialized()

    shutdown_calls = []
    monkeypatch.setattr(interactor, "ExitCallback", lambda: shutdown_calls.append("exit"))
    monkeypatch.setattr(interactor, "TerminateApp", lambda: shutdown_calls.append("terminate"))
    plt.close()

    assert shutdown_calls == []
    assert plt.interactor is None
    assert plt.window is None


@pytest.mark.parametrize("deep", [False, True])
def test_clear_only_removes_objects_from_the_requested_renderer(deep) -> None:
    with Plotter(shape=(1, 2), offscreen=True) as plt:
        left, right, shared = Sphere(), Cube(), Text2D("shared")
        plt.add(left, shared, at=0).add(right, shared, at=1)

        assert plt.clear(at=1, deep=deep) is plt
        assert plt.renderer is plt.renderers[0]
        assert plt.renderers[0].HasViewProp(left.actor)
        assert plt.renderers[0].HasViewProp(shared.actor)
        assert not plt.get_actors(at=1, include_non_pickables=True)
        assert plt.objects == [left, shared]
        assert right.rendered_at == set()
        assert shared.rendered_at == {0}
        assert shared.text() == "shared"


def test_offscreen_close_finalizes_window_once(monkeypatch) -> None:
    plt = Plotter(offscreen=True)
    finalize = plt.window.Finalize
    finalized = []

    def record_finalize():
        finalized.append(True)
        finalize()

    monkeypatch.setattr(plt.window, "Finalize", record_finalize)
    with plt:
        assert plt.interactor is None

    assert plt.close() is plt
    assert finalized == [True]
    assert plt.window is None
    assert plt.camera is None
    assert plt.renderer is None
    assert not plt.renderers


@pytest.mark.rendering
def test_plotter_button_registers_actor_and_pick_callback() -> None:
    plt = Plotter(offscreen=True)
    plt.interactor = vtki.vtkRenderWindowInteractor()
    plt.interactor.SetRenderWindow(plt.window)
    plt.interactor.Initialize()
    picked = []

    try:
        button = plt.add_button(lambda actor, event: picked.append((actor, event)))

        assert button in plt.buttons
        assert plt.renderer.HasViewProp(button.actor)
        button.actor.InvokeEvent("PickEvent")
        assert picked == [(button.actor, "PickEvent")]
    finally:
        plt.close()
