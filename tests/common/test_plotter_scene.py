#!/usr/bin/env python3
from __future__ import annotations

"""Regression checks for Plotter scene object lookup."""

import numpy as np
import pytest

import vedo.vtkclasses as vtki
from vedo import Arrows, Plotter, Sphere


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
