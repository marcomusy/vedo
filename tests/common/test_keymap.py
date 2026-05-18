#!/usr/bin/env python3
from __future__ import annotations

"""Regression checks for default plotter key handling."""

from vedo.plotter import keymap


class _FakeInteractor:
    def __init__(self, key: str, *, ctrl: bool = False, shift: bool = False):
        self.key = key
        self.ctrl = ctrl
        self.shift = shift

    def GetKeySym(self):
        return self.key

    def GetShiftKey(self):
        return self.shift

    def GetControlKey(self):
        return self.ctrl

    def GetAltKey(self):
        return False

    def GetEventPosition(self):
        return 0, 0

    def FindPokedRenderer(self, _x, _y):
        return object()

    def Render(self):
        pass


class _FakeCutMesh:
    def __init__(self):
        self.written = []

    def write(self, filename):
        self.written.append(filename)


class _FakeSourceMesh:
    def __init__(self, calls):
        self.calls = calls
        self.filename = "/tmp/bunny_cut.stl"

    def apply_transform_from_actor(self):
        self.calls.append("apply_transform_from_actor")


class _FakeCutter:
    def __init__(self, cut_mesh):
        self.calls = []
        self.mesh = _FakeSourceMesh(self.calls)
        self.cut_mesh = cut_mesh
        self.invert = None

    def get_cut_mesh(self, invert=False):
        self.calls.append("get_cut_mesh")
        self.invert = invert
        return self.cut_mesh


class _FakePlotter:
    def __init__(self, cutter):
        self.cutter_widget = cutter


def test_ctrl_s_saves_active_cutter_mesh(monkeypatch) -> None:
    notices = []
    monkeypatch.setattr(
        keymap,
        "_print_keymap_notice",
        lambda *args, **kwargs: notices.append((args, kwargs)),
    )
    cut_mesh = _FakeCutMesh()
    cutter = _FakeCutter(cut_mesh)
    plotter = _FakePlotter(cutter)
    iren = _FakeInteractor("s", ctrl=True)

    keymap.handle_default_keypress(plotter, iren, None)

    assert cutter.invert is False
    assert cutter.calls == ["apply_transform_from_actor", "get_cut_mesh"]
    assert cut_mesh.written == ["bunny_cut.stl"]
    assert notices[0][0][0] == "Cut Mesh Saved"


def test_ctrl_shift_s_also_saves_active_cutter_mesh(monkeypatch) -> None:
    monkeypatch.setattr(keymap, "_print_keymap_notice", lambda *args, **kwargs: None)
    cut_mesh = _FakeCutMesh()
    plotter = _FakePlotter(_FakeCutter(cut_mesh))
    iren = _FakeInteractor("s", ctrl=True, shift=True)

    keymap.handle_default_keypress(plotter, iren, None)

    assert cut_mesh.written == ["bunny_cut.stl"]
