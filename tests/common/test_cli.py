#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace

import pytest

import vedo.cli as cli


def test_classify_output_path() -> None:
    assert cli._classify_output_path("figure.png") == "image"
    assert cli._classify_output_path("scene.html") == "scene"
    assert cli._classify_output_path("scene.xyz") is None


def test_main_rejects_unsupported_output(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys, "argv", ["vedo", "mesh.vtk", "--output", "mesh.gif"])
    with pytest.raises(SystemExit):
        cli.main()


def test_main_rejects_backend_without_html_output(monkeypatch) -> None:
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["vedo", "mesh.vtk", "--output", "mesh.png", "--backend", "threejs"],
    )
    with pytest.raises(SystemExit):
        cli.main()


def test_write_cli_output_dispatches_screenshot(monkeypatch) -> None:
    screenshot_calls = []
    export_calls = []
    log_messages = []

    class DummyPlotter:
        def screenshot(self, filename, scale=1):
            screenshot_calls.append((filename, scale))

    monkeypatch.setattr(
        cli,
        "vedo",
        SimpleNamespace(
            file_io=SimpleNamespace(
                export_window=lambda *args, **kwargs: export_calls.append((args, kwargs))
            ),
            logger=SimpleNamespace(info=lambda message: log_messages.append(message)),
        ),
    )

    args = SimpleNamespace(output="frame.png", scale=3, backend=None)
    assert cli._write_cli_output(DummyPlotter(), args) is True
    assert screenshot_calls == [("frame.png", 3)]
    assert export_calls == []
    assert log_messages == ["Saved output to frame.png"]


def test_write_cli_output_dispatches_scene_export(monkeypatch) -> None:
    export_calls = []
    plotter = SimpleNamespace()

    monkeypatch.setattr(
        cli,
        "vedo",
        SimpleNamespace(
            file_io=SimpleNamespace(
                export_window=lambda *args, **kwargs: export_calls.append((args, kwargs))
            ),
            logger=SimpleNamespace(info=lambda _message: None),
        ),
    )

    args = SimpleNamespace(output="scene.html", scale=1, backend="threejs")
    cli._write_cli_output(plotter, args)

    assert len(export_calls) == 1
    call_args, call_kwargs = export_calls[0]
    assert call_args == ("scene.html",)
    assert call_kwargs == {"plt": plotter, "backend": "threejs"}


def test_show_and_finalize_forces_noninteractive_export(monkeypatch) -> None:
    write_calls = []

    class DummyPlotter:
        def __init__(self) -> None:
            self.show_calls = []
            self.closed = False

        def show(self, *objects, **kwargs):
            self.show_calls.append((objects, kwargs))
            return self

        def close(self):
            self.closed = True

    monkeypatch.setattr(cli, "_write_cli_output", lambda plt, args: write_calls.append((plt, args)))

    plotter = DummyPlotter()
    args = SimpleNamespace(output="frame.png", offscreen=False)
    cli._show_and_finalize(plotter, args, "mesh", interactive=True)

    assert plotter.show_calls == [(("mesh",), {"interactive": False})]
    assert len(write_calls) == 1
    assert plotter.closed is True


def test_main_parses_export_options(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(cli, "_ensure_cli_runtime", lambda: None)
    monkeypatch.setattr(cli, "draw_scene", lambda args: captured.update(vars(args)))
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["vedo", "mesh.vtk", "--output", "mesh.png", "--scale", "2", "--offscreen"],
    )

    assert cli.main() == 0
    assert captured["files"] == ["mesh.vtk"]
    assert captured["output"] == "mesh.png"
    assert captured["scale"] == 2
    assert captured["offscreen"] is True
