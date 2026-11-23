#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import shlex
import warnings
from types import SimpleNamespace

import pytest

import vedo.cli as cli
from vedo.file_io.numpy_io import NumpyIntegrityError, UnsafePickleError, _warn_unsafe_pickle


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_classify_output_path() -> None:
    assert cli._classify_output_path("figure.png") == "image"
    assert cli._classify_output_path("scene.html") == "scene"
    assert cli._classify_output_path("scene.xyz") is None


def test_cli_help_is_plain_for_non_tty() -> None:
    parser = cli.get_parser()
    stream = io.StringIO()
    parser.print_help(stream)
    assert "\x1b[" not in stream.getvalue()


def test_cli_help_is_colored_for_tty(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    parser = cli.get_parser()
    stream = TtyStringIO()
    parser.print_help(stream)
    assert "\x1b[" in stream.getvalue()


def test_cli_help_respects_no_color(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    parser = cli.get_parser()
    stream = TtyStringIO()
    parser.print_help(stream)
    assert "\x1b[" not in stream.getvalue()


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


@pytest.mark.parametrize("convert", [False, True])
def test_main_explains_pickle_opt_in_without_traceback(monkeypatch, capsys, convert):
    filename = "scene's depth ratios.npz"
    arguments = ["--convert", filename, "--to", "vtk"] if convert else [filename]
    monkeypatch.setattr(cli.sys, "argv", ["vedo", *arguments])
    monkeypatch.setattr(cli, "_ensure_cli_runtime", lambda: None)

    def reject_pickle(args):
        raise UnsafePickleError("Internal loader details")

    monkeypatch.setattr(cli, "exe_convert" if convert else "draw_scene", reject_pickle)
    assert cli.main() == 1
    output = capsys.readouterr()
    assert not output.out
    assert "can execute code" in output.err
    assert "Only if you trust" in output.err
    assert "Internal loader details" not in output.err
    command = output.err.splitlines()[-1].strip()
    if os.name == "nt":
        assert command.startswith("set VEDO_ALLOW_UNSAFE_PICKLE=1 && ")
    else:
        assert shlex.split(command) == ["VEDO_ALLOW_UNSAFE_PICKLE=1", "vedo", *arguments]


def test_main_reports_font_integrity_error_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["vedo", "scene.npz"])
    monkeypatch.setattr(cli, "_ensure_cli_runtime", lambda: None)

    def reject_font(args):
        raise NumpyIntegrityError("Font checksum mismatch; refusing to load it.")

    monkeypatch.setattr(cli, "draw_scene", reject_font)
    assert cli.main() == 1
    output = capsys.readouterr()
    assert not output.out
    assert output.err == "vedo: loading blocked: Font checksum mismatch; refusing to load it.\n"


def test_loading_error_contains_entire_message_inside_panel(monkeypatch):
    from rich.text import Text

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = TtyStringIO()
    monkeypatch.setattr(cli.sys, "stderr", stream)
    command = "VEDO_ALLOW_UNSAFE_PICKLE=1 vedo '[scene].npz'"
    cli._print_loading_error("Pickle can execute code.", command=command)
    output = stream.getvalue()
    assert "\x1b[" in output
    assert "Loading blocked" in output
    assert command in output
    lines = Text.from_ansi(output).plain.splitlines()
    assert lines[-1].startswith("╰")
    assert all(line.startswith("│") and line.endswith("│") for line in lines[1:-1])


def test_cli_colorizes_unsafe_pickle_warning_and_preserves_other_warnings(monkeypatch):
    from rich.text import Text

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(cli.sys, "argv", ["vedo", "scene.npz"])
    monkeypatch.setattr(cli, "_ensure_cli_runtime", lambda: None)
    stream = TtyStringIO()
    monkeypatch.setattr(cli.sys, "stderr", stream)
    original_showwarning = warnings.showwarning

    def load_trusted_scene(args):
        _warn_unsafe_pickle()
        _warn_unsafe_pickle()
        warnings.warn("Unrelated warning", UserWarning)

    monkeypatch.setattr(cli, "draw_scene", load_trusted_scene)
    _warn_unsafe_pickle.cache_clear()
    try:
        with pytest.warns(UserWarning, match="Unrelated warning"):
            assert cli.main() == 0
    finally:
        _warn_unsafe_pickle.cache_clear()
    output = stream.getvalue()
    assert "\x1b[" in output
    plain = Text.from_ansi(output).plain
    paragraph = " ".join(line.strip("│ ") for line in plain.splitlines()[1:-1])
    assert plain.count("Unsafe pickle enabled") == 1
    assert "VEDO_ALLOW_UNSAFE_PICKLE=1" in plain
    assert "execute Python code" in paragraph
    assert "whole process" in paragraph
    assert "numpy_io.py" not in plain
    assert "warnings.warn" not in plain
    assert warnings.showwarning is original_showwarning


@pytest.mark.parametrize("setting", ["NO_COLOR", "TERM"])
def test_loading_error_respects_terminal_color_settings(monkeypatch, setting):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv(setting, "1" if setting == "NO_COLOR" else "dumb")
    stream = TtyStringIO()
    monkeypatch.setattr(cli.sys, "stderr", stream)
    cli._print_loading_error("Pickle can execute code.", command="vedo scene.npz")
    assert "\x1b[" not in stream.getvalue()
    assert "vedo: loading blocked:" in stream.getvalue()


def test_main_preserves_unexpected_exceptions(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["vedo", "scene.npz"])
    monkeypatch.setattr(cli, "_ensure_cli_runtime", lambda: None)

    def fail(args):
        raise ValueError("unexpected failure")

    monkeypatch.setattr(cli, "draw_scene", fail)
    with pytest.raises(ValueError, match="unexpected failure"):
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
