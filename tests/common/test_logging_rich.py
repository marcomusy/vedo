#!/usr/bin/env python3
from __future__ import annotations

import io
import logging

from rich.console import Console
from rich.logging import RichHandler

import vedo


def test_default_logger_uses_rich_handler() -> None:
    default_handlers = [
        handler
        for handler in vedo.logger.handlers
        if getattr(handler, "_vedo_default_handler", False)
    ]

    assert len(default_handlers) == 1
    assert isinstance(default_handlers[0], RichHandler)
    assert vedo.logger.level == logging.INFO
    assert vedo.logger.propagate is False


def test_default_logger_renders_vedo_prefix() -> None:
    handler = next(
        handler
        for handler in vedo.logger.handlers
        if getattr(handler, "_vedo_default_handler", False)
    )
    assert isinstance(handler, RichHandler)

    buffer = io.StringIO()
    original_console = handler.console
    handler.console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=120,
    )
    try:
        vedo.logger.error("rich logger smoke test")
    finally:
        handler.console = original_console

    rich_text = buffer.getvalue()
    assert "ERROR" in rich_text
    assert "[vedo.test_logging_rich:" in rich_text
    assert "rich logger smoke test" in rich_text
