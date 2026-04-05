#!/usr/bin/env python3
from __future__ import annotations

"""Guardrails for the slow VTK import notice."""

import contextlib
import io
import time
import types

import vedo.vtkclasses as vtki


def _reset_notice_state() -> None:
    timer = vtki._slow_load_notice_timer
    if timer is not None:
        timer.cancel()
    vtki._slow_load_notice_timer = None
    vtki._slow_load_notice_shown = False
    vtki._slow_load_depth = 0


def test_vtkclasses_slow_notice() -> None:
    original_delay = vtki._SLOW_LOAD_NOTICE_DELAY
    original_import_module = vtki.import_module
    slow_module_name = "vtkmodules.vtkSlowNoticeModule"
    fast_module_name = "vtkmodules.vtkFastNoticeModule"
    slow_symbol = "vtkSlowNoticeThing"
    fast_symbol = "vtkFastNoticeThing"

    try:
        vtki._SLOW_LOAD_NOTICE_DELAY = 0.05
        vtki.location[slow_symbol] = "vtkSlowNoticeModule"
        vtki.location[fast_symbol] = "vtkFastNoticeModule"
        vtki.module_cache.pop(slow_module_name, None)
        vtki.module_cache.pop(fast_module_name, None)

        slow_module = types.SimpleNamespace(**{slow_symbol: object()})
        fast_module = types.SimpleNamespace(**{fast_symbol: object()})

        def fake_import_module(module_name: str):
            if module_name == slow_module_name:
                time.sleep(0.10)
                return slow_module
            if module_name == fast_module_name:
                return fast_module
            return original_import_module(module_name)

        vtki.import_module = fake_import_module

        _reset_notice_state()
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            assert vtki.get_class(fast_symbol) is fast_module.vtkFastNoticeThing
        assert "please wait" not in stderr.getvalue().lower()

        _reset_notice_state()
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            assert vtki.get_class(slow_symbol) is slow_module.vtkSlowNoticeThing
        assert "please wait" in stderr.getvalue().lower()

        vtki.module_cache.pop(slow_module_name, None)
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            assert vtki.get_class(slow_symbol) is slow_module.vtkSlowNoticeThing
        assert "please wait" not in stderr.getvalue().lower()
    finally:
        _reset_notice_state()
        vtki._SLOW_LOAD_NOTICE_DELAY = original_delay
        vtki.import_module = original_import_module
        vtki.location.pop(slow_symbol, None)
        vtki.location.pop(fast_symbol, None)
        vtki.module_cache.pop(slow_module_name, None)
        vtki.module_cache.pop(fast_module_name, None)
