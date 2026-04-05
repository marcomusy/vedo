from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def restore_vedo_settings():
    import vedo

    snapshot = {
        key: getattr(vedo.settings, key)
        for key in getattr(vedo.settings, "__slots__", ())
        if hasattr(vedo.settings, key)
    }
    yield
    for key, value in snapshot.items():
        setattr(vedo.settings, key, value)

    if vedo.plotter_instance is not None:
        try:
            vedo.plotter_instance.close()
        except Exception:
            pass
