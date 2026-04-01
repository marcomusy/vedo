#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Shared helpers for plain-text and rich object summaries."""

import sys
import numpy as np


def summary_title(obj, address: str | None = None) -> str:
    """Build the standard summary title for an object."""
    if address is None:
        address_getter = getattr(obj, "_summary_address", None)
        if callable(address_getter):
            address = address_getter()
        else:
            address = hex(id(obj))
    return f"{obj.__class__.__module__}.{obj.__class__.__name__} at ({address})"


def summary_panel(obj, rows, color: str = "white", expand: bool = False):
    """Return a rich panel for the provided summary rows."""
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=False, box=None, pad_edge=False, expand=False)
    table.add_column("Field", style=f"bold {color}", no_wrap=True)
    table.add_column("Value", style=color)
    for field, value in rows:
        table.add_row(str(field), str(value))
    return Panel(
        table,
        title=summary_title(obj),
        title_align="left",
        expand=expand,
        border_style=f"bold {color}",
    )


def summary_text(obj, rows) -> str:
    """Return a plain-text summary for the provided rows."""
    rows = [(str(field), str(value)) for field, value in rows]
    width = max((len(field) for field, _ in rows), default=0)
    width = max(width, 14)

    out = [summary_title(obj)]
    for field, value in rows:
        label = field.ljust(width)
        if "\n" in value:
            out.append(f"{label}:")
            indent = " " * (width + 2)
            out.extend(f"{indent}{line}" for line in value.splitlines())
        else:
            out.append(f"{label}: {value}")
    return "\n".join(out)


def summary_string(obj, rows, color: str = "white", expand: bool = False) -> str:
    """Use rich formatting in an interactive terminal, else return plain text."""
    fallback = summary_text(obj, rows)
    if (
        sys.stdout is None
        or not hasattr(sys.stdout, "isatty")
        or not sys.stdout.isatty()
    ):
        return fallback
    try:
        from io import StringIO
        from rich.console import Console

        buffer = StringIO()
        Console(file=buffer, force_terminal=True).print(
            summary_panel(obj, rows, color=color, expand=expand)
        )
        return buffer.getvalue().rstrip()
    except Exception:
        return fallback


def format_bounds(bounds, precision_func, digits: int = 3) -> str:
    """Format a bounds tuple as x/y/z ranges."""
    bx1, bx2 = precision_func(bounds[0], digits), precision_func(bounds[1], digits)
    by1, by2 = precision_func(bounds[2], digits), precision_func(bounds[3], digits)
    bz1, bz2 = precision_func(bounds[4], digits), precision_func(bounds[5], digits)
    return f"x=({bx1}, {bx2}), y=({by1}, {by2}), z=({bz1}, {bz2})"


def active_array_label(dataset, association: str, key: str, base_label: str) -> str:
    """Return the label for a point/cell array, marking the active role when needed."""
    if association.startswith("p"):
        data = dataset.GetPointData()
    else:
        data = dataset.GetCellData()
    scalars = data.GetScalars()
    vectors = data.GetVectors()
    tensors = data.GetTensors()
    if scalars and scalars.GetName() == key:
        return base_label + " *"
    if vectors and vectors.GetName() == key:
        return base_label + " **"
    if tensors and tensors.GetName() == key:
        return base_label + " ***"
    return base_label


def summarize_array(
    arr, precision_func, *, include_range: bool = True, dim_label: str = "dim"
) -> str:
    """Summarize a numpy array for object-printing."""
    dim = arr.shape[1] if arr.ndim > 1 else 1
    value = f"({arr.dtype}), {dim_label}={dim}"
    if include_range and len(arr) > 0 and dim == 1:
        if "int" in arr.dtype.name:
            rng = f"{arr.min()}, {arr.max()}"
        else:
            rng = precision_func(arr.min(), 3) + ", " + precision_func(arr.max(), 3)
        value += f", range=({rng})"
    elif include_range and len(arr) > 0 and arr.ndim >= 1:
        try:
            arr_min = np.nanmin(arr)
            arr_max = np.nanmax(arr)
            if np.isfinite(arr_min) and np.isfinite(arr_max):
                rng = precision_func(arr_min, 3) + ", " + precision_func(arr_max, 3)
                value += f", range=({rng})"
        except (TypeError, ValueError):
            pass
    return value
