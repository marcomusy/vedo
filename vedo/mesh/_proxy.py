#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Runtime-safe Mesh proxy for mixins to avoid circular imports."""

import vedo


class _MeshMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, vedo.mesh.Mesh)


class Mesh(metaclass=_MeshMeta):
    def __new__(cls, *args, **kwargs):
        return vedo.mesh.Mesh(*args, **kwargs)
