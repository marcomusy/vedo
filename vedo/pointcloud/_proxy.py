#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-safe Points proxy to avoid circular imports in mixins."""

import vedo


class _PointsMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, vedo.pointcloud.Points)


class Points(metaclass=_PointsMeta):
    def __new__(cls, *args, **kwargs):
        return vedo.pointcloud.Points(*args, **kwargs)
