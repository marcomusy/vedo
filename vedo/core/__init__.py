#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility facade for vedo core algorithm mixins."""

from vedo.core.data import DataArrayHelper, _get_data_legacy_format
from vedo.core.common import CommonAlgorithms
from vedo.core.points import PointAlgorithms
from vedo.core.volume import VolumeAlgorithms

__all__ = [
    "DataArrayHelper",
    "CommonAlgorithms",
    "PointAlgorithms",
    "VolumeAlgorithms",
]
