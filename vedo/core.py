#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility facade for vedo core algorithm mixins."""

from vedo.core_data import DataArrayHelper, _get_data_legacy_format
from vedo.core_common import CommonAlgorithms
from vedo.core_points import PointAlgorithms
from vedo.core_volume import VolumeAlgorithms

__all__ = [
    "DataArrayHelper",
    "CommonAlgorithms",
    "PointAlgorithms",
    "VolumeAlgorithms",
]
