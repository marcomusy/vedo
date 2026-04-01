#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Event data object used by the Plotter callback system."""

from vedo.core.summary import summary_panel, summary_string


class Event:
    """
    This class holds the info from an event in the window, works as dictionary too.
    """

    __slots__ = [
        "name",
        "title",
        "id",
        "timerid",
        "time",
        "priority",
        "at",
        "object",
        "actor",
        "picked3d",
        "keypress",
        "picked2d",
        "delta2d",
        "angle2d",
        "speed2d",
        "delta3d",
        "speed3d",
        "isPoints",
        "isMesh",
        "isAssembly",
        "isVolume",
        "isImage",
        "isActor2D",
    ]

    def __init__(self):
        self.name = "event"
        self.title = ""
        self.id = 0
        self.timerid = 0
        self.time = 0
        self.priority = 0
        self.at = 0
        self.object = None
        self.actor = None
        self.picked3d = ()
        self.keypress = ""
        self.picked2d = ()
        self.delta2d = ()
        self.angle2d = 0
        self.speed2d = ()
        self.delta3d = ()
        self.speed3d = 0
        self.isPoints = False
        self.isMesh = False
        self.isAssembly = False
        self.isVolume = False
        self.isImage = False
        self.isActor2D = False

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return summary_string(self, self._summary_rows())

    def __repr__(self):
        return self.__str__()

    def __rich__(self):
        return summary_panel(self, self._summary_rows())

    def _summary_rows(self):
        rows = []
        for n in self.__slots__:
            if n == "actor":
                continue
            rows.append((n, str(self[n]).replace("\n", "")[:65].rstrip()))
        return rows

    def keys(self):
        """Return the list of keys."""
        return self.__slots__
