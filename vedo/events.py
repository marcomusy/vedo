#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event data object used by the Plotter callback system."""


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
        import vedo

        module = self.__class__.__module__
        name = self.__class__.__name__
        out = vedo.printc(
            f"{module}.{name} at ({hex(id(self))})".ljust(75),
            bold=True, invert=True, return_string=True,
        )
        out += "\x1b[0m"
        for n in self.__slots__:
            if n == "actor":
                continue
            out += f"{n}".ljust(11) + ": "
            val = str(self[n]).replace("\n", "")[:65].rstrip()
            if val == "True":
                out += "\x1b[32;1m"
            elif val == "False":
                out += "\x1b[31;1m"
            out += val + "\x1b[0m\n"
        return out.rstrip()

    def keys(self):
        """Return the list of keys."""
        return self.__slots__
