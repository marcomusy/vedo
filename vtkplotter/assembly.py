from __future__ import division, print_function

import numpy as np
import vtk
import vtkplotter.docs as docs
from vtkplotter.base import ActorBase

__doc__ = (
    """
Submodule extending the ``vtkAssembly`` object functionality.
"""
    + docs._defs
)

__all__ = ["Assembly"]


#################################################
class Assembly(vtk.vtkAssembly, ActorBase):
    """Group many meshes as a single new mesh as a ``vtkAssembly``.

    |gyroscope1| |gyroscope1.py|_
    """

    def __init__(self, meshs):

        vtk.vtkAssembly.__init__(self)
        ActorBase.__init__(self)

        self.actors = meshs

        if len(meshs) and hasattr(meshs[0], "base"):
            self.base = meshs[0].base
            self.top = meshs[0].top
        else:
            self.base = None
            self.top = None

        for a in meshs:
            if a:
                self.AddPart(a)

    def __add__(self, meshs):
        if isinstance(meshs, list):
            for a in meshs:
                self.AddPart(self)
        elif isinstance(meshs, vtk.vtkAssembly):
            acts = meshs.getMeshes()
            for a in acts:
                self.AddPart(a)
        else:  # meshs=one mesh
            self.AddPart(meshs)
        return self

    
    def getActors(self):
        """Obsolete, use getMeshes() instead."""
        print("WARNING: getActors() is obsolete, use getMeshes() instead.")
        return self.getMeshes()

        
    def getMeshes(self):
        """Unpack the list of ``Mesh`` objects from a ``Assembly``."""
        return self.actors

    def getMesh(self, i):
        """Get `i-th` ``Mesh`` object from a ``Assembly``."""
        return self.actors[i]

    def diagonalSize(self):
        """Return the maximum diagonal size of the ``Mesh`` objects in ``Assembly``."""
        szs = [a.diagonalSize() for a in self.actors]
        return np.max(szs)

    def lighting(self, style='', ambient=None, diffuse=None,
                 specular=None, specularPower=None, specularColor=None, enabled=True):
        """Set the lighting type to all ``Mesh`` in the ``Assembly`` object.

        :param str style: preset style, can be `[metallic, plastic, shiny, glossy]`
        :param float ambient: ambient fraction of emission [0-1]
        :param float diffuse: emission of diffused light in fraction [0-1]
        :param float specular: fraction of reflected light [0-1]
        :param float specularPower: precision of reflection [1-100]
        :param color specularColor: color that is being reflected by the surface
        :param bool enabled: enable/disable all surface light emission
        """
        for a in self.actors:
            a.lighting(style, ambient, diffuse,
                       specular, specularPower, specularColor, enabled)
        return self

