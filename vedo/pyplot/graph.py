#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Graph plotting utilities."""

from typing_extensions import Self
import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import settings
from vedo.core.transformations import cart2spher, spher2cart
from vedo import addons
from vedo import colors
from vedo import utils
from vedo import shapes
from vedo.pointcloud import Points, merge
from vedo.mesh import Mesh
from vedo.assembly import Assembly

class DirectedGraph(Assembly):
    """
    Support for Directed Graphs.
    """

    def __init__(self, **kargs):
        """
        A graph consists of a collection of nodes (without postional information)
        and a collection of edges connecting pairs of nodes.
        The task is to determine the node positions only based on their connections.

        This class is derived from class `Assembly`, and it assembles 4 Mesh objects
        representing the graph, the node labels, edge labels and edge arrows.

        Arguments:
            c : (color)
                Color of the Graph
            n : (int)
                number of the initial set of nodes
            layout : (int, str)
                layout in
                `['2d', 'fast2d', 'clustering2d', 'circular', 'circular3d', 'cone', 'force', 'tree']`.
                Each of these layouts has different available options.

        ---------------------------------------------------------------
        .. note:: Options for layouts '2d', 'fast2d' and 'clustering2d'

        Arguments:
            seed : (int)
                seed of the random number generator used to jitter point positions
            rest_distance : (float)
                manually set the resting distance
            nmax : (int)
                the maximum number of iterations to be used
            zrange : (list)
                expand 2d graph along z axis.

        ---------------------------------------------------------------
        .. note:: Options for layouts 'circular', and 'circular3d':

        Arguments:
            radius : (float)
                set the radius of the circles
            height : (float)
                set the vertical (local z) distance between the circles
            zrange : (float)
                expand 2d graph along z axis

        ---------------------------------------------------------------
        .. note:: Options for layout 'cone'

        Arguments:
            compactness : (float)
                ratio between the average width of a cone in the tree,
                and the height of the cone.
            compression : (bool)
                put children closer together, possibly allowing sub-trees to overlap.
                This is useful if the tree is actually the spanning tree of a graph.
            spacing : (float)
                space between layers of the tree

        ---------------------------------------------------------------
        .. note:: Options for layout 'force'

        Arguments:
            seed : (int)
                seed the random number generator used to jitter point positions
            bounds : (list)
                set the region in space in which to place the final graph
            nmax : (int)
                the maximum number of iterations to be used
            three_dimensional : (bool)
                allow optimization in the 3rd dimension too
            random_initial_points : (bool)
                use random positions within the graph bounds as initial points

        Examples:
            - [lineage_graph.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/graph_lineage.py)

                ![](https://vedo.embl.es/images/pyplot/graph_lineage.png)

            - [graph_network.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/graph_network.py)

                ![](https://vedo.embl.es/images/pyplot/graph_network.png)
        """

        super().__init__()

        self.nodes = []
        self.edges = []

        self._node_labels = []  # holds strings
        self._edge_labels = []
        self.edge_orientations = []
        self.edge_glyph_position = 0.6

        self.zrange = 0.0

        self.rotX = 0
        self.rotY = 0
        self.rotZ = 0

        self.arrow_scale = 0.15
        self.node_label_scale = None
        self.node_label_justify = "bottom-left"

        self.edge_label_scale = None

        self.mdg = vtki.new("MutableDirectedGraph")

        n = kargs.pop("n", 0)
        for _ in range(n):
            self.add_node()

        self._c = kargs.pop("c", (0.3, 0.3, 0.3))

        self.gl = vtki.new("GraphLayout")

        self.font = kargs.pop("font", "")

        s = kargs.pop("layout", "2d")
        if isinstance(s, int):
            ss = ["2d", "fast2d", "clustering2d", "circular", "circular3d", "cone", "force", "tree"]
            s = ss[s]
        self.layout = s

        if "2d" in s:
            if "clustering" in s:
                self.strategy = vtki.new("Clustering2DLayoutStrategy")
            elif "fast" in s:
                self.strategy = vtki.new("Fast2DLayoutStrategy")
            else:
                self.strategy = vtki.new("Simple2DLayoutStrategy")
            self.rotX = 180
            opt = kargs.pop("rest_distance", None)
            if opt is not None:
                self.strategy.SetRestDistance(opt)
            opt = kargs.pop("seed", None)
            if opt is not None:
                self.strategy.SetRandomSeed(opt)
            opt = kargs.pop("nmax", None)
            if opt is not None:
                self.strategy.SetMaxNumberOfIterations(opt)
            self.zrange = kargs.pop("zrange", 0)

        elif "circ" in s:
            if "3d" in s:
                self.strategy = vtki.new("Simple3DCirclesStrategy")
                self.strategy.SetDirection(0, 0, -1)
                self.strategy.SetAutoHeight(True)
                self.strategy.SetMethod(1)
                self.rotX = -90
                opt = kargs.pop("radius", None)  # float
                if opt is not None:
                    self.strategy.SetMethod(0)
                    self.strategy.SetRadius(opt)  # float
                opt = kargs.pop("height", None)
                if opt is not None:
                    self.strategy.SetAutoHeight(False)
                    self.strategy.SetHeight(opt)  # float
            else:
                self.strategy = vtki.new("CircularLayoutStrategy")
                self.zrange = kargs.pop("zrange", 0)

        elif "cone" in s:
            self.strategy = vtki.new("ConeLayoutStrategy")
            self.rotX = 180
            opt = kargs.pop("compactness", None)
            if opt is not None:
                self.strategy.SetCompactness(opt)
            opt = kargs.pop("compression", None)
            if opt is not None:
                self.strategy.SetCompression(opt)
            opt = kargs.pop("spacing", None)
            if opt is not None:
                self.strategy.SetSpacing(opt)

        elif "force" in s:
            self.strategy = vtki.new("ForceDirectedLayoutStrategy")
            opt = kargs.pop("seed", None)
            if opt is not None:
                self.strategy.SetRandomSeed(opt)
            opt = kargs.pop("bounds", None)
            if opt is not None:
                self.strategy.SetAutomaticBoundsComputation(False)
                self.strategy.SetGraphBounds(opt)  # list
            opt = kargs.pop("nmax", None)
            if opt is not None:
                self.strategy.SetMaxNumberOfIterations(opt)  # int
            opt = kargs.pop("three_dimensional", True)
            if opt is not None:
                self.strategy.SetThreeDimensionalLayout(opt)  # bool
            opt = kargs.pop("random_initial_points", None)
            if opt is not None:
                self.strategy.SetRandomInitialPoints(opt)  # bool

        elif "tree" in s:
            self.strategy = vtki.new("SpanTreeLayoutStrategy")
            self.rotX = 180

        else:
            vedo.logger.error(f"Cannot understand layout {s}. Available layouts:")
            vedo.logger.error("[2d,fast2d,clustering2d,circular,circular3d,cone,force,tree]")
            raise RuntimeError()

        self.gl.SetLayoutStrategy(self.strategy)

        if len(kargs) > 0:
            vedo.logger.error(f"Cannot understand options: {kargs}")

    def add_node(self, label="id") -> int:
        """Add a new node to the `Graph`."""
        v = self.mdg.AddVertex()  # vtk calls it vertex..
        self.nodes.append(v)
        if label == "id":
            label = int(v)
        self._node_labels.append(str(label))
        return v

    def add_edge(self, v1, v2, label="") -> int:
        """Add a new edge between to nodes.
        An extra node is created automatically if needed."""
        nv = len(self.nodes)
        if v1 >= nv:
            for _ in range(nv, v1 + 1):
                self.add_node()
        nv = len(self.nodes)
        if v2 >= nv:
            for _ in range(nv, v2 + 1):
                self.add_node()
        e = self.mdg.AddEdge(v1, v2)
        self.edges.append(e)
        self._edge_labels.append(str(label))
        return e

    def add_child(self, v, node_label="id", edge_label="") -> int:
        """Add a new edge to a new node as its child.
        The extra node is created automatically if needed."""
        nv = len(self.nodes)
        if v >= nv:
            for _ in range(nv, v + 1):
                self.add_node()
        child = self.mdg.AddChild(v)
        self.edges.append((v, child))
        self.nodes.append(child)
        if node_label == "id":
            node_label = int(child)
        self._node_labels.append(str(node_label))
        self._edge_labels.append(str(edge_label))
        return child

    def build(self):
        """
        Build the `DirectedGraph(Assembly)`.
        Accessory objects are also created for labels and arrows.
        """
        self.gl.SetZRange(self.zrange)
        self.gl.SetInputData(self.mdg)
        self.gl.Update()

        gr2poly = vtki.new("GraphToPolyData")
        gr2poly.EdgeGlyphOutputOn()
        gr2poly.SetEdgeGlyphPosition(self.edge_glyph_position)
        gr2poly.SetInputData(self.gl.GetOutput())
        gr2poly.Update()

        dgraph = Mesh(gr2poly.GetOutput(0))
        # dgraph.clean() # WRONG!!! dont uncomment
        dgraph.flat().color(self._c).lw(2)
        dgraph.name = "DirectedGraph"

        diagsz = self.diagonal_size() / 1.42
        if not diagsz:
            return None

        dgraph.scale(1 / diagsz)
        if self.rotX:
            dgraph.rotate_x(self.rotX)
        if self.rotY:
            dgraph.rotate_y(self.rotY)
        if self.rotZ:
            dgraph.rotate_z(self.rotZ)

        vecs = gr2poly.GetOutput(1).GetPointData().GetVectors()
        self.edge_orientations = utils.vtk2numpy(vecs)

        # Use Glyph3D to repeat the glyph on all edges.
        arrows = None
        if self.arrow_scale:
            arrow_source = vtki.new("GlyphSource2D")
            arrow_source.SetGlyphTypeToEdgeArrow()
            arrow_source.SetScale(self.arrow_scale)
            arrow_source.Update()
            arrow_glyph = vtki.vtkGlyph3D()
            arrow_glyph.SetInputData(0, gr2poly.GetOutput(1))
            arrow_glyph.SetInputData(1, arrow_source.GetOutput())
            arrow_glyph.Update()
            arrows = Mesh(arrow_glyph.GetOutput())
            arrows.scale(1 / diagsz)
            arrows.lighting("off").color(self._c)
            if self.rotX:
                arrows.rotate_x(self.rotX)
            if self.rotY:
                arrows.rotate_y(self.rotY)
            if self.rotZ:
                arrows.rotate_z(self.rotZ)
            arrows.name = "DirectedGraphArrows"

        node_labels = None
        if self._node_labels:
            node_labels = dgraph.labels(
                self._node_labels,
                scale=self.node_label_scale,
                precision=0,
                font=self.font,
                justify=self.node_label_justify,
            )
            node_labels.color(self._c).pickable(True)
            node_labels.name = "DirectedGraphNodeLabels"

        edge_labels = None
        if self._edge_labels:
            edge_labels = dgraph.labels(
                self._edge_labels, on="cells", scale=self.edge_label_scale, precision=0, font=self.font
            )
            edge_labels.color(self._c).pickable(True)
            edge_labels.name = "DirectedGraphEdgeLabels"

        super().__init__([dgraph, node_labels, edge_labels, arrows])
        self.name = "DirectedGraphAssembly"
        return self

__all__ = ["DirectedGraph"]
