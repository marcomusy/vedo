from vedo import *

class SubMesh:
    """
    Cut out a submesh and glue it back, possibly with updated vertices, to the original mesh.
    The number and ordering of vertices in the resultant mesh are preserved.
    Class properties:
    * original_mesh
    * submesh: sub-mesh cut from original_mesh
    * mesh: the resultant mesh with submesh glued back
    * old_pids: indices of the submesh vertices which are also the mesh vertices
    * new_pids: indices of the new vertices added to submesh along the cut lines
    * cut: the pointcloud of the new vertices
    * dist2cut: distances from the original vertices in the submesh (old_pids) to the cut
    """
    def __init__(self, msh: Mesh, cut_fn_name: str, **kwargs):
        """
        :param msh: a Mesh
        :param cut_fn_name: Mesh method name to cut the Mesh
        :param kwargs: keyworded arguments to the cut meshod
        """
        self.original_mesh = msh
        self.mesh = msh.clone()
        self.mesh.pointdata['pids'] = np.arange(self.mesh.nvertices)
        self.submesh = getattr(self.mesh.clone(), cut_fn_name)(**kwargs)
        verts = Points(self.mesh.vertices)
        self.old_pids = []
        self.new_pids = []
        for i, v in enumerate(self.submesh.vertices):
            if Point(v).distance_to(verts) < 1e-3:
                self.old_pids.append(i)
            else:
                self.new_pids.append(i)
        self.cut = Points(self.submesh.vertices[self.new_pids])
        self.dist2cut = dict()

    def glue_(self, radius, align):
        """
        Glue submesh with possibly modified vertex positions back to the original mesh.
        :param radius: smoothing radius. The vertices of submesh which were originally
        at the distance smaller than radius, are interpolated between the original and new positions proportionally
        to the distance
        :param align: align the cut of submesh to the cut of the original mesh before gluing
        :return: mesh with submesh glued back
        """
        sm = self.submesh.clone()
        if align:
            sm.align_with_landmarks(self.submesh.vertices[self.new_pids], self.cut.vertices, rigid=True)
        if radius > 0:
            if len(self.dist2cut) == 0:  # pre-compute the distances for interactive gluing
                for i in self.old_pids:
                    pos = self.original_mesh.vertices[self.submesh.pointdata['pids'][i]]
                    self.dist2cut[i] = Point(pos).distance_to(self.cut).item()
            for i in self.old_pids:
                d = min(self.dist2cut[i] / radius, 1.)
                self.mesh.vertices[self.submesh.pointdata['pids'][i]] = (
                        d * sm.vertices[i] + (1-d) * self.original_mesh.vertices[self.submesh.pointdata['pids'][i]])
        else:
            for i in self.old_pids:
                self.mesh.vertices[self.submesh.pointdata['pids'][i]] = sm.vertices[i]

        self.mesh.pointdata.remove('pids')

    def glue(self, radius: float=0, mesh_col="wheat", align=False, interactive=False):
        """
        Glue submesh with possibly modified vertex positions back to the original mesh.
        :param radius: smoothing radius. The vertices of submesh which were originally
        at the distance smaller than radius, are interpolated between the original and new positions proportionally
        to the distance
        :param mesh_col: colour of the mesh in the plot
        :param align: align the cut of submesh to the cut of the original mesh before gluing
        :param interactive: open an interactive plot to adjust the smoothing radius
        :return: mesh with submesh glued back
        """
        self.glue_(radius=radius, align=align)
        if not interactive:
            return
        else:
            if len(self.dist2cut) == 0:  # pre-compute the distances for interactive gluing
                for i in self.old_pids:
                    pos = self.original_mesh.vertices[self.submesh.pointdata['pids'][i]]
                    self.dist2cut[i] = Point(pos).distance_to(self.cut).item()

            self.plt = Plotter()
            self.plt += self.mesh.c(mesh_col)

            def stitch(widget, event):
                self.glue_(radius=widget.value**2, align=align)
                self.plt -= self.mesh
                self.plt += self.mesh.c(mesh_col)

            self.plt.add_slider(
                stitch,
                value=radius,
                xmin=0,
                xmax=np.array(list(self.dist2cut.values())).max()**0.5 *2,
                pos="bottom",
                title="Smoothing radius",
            )
            self.plt.show(interactive=True).close()


S = Sphere(r=1, res=50).lw(1).flat()
box = Cube(side=1.5).wireframe()
cups = SubMesh(S, 'cut_with_box', bounds=box, invert=True)
cups.submesh.scale(1.2)  # alter the submesh
cups.glue(radius=0.2, mesh_col="coral", interactive=True)


man = Mesh(dataurl+"man.vtk").rotate_x(-90)
man.color('w').lw(1).flat()
cut_height = 1.20
head = SubMesh(man, 'cut_with_plane', origin=(0, cut_height, 0), normal=(0, 1, 0))
# modify the head:
head.submesh.scale(1.2, origin=(0,cut_height,0)).shift((0, 0.05, 0))
head.glue(interactive=True)