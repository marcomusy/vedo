#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import vedo.vtkclasses as vtki # a wrapper for lazy imports

__docformat__ = "google"

__doc__ = """
Submodule to work with linear and non-linear transformations<br>

![](https://vedo.embl.es/images/feats/transforms.png)
"""

__all__ = [
    "LinearTransform",
    "NonLinearTransform",
    "TransformInterpolator",
    "spher2cart",
    "cart2spher",
    "cart2cyl",
    "cyl2cart",
    "cyl2spher",
    "spher2cyl",
    "cart2pol",
    "pol2cart",
]

###################################################
def _is_sequence(arg):
    if hasattr(arg, "strip"):
        return False
    if hasattr(arg, "__getslice__"):
        return True
    if hasattr(arg, "__iter__"):
        return True
    return False


###################################################
class LinearTransform:
    """Work with linear transformations."""

    def __init__(self, T=None):
        """
        Define a linear transformation.
        Can be saved to file and reloaded.

        Arguments:
            T : (vtkTransform, numpy array)
                input transformation. Defaults to unit.

        Example:
            ```python
            from vedo import *
            settings.use_parallel_projection = True

            LT = LinearTransform()
            LT.translate([3,0,1]).rotate_z(45)
            LT.comment = "shifting by (3,0,1) and rotating by 45 deg"
            print(LT)

            sph = Sphere(r=0.2)
            sph.apply_transform(LT) # same as: LT.move(s1)
            print(sph.transform)

            show(Point([0,0,0]), sph, str(LT.matrix), axes=1).close()
            ```
        """
        self.name = "LinearTransform"
        self.filename = ""
        self.comment = ""

        if T is None:
            T = vtki.vtkTransform()

        elif isinstance(T, vtki.vtkMatrix4x4):
            S = vtki.vtkTransform()
            S.SetMatrix(T)
            T = S

        elif isinstance(T, vtki.vtkLandmarkTransform):
            S = vtki.vtkTransform()
            S.SetMatrix(T.GetMatrix())
            T = S

        elif _is_sequence(T):
            S = vtki.vtkTransform()
            M = vtki.vtkMatrix4x4()
            n = len(T)
            for i in range(n):
                for j in range(n):
                    M.SetElement(i, j, T[i][j])
            S.SetMatrix(M)
            T = S

        elif isinstance(T, vtki.vtkLinearTransform):
            S = vtki.vtkTransform()
            S.DeepCopy(T)
            T = S

        elif isinstance(T, LinearTransform):
            S = vtki.vtkTransform()
            S.DeepCopy(T.T)
            T = S

        elif isinstance(T, str):
            import json
            self.filename = str(T)
            try:
                with open(self.filename, "r") as read_file:
                    D = json.load(read_file)
                self.name = D["name"]
                self.comment = D["comment"]
                matrix = np.array(D["matrix"])
            except json.decoder.JSONDecodeError:
                ### assuming legacy vedo format E.g.:
                # aligned by manual_align.py
                # 0.8026854838223 -0.0789823873914 -0.508476844097  38.17377632072
                # 0.0679734082661  0.9501827489452 -0.040289803376 -69.53864247951
                # 0.5100652300642 -0.0023313569781  0.805555043665 -81.20317788519
                # 0.0 0.0 0.0 1.0
                with open(self.filename, "r", encoding="UTF-8") as read_file:
                    lines = read_file.readlines()
                    i = 0
                    matrix = np.eye(4)
                    for l in lines:
                        if l.startswith("#"):
                            self.comment = l.replace("#", "").strip()
                            continue
                        vals = l.split(" ")
                        for j in range(len(vals)):
                            v = vals[j].replace("\n", "")
                            if v != "":
                                matrix[i, j] = float(v)
                        i += 1
            T = vtki.vtkTransform()
            m = vtki.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    m.SetElement(i, j, matrix[i][j])
            T.SetMatrix(m)

        self.T = T
        self.T.PostMultiply()
        self.inverse_flag = False

    def __str__(self):
        module = self.__class__.__module__
        name = self.__class__.__name__
        s = f"\x1b[7m\x1b[1m{module}.{name} at ({hex(id(self))})".ljust(75) + "\x1b[0m"
        s += "\nname".ljust(15) + ": " + self.name
        if self.filename:
            s += "\nfilename".ljust(15) + ": " + self.filename
        if self.comment:
            s += "\ncomment".ljust(15) + f': \x1b[3m"{self.comment}"\x1b[0m'
        s += f"\nconcatenations".ljust(15) + f": {self.ntransforms}"
        s += "\ninverse flag".ljust(15) + f": {bool(self.inverse_flag)}"
        arr = np.array2string(self.matrix,
            separator=', ', precision=6, suppress_small=True)
        s += "\nmatrix 4x4".ljust(15) + f":\n{arr}"
        return s

    def __repr__(self):
        return self.__str__()

    def print(self):
        """Print transformation."""
        print(self.__str__())
        return self

    def __call__(self, obj):
        """
        Apply transformation to object or single point.
        Same as `move()` except that a copy is returned.
        """
        return self.move(obj.copy())
    
    def transform_point(self, p):
        """
        Apply transformation to a single point.
        """
        if len(p) == 2:
            p = [p[0], p[1], 0]
        return np.array(self.T.TransformFloatPoint(p))

    def move(self, obj):
        """
        Apply transformation to object or single point.

        Note:
            When applying a transformation to a mesh, the mesh is modified in place.
            If you want to keep the original mesh unchanged, use `clone()` method.

        Example:
            ```python
            from vedo import *
            settings.use_parallel_projection = True

            LT = LinearTransform()
            LT.translate([3,0,1]).rotate_z(45)
            print(LT)

            s = Sphere(r=0.2)
            LT.move(s)
            # same as:
            # s.apply_transform(LT)

            zero = Point([0,0,0])
            show(s, zero, axes=1).close()
            ```
        """
        if _is_sequence(obj):
            n = len(obj)
            if n == 2:
                obj = [obj[0], obj[1], 0]
            return np.array(self.T.TransformFloatPoint(obj))

        obj.apply_transform(self)
        return obj

    def reset(self):
        """Reset transformation."""
        self.T.Identity()
        return self
    
    def compute_main_axes(self):
        """
        Compute main axes of the transformation matrix.
        These are the axes of the ellipsoid that is the 
        image of the unit sphere under the transformation.

        Example:
        ```python
        from vedo import *
        settings.use_parallel_projection = True

        M = np.random.rand(3,3)-0.5
        print(M)
        print(" M@[1,0,0] =", M@[1,1,0])

        ######################
        A = LinearTransform(M)
        print(A)
        pt = Point([1,1,0])
        print(A(pt).vertices[0], "is the same as", A([1,1,0]))

        maxes = A.compute_main_axes()

        arr1 = Arrow([0,0,0], maxes[0]).c('r')
        arr2 = Arrow([0,0,0], maxes[1]).c('g')
        arr3 = Arrow([0,0,0], maxes[2]).c('b')

        sphere1 = Sphere().wireframe().lighting('off')
        sphere1.cmap('hot', sphere1.vertices[:,2])

        sphere2 = sphere1.clone().apply_transform(A)

        show([sphere1, [sphere2, arr1, arr2, arr3]], N=2, axes=1, bg='bb')
        ```
        """
        m = self.matrix3x3
        eigval, eigvec = np.linalg.eig(m @ m.T)
        eigval = np.sqrt(eigval)
        return  np.array([
            eigvec[:,0] * eigval[0],
            eigvec[:,1] * eigval[1],
            eigvec[:,2] * eigval[2],
        ])

    def pop(self):
        """Delete the transformation on the top of the stack
        and sets the top to the next transformation on the stack."""
        self.T.Pop()
        return self

    def is_identity(self):
        """Check if the transformation is the identity."""
        m = self.T.GetMatrix()
        M = [[m.GetElement(i, j) for j in range(4)] for i in range(4)]
        if np.allclose(M - np.eye(4), 0):
            return True
        return False

    def invert(self):
        """Invert the transformation. Acts in-place."""
        self.T.Inverse()
        self.inverse_flag = bool(self.T.GetInverseFlag())
        return self

    def compute_inverse(self):
        """Compute the inverse."""
        t = self.clone()
        t.invert()
        return t

    def transpose(self):
        """Transpose the transformation. Acts in-place."""
        M = vtki.vtkMatrix4x4()
        self.T.GetTranspose(M)
        self.T.SetMatrix(M)
        return self

    def copy(self):
        """Return a copy of the transformation. Alias of `clone()`."""
        return self.clone()

    def clone(self):
        """Clone transformation to make an exact copy."""
        return LinearTransform(self.T)

    def concatenate(self, T, pre_multiply=False):
        """
        Post-multiply (by default) 2 transfomations.
        T can also be a 4x4 matrix or 3x3 matrix.

        Example:
            ```python
            from vedo import LinearTransform

            A = LinearTransform()
            A.rotate_x(45)
            A.translate([7,8,9])
            A.translate([10,10,10])
            A.name = "My transformation A"
            print(A)

            B = A.compute_inverse()
            B.shift([1,2,3])
            B.name = "My transformation B (shifted inverse of A)"
            print(B)

            # A is applied first, then B
            # print("A.concatenate(B)", A.concatenate(B))

            # B is applied first, then A
            print(B*A)
            ```
        """
        if _is_sequence(T):
            S = vtki.vtkTransform()
            M = vtki.vtkMatrix4x4()
            n = len(T)
            for i in range(n):
                for j in range(n):
                    M.SetElement(i, j, T[i][j])
            S.SetMatrix(M)
            T = S

        if pre_multiply:
            self.T.PreMultiply()
        try:
            self.T.Concatenate(T)
        except:
            self.T.Concatenate(T.T)
        self.T.PostMultiply()
        return self

    def __mul__(self, A):
        """Pre-multiply 2 transfomations."""
        return self.concatenate(A, pre_multiply=True)

    def get_concatenated_transform(self, i):
        """Get intermediate matrix by concatenation index."""
        return LinearTransform(self.T.GetConcatenatedTransform(i))

    @property
    def ntransforms(self):
        """Get the number of concatenated transforms."""
        return self.T.GetNumberOfConcatenatedTransforms()

    def translate(self, p):
        """Translate, same as `shift`."""
        if len(p) == 2:
            p = [p[0], p[1], 0]
        self.T.Translate(p)
        return self

    def shift(self, p):
        """Shift, same as `translate`."""
        return self.translate(p)

    def scale(self, s, origin=True):
        """Scale."""
        if not _is_sequence(s):
            s = [s, s, s]

        if origin is True:
            p = np.array(self.T.GetPosition())
            if np.linalg.norm(p) > 0:
                self.T.Translate(-p)
                self.T.Scale(*s)
                self.T.Translate(p)
            else:
                self.T.Scale(*s)

        elif _is_sequence(origin):
            origin = np.asarray(origin)
            self.T.Translate(-origin)
            self.T.Scale(*s)
            self.T.Translate(origin)

        else:
            self.T.Scale(*s)
        return self

    def rotate(self, angle, axis=(1, 0, 0), point=(0, 0, 0), rad=False):
        """
        Rotate around an arbitrary `axis` passing through `point`.

        Example:
            ```python
            from vedo import *
            c1 = Cube()
            c2 = c1.clone().c('violet').alpha(0.5) # copy of c1
            v = vector(0.2, 1, 0)
            p = vector(1.0, 0, 0)  # axis passes through this point
            c2.rotate(90, axis=v, point=p)
            l = Line(p-v, p+v).c('red5').lw(3)
            show(c1, l, c2, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/rotate_axis.png)
        """
        if not angle:
            return self
        if rad:
            anglerad = angle
        else:
            anglerad = np.deg2rad(angle)
        axis = np.asarray(axis) / np.linalg.norm(axis)
        a = np.cos(anglerad / 2)
        b, c, d = -axis * np.sin(anglerad / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        R = np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )
        rv = np.dot(R, self.T.GetPosition() - np.asarray(point)) + point

        if rad:
            angle *= 180.0 / np.pi
        # this vtk method only rotates in the origin of the object:
        self.T.RotateWXYZ(angle, axis[0], axis[1], axis[2])
        self.T.Translate(rv - np.array(self.T.GetPosition()))
        return self

    def _rotatexyz(self, axe, angle, rad, around):
        if not angle:
            return self
        if rad:
            angle *= 180 / np.pi

        rot = dict(x=self.T.RotateX, y=self.T.RotateY, z=self.T.RotateZ)

        if around is None:
            # rotate around its origin
            rot[axe](angle)
        else:
            # displacement needed to bring it back to the origin
            self.T.Translate(-np.asarray(around))
            rot[axe](angle)
            self.T.Translate(around)
        return self

    def rotate_x(self, angle, rad=False, around=None):
        """
        Rotate around x-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        return self._rotatexyz("x", angle, rad, around)

    def rotate_y(self, angle, rad=False, around=None):
        """
        Rotate around y-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        return self._rotatexyz("y", angle, rad, around)

    def rotate_z(self, angle, rad=False, around=None):
        """
        Rotate around z-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        return self._rotatexyz("z", angle, rad, around)

    def set_position(self, p):
        """Set position."""
        if len(p) == 2:
            p = np.array([p[0], p[1], 0])
        q = np.array(self.T.GetPosition())
        self.T.Translate(p - q)
        return self

    # def set_scale(self, s):
    #     """Set absolute scale."""
    #     if not _is_sequence(s):
    #         s = [s, s, s]
    #     s0, s1, s2 = 1, 1, 1
    #     b = self.T.GetScale()
    #     print(b)
    #     if b[0]:
    #         s0 = s[0] / b[0]
    #     if b[1]:
    #         s1 = s[1] / b[1]
    #     if b[2]:
    #         s2 = s[2] / b[2]
    #     self.T.Scale(s0, s1, s2)
    #     print()
    #     return self

    def get_scale(self):
        """Get current scale."""
        return np.array(self.T.GetScale())

    @property
    def orientation(self):
        """Compute orientation."""
        return np.array(self.T.GetOrientation())

    @property
    def position(self):
        """Compute position."""
        return np.array(self.T.GetPosition())

    @property
    def matrix(self):
        """Get the 4x4 trasformation matrix."""
        m = self.T.GetMatrix()
        M = [[m.GetElement(i, j) for j in range(4)] for i in range(4)]
        return np.array(M)

    @matrix.setter
    def matrix(self, M):
        """Set trasformation by assigning a 4x4 or 3x3 numpy matrix."""
        m = vtki.vtkMatrix4x4()
        n = len(M)
        for i in range(n):
            for j in range(n):
                m.SetElement(i, j, M[i][j])
        self.T.SetMatrix(m)

    @property
    def matrix3x3(self):
        """Get the 3x3 trasformation matrix."""
        m = self.T.GetMatrix()
        M = [[m.GetElement(i, j) for j in range(3)] for i in range(3)]
        return np.array(M)

    def write(self, filename="transform.mat"):
        """Save transformation to ASCII file."""
        import json
        m = self.T.GetMatrix()
        M = [[m.GetElement(i, j) for j in range(4)] for i in range(4)]
        arr = np.array(M)
        dictionary = {
            "name": self.name,
            "comment": self.comment,
            "matrix": arr.astype(float).tolist(),
            "ntransforms": self.ntransforms,
        }
        with open(filename, "w") as outfile:
            json.dump(dictionary, outfile, sort_keys=True, indent=2)

    def reorient(
        self, initaxis, newaxis, around=(0, 0, 0), rotation=0, rad=False, xyplane=True
    ):
        """
        Set/Get object orientation.

        Arguments:
            rotation : (float)
                rotate object around newaxis.
            concatenate : (bool)
                concatenate the orientation operation with the previous existing transform (if any)
            rad : (bool)
                set to True if angle is expressed in radians.
            xyplane : (bool)
                make an extra rotation to keep the object aligned to the xy-plane
        """
        newaxis = np.asarray(newaxis) / np.linalg.norm(newaxis)
        initaxis = np.asarray(initaxis) / np.linalg.norm(initaxis)

        if not np.any(initaxis - newaxis):
            return self

        if not np.any(initaxis + newaxis):
            print("Warning: in reorient() initaxis and newaxis are parallel")
            newaxis += np.array([0.0000001, 0.0000002, 0])
            angleth = np.pi
        else:
            angleth = np.arccos(np.dot(initaxis, newaxis))
        crossvec = np.cross(initaxis, newaxis)

        p = np.asarray(around)
        self.T.Translate(-p)
        if rotation:
            if rad:
                rotation = np.rad2deg(rotation)
            self.T.RotateWXYZ(rotation, initaxis)

        self.T.RotateWXYZ(np.rad2deg(angleth), crossvec)

        if xyplane:
            self.T.RotateWXYZ(-self.orientation[0] * 1.4142, newaxis)

        self.T.Translate(p)
        return self


###################################################
class NonLinearTransform:
    """Work with non-linear transformations."""

    def __init__(self, T=None, **kwargs):
        """
        Define a non-linear transformation.
        Can be saved to file and reloaded.

        Arguments:
            T : (vtkThinPlateSplineTransform, str, dict)
                vtk transformation.
                If T is a string, it is assumed to be a filename.
                If T is a dictionary, it is assumed to be a set of keyword arguments.
                Defaults to None.
            **kwargs : (dict)
                keyword arguments to define the transformation.
                The following keywords are accepted:
                - name : (str) name of the transformation
                - comment : (str) comment
                - source_points : (list) source points
                - target_points : (list) target points
                - mode : (str) either '2d' or '3d'
                - sigma : (float) sigma parameter

        Example:
            ```python
            from vedo import *
            settings.use_parallel_projection = True

            NLT = NonLinearTransform()
            NLT.source_points = [[-2,0,0], [1,2,1], [2,-2,2]]
            NLT.target_points = NLT.source_points + np.random.randn(3,3)*0.5
            NLT.mode = '3d'
            print(NLT)

            s1 = Sphere()
            NLT.move(s1)
            # same as:
            # s1.apply_transform(NLT)

            arrs = Arrows(NLT.source_points, NLT.target_points)
            show(s1, arrs, Sphere().alpha(0.1), axes=1).close()
            ```
        """

        self.name = "NonLinearTransform"
        self.filename = ""
        self.comment = ""

        if T is None and len(kwargs) == 0:
            T = vtki.vtkThinPlateSplineTransform()

        elif isinstance(T, vtki.vtkThinPlateSplineTransform):
            S = vtki.vtkThinPlateSplineTransform()
            S.DeepCopy(T)
            T = S

        elif isinstance(T, NonLinearTransform):
            S = vtki.vtkThinPlateSplineTransform()
            S.DeepCopy(T.T)
            T = S

        elif isinstance(T, str):
            import json
            filename = str(T)
            self.filename = filename
            with open(filename, "r") as read_file:
                D = json.load(read_file)
            self.name = D["name"]
            self.comment = D["comment"]
            source = D["source_points"]
            target = D["target_points"]
            mode = D["mode"]
            sigma = D["sigma"]

            T = vtki.vtkThinPlateSplineTransform()
            vptss = vtki.vtkPoints()
            for p in source:
                if len(p) == 2:
                    p = [p[0], p[1], 0.0]
                vptss.InsertNextPoint(p)
            T.SetSourceLandmarks(vptss)
            vptst = vtki.vtkPoints()
            for p in target:
                if len(p) == 2:
                    p = [p[0], p[1], 0.0]
                vptst.InsertNextPoint(p)
            T.SetTargetLandmarks(vptst)
            T.SetSigma(sigma)
            if mode == "2d":
                T.SetBasisToR2LogR()
            elif mode == "3d":
                T.SetBasisToR()
            else:
                print(f'In {filename} mode can be either "2d" or "3d"')

        elif len(kwargs) > 0:
            T = kwargs.copy()
            self.name = T.pop("name", "NonLinearTransform")
            self.comment = T.pop("comment", "")
            source = T.pop("source_points", [])
            target = T.pop("target_points", [])
            mode = T.pop("mode", "3d")
            sigma = T.pop("sigma", 1.0)
            if len(T) > 0:
                print("Warning: NonLinearTransform got unexpected keyword arguments:")
                print(T)

            T = vtki.vtkThinPlateSplineTransform()
            vptss = vtki.vtkPoints()
            for p in source:
                if len(p) == 2:
                    p = [p[0], p[1], 0.0]
                vptss.InsertNextPoint(p)
            T.SetSourceLandmarks(vptss)
            vptst = vtki.vtkPoints()
            for p in target:
                if len(p) == 2:
                    p = [p[0], p[1], 0.0]
                vptst.InsertNextPoint(p)
            T.SetTargetLandmarks(vptst)
            T.SetSigma(sigma)
            if mode == "2d":
                T.SetBasisToR2LogR()
            elif mode == "3d":
                T.SetBasisToR()
            else:
                print(f'In {filename} mode can be either "2d" or "3d"')

        self.T = T
        self.inverse_flag = False

    def __str__(self):
        module = self.__class__.__module__
        name = self.__class__.__name__
        s = f"\x1b[7m\x1b[1m{module}.{name} at ({hex(id(self))})".ljust(75) + "\x1b[0m\n"
        s += "name".ljust(9) + ": "  + self.name + "\n"
        if self.filename:
            s += "filename".ljust(9) + ": " + self.filename + "\n"
        if self.comment:
            s += "comment".ljust(9) + f': \x1b[3m"{self.comment}"\x1b[0m\n'
        s += f"mode".ljust(9)  + f": {self.mode}\n"
        s += f"sigma".ljust(9) + f": {self.sigma}\n"
        p = self.source_points
        q = self.target_points
        s += f"sources".ljust(9) + f": {p.size}, bounds {np.min(p, axis=0)}, {np.max(p, axis=0)}\n"
        s += f"targets".ljust(9) + f": {q.size}, bounds {np.min(q, axis=0)}, {np.max(q, axis=0)}"
        return s

    def __repr__(self):
        return self.__str__()

    def print(self):
        """Print transformation."""
        print(self.__str__())
        return self

    def update(self):
        """Update transformation."""
        self.T.Update()
        return self

    @property
    def position(self):
        """
        Trying to get the position of a `NonLinearTransform` always returns [0,0,0].
        """
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # @position.setter
    # def position(self, p):
    #     """
    #     Trying to set position of a `NonLinearTransform`
    #     has no effect and prints a warning.

    #     Use clone() method to create a copy of the object,
    #     or reset it with 'object.transform = vedo.LinearTransform()'
    #     """
    #     print("Warning: NonLinearTransform has no position.")
    #     print("  Use clone() method to create a copy of the object,")
    #     print("  or reset it with 'object.transform = vedo.LinearTransform()'")

    @property
    def source_points(self):
        """Get the source points."""
        pts = self.T.GetSourceLandmarks()
        vpts = []
        if pts:
            for i in range(pts.GetNumberOfPoints()):
                vpts.append(pts.GetPoint(i))
        return np.array(vpts, dtype=np.float32)

    @property
    def target_points(self):
        """Get the target points."""
        pts = self.T.GetTargetLandmarks()
        vpts = []
        for i in range(pts.GetNumberOfPoints()):
            vpts.append(pts.GetPoint(i))
        return np.array(vpts, dtype=np.float32)

    @source_points.setter
    def source_points(self, pts):
        """Set source points."""
        if _is_sequence(pts):
            pass
        else:
            pts = pts.vertices
        vpts = vtki.vtkPoints()
        for p in pts:
            if len(p) == 2:
                p = [p[0], p[1], 0.0]
            vpts.InsertNextPoint(p)
        self.T.SetSourceLandmarks(vpts)

    @target_points.setter
    def target_points(self, pts):
        """Set target points."""
        if _is_sequence(pts):
            pass
        else:
            pts = pts.vertices
        vpts = vtki.vtkPoints()
        for p in pts:
            if len(p) == 2:
                p = [p[0], p[1], 0.0]
            vpts.InsertNextPoint(p)
        self.T.SetTargetLandmarks(vpts)

    @property
    def sigma(self) -> float:
        """Set sigma."""
        return self.T.GetSigma()

    @sigma.setter
    def sigma(self, s):
        """Get sigma."""
        self.T.SetSigma(s)

    @property
    def mode(self) -> str:
        """Get mode."""
        m = self.T.GetBasis()
        # print("T.GetBasis()", m, self.T.GetBasisAsString())
        if m == 2:
            return "2d"
        elif m == 1:
            return "3d"
        else:
            print("Warning: NonLinearTransform has no valid mode.")
            return ""

    @mode.setter
    def mode(self, m):
        """Set mode."""
        if m == "3d":
            self.T.SetBasisToR()
        elif m == "2d":
            self.T.SetBasisToR2LogR()
        else:
            print('In NonLinearTransform mode can be either "2d" or "3d"')

    def clone(self):
        """Clone transformation to make an exact copy."""
        return NonLinearTransform(self.T)

    def write(self, filename):
        """Save transformation to ASCII file."""
        import json

        dictionary = {
            "name": self.name,
            "comment": self.comment,
            "mode": self.mode,
            "sigma": self.sigma,
            "source_points": self.source_points.astype(float).tolist(),
            "target_points": self.target_points.astype(float).tolist(),
        }
        with open(filename, "w") as outfile:
            json.dump(dictionary, outfile, sort_keys=True, indent=2)

    def invert(self):
        """Invert transformation."""
        self.T.Inverse()
        self.inverse_flag = bool(self.T.GetInverseFlag())
        return self

    def compute_inverse(self):
        """Compute inverse."""
        t = self.clone()
        t.invert()
        return t

    def __call__(self, obj):
        """
        Apply transformation to object or single point.
        Same as `move()` except that a copy is returned.
        """
        # use copy here not clone in case user passes a numpy array
        return self.move(obj.copy())

    def compute_main_axes(self, pt=(0,0,0), ds=1):
        """
        Compute main axes of the transformation.
        These are the axes of the ellipsoid that is the 
        image of the unit sphere under the transformation.

        Arguments:
            pt : (list)
                point to compute the axes at.
            ds : (float)
                step size to compute the axes.
        """
        if len(pt) == 2:
            pt = [pt[0], pt[1], 0]
        pt = np.asarray(pt)
        m = np.array([
            self.move(pt + [ds,0,0]),
            self.move(pt + [0,ds,0]),
            self.move(pt + [0,0,ds]),
        ])
        eigval, eigvec = np.linalg.eig(m @ m.T)
        eigval = np.sqrt(eigval)
        return np.array([
            eigvec[:, 0] * eigval[0],
            eigvec[:, 1] * eigval[1],
            eigvec[:, 2] * eigval[2],
        ])

    def transform_point(self, p):
        """
        Apply transformation to a single point.
        """
        if len(p) == 2:
            p = [p[0], p[1], 0]
        return np.array(self.T.TransformFloatPoint(p))

    def move(self, obj):
        """
        Apply transformation to the argument object.

        Note:
            When applying a transformation to a mesh, the mesh is modified in place.
            If you want to keep the original mesh unchanged, use the `clone()` method.

        Example:
            ```python
            from vedo import *
            np.random.seed(0)
            settings.use_parallel_projection = True

            NLT = NonLinearTransform()
            NLT.source_points = [[-2,0,0], [1,2,1], [2,-2,2]]
            NLT.target_points = NLT.source_points + np.random.randn(3,3)*0.5
            NLT.mode = '3d'
            print(NLT)

            s1 = Sphere()
            NLT.move(s1)
            # same as:
            # s1.apply_transform(NLT)

            arrs = Arrows(NLT.source_points, NLT.target_points)
            show(s1, arrs, Sphere().alpha(0.1), axes=1).close()
            ```
        """
        if _is_sequence(obj):
            return self.transform_point(obj)
        obj.apply_transform(self)
        return obj

########################################################################
class TransformInterpolator:
    """
    Interpolate between a set of linear transformations.
    
    Position, scale and orientation (i.e., rotations) are interpolated separately,
    and can be interpolated linearly or with a spline function.
    Note that orientation is interpolated using quaternions via
    SLERP (spherical linear interpolation) or the special `vtkQuaternionSpline` class.

    To use this class, add at least two pairs of (t, transformation) with the add() method.
    Then interpolate the transforms with the `TransformInterpolator(t)` call method,
    where "t" must be in the range of `(min, max)` times specified by the add() method.

    Example:
        ```python
        from vedo import *

        T0 = LinearTransform()
        T1 = LinearTransform().rotate_x(90).shift([12,0,0])

        TRI = TransformInterpolator("linear")
        TRI.add(0, T0)
        TRI.add(1, T1)

        plt = Plotter(axes=1)
        for i in range(11):
            t = i/10
            T = TRI(t)
            plt += Cube().color(i).apply_transform(T)
        plt.show().close()
        ```
        ![](https://vedo.embl.es/images/other/transf_interp.png)
    """
    def __init__(self, mode="linear"):
        """
        Interpolate between two or more linear transformations.
        """
        self.vtk_interpolator = vtki.new("TransformInterpolator")
        self.mode(mode)
        self.TS = []

    def __call__(self, t):
        """
        Get the intermediate transformation at time `t`.
        """
        xform = vtki.vtkTransform()
        self.vtk_interpolator.InterpolateTransform(t, xform)
        return LinearTransform(xform)

    def add(self, t, T):
        """Add intermediate transformations."""
        try:
            # in case a vedo object is passed
            T = T.transform
        except AttributeError:
            pass
        self.TS.append(T)
        self.vtk_interpolator.AddTransform(t, T.T)
        return self

    def remove(self, t):
        """Remove intermediate transformations."""
        self.TS.pop(T)
        self.vtk_interpolator.RemoveTransform(t)
        return self
    
    def trange(self):
        """Get interpolation range."""
        tmin = self.vtk_interpolator.GetMinimumT()
        tmax = self.vtk_interpolator.GetMaximumT()
        return np.array([tmin, tmax])
    
    def clear(self):
        """Clear all intermediate transformations."""
        self.TS = []
        self.vtk_interpolator.Initialize()
        return self
    
    def mode(self, m):
        """Set interpolation mode ('linear' or 'spline')."""
        if m == "linear":
            self.vtk_interpolator.SetInterpolationTypeToLinear()
        elif m == "spline":
            self.vtk_interpolator.SetInterpolationTypeToSpline()
        else:
            print('In TransformInterpolator mode can be either "linear" or "spline"')
        return self
    
    @property
    def ntransforms(self):
        """Get number of transformations."""
        return self.vtk_interpolator.GetNumberOfTransforms()


########################################################################
# 2d ######
def cart2pol(x, y):
    """2D Cartesian to Polar coordinates conversion."""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return np.array([rho, theta])


def pol2cart(rho, theta):
    """2D Polar to Cartesian coordinates conversion."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.array([x, y])


########################################################################
# 3d ######
def cart2spher(x, y, z):
    """3D Cartesian to Spherical coordinate conversion."""
    hxy = np.hypot(x, y)
    rho = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    return np.array([rho, theta, phi])


def spher2cart(rho, theta, phi):
    """3D Spherical to Cartesian coordinate conversion."""
    st = np.sin(theta)
    sp = np.sin(phi)
    ct = np.cos(theta)
    cp = np.cos(phi)
    rst = rho * st
    x = rst * cp
    y = rst * sp
    z = rho * ct
    return np.array([x, y, z])


def cart2cyl(x, y, z):
    """3D Cartesian to Cylindrical coordinate conversion."""
    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return np.array([rho, theta, z])


def cyl2cart(rho, theta, z):
    """3D Cylindrical to Cartesian coordinate conversion."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.array([x, y, z])


def cyl2spher(rho, theta, z):
    """3D Cylindrical to Spherical coordinate conversion."""
    rhos = np.sqrt(rho * rho + z * z)
    phi = np.arctan2(rho, z)
    return np.array([rhos, phi, theta])


def spher2cyl(rho, theta, phi):
    """3D Spherical to Cylindrical coordinate conversion."""
    rhoc = rho * np.sin(theta)
    z = rho * np.cos(theta)
    return np.array([rhoc, phi, z])
