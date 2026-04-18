#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing_extensions import Self
from warnings import warn
import numpy as np
import vedo.vtkclasses as vtki  # a wrapper for lazy imports
from vedo.core.summary import summary_panel, summary_string

__docformat__ = "google"

__doc__ = """
Submodule to work with linear and non-linear transformations<br>

![](https://vedo.embl.es/images/feats/transforms.png)
"""

__all__ = [
    "Quaternion",
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
    # Kept local to avoid a circular import: vedo.utils imports vedo at module
    # level, so importing is_sequence from there during vedo initialisation
    # would produce a partially-initialised module.  The body is identical to
    # vedo.utils.is_sequence — keep them in sync if either changes.
    if hasattr(arg, "strip"):
        return False
    if hasattr(arg, "__getslice__"):
        return True
    if hasattr(arg, "__iter__"):
        return True
    return False


def _is_vtk_quaternion(arg) -> bool:
    """Return ``True`` for VTK quaternion specializations."""
    return all(
        hasattr(arg, name) for name in ("GetW", "GetX", "GetY", "GetZ", "ToMatrix3x3")
    )


###################################################
class LinearTransform:
    """Work with linear transformations."""

    def __init__(self, T=None) -> None:
        """
        Define a linear transformation.
        Can be saved to file and reloaded.

        Args:
            T (str, vtkTransform, numpy array):
                input transformation. Defaults to unit.

        Examples:
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
                    for line in lines:
                        if line.startswith("#"):
                            self.comment = line.replace("#", "").strip()
                            continue
                        vals = line.split(" ")
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

    def _summary_rows(self):
        rows = [("name", self.name)]
        if self.filename:
            rows.append(("filename", self.filename))
        if self.comment:
            rows.append(("comment", self.comment))
        rows.append(("concatenations", str(self.ntransforms)))
        rows.append(("inverse flag", str(bool(self.inverse_flag))))
        rows.append(
            (
                "matrix 4x4",
                np.array2string(
                    self.matrix, separator=", ", precision=6, suppress_small=True
                ),
            )
        )
        return rows

    def __str__(self):
        return summary_string(self, self._summary_rows())

    def __repr__(self):
        return self.__str__()

    def __rich__(self):
        return summary_panel(self, self._summary_rows())

    def print(self) -> LinearTransform:
        """Print transformation."""
        print(self)
        return self

    def __call__(self, obj):
        """
        Apply transformation to object or single point.
        Same as `move()` except that a copy is returned.
        """
        return self.move(obj.copy())

    def transform_point(self, p) -> np.ndarray:
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

        Examples:
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

    def reset(self) -> Self:
        """Reset transformation."""
        self.T.Identity()
        return self

    def compute_main_axes(self) -> np.ndarray:
        """
        Compute main axes of the transformation matrix.
        These are the axes of the ellipsoid that is the
        image of the unit sphere under the transformation.

        Examples:
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
        print(A(pt).coordinates[0], "is the same as", A([1,1,0]))

        maxes = A.compute_main_axes()

        arr1 = Arrow([0,0,0], maxes[0]).c('r')
        arr2 = Arrow([0,0,0], maxes[1]).c('g')
        arr3 = Arrow([0,0,0], maxes[2]).c('b')

        sphere1 = Sphere().wireframe().lighting('off')
        sphere1.cmap('hot', sphere1.coordinates[:,2])

        sphere2 = sphere1.clone().apply_transform(A)

        show([sphere1, [sphere2, arr1, arr2, arr3]], N=2, axes=1, bg='bb')
        ```
        """
        m = self.matrix3x3
        eigval, eigvec = np.linalg.eig(m @ m.T)
        eigval = np.sqrt(eigval)
        return np.array(
            [
                eigvec[:, 0] * eigval[0],
                eigvec[:, 1] * eigval[1],
                eigvec[:, 2] * eigval[2],
            ]
        )

    def pop(self) -> Self:
        """Delete the transformation on the top of the stack
        and sets the top to the next transformation on the stack."""
        self.T.Pop()
        return self

    def is_identity(self) -> bool:
        """Check if the transformation is the identity."""
        m = self.T.GetMatrix()
        M = [[m.GetElement(i, j) for j in range(4)] for i in range(4)]
        if np.allclose(M - np.eye(4), 0):
            return True
        return False

    def invert(self) -> Self:
        """Invert the transformation. Acts in-place."""
        self.T.Inverse()
        self.inverse_flag = bool(self.T.GetInverseFlag())
        return self

    def compute_inverse(self) -> LinearTransform:
        """Compute the inverse."""
        t = self.clone()
        t.invert()
        return t

    def transpose(self) -> Self:
        """Transpose the transformation. Acts in-place."""
        M = vtki.vtkMatrix4x4()
        self.T.GetTranspose(M)
        self.T.SetMatrix(M)
        return self

    def copy(self) -> LinearTransform:
        """Return a copy of the transformation. Alias of `clone()`."""
        return self.clone()

    def clone(self) -> LinearTransform:
        """Clone transformation to make an exact copy."""
        return LinearTransform(self.T)

    def concatenate(self, T, pre_multiply=False) -> Self:
        """
        Post-multiply (by default) 2 transfomations.
        T can also be a 4x4 matrix or 3x3 matrix.

        Examples:
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
        transform = T.T if hasattr(T, "T") else T
        self.T.Concatenate(transform)
        self.T.PostMultiply()
        return self

    def __mul__(self, A):
        """Pre-multiply 2 transfomations."""
        return self.concatenate(A, pre_multiply=True)

    def get_concatenated_transform(self, i) -> LinearTransform:
        """Get intermediate matrix by concatenation index."""
        return LinearTransform(self.T.GetConcatenatedTransform(i))

    @property
    def ntransforms(self) -> int:
        """Get the number of concatenated transforms."""
        return self.T.GetNumberOfConcatenatedTransforms()

    def translate(self, p) -> Self:
        """Translate, same as `shift`."""
        if len(p) == 2:
            p = [p[0], p[1], 0]
        self.T.Translate(p)
        return self

    def shift(self, p) -> Self:
        """Shift, same as `translate`."""
        return self.translate(p)

    def scale(self, s, origin=True) -> Self:
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

    def rotate(self, angle, axis=(1, 0, 0), point=(0, 0, 0), rad=False) -> Self:
        """
        Rotate around an arbitrary `axis` passing through `point`.

        Examples:
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
        axis = np.asarray(axis, dtype=float)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            return self
        if rad:
            anglerad = angle
        else:
            anglerad = np.deg2rad(angle)

        axis = axis / axis_norm
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

    def rotate_x(self, angle: float, rad=False, around=None) -> Self:
        """
        Rotate around x-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        return self._rotatexyz("x", angle, rad, around)

    def rotate_y(self, angle: float, rad=False, around=None) -> Self:
        """
        Rotate around y-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        return self._rotatexyz("y", angle, rad, around)

    def rotate_z(self, angle: float, rad=False, around=None) -> Self:
        """
        Rotate around z-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        return self._rotatexyz("z", angle, rad, around)

    def set_position(self, p) -> Self:
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

    def get_scale(self) -> np.ndarray:
        """Get current scale."""
        return np.array(self.T.GetScale())

    @property
    def orientation(self) -> np.ndarray:
        """Compute orientation."""
        return np.array(self.T.GetOrientation())

    @property
    def position(self) -> np.ndarray:
        """Compute position."""
        return np.array(self.T.GetPosition())

    @property
    def matrix(self) -> np.ndarray:
        """Get the 4x4 trasformation matrix."""
        m = self.T.GetMatrix()
        M = [[m.GetElement(i, j) for j in range(4)] for i in range(4)]
        return np.array(M)

    @matrix.setter
    def matrix(self, M) -> None:
        """Set trasformation by assigning a 4x4 or 3x3 numpy matrix."""
        n = len(M)
        m = vtki.vtkMatrix4x4()
        for i in range(n):
            for j in range(n):
                m.SetElement(i, j, M[i][j])
        self.T.SetMatrix(m)

    @property
    def matrix3x3(self) -> np.ndarray:
        """Get the 3x3 trasformation matrix."""
        m = self.T.GetMatrix()
        M = [[m.GetElement(i, j) for j in range(3)] for i in range(3)]
        return np.array(M)

    def write(self, filename="transform.mat") -> Self:
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
        return self

    def reorient(
        self, initaxis, newaxis, around=(0, 0, 0), rotation=0.0, rad=False, xyplane=True
    ) -> Self:
        """
        Set/Get object orientation.

        Args:
            rotation (float):
                rotate object around newaxis.
            concatenate (bool):
                concatenate the orientation operation with the previous existing transform (if any)
            rad (bool):
                set to True if angle is expressed in radians.
            xyplane (bool):
                make an extra rotation to keep the object aligned to the xy-plane
        """
        newaxis = np.asarray(newaxis) / np.linalg.norm(newaxis)
        initaxis = np.asarray(initaxis) / np.linalg.norm(initaxis)

        if not np.any(initaxis - newaxis):
            return self

        if not np.any(initaxis + newaxis):
            warn("In reorient() initaxis and newaxis are parallel", stacklevel=2)
            newaxis += np.array([0.0000001, 0.0000002, 0.0])
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
class Quaternion:
    """Work with quaternion rotations."""

    def __init__(self, q=None, *, axis=None, angle=0.0, rad=False, xyzw=False) -> None:
        """
        Define a quaternion rotation.

        Args:
            q (Quaternion, vtkQuaternion, vtkLinearTransform, vtkMatrix4x4, sequence):
                input quaternion in ``(w, x, y, z)`` order by default, or a 3x3 rotation matrix.
            axis (list):
                optional rotation axis to build the quaternion from axis-angle form.
            angle (float):
                rotation angle associated to ``axis``.
            rad (bool):
                set to ``True`` if ``angle`` is expressed in radians.
            xyzw (bool):
                interpret a 4-sequence input as ``(x, y, z, w)``.
        """
        self.name = "Quaternion"
        self.T = vtki.vtkQuaterniond()
        self.T.ToIdentity()

        if axis is not None:
            if q is not None:
                raise ValueError(
                    "Quaternion() accepts either q=... or axis/angle, not both"
                )
            self.set_axis_angle(angle, axis, rad=rad)
            return

        if q is None:
            return

        if isinstance(q, Quaternion):
            self.T = vtki.vtkQuaterniond(q.T)
            return

        if _is_vtk_quaternion(q):
            self.T = vtki.vtkQuaterniond(q.GetW(), q.GetX(), q.GetY(), q.GetZ())
            return

        if isinstance(q, LinearTransform):
            self.matrix3x3 = q.matrix3x3
            return

        if isinstance(q, vtki.vtkMatrix4x4):
            self.matrix3x3 = np.array(
                [[q.GetElement(i, j) for j in range(3)] for i in range(3)],
                dtype=float,
            )
            return

        if isinstance(q, vtki.vtkLinearTransform):
            self.matrix3x3 = np.array(
                [[q.GetMatrix().GetElement(i, j) for j in range(3)] for i in range(3)],
                dtype=float,
            )
            return

        if _is_sequence(q):
            arr = np.asarray(q, dtype=float)
            if arr.shape == (4,):
                self.set(arr, xyzw=xyzw)
                return
            if arr.shape == (3, 3):
                self.matrix3x3 = arr
                return
            raise ValueError("Quaternion() expects a 4-sequence or a 3x3 matrix")

        raise TypeError(f"Cannot build Quaternion from {type(q).__name__}")

    def _summary_rows(self):
        angle, axis = self.angle_axis()
        return [
            ("q (wxyz)", np.array2string(self.wxyz, precision=6, separator=", ")),
            ("q (xyzw)", np.array2string(self.xyzw, precision=6, separator=", ")),
            ("angle", f"{angle:.6f} deg"),
            ("axis", np.array2string(axis, precision=6, separator=", ")),
        ]

    def __str__(self):
        return summary_string(self, self._summary_rows())

    def __repr__(self):
        return self.__str__()

    def __rich__(self):
        return summary_panel(self, self._summary_rows())

    def print(self) -> Quaternion:
        """Print quaternion details."""
        print(self)
        return self

    def __call__(self, p) -> np.ndarray:
        """Rotate a single 2D or 3D vector."""
        return self.rotate(p)

    @classmethod
    def from_xyzw(cls, q) -> Quaternion:
        """Build a quaternion from ``(x, y, z, w)`` components."""
        return cls(q, xyzw=True)

    @classmethod
    def from_axis_angle(cls, angle, axis=(1, 0, 0), rad=False) -> Quaternion:
        """Build a quaternion from axis-angle form."""
        return cls(axis=axis, angle=angle, rad=rad)

    def copy(self) -> Quaternion:
        """Return a copy of the quaternion. Alias of ``clone()``."""
        return self.clone()

    def clone(self) -> Quaternion:
        """Clone the quaternion to make an exact copy."""
        return Quaternion(self.T)

    def reset(self) -> Self:
        """Reset quaternion to identity."""
        self.T.ToIdentity()
        return self

    def set(self, q, xyzw=False) -> Self:
        """Set quaternion components."""
        arr = np.asarray(q, dtype=float).ravel()
        if arr.shape != (4,):
            raise ValueError("Quaternion.set() expects a 4-sequence")
        if xyzw:
            self.T.Set(arr[3], arr[0], arr[1], arr[2])
        else:
            self.T.Set(arr)
        return self

    def set_axis_angle(self, angle, axis=(1, 0, 0), rad=False) -> Self:
        """Set quaternion from axis-angle form."""
        axis = np.asarray(axis, dtype=float)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            raise ValueError("Quaternion axis cannot have zero length")
        axis = axis / axis_norm
        if not rad:
            angle = np.deg2rad(angle)
        self.T.SetRotationAngleAndAxis(angle, axis)
        return self

    def angle_axis(self, rad=False) -> tuple[float, np.ndarray]:
        """Return the quaternion as ``(angle, axis)``."""
        axis = np.zeros(3, dtype=float)
        angle = float(self.T.GetRotationAngleAndAxis(axis))
        if not rad:
            angle = np.rad2deg(angle)
        return angle, axis

    def normalize(self) -> Self:
        """Normalize the quaternion in place."""
        self.T.Normalize()
        return self

    def normalized(self) -> Quaternion:
        """Return a normalized copy of the quaternion."""
        return Quaternion(self.T.Normalized())

    def conjugate(self) -> Self:
        """Conjugate the quaternion in place."""
        self.T.Conjugate()
        return self

    def conjugated(self) -> Quaternion:
        """Return the conjugated quaternion."""
        return Quaternion(self.T.Conjugated())

    def invert(self) -> Self:
        """Invert the quaternion in place."""
        self.T.Invert()
        return self

    def inverse(self) -> Quaternion:
        """Return the inverse quaternion."""
        return Quaternion(self.T.Inverse())

    def slerp(self, t: float, q) -> Quaternion:
        """Spherically interpolate towards quaternion ``q``."""
        q0 = self.normalized().wxyz
        q1 = Quaternion(q).normalized().wxyz
        dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))

        # Flip the second quaternion so we stay on the shortest arc.
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        if dot > 0.9995:
            out = q0 + t * (q1 - q0)
            out /= np.linalg.norm(out)
            return Quaternion(out)

        theta0 = np.arccos(dot)
        theta = theta0 * t
        sin_theta0 = np.sin(theta0)
        out = np.sin(theta0 - theta) / sin_theta0 * q0 + np.sin(theta) / sin_theta0 * q1
        return Quaternion(out)

    def rotate(self, p) -> np.ndarray:
        """Rotate a single 2D or 3D vector."""
        p = np.asarray(p, dtype=float)
        if p.shape == (2,):
            p = np.array([p[0], p[1], 0.0], dtype=float)
        if p.shape != (3,):
            raise ValueError("Quaternion.rotate() expects a 2D or 3D vector")
        return self.matrix3x3 @ p

    transform_point = rotate

    def to_transform(self) -> LinearTransform:
        """Convert the quaternion to a ``LinearTransform``."""
        return LinearTransform(self.matrix3x3)

    @property
    def w(self) -> float:
        return float(self.T.GetW())

    @w.setter
    def w(self, value) -> None:
        self.T.SetW(float(value))

    @property
    def x(self) -> float:
        return float(self.T.GetX())

    @x.setter
    def x(self, value) -> None:
        self.T.SetX(float(value))

    @property
    def y(self) -> float:
        return float(self.T.GetY())

    @y.setter
    def y(self, value) -> None:
        self.T.SetY(float(value))

    @property
    def z(self) -> float:
        return float(self.T.GetZ())

    @z.setter
    def z(self, value) -> None:
        self.T.SetZ(float(value))

    @property
    def wxyz(self) -> np.ndarray:
        """Get the quaternion as ``(w, x, y, z)``."""
        return np.array([self.w, self.x, self.y, self.z], dtype=float)

    @wxyz.setter
    def wxyz(self, q) -> None:
        self.set(q)

    @property
    def xyzw(self) -> np.ndarray:
        """Get the quaternion as ``(x, y, z, w)``."""
        return np.array([self.x, self.y, self.z, self.w], dtype=float)

    @xyzw.setter
    def xyzw(self, q) -> None:
        self.set(q, xyzw=True)

    @property
    def norm(self) -> float:
        """Get the quaternion norm."""
        return float(self.T.Norm())

    @property
    def squared_norm(self) -> float:
        """Get the squared quaternion norm."""
        return float(self.T.SquaredNorm())

    @property
    def matrix3x3(self) -> np.ndarray:
        """Get the 3x3 rotation matrix."""
        M = np.zeros((3, 3), dtype=float)
        self.T.ToMatrix3x3(M)
        return M

    @matrix3x3.setter
    def matrix3x3(self, M) -> None:
        """Set quaternion from a 3x3 rotation matrix."""
        arr = np.asarray(M, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError("Quaternion.matrix3x3 expects a 3x3 matrix")
        self.T.FromMatrix3x3(arr)

    @property
    def matrix(self) -> np.ndarray:
        """Alias of ``matrix3x3``."""
        return self.matrix3x3

    @matrix.setter
    def matrix(self, M) -> None:
        self.matrix3x3 = M


###################################################
class NonLinearTransform:
    """Work with non-linear transformations."""

    def __init__(self, T=None, **kwargs) -> None:
        """
        Define a non-linear transformation.
        Can be saved to file and reloaded.

        Args:
            T (vtkThinPlateSplineTransform, str, dict):
                vtk transformation.
                If T is a string, it is assumed to be a filename.
                If T is a dictionary, it is assumed to be a set of keyword arguments.
                Defaults to None.
            **kwargs (dict):
                keyword arguments to define the transformation.
                The following keywords are accepted:
                - name : (str) name of the transformation
                - comment : (str) comment
                - source_points : (list) source points
                - target_points : (list) target points
                - mode : (str) either '2d' or '3d'
                - sigma : (float) sigma parameter

        Examples:
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
                warn(f'In {filename} mode can be either "2d" or "3d"', stacklevel=2)

        elif len(kwargs) > 0:
            T = kwargs.copy()
            self.name = T.pop("name", "NonLinearTransform")
            self.comment = T.pop("comment", "")
            source = T.pop("source_points", [])
            target = T.pop("target_points", [])
            mode = T.pop("mode", "3d")
            sigma = T.pop("sigma", 1.0)
            if len(T) > 0:
                warn(
                    f"NonLinearTransform got unexpected keyword arguments: {sorted(T.keys())}",
                    stacklevel=2,
                )

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
                warn('Mode can be either "2d" or "3d"', stacklevel=2)

        self.T = T
        self.inverse_flag = False

    def _summary_rows(self):
        p = self.source_points
        q = self.target_points
        rows = [("name", self.name)]
        if self.filename:
            rows.append(("filename", self.filename))
        if self.comment:
            rows.append(("comment", self.comment))
        rows.append(("mode", self.mode))
        rows.append(("sigma", str(self.sigma)))
        if len(p):
            rows.append(
                (
                    "sources",
                    f"{len(p)}, bounds {np.min(p, axis=0)}, {np.max(p, axis=0)}",
                )
            )
        else:
            rows.append(("sources", "0"))
        if len(q):
            rows.append(
                (
                    "targets",
                    f"{len(q)}, bounds {np.min(q, axis=0)}, {np.max(q, axis=0)}",
                )
            )
        else:
            rows.append(("targets", "0"))
        return rows

    def __str__(self):
        return summary_string(self, self._summary_rows())

    def __repr__(self):
        return self.__str__()

    def __rich__(self):
        return summary_panel(self, self._summary_rows())

    def print(self) -> Self:
        """Print transformation."""
        print(self)
        return self

    def update(self) -> Self:
        """Update transformation."""
        self.T.Update()
        return self

    @property
    def position(self) -> np.ndarray:
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
    def source_points(self) -> np.ndarray:
        """Get the source points."""
        pts = self.T.GetSourceLandmarks()
        vpts = []
        if pts:
            for i in range(pts.GetNumberOfPoints()):
                vpts.append(pts.GetPoint(i))
        if not vpts:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(vpts, dtype=np.float32)

    @source_points.setter
    def source_points(self, pts):
        """Set source points."""
        if _is_sequence(pts):
            pass
        else:
            pts = pts.coordinates
        vpts = vtki.vtkPoints()
        for p in pts:
            if len(p) == 2:
                p = [p[0], p[1], 0.0]
            vpts.InsertNextPoint(p)
        self.T.SetSourceLandmarks(vpts)

    @property
    def target_points(self) -> np.ndarray:
        """Get the target points."""
        pts = self.T.GetTargetLandmarks()
        vpts = []
        if pts:
            for i in range(pts.GetNumberOfPoints()):
                vpts.append(pts.GetPoint(i))
        if not vpts:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(vpts, dtype=np.float32)

    @target_points.setter
    def target_points(self, pts):
        """Set target points."""
        if _is_sequence(pts):
            pass
        else:
            pts = pts.coordinates
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
            warn("NonLinearTransform has no valid mode.", stacklevel=2)
            return ""

    @mode.setter
    def mode(self, m):
        """Set mode."""
        if m == "3d":
            self.T.SetBasisToR()
        elif m == "2d":
            self.T.SetBasisToR2LogR()
        else:
            warn('In NonLinearTransform mode can be either "2d" or "3d"', stacklevel=2)

    def clone(self) -> NonLinearTransform:
        """Clone transformation to make an exact copy."""
        return NonLinearTransform(self.T)

    def write(self, filename) -> Self:
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
        return self

    def invert(self) -> NonLinearTransform:
        """Invert transformation."""
        self.T.Inverse()
        self.inverse_flag = bool(self.T.GetInverseFlag())
        return self

    def compute_inverse(self) -> Self:
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

    def compute_main_axes(self, pt=(0, 0, 0), ds=1) -> np.ndarray:
        """
        Compute main axes of the transformation.
        These are the axes of the ellipsoid that is the
        image of the unit sphere under the transformation.

        Args:
            pt (list):
                point to compute the axes at.
            ds (float):
                step size to compute the axes.
        """
        if len(pt) == 2:
            pt = [pt[0], pt[1], 0]
        pt = np.asarray(pt)
        m = np.array(
            [
                self.move(pt + [ds, 0, 0]),
                self.move(pt + [0, ds, 0]),
                self.move(pt + [0, 0, ds]),
            ]
        )
        eigval, eigvec = np.linalg.eig(m @ m.T)
        eigval = np.sqrt(eigval)
        return np.array(
            [
                eigvec[:, 0] * eigval[0],
                eigvec[:, 1] * eigval[1],
                eigvec[:, 2] * eigval[2],
            ]
        )

    def transform_point(self, p) -> np.ndarray:
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

        Examples:
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

    Examples:
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

    def __init__(self, mode="linear") -> None:
        """
        Interpolate between two or more linear transformations.
        """
        self.vtk_interpolator = vtki.new("TransformInterpolator")
        self.mode(mode)
        self.TS: list[LinearTransform] = []

    def _mode_name(self) -> str:
        mapping = {
            0: "linear",
            1: "spline",
            2: "manual",
        }
        return mapping.get(self.vtk_interpolator.GetInterpolationType(), "unknown")

    def _summary_rows(self):
        rows = [
            ("mode", self._mode_name()),
            ("ntransforms", str(self.ntransforms)),
        ]
        if self.ntransforms:
            tmin, tmax = self.trange()
            rows.append(("trange", f"[{tmin}, {tmax}]"))
        else:
            rows.append(("trange", "[]"))
        return rows

    def __str__(self):
        return summary_string(self, self._summary_rows())

    def __repr__(self):
        return self.__str__()

    def __rich__(self):
        return summary_panel(self, self._summary_rows())

    def print(self) -> TransformInterpolator:
        """Print interpolator details."""
        print(self)
        return self

    def __call__(self, t):
        """
        Get the intermediate transformation at time `t`.
        """
        xform = vtki.vtkTransform()
        self.vtk_interpolator.InterpolateTransform(t, xform)
        return LinearTransform(xform)

    def add(self, t, T) -> TransformInterpolator:
        """Add intermediate transformations."""
        try:
            # in case a vedo object is passed
            T = T.transform
        except AttributeError:
            pass
        if isinstance(T, LinearTransform):
            LT = T
        elif isinstance(T, vtki.vtkLinearTransform):
            LT = LinearTransform(T)
        else:
            raise TypeError(
                "TransformInterpolator.add() expects LinearTransform or vtkLinearTransform"
            )

        self.TS.append(LT)
        self.vtk_interpolator.AddTransform(t, LT.T)
        return self

    # def remove(self, t) -> TransformInterpolator:
    #     """Remove intermediate transformations."""
    #     self.TS.pop(t)
    #     self.vtk_interpolator.RemoveTransform(t)
    #     return self

    def trange(self) -> np.ndarray:
        """Get interpolation range."""
        tmin = self.vtk_interpolator.GetMinimumT()
        tmax = self.vtk_interpolator.GetMaximumT()
        return np.array([tmin, tmax])

    def clear(self) -> TransformInterpolator:
        """Clear all intermediate transformations."""
        self.TS = []
        self.vtk_interpolator.Initialize()
        return self

    def mode(self, m) -> TransformInterpolator:
        """Set interpolation mode ('linear' or 'spline')."""
        if m == "linear":
            self.vtk_interpolator.SetInterpolationTypeToLinear()
        elif m == "spline":
            self.vtk_interpolator.SetInterpolationTypeToSpline()
        else:
            warn(
                'In TransformInterpolator mode can be either "linear" or "spline"',
                stacklevel=2,
            )
        return self

    @property
    def ntransforms(self) -> int:
        """Get number of transformations."""
        return self.vtk_interpolator.GetNumberOfTransforms()


########################################################################
# 2d ######
def cart2pol(x, y) -> np.ndarray:
    """2D Cartesian to Polar coordinates conversion."""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return np.array([rho, theta])


def pol2cart(rho, theta) -> np.ndarray:
    """2D Polar to Cartesian coordinates conversion."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.array([x, y])


########################################################################
# 3d ######
def cart2spher(x, y, z) -> np.ndarray:
    """3D Cartesian to Spherical coordinate conversion."""
    hxy = np.hypot(x, y)
    rho = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    return np.array([rho, theta, phi])


def spher2cart(rho, theta, phi) -> np.ndarray:
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


def cart2cyl(x, y, z) -> np.ndarray:
    """3D Cartesian to Cylindrical coordinate conversion."""
    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return np.array([rho, theta, z])


def cyl2cart(rho, theta, z) -> np.ndarray:
    """3D Cylindrical to Cartesian coordinate conversion."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.array([x, y, z])


def cyl2spher(rho, theta, z) -> np.ndarray:
    """3D Cylindrical to Spherical coordinate conversion."""
    rhos = np.sqrt(rho * rho + z * z)
    phi = np.arctan2(rho, z)
    return np.array([rhos, phi, theta])


def spher2cyl(rho, theta, phi) -> np.ndarray:
    """3D Spherical to Cylindrical coordinate conversion."""
    rhoc = rho * np.sin(theta)
    z = rho * np.cos(theta)
    return np.array([rhoc, phi, z])
