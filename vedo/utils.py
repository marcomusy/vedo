#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np

from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray
import vedo.vtkclasses as vtki

import vedo


__docformat__ = "google"

__doc__ = "Utilities submodule."

__all__ = [
    "OperationNode",
    "ProgressBar",
    "progressbar",
    "Minimizer",
    "geometry",
    "is_sequence",
    "lin_interpolate",
    "vector",
    "mag",
    "mag2",
    "versor",
    "precision",
    "round_to_digit",
    "point_in_triangle",
    "point_line_distance",
    "closest",
    "grep",
    "make_bands",
    "pack_spheres",
    "humansort",
    "print_histogram",
    "print_inheritance_tree",
    "camera_from_quaternion",
    "camera_from_neuroglancer",
    "camera_from_dict",
    "camera_to_dict",
    "oriented_camera",
    "vedo2trimesh",
    "trimesh2vedo",
    "vedo2meshlab",
    "meshlab2vedo",
    "vedo2open3d",
    "open3d2vedo",
    "vtk2numpy",
    "numpy2vtk",
    "get_uv",
    "andrews_curves",
]


###########################################################################
array_types = {}
array_types[vtki.VTK_UNSIGNED_CHAR] = ("UNSIGNED_CHAR",  "np.uint8")
array_types[vtki.VTK_UNSIGNED_SHORT]= ("UNSIGNED_SHORT", "np.uint16")
array_types[vtki.VTK_UNSIGNED_INT]  = ("UNSIGNED_INT",   "np.uint32")
array_types[vtki.VTK_UNSIGNED_LONG_LONG] = ("UNSIGNED_LONG_LONG", "np.uint64")
array_types[vtki.VTK_CHAR]          = ("CHAR",           "np.int8")
array_types[vtki.VTK_SHORT]         = ("SHORT",          "np.int16")
array_types[vtki.VTK_INT]           = ("INT",            "np.int32")
array_types[vtki.VTK_LONG]          = ("LONG",           "") # ??
array_types[vtki.VTK_LONG_LONG]     = ("LONG_LONG",      "np.int64")
array_types[vtki.VTK_FLOAT]         = ("FLOAT",          "np.float32")
array_types[vtki.VTK_DOUBLE]        = ("DOUBLE",         "np.float64")
array_types[vtki.VTK_SIGNED_CHAR]   = ("SIGNED_CHAR",    "np.int8")
array_types[vtki.VTK_ID_TYPE]       = ("ID",             "np.int64")


###########################################################################
class OperationNode:
    """
    Keep track of the operations which led to a final state.
    """
    # https://www.graphviz.org/doc/info/shapes.html#html
    # Mesh     #e9c46a
    # Follower #d9ed92
    # Volume, UnstructuredGrid #4cc9f0
    # TetMesh  #9e2a2b
    # File     #8a817c
    # Image  #f28482
    # Assembly #f08080

    def __init__(
        self, operation, parents=(), comment="", shape="none", c="#e9c46a", style="filled"
    ):
        """
        Keep track of the operations which led to a final object.
        This allows to show the `pipeline` tree for any `vedo` object with e.g.:

        ```python
        from vedo import *
        sp = Sphere()
        sp.clean().subdivide()
        sp.pipeline.show()
        ```

        Arguments:
            operation : (str, class)
                descriptor label, if a class is passed then grab its name
            parents : (list)
                list of the parent classes the object comes from
            comment : (str)
                a second-line text description
            shape : (str)
                shape of the frame, check out [this link.](https://graphviz.org/doc/info/shapes.html)
            c : (hex)
                hex color
            style : (str)
                comma-separated list of styles

        Example:
            ```python
            from vedo.utils import OperationNode

            op_node1 = OperationNode("Operation1", c="lightblue")
            op_node2 = OperationNode("Operation2")
            op_node3 = OperationNode("Operation3", shape='diamond')
            op_node4 = OperationNode("Operation4")
            op_node5 = OperationNode("Operation5")
            op_node6 = OperationNode("Result", c="lightgreen")

            op_node3.add_parent(op_node1)
            op_node4.add_parent(op_node1)
            op_node3.add_parent(op_node2)
            op_node5.add_parent(op_node2)
            op_node6.add_parent(op_node3)
            op_node6.add_parent(op_node5)
            op_node6.add_parent(op_node1)

            op_node6.show(orientation="TB")
            ```
            ![](https://vedo.embl.es/images/feats/operation_node.png)
        """
        if not vedo.settings.enable_pipeline:
            return

        if isinstance(operation, str):
            self.operation = operation
        else:
            self.operation = operation.__class__.__name__
        self.operation_plain = str(self.operation)

        pp = []  # filter out invalid stuff
        for p in parents:
            if hasattr(p, "pipeline"):
                pp.append(p.pipeline)
        self.parents = pp

        if comment:
            self.operation = f"<{self.operation}<BR/><SUB><I>{comment}</I></SUB>>"

        self.dot = None
        self.time = time.time()
        self.shape = shape
        self.style = style
        self.color = c
        self.counts = 0

    def add_parent(self, parent):
        self.parents.append(parent)

    def _build_tree(self, dot):
        dot.node(
            str(id(self)),
            label=self.operation,
            shape=self.shape,
            color=self.color,
            style=self.style,
        )
        for parent in self.parents:
            if parent:
                t = f"{self.time - parent.time: .1f}s"
                dot.edge(str(id(parent)), str(id(self)), label=t)
                parent._build_tree(dot)

    def __repr__(self):
        try:
            from treelib import Tree
        except ImportError:
            vedo.logger.error(
                "To use this functionality please install treelib:"
                "\n pip install treelib"
            )
            return ""

        def _build_tree(parent):
            for par in parent.parents:
                if par:
                    op = par.operation_plain
                    tree.create_node(
                        op, op + str(par.time), parent=parent.operation_plain + str(parent.time)
                    )
                    _build_tree(par)
        try:
            tree = Tree()
            tree.create_node(self.operation_plain, self.operation_plain + str(self.time))
            _build_tree(self)
            out = tree.show(stdout=False)
        except:
            out = f"Sorry treelib failed to build the tree for '{self.operation_plain}()'."
        return out

    def print(self):
        """Print the tree of operations."""
        print(self.__str__())

    def show(self, orientation="LR", popup=True):
        """Show the graphviz output for the pipeline of this object"""
        if not vedo.settings.enable_pipeline:
            return

        try:
            from graphviz import Digraph
        except ImportError:
            vedo.logger.error("please install graphviz with command\n pip install graphviz")
            return

        # visualize the entire tree
        dot = Digraph(
            node_attr={"fontcolor": "#201010", "fontname": "Helvetica", "fontsize": "12"},
            edge_attr={"fontname": "Helvetica", "fontsize": "6", "arrowsize": "0.4"},
        )
        dot.attr(rankdir=orientation)

        self.counts = 0
        self._build_tree(dot)
        self.dot = dot

        home_dir = os.path.expanduser("~")
        gpath = os.path.join(
            home_dir, vedo.settings.cache_directory, "vedo", "pipeline_graphviz")

        dot.render(gpath, view=popup)


###########################################################################
class ProgressBar:
    """
    Class to print a progress bar.
    """

    def __init__(
        self,
        start,
        stop,
        step=1,
        c=None,
        bold=True,
        italic=False,
        title="",
        eta=True,
        delay=-1,
        width=25,
        char="\U00002501",
        char_back="\U00002500",
    ):
        """
        Class to print a progress bar with optional text message.

        Check out also function `progressbar()`.

        Arguments:
            start : (int)
                starting value
            stop : (int)
                stopping value
            step : (int)
                step value
            c : (str)
                color in hex format
            title : (str)
                title text
            eta : (bool)
                estimate time of arrival
            delay : (float)
                minimum time before printing anything,
                if negative use the default value
                as set in `vedo.settings.progressbar_delay`
            width : (int)
                width of the progress bar
            char : (str)
                character to use for the progress bar
            char_back : (str)
                character to use for the background of the progress bar

        Example:
            ```python
            import time
            from vedo import ProgressBar
            pb = ProgressBar(0,40, c='r')
            for i in pb.range():
                time.sleep(0.1)
                pb.print()
            ```
            ![](https://user-images.githubusercontent.com/32848391/51858823-ed1f4880-2335-11e9-8788-2d102ace2578.png)
        """
        self.char = char
        self.char_back = char_back

        self.title = title + " "
        if title:
            self.title = " " + self.title

        if delay < 0:
            delay = vedo.settings.progressbar_delay

        self.start = start
        self.stop = stop
        self.step = step

        self.color = c
        self.bold = bold
        self.italic = italic
        self.width = width
        self.pbar = ""
        self.percent = 0.0
        self.percent_int = 0
        self.eta = eta
        self.delay = delay

        self.t0 = time.time()
        self._remaining = 1e10

        self._update(0)

        self._counts = 0
        self._oldbar = ""
        self._lentxt = 0
        self._range = np.arange(start, stop, step)

    def print(self, txt="", c=None):
        """Print the progress bar with an optional message."""
        if not c:
            c = self.color

        self._update(self._counts + self.step)

        if self.delay:
            if time.time() - self.t0 < self.delay:
                return

        if self.pbar != self._oldbar:
            self._oldbar = self.pbar

            if self.eta and self._counts > 1:

                tdenom = time.time() - self.t0
                if tdenom:
                    vel = self._counts / tdenom
                    self._remaining = (self.stop - self._counts) / vel
                else:
                    vel = 1
                    self._remaining = 0.0

                if self._remaining > 60:
                    mins = int(self._remaining / 60)
                    secs = self._remaining - 60 * mins
                    mins = f"{mins}m"
                    secs = f"{int(secs + 0.5)}s "
                else:
                    mins = ""
                    secs = f"{int(self._remaining + 0.5)}s "

                vel = round(vel, 1)
                eta = f"eta: {mins}{secs}({vel} it/s) "
                if self._remaining < 0.5:
                    dt = time.time() - self.t0
                    if dt > 60:
                        mins = int(dt / 60)
                        secs = dt - 60 * mins
                        mins = f"{mins}m"
                        secs = f"{int(secs + 0.5)}s "
                    else:
                        mins = ""
                        secs = f"{int(dt + 0.5)}s "
                    eta = f"elapsed: {mins}{secs}({vel} it/s)        "
                    txt = ""
            else:
                eta = ""

            eraser = " " * self._lentxt + "\b" * self._lentxt

            s = f"{self.pbar} {eraser}{eta}{txt}\r"
            vedo.printc(s, c=c, bold=self.bold, italic=self.italic, end="")
            if self.percent > 99.999:
                print("")

            self._lentxt = len(txt)

    def range(self):
        """Return the range iterator."""
        return self._range

    def _update(self, counts):
        if counts < self.start:
            counts = self.start
        elif counts > self.stop:
            counts = self.stop
        self._counts = counts

        self.percent = (self._counts - self.start) * 100.0

        delta = self.stop - self.start
        if delta:
            self.percent /= delta
        else:
            self.percent = 0.0

        self.percent_int = int(round(self.percent))
        af = self.width - 2
        nh = int(round(self.percent_int / 100 * af))
        pbar_background = "\x1b[2m" + self.char_back * (af - nh)
        self.pbar = f"{self.title}{self.char * (nh-1)}{pbar_background}"
        if self.percent < 100.0:
            ps = f" {self.percent_int}%"
        else:
            ps = ""
        self.pbar += ps


#####################################
def progressbar(
        iterable,
        c=None, bold=True, italic=False, title="",
        eta=True, width=25, delay=-1,
    ):
    """
    Function to print a progress bar with optional text message.

    Use delay to set a minimum time before printing anything.
    If delay is negative, then use the default value
    as set in `vedo.settings.progressbar_delay`.

    Arguments:
        start : (int)
            starting value
        stop : (int)
            stopping value
        step : (int)
            step value
        c : (str)
            color in hex format
        title : (str)
            title text
        eta : (bool)
            estimate time of arrival
        delay : (float)
            minimum time before printing anything,
            if negative use the default value
            set in `vedo.settings.progressbar_delay`
        width : (int)
            width of the progress bar
        char : (str)
            character to use for the progress bar
        char_back : (str)
            character to use for the background of the progress bar

    Example:
        ```python
        import time
        for i in progressbar(range(100), c='r'):
            time.sleep(0.1)
        ```
        ![](https://user-images.githubusercontent.com/32848391/51858823-ed1f4880-2335-11e9-8788-2d102ace2578.png)
    """
    try:
        if is_number(iterable):
            total = int(iterable)
            iterable = range(total)
        else:
            total = len(iterable)
    except TypeError:
        iterable = list(iterable)
        total = len(iterable)

    pb = ProgressBar(
        0, total, c=c, bold=bold, italic=italic, title=title,
        eta=eta, delay=delay, width=width,
    )
    for item in iterable:
        pb.print()
        yield item


###########################################################
class Minimizer:
    """
    A function minimizer that uses the Nelder-Mead method.

    The algorithm constructs an n-dimensional simplex in parameter
    space (i.e. a tetrahedron if the number or parameters is 3)
    and moves the vertices around parameter space until
    a local minimum is found. The amoeba method is robust,
    reasonably efficient, but is not guaranteed to find
    the global minimum if several local minima exist.
    
    Arguments:
        function : (callable)
            the function to minimize
        max_iterations : (int)
            the maximum number of iterations
        contraction_ratio : (float)
            The contraction ratio.
            The default value of 0.5 gives fast convergence,
            but larger values such as 0.6 or 0.7 provide greater stability.
        expansion_ratio : (float)
            The expansion ratio.
            The default value is 2.0, which provides rapid expansion.
            Values between 1.1 and 2.0 are valid.
        tol : (float)
            the tolerance for convergence
    
    Example:
        - [nelder-mead.py](https://github.com/marcomusy/vedo/blob/master/examples/others/nelder-mead.py)
    """
    def __init__(
            self,
            function=None,
            max_iterations=10000,
            contraction_ratio=0.5,
            expansion_ratio=2.0,
            tol=1e-5,
        ):
        self.function = function
        self.tolerance = tol
        self.contraction_ratio = contraction_ratio
        self.expansion_ratio = expansion_ratio
        self.max_iterations = max_iterations
        self.minimizer = vtki.new("AmoebaMinimizer")
        self.minimizer.SetFunction(self._vtkfunc)
        self.results = {}
        self.parameters_path = []
        self.function_path = []

    def _vtkfunc(self):
        n = self.minimizer.GetNumberOfParameters()
        ain = [self.minimizer.GetParameterValue(i) for i in range(n)]
        r = self.function(ain)
        self.minimizer.SetFunctionValue(r)
        self.parameters_path.append(ain)
        self.function_path.append(r)
        return r
    
    def eval(self, parameters=()):
        """
        Evaluate the function at the current or given parameters.
        """
        if len(parameters) == 0:
            return self.minimizer.EvaluateFunction()
        self.set_parameters(parameters)
        return self.function(parameters)
    
    def set_parameter(self, name, value, scale=1.0):
        """
        Set the parameter value.
        The initial amount by which the parameter
        will be modified during the search for the minimum. 
        """
        self.minimizer.SetParameterValue(name, value)
        self.minimizer.SetParameterScale(name, scale)
    
    def set_parameters(self, parameters):
        """
        Set the parameters names and values from a dictionary.
        """
        for name, value in parameters.items():
            if len(value) == 2:
                self.set_parameter(name, value[0], value[1])
            else:
                self.set_parameter(name, value)
    
    def minimize(self):
        """
        Minimize the input function.

        Returns:
            dict : 
                the minimization results
            init_parameters : (dict)
                the initial parameters
            parameters : (dict)
                the final parameters
            min_value : (float)
                the minimum value
            iterations : (int)
                the number of iterations
            max_iterations : (int)
                the maximum number of iterations
            tolerance : (float)
                the tolerance for convergence
            convergence_flag : (int)
                zero if the tolerance stopping
                criterion has been met.
            parameters_path : (np.array)
                the path of the minimization
                algorithm in parameter space
            function_path : (np.array)
                the path of the minimization
                algorithm in function space
            hessian : (np.array)
                the Hessian matrix of the
                function at the minimum
            parameter_errors : (np.array)
                the errors on the parameters
        """
        n = self.minimizer.GetNumberOfParameters()
        out = [(
            self.minimizer.GetParameterName(i),
            (self.minimizer.GetParameterValue(i),
             self.minimizer.GetParameterScale(i))
        ) for i in range(n)]
        self.results["init_parameters"] = dict(out)

        self.minimizer.SetTolerance(self.tolerance)
        self.minimizer.SetContractionRatio(self.contraction_ratio)
        self.minimizer.SetExpansionRatio(self.expansion_ratio)
        self.minimizer.SetMaxIterations(self.max_iterations)

        self.minimizer.Minimize()
        self.results["convergence_flag"] = not bool(self.minimizer.Iterate())

        out = [(
            self.minimizer.GetParameterName(i),
            self.minimizer.GetParameterValue(i),
        ) for i in range(n)]

        self.results["parameters"] = dict(out)
        self.results["min_value"] = self.minimizer.GetFunctionValue()
        self.results["iterations"] = self.minimizer.GetIterations()
        self.results["max_iterations"] = self.minimizer.GetMaxIterations()
        self.results["tolerance"] = self.minimizer.GetTolerance()
        self.results["expansion_ratio"] = self.expansion_ratio
        self.results["contraction_ratio"] = self.contraction_ratio
        self.results["parameters_path"] = np.array(self.parameters_path)
        self.results["function_path"] = np.array(self.function_path)
        self.results["hessian"] = np.zeros((n,n))
        self.results["parameter_errors"] = np.zeros(n)
        return self.results

    def compute_hessian(self, epsilon=0):
        """
        Compute the Hessian matrix of `function` at the
        minimum numerically.

        Arguments:
            epsilon : (float)
                Step size used for numerical approximation.

        Returns:
            array: Hessian matrix of `function` at minimum.
        """
        if not epsilon:
            epsilon = self.tolerance * 10
        n = self.minimizer.GetNumberOfParameters()
        x0 = [self.minimizer.GetParameterValue(i) for i in range(n)]
        hessian = np.zeros((n, n))
        for i in vedo.progressbar(n, title="Computing Hessian", delay=2):
            for j in range(n):
                xijp = np.copy(x0)
                xijp[i] += epsilon
                xijp[j] += epsilon
                xijm = np.copy(x0)
                xijm[i] += epsilon
                xijm[j] -= epsilon
                xjip = np.copy(x0)
                xjip[i] -= epsilon
                xjip[j] += epsilon
                xjim = np.copy(x0)
                xjim[i] -= epsilon
                xjim[j] -= epsilon
                # Second derivative approximation
                fijp = self.function(xijp)
                fijm = self.function(xijm)
                fjip = self.function(xjip)
                fjim = self.function(xjim)
                hessian[i, j] = (fijp - fijm - fjip + fjim) / (2 * epsilon**2)
        self.results["hessian"] = hessian
        try:
            ihess = np.linalg.inv(hessian)
            self.results["parameter_errors"] = np.sqrt(np.diag(ihess))
        except:
            vedo.logger.warning("Cannot compute hessian for parameter errors")
            self.results["parameter_errors"] = np.zeros(n)
        return hessian

    def __str__(self) -> str:
        out = vedo.printc(
            f"vedo.utils.Minimizer at ({hex(id(self))})".ljust(75),
            bold=True, invert=True, return_string=True,
        )
        out += "Function name".ljust(20) + self.function.__name__ + "()\n"
        out += "-------- parameters initial value -----------\n"
        out += "Name".ljust(20) + "Value".ljust(20) + "Scale\n"
        for name, value in self.results["init_parameters"].items():
            out += name.ljust(20) + str(value[0]).ljust(20) + str(value[1]) + "\n"
        out += "-------- parameters final value --------------\n"
        for name, value in self.results["parameters"].items():
            out += name.ljust(20) + f"{value:.6f}"
            ierr = list(self.results["parameters"]).index(name)
            err = self.results["parameter_errors"][ierr]
            if err:
                out += f" ± {err:.4f}"
            out += "\n"
        out += "Value at minimum".ljust(20)+ f'{self.results["min_value"]}\n'
        out += "Iterations".ljust(20)      + f'{self.results["iterations"]}\n'
        out += "Max iterations".ljust(20)  + f'{self.results["max_iterations"]}\n'
        out += "Convergence flag".ljust(20)+ f'{self.results["convergence_flag"]}\n'
        out += "Tolerance".ljust(20)       + f'{self.results["tolerance"]}\n'
        try:
            arr = np.array2string(
                self.compute_hessian(),
                separator=', ', precision=6, suppress_small=True,
            )
            out += "Hessian Matrix:\n" + arr
        except:
            out += "Hessian Matrix: (not available)"
        return out


###########################################################
def andrews_curves(M, res=100):
    """
    Computes the [Andrews curves](https://en.wikipedia.org/wiki/Andrews_plot)
    for the provided data.

    The input array is an array of shape (n,m) where n is the number of
    features and m is the number of observations.
    
    Arguments:
        M : (ndarray)
            the data matrix (or data vector).
        res : (int)
            the resolution (n. of points) of the output curve.
    
    Example:
        - [andrews_cluster.py](https://github.com/marcomusy/vedo/blob/master/examples/pyplot/andrews_cluster.py)
    
        ![](https://vedo.embl.es/images/pyplot/andrews_cluster.png)
    """
    # Credits:
    # https://gist.github.com/ryuzakyl/12c221ff0e54d8b1ac171c69ea552c0a
    M = np.asarray(M)
    m = int(res + 0.5)

    # getting data vectors
    X = np.reshape(M, (1, -1)) if len(M.shape) == 1 else M.copy()
    _rows, n = X.shape

    # andrews curve dimension (n. theta angles)
    t = np.linspace(-np.pi, np.pi, m)

    # m: range of values for angle theta
    # n: amount of components of the Fourier expansion
    A = np.empty((m, n))

    # setting first column of A
    A[:, 0] = [1/np.sqrt(2)] * m

    # filling columns of A
    for i in range(1, n):
        # computing the scaling coefficient for angle theta
        c = np.ceil(i / 2)
        # computing i-th column of matrix A
        col = np.sin(c * t) if i % 2 == 1 else np.cos(c * t)
        # setting column in matrix A
        A[:, i] = col[:]

    # computing Andrews curves for provided data
    andrew_curves = np.dot(A, X.T).T

    # returning the Andrews Curves (raveling if needed)
    return np.ravel(andrew_curves) if andrew_curves.shape[0] == 1 else andrew_curves


###########################################################
def numpy2vtk(arr, dtype=None, deep=True, name=""):
    """
    Convert a numpy array into a `vtkDataArray`.
    Use `dtype='id'` for `vtkIdTypeArray` objects.
    """
    # https://github.com/Kitware/VTK/blob/master/Wrapping/Python/vtkmodules/util/numpy_support.py
    if arr is None:
        return None

    arr = np.ascontiguousarray(arr)

    if dtype == "id":
        varr = numpy_to_vtkIdTypeArray(arr.astype(np.int64), deep=deep)
    elif dtype:
        varr = numpy_to_vtk(arr.astype(dtype), deep=deep)
    else:
        # let numpy_to_vtk() decide what is best type based on arr type
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8)
        varr = numpy_to_vtk(arr, deep=deep)

    if name:
        varr.SetName(name)
    return varr

def vtk2numpy(varr):
    """Convert a `vtkDataArray`, `vtkIdList` or `vtTransform` into a numpy array."""
    if varr is None:
        return np.array([])
    if isinstance(varr, vtki.vtkIdList):
        return np.array([varr.GetId(i) for i in range(varr.GetNumberOfIds())])
    elif isinstance(varr, vtki.vtkBitArray):
        carr = vtki.vtkCharArray()
        carr.DeepCopy(varr)
        varr = carr
    elif isinstance(varr, vtki.vtkHomogeneousTransform):
        try:
            varr = varr.GetMatrix()
        except AttributeError:
            pass
        n = 4
        M = [[varr.GetElement(i, j) for j in range(n)] for i in range(n)]
        return np.array(M)
    return vtk_to_numpy(varr)


def make3d(pts):
    """
    Make an array which might be 2D to 3D.

    Array can also be in the form `[allx, ally, allz]`.
    """
    if pts is None:
        return np.array([])
    pts = np.asarray(pts)

    if pts.dtype == "object":
        raise ValueError("Cannot form a valid numpy array, input may be non-homogenous")

    if pts.size == 0:  # empty list
        return pts

    if pts.ndim == 1:
        if pts.shape[0] == 2:
            return np.hstack([pts, [0]]).astype(pts.dtype)
        elif pts.shape[0] == 3:
            return pts
        else:
            raise ValueError

    if pts.shape[1] == 3:
        return pts

    # if 2 <= pts.shape[0] <= 3 and pts.shape[1] > 3:
    #     pts = pts.T

    if pts.shape[1] == 2:
        return np.c_[pts, np.zeros(pts.shape[0], dtype=pts.dtype)]

    if pts.shape[1] != 3:
        raise ValueError(f"input shape is not supported: {pts.shape}")
    return pts


def geometry(obj, extent=None):
    """
    Apply the `vtkGeometryFilter` to the input object.
    This is a general-purpose filter to extract geometry (and associated data)
    from any type of dataset.
    This filter also may be used to convert any type of data to polygonal type.
    The conversion process may be less than satisfactory for some 3D datasets.
    For example, this filter will extract the outer surface of a volume
    or structured grid dataset.

    Returns a `vedo.Mesh` object.

    Set `extent` as the `[xmin,xmax, ymin,ymax, zmin,zmax]` bounding box to clip data.
    """
    gf = vtki.new("GeometryFilter")
    gf.SetInputData(obj)
    if extent is not None:
        gf.SetExtent(extent)
    gf.Update()
    return vedo.Mesh(gf.GetOutput())


def buildPolyData(vertices, faces=None, lines=None, strips=None, index_offset=0, tetras=False):
    """
    Build a `vtkPolyData` object from a list of vertices
    where faces represents the connectivity of the polygonal mesh.
    Lines and triangle strips can also be specified.

    E.g. :
        - `vertices=[[x1,y1,z1],[x2,y2,z2], ...]`
        - `faces=[[0,1,2], [1,2,3], ...]`
        - `lines=[[0,1], [1,2,3,4], ...]`
        - `strips=[[0,1,2,3,4,5], [2,3,9,7,4], ...]`

    A flat list of faces can be passed as `faces=[3, 0,1,2, 4, 1,2,3,4, ...]`.
    For lines use `lines=[2, 0,1, 4, 1,2,3,4, ...]`.

    Use `index_offset=1` if face numbering starts from 1 instead of 0.

    If `tetras=True`, interpret 4-point faces as tetrahedrons instead of surface quads.
    """
    if is_sequence(faces) and len(faces) == 0:
        faces=None
    if is_sequence(lines) and len(lines) == 0:
        lines=None
    if is_sequence(strips) and len(strips) == 0:
        strips=None

    poly = vtki.vtkPolyData()

    if len(vertices) == 0:
        return poly

    vertices = make3d(vertices)
    source_points = vtki.vtkPoints()
    source_points.SetData(numpy2vtk(vertices, dtype=np.float32))
    poly.SetPoints(source_points)

    if lines is not None:
        # Create a cell array to store the lines in and add the lines to it
        linesarr = vtki.vtkCellArray()
        if is_sequence(lines[0]):  # assume format [(id0,id1),..]
            for iline in lines:
                for i in range(0, len(iline) - 1):
                    i1, i2 = iline[i], iline[i + 1]
                    if i1 != i2:
                        vline = vtki.vtkLine()
                        vline.GetPointIds().SetId(0, i1)
                        vline.GetPointIds().SetId(1, i2)
                        linesarr.InsertNextCell(vline)
        else:  # assume format [id0,id1,...]
            # print("buildPolyData: assuming lines format [id0,id1,...]", lines)
            # TODO CORRECT THIS CASE, MUST BE [2, id0,id1,...]
            for i in range(0, len(lines) - 1):
                vline = vtki.vtkLine()
                vline.GetPointIds().SetId(0, lines[i])
                vline.GetPointIds().SetId(1, lines[i + 1])
                linesarr.InsertNextCell(vline)
        poly.SetLines(linesarr)

    if faces is not None:
        source_polygons = vtki.vtkCellArray()
        if isinstance(faces, np.ndarray) or not is_ragged(faces):
            ##### all faces are composed of equal nr of vtxs, FAST
            faces = np.asarray(faces)
            ast = np.int32
            if vtki.vtkIdTypeArray().GetDataTypeSize() != 4:
                ast = np.int64

            if faces.ndim > 1:
                nf, nc = faces.shape
                hs = np.hstack((np.zeros(nf)[:, None] + nc, faces))
            else:
                nf = faces.shape[0]
                hs = faces
            arr = numpy_to_vtkIdTypeArray(hs.astype(ast).ravel(), deep=True)
            source_polygons.SetCells(nf, arr)

        else:
            ############################# manually add faces, SLOW
            for f in faces:
                n = len(f)

                if n == 3:
                    ele = vtki.vtkTriangle()
                    pids = ele.GetPointIds()
                    for i in range(3):
                        pids.SetId(i, f[i] - index_offset)
                    source_polygons.InsertNextCell(ele)

                elif n == 4 and tetras:
                    ele0 = vtki.vtkTriangle()
                    ele1 = vtki.vtkTriangle()
                    ele2 = vtki.vtkTriangle()
                    ele3 = vtki.vtkTriangle()
                    if index_offset:
                        for i in [0, 1, 2, 3]:
                            f[i] -= index_offset
                    f0, f1, f2, f3 = f
                    pid0 = ele0.GetPointIds()
                    pid1 = ele1.GetPointIds()
                    pid2 = ele2.GetPointIds()
                    pid3 = ele3.GetPointIds()

                    pid0.SetId(0, f0)
                    pid0.SetId(1, f1)
                    pid0.SetId(2, f2)

                    pid1.SetId(0, f0)
                    pid1.SetId(1, f1)
                    pid1.SetId(2, f3)

                    pid2.SetId(0, f1)
                    pid2.SetId(1, f2)
                    pid2.SetId(2, f3)

                    pid3.SetId(0, f2)
                    pid3.SetId(1, f3)
                    pid3.SetId(2, f0)

                    source_polygons.InsertNextCell(ele0)
                    source_polygons.InsertNextCell(ele1)
                    source_polygons.InsertNextCell(ele2)
                    source_polygons.InsertNextCell(ele3)

                else:
                    ele = vtki.vtkPolygon()
                    pids = ele.GetPointIds()
                    pids.SetNumberOfIds(n)
                    for i in range(n):
                        pids.SetId(i, f[i] - index_offset)
                    source_polygons.InsertNextCell(ele)

        poly.SetPolys(source_polygons)
    
    if strips is not None:
        tscells = vtki.vtkCellArray()
        for strip in strips:
            # create a triangle strip
            # https://vtk.org/doc/nightly/html/classvtkTriangleStrip.html
            n = len(strip)
            tstrip = vtki.vtkTriangleStrip()
            tstrip_ids = tstrip.GetPointIds()
            tstrip_ids.SetNumberOfIds(n)
            for i in range(n):
                tstrip_ids.SetId(i, strip[i] - index_offset)
            tscells.InsertNextCell(tstrip)
        poly.SetStrips(tscells)

    if faces is None and lines is None and strips is None:
        source_vertices = vtki.vtkCellArray()
        for i in range(len(vertices)):
            source_vertices.InsertNextCell(1)
            source_vertices.InsertCellPoint(i)
        poly.SetVerts(source_vertices)

    # print("buildPolyData \n",
    #     poly.GetNumberOfPoints(),
    #     poly.GetNumberOfCells(), # grand total
    #     poly.GetNumberOfLines(),
    #     poly.GetNumberOfPolys(),
    #     poly.GetNumberOfStrips(),
    #     poly.GetNumberOfVerts(),
    # )
    return poly


##############################################################################
def get_font_path(font):
    """Internal use."""
    if font in vedo.settings.font_parameters.keys():
        if vedo.settings.font_parameters[font]["islocal"]:
            fl = os.path.join(vedo.fonts_path, f"{font}.ttf")
        else:
            try:
                fl = vedo.file_io.download(f"https://vedo.embl.es/fonts/{font}.ttf", verbose=False)
            except:
                vedo.logger.warning(f"Could not download https://vedo.embl.es/fonts/{font}.ttf")
                fl = os.path.join(vedo.fonts_path, "Normografo.ttf")
    else:
        if font.startswith("https://"):
            fl = vedo.file_io.download(font, verbose=False)
        elif os.path.isfile(font):
            fl = font  # assume user is passing a valid file
        else:
            if font.endswith(".ttf"):
                vedo.logger.error(
                    f"Could not set font file {font}"
                    f"-> using default: {vedo.settings.default_font}"
                )
            else:
                vedo.settings.default_font = "Normografo"
                vedo.logger.error(
                    f"Could not set font name {font}"
                    f" -> using default: Normografo\n"
                    f"Check out https://vedo.embl.es/fonts for additional fonts\n"
                    f"Type 'vedo -r fonts' to see available fonts"
                )
            fl = get_font_path(vedo.settings.default_font)
    return fl


def is_sequence(arg):
    """Check if the input is iterable."""
    if hasattr(arg, "strip"):
        return False
    if hasattr(arg, "__getslice__"):
        return True
    if hasattr(arg, "__iter__"):
        return True
    return False


def is_ragged(arr, deep=False):
    """
    A ragged or inhomogeneous array in Python is an array
    with arrays of different lengths as its elements.
    To check if an array is ragged,we iterate through the elements
    and check if their lengths are the same.

    Example:
    ```python
    arr = [[1, 2, 3], [[4, 5], [6], 1], [7, 8, 9]]
    print(is_ragged(arr, deep=True))  # output: True
    ```
    """
    n = len(arr)
    if n == 0:
        return False
    if is_sequence(arr[0]):
        length = len(arr[0])
        for i in range(1, n):
            if len(arr[i]) != length or (deep and is_ragged(arr[i])):
                return True
        return False
    return False


def flatten(list_to_flatten):
    """Flatten out a list."""

    def _genflatten(lst):
        for elem in lst:
            if isinstance(elem, (list, tuple)):
                for x in flatten(elem):
                    yield x
            else:
                yield elem

    return list(_genflatten(list_to_flatten))


def humansort(alist):
    """
    Sort in place a given list the way humans expect.

    E.g. `['file11', 'file1'] -> ['file1', 'file11']`

    .. warning:: input list is modified in-place by this function.
    """
    import re

    def alphanum_key(s):
        # Turn a string into a list of string and number chunks.
        # e.g. "z23a" -> ["z", 23, "a"]
        def tryint(s):
            if s.isdigit():
                return int(s)
            return s

        return [tryint(c) for c in re.split("([0-9]+)", s)]

    alist.sort(key=alphanum_key)
    return alist  # NB: input list is modified


def sort_by_column(arr, nth, invert=False):
    """Sort a numpy array by its `n-th` column."""
    arr = np.asarray(arr)
    arr = arr[arr[:, nth].argsort()]
    if invert:
        return np.flip(arr, axis=0)
    return arr


def point_in_triangle(p, p1, p2, p3):
    """
    Return True if a point is inside (or above/below)
    a triangle defined by 3 points in space.
    """
    p1 = np.array(p1)
    u = p2 - p1
    v = p3 - p1
    n = np.cross(u, v)
    w = p - p1
    ln = np.dot(n, n)
    if not ln:
        return None  # degenerate triangle
    gamma = (np.dot(np.cross(u, w), n)) / ln
    if 0 < gamma < 1:
        beta = (np.dot(np.cross(w, v), n)) / ln
        if 0 < beta < 1:
            alpha = 1 - gamma - beta
            if 0 < alpha < 1:
                return True
    return False


def intersection_ray_triangle(P0, P1, V0, V1, V2):
    """
    Fast intersection between a directional ray defined by `P0,P1`
    and triangle `V0, V1, V2`.

    Returns the intersection point or
    - `None` if triangle is degenerate, or ray is  parallel to triangle plane.
    - `False` if no intersection, or ray direction points away from triangle.
    """
    # Credits: http://geomalgorithms.com/a06-_intersect-2.html
    # Get triangle edge vectors and plane normal
    # todo : this is slow should check
    # https://vtk.org/doc/nightly/html/classvtkCell.html
    V0 = np.asarray(V0, dtype=float)
    P0 = np.asarray(P0, dtype=float)
    u = V1 - V0
    v = V2 - V0
    n = np.cross(u, v)
    if not np.abs(v).sum():  # triangle is degenerate
        return None  # do not deal with this case

    rd = P1 - P0  # ray direction vector
    w0 = P0 - V0
    a = -np.dot(n, w0)
    b = np.dot(n, rd)
    if not b:  # ray is  parallel to triangle plane
        return None

    # Get intersect point of ray with triangle plane
    r = a / b
    if r < 0.0:  # ray goes away from triangle
        return False  #  => no intersect

    # Gor a segment, also test if (r > 1.0) => no intersect
    I = P0 + r * rd  # intersect point of ray and plane

    # is I inside T?
    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    w = I - V0
    wu = np.dot(w, u)
    wv = np.dot(w, v)
    D = uv * uv - uu * vv

    # Get and test parametric coords
    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0:  # I is outside T
        return False
    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0:  # I is outside T
        return False
    return I  # I is in T


def triangle_solver(**input_dict):
    """
    Solve a triangle from any 3 known elements.
    (Note that there might be more than one solution or none).
    Angles are in radians.

    Example:
    ```python
    print(triangle_solver(a=3, b=4, c=5))
    print(triangle_solver(a=3, ac=0.9273, ab=1.5716))
    print(triangle_solver(a=3, b=4, ab=1.5716))
    print(triangle_solver(b=4, bc=.64, ab=1.5716))
    print(triangle_solver(c=5, ac=.9273, bc=0.6435))
    print(triangle_solver(a=3, c=5, bc=0.6435))
    print(triangle_solver(b=4, c=5, ac=0.927))
    ```
    """
    a = input_dict.get("a")
    b = input_dict.get("b")
    c = input_dict.get("c")
    ab = input_dict.get("ab")
    bc = input_dict.get("bc")
    ac = input_dict.get("ac")

    if ab and bc:
        ac = np.pi - bc - ab
    elif bc and ac:
        ab = np.pi - bc - ac
    elif ab and ac:
        bc = np.pi - ab - ac

    if a is not None and b is not None and c is not None:
        ab = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
        sinab = np.sin(ab)
        ac = np.arcsin(a / c * sinab)
        bc = np.arcsin(b / c * sinab)

    elif a is not None and b is not None and ab is not None:
        c = np.sqrt(a ** 2 + b ** 2 - 2 * a * b * np.cos(ab))
        sinab = np.sin(ab)
        ac = np.arcsin(a / c * sinab)
        bc = np.arcsin(b / c * sinab)

    elif a is not None and ac is not None and ab is not None:
        h = a * np.sin(ac)
        b = h / np.sin(bc)
        c = b * np.cos(bc) + a * np.cos(ac)

    elif b is not None and bc is not None and ab is not None:
        h = b * np.sin(bc)
        a = h / np.sin(ac)
        c = np.sqrt(a * a + b * b)

    elif c is not None and ac is not None and bc is not None:
        h = c * np.sin(bc)
        b1 = c * np.cos(bc)
        b2 = h / np.tan(ab)
        b = b1 + b2
        a = np.sqrt(b2 * b2 + h * h)

    elif a is not None and c is not None and bc is not None:
        # double solution
        h = c * np.sin(bc)
        k = np.sqrt(a * a - h * h)
        omega = np.arcsin(k / a)
        cosbc = np.cos(bc)
        b = c * cosbc - k
        phi = np.pi / 2 - bc - omega
        ac = phi
        ab = np.pi - ac - bc
        if k:
            b2 = c * cosbc + k
            ac2 = phi + 2 * omega
            ab2 = np.pi - ac2 - bc
            return [
                {"a": a, "b": b, "c": c, "ab": ab, "bc": bc, "ac": ac},
                {"a": a, "b": b2, "c": c, "ab": ab2, "bc": bc, "ac": ac2},
            ]

    elif b is not None and c is not None and ac is not None:
        # double solution
        h = c * np.sin(ac)
        k = np.sqrt(b * b - h * h)
        omega = np.arcsin(k / b)
        cosac = np.cos(ac)
        a = c * cosac - k
        phi = np.pi / 2 - ac - omega
        bc = phi
        ab = np.pi - bc - ac
        if k:
            a2 = c * cosac + k
            bc2 = phi + 2 * omega
            ab2 = np.pi - ac - bc2
            return [
                {"a": a, "b": b, "c": c, "ab": ab, "bc": bc, "ac": ac},
                {"a": a2, "b": b, "c": c, "ab": ab2, "bc": bc2, "ac": ac},
            ]

    else:
        vedo.logger.error(f"Case {input_dict} is not supported.")
        return []

    return [{"a": a, "b": b, "c": c, "ab": ab, "bc": bc, "ac": ac}]


#############################################################################
def point_line_distance(p, p1, p2):
    """
    Compute the distance of a point to a line (not the segment)
    defined by `p1` and `p2`.
    """
    return np.sqrt(vtki.vtkLine.DistanceToLine(p, p1, p2))

def line_line_distance(p1, p2, q1, q2):
    """
    Compute the distance of a line to a line (not the segment)
    defined by `p1` and `p2` and `q1` and `q2`.

    Returns the distance,
    the closest point on line 1, the closest point on line 2.
    Their parametric coords (-inf <= t0, t1 <= inf) are also returned.
    """
    closest_pt1 = [0,0,0]
    closest_pt2 = [0,0,0]
    t1, t2 = 0.0, 0.0
    d = vtki.vtkLine.DistanceBetweenLines(
        p1, p2, q1, q2, closest_pt1, closest_pt2, t1, t2)
    return np.sqrt(d), closest_pt1, closest_pt2, t1, t2

def segment_segment_distance(p1, p2, q1, q2):
    """
    Compute the distance of a segment to a segment
    defined by `p1` and `p2` and `q1` and `q2`.

    Returns the distance,
    the closest point on line 1, the closest point on line 2.
    Their parametric coords (-inf <= t0, t1 <= inf) are also returned.
    """
    closest_pt1 = [0,0,0]
    closest_pt2 = [0,0,0]
    t1, t2 = 0.0, 0.0
    d = vtki.vtkLine.DistanceBetweenLineSegments(
        p1, p2, q1, q2, closest_pt1, closest_pt2, t1, t2)
    return np.sqrt(d), closest_pt1, closest_pt2, t1, t2


def closest(point, points, n=1, return_ids=False, use_tree=False):
    """
    Returns the distances and the closest point(s) to the given set of points.
    Needs `scipy.spatial` library.

    Arguments:
        n : (int)
            the nr of closest points to return
        return_ids : (bool)
            return the ids instead of the points coordinates
        use_tree : (bool)
            build a `scipy.spatial.KDTree`.
            An already existing one can be passed to avoid rebuilding.
    """
    from scipy.spatial import distance, KDTree

    points = np.asarray(points)
    if n == 1:
        dists = distance.cdist([point], points)
        closest_idx = np.argmin(dists)
    else:
        if use_tree:
            if isinstance(use_tree, KDTree):  # reuse
                tree = use_tree
            else:
                tree = KDTree(points)
            dists, closest_idx = tree.query([point], k=n)
            closest_idx = closest_idx[0]
        else:
            dists = distance.cdist([point], points)
            closest_idx = np.argsort(dists)[0][:n]
    if return_ids:
        return dists, closest_idx
    else:
        return dists, points[closest_idx]


#############################################################################
def lin_interpolate(x, rangeX, rangeY):
    """
    Interpolate linearly the variable `x` in `rangeX` onto the new `rangeY`.
    If `x` is a 3D vector the linear weight is the distance to the two 3D `rangeX` vectors.

    E.g. if `x` runs in `rangeX=[x0,x1]` and I want it to run in `rangeY=[y0,y1]` then

    `y = lin_interpolate(x, rangeX, rangeY)` will interpolate `x` onto `rangeY`.

    Examples:
        - [lin_interpolate.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/lin_interpolate.py)

            ![](https://vedo.embl.es/images/basic/linInterpolate.png)
    """
    if is_sequence(x):
        x = np.asarray(x)
        x0, x1 = np.asarray(rangeX)
        y0, y1 = np.asarray(rangeY)
        dx = x1 - x0
        dxn = np.linalg.norm(dx)
        if not dxn:
            return y0
        s = np.linalg.norm(x - x0) / dxn
        t = np.linalg.norm(x - x1) / dxn
        st = s + t
        out = y0 * (t / st) + y1 * (s / st)

    else:  # faster

        x0 = rangeX[0]
        dx = rangeX[1] - x0
        if not dx:
            return rangeY[0]
        s = (x - x0) / dx
        out = rangeY[0] * (1 - s) + rangeY[1] * s
    return out


def get_uv(p, x, v):
    """
    Obtain the texture uv-coords of a point p belonging to a face that has point
    coordinates (x0, x1, x2) with the corresponding uv-coordinates v=(v0, v1, v2).
    All p and x0,x1,x2 are 3D-vectors, while v are their 2D uv-coordinates.

    Example:
        ```python
        from vedo import *

        pic = Image(dataurl+"coloured_cube_faces.jpg")
        cb = Mesh(dataurl+"coloured_cube.obj").lighting("off").texture(pic)

        cbpts = cb.vertices
        faces = cb.cells
        uv = cb.pointdata["Material"]

        pt = [-0.2, 0.75, 2]
        pr = cb.closest_point(pt)

        idface = cb.closest_point(pt, return_cell_id=True)
        idpts = faces[idface]
        uv_face = uv[idpts]

        uv_pr = utils.get_uv(pr, cbpts[idpts], uv_face)
        print("interpolated uv =", uv_pr)

        sx, sy = pic.dimensions()
        i_interp_uv = uv_pr * [sy, sx]
        ix, iy = i_interp_uv.astype(int)
        mpic = pic.tomesh()
        rgba = mpic.pointdata["RGBA"].reshape(sy, sx, 3)
        print("color =", rgba[ix, iy])

        show(
            [[cb, Point(pr), cb.labels("Material")],
                [pic, Point(i_interp_uv)]],
            N=2, axes=1, sharecam=False,
        ).close()
        ```
        ![](https://vedo.embl.es/images/feats/utils_get_uv.png)
    """
    # Vector vp=p-x0 is representable as alpha*s + beta*t,
    # where s = x1-x0 and t = x2-x0, in matrix form
    # vp = [alpha, beta] . matrix(s,t)
    # M = matrix(s,t) is 2x3 matrix, so (alpha, beta) can be found by
    # inverting any of its minor A with non-zero determinant.
    # Once found, uv-coords of p are vt0 + alpha (vt1-v0) + beta (vt2-v0)

    p = np.asarray(p)
    x0, x1, x2 = np.asarray(x)[:3]
    vt0, vt1, vt2 = np.asarray(v)[:3]

    s = x1 - x0
    t = x2 - x0
    vs = vt1 - vt0
    vt = vt2 - vt0
    vp = p - x0

    # finding a minor with independent rows
    M = np.matrix([s, t])
    mnr = [0, 1]
    A = M[:, mnr]
    if np.abs(np.linalg.det(A)) < 0.000001:
        mnr = [0, 2]
        A = M[:, mnr]
        if np.abs(np.linalg.det(A)) < 0.000001:
            mnr = [1, 2]
            A = M[:, mnr]
    Ainv = np.linalg.inv(A)
    alpha_beta = vp[mnr].dot(Ainv)  # [alpha, beta]
    return np.asarray(vt0 + alpha_beta.dot(np.matrix([vs, vt])))[0]


def vector(x, y=None, z=0.0, dtype=np.float64):
    """
    Return a 3D numpy array representing a vector.

    If `y` is `None`, assume input is already in the form `[x,y,z]`.
    """
    if y is None:  # assume x is already [x,y,z]
        return np.asarray(x, dtype=dtype)
    return np.array([x, y, z], dtype=dtype)


def versor(x, y=None, z=0.0, dtype=np.float64):
    """Return the unit vector. Input can be a list of vectors."""
    v = vector(x, y, z, dtype)
    if isinstance(v[0], np.ndarray):
        return np.divide(v, mag(v)[:, None])
    return v / mag(v)


def mag(v):
    """Get the magnitude of a vector or array of vectors."""
    v = np.asarray(v)
    if v.ndim == 1:
        return np.linalg.norm(v)
    return np.linalg.norm(v, axis=1)


def mag2(v):
    """Get the squared magnitude of a vector or array of vectors."""
    v = np.asarray(v)
    if v.ndim == 1:
        return np.square(v).sum()
    return np.square(v).sum(axis=1)


def is_integer(n):
    """Check if input is an integer."""
    try:
        float(n)
    except (ValueError, TypeError):
        return False
    else:
        return float(n).is_integer()


def is_number(n):
    """Check if input is a number"""
    try:
        float(n)
        return True
    except (ValueError, TypeError):
        return False


def round_to_digit(x, p):
    """Round a real number to the specified number of significant digits."""
    if not x:
        return 0
    r = np.round(x, p - int(np.floor(np.log10(abs(x)))) - 1)
    if int(r) == r:
        return int(r)
    return r


def pack_spheres(bounds, radius):
    """
    Packing spheres into a bounding box.
    Returns a numpy array of sphere centers.
    """
    h = 0.8164965 / 2
    d = 0.8660254
    a = 0.288675135

    if is_sequence(bounds):
        x0, x1, y0, y1, z0, z1 = bounds
    else:
        x0, x1, y0, y1, z0, z1 = bounds.bounds()

    x = np.arange(x0, x1, radius)
    nul = np.zeros_like(x)
    nz = int((z1 - z0) / radius / h / 2 + 1.5)
    ny = int((y1 - y0) / radius / d + 1.5)

    pts = []
    for iz in range(nz):
        z = z0 + nul + iz * h * radius
        dx, dy, dz = [radius * 0.5, radius * a, iz * h * radius]
        for iy in range(ny):
            y = y0 + nul + iy * d * radius
            if iy % 2:
                xs = x
            else:
                xs = x + radius * 0.5
            if iz % 2:
                p = np.c_[xs, y, z] + [dx, dy, dz]
            else:
                p = np.c_[xs, y, z] + [0, 0, dz]
            pts += p.tolist()
    return np.array(pts)


def precision(x, p, vrange=None, delimiter="e"):
    """
    Returns a string representation of `x` formatted to precision `p`.

    Set `vrange` to the range in which x exists (to snap x to '0' if below precision).
    """
    # Based on the webkit javascript implementation
    # `from here <https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp>`_,
    # and implemented by `randlet <https://github.com/randlet/to-precision>`_.
    # Modified for vedo by M.Musy 2020

    if isinstance(x, str):  # do nothing
        return x

    if is_sequence(x):
        out = "("
        nn = len(x) - 1
        for i, ix in enumerate(x):

            try:
                if np.isnan(ix):
                    return "NaN"
            except:
                # cannot handle list of list
                continue

            out += precision(ix, p)
            if i < nn:
                out += ", "
        return out + ")"  ############ <--

    try:
        if np.isnan(x):
            return "NaN"
    except TypeError:
        return "NaN"

    x = float(x)

    if x == 0.0 or (vrange is not None and abs(x) < vrange / pow(10, p)):
        return "0"

    out = []
    if x < 0:
        out.append("-")
        x = -x

    e = int(np.log10(x))
    # tens = np.power(10, e - p + 1)
    tens = 10 ** (e - p + 1)
    n = np.floor(x / tens)

    # if n < np.power(10, p - 1):
    if n < 10 ** (p - 1):
        e = e - 1
        # tens = np.power(10, e - p + 1)
        tens = 10 ** (e - p + 1)
        n = np.floor(x / tens)

    if abs((n + 1.0) * tens - x) <= abs(n * tens - x):
        n = n + 1

    # if n >= np.power(10, p):
    if n >= 10 ** p:
        n = n / 10.0
        e = e + 1

    m = "%.*g" % (p, n)
    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append(delimiter)
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p - 1):
        out.append(m)
    elif e >= 0:
        out.append(m[: e + 1])
        if e + 1 < len(m):
            out.append(".")
            out.extend(m[e + 1 :])
    else:
        out.append("0.")
        out.extend(["0"] * -(e + 1))
        out.append(m)
    return "".join(out)


##################################################################################
def grep(filename, tag, column=None, first_occurrence_only=False):
    """Greps the line in a file that starts with a specific `tag` string inside the file."""
    import re

    with open(filename, "r", encoding="UTF-8") as afile:
        content = []
        for line in afile:
            if re.search(tag, line):
                c = line.split()
                c[-1] = c[-1].replace("\n", "")
                if column is not None:
                    c = c[column]
                content.append(c)
                if first_occurrence_only:
                    break
    return content

def parse_pattern(query, strings_to_parse) -> list:
    """
    Parse a pattern query to a list of strings.
    The query string can contain wildcards like * and ?.

    Arguments:
        query : (str)
            the query to parse
        strings_to_parse : (str/list)
            the string or list of strings to parse

    Returns:
        a list of booleans, one for each string in strings_to_parse

    Example:
        >>> query = r'*Sphere 1?3*'
        >>> strings = ["Sphere 143 red", "Sphere 13 red", "Sphere 123", "ASphere 173"]
        >>> parse_pattern(query, strings)
        [True, True, False, False]
    """
    from re import findall as re_findall
    if not isinstance(query, str):
        return [False]

    if not is_sequence(strings_to_parse):
        strings_to_parse = [strings_to_parse]

    outs = []
    for sp in strings_to_parse:
        if not isinstance(sp, str):
            outs.append(False)
            continue

        s = query
        if s.startswith("*"):
            s = s[1:]
        else:
            s = "^" + s

        t = ""
        if not s.endswith("*"):
            t = "$"
        else:
            s = s[:-1]

        pattern = s.replace('?', r'\w').replace(' ', r'\s').replace("*", r"\w+") + t

        # Search for the pattern in the input string
        match = re_findall(pattern, sp)
        out = bool(match)
        outs.append(out)
        # Print the matches for debugging
        print("pattern", pattern, "in:", strings_to_parse)
        print("matches", match, "result:", out)
    return outs

def print_histogram(
    data,
    bins=10,
    height=10,
    logscale=False,
    minbin=0,
    horizontal=True,
    char="\U00002589",
    c=None,
    bold=True,
    title="histogram",
    spacer="",
):
    """
    Ascii histogram printing.

    Input can be a `vedo.Volume` or `vedo.Mesh`.
    Returns the raw data before binning (useful when passing vtk objects).

    Arguments:
        bins : (int)
            number of histogram bins
        height : (int)
            height of the histogram in character units
        logscale : (bool)
            use logscale for frequencies
        minbin : (int)
            ignore bins before minbin
        horizontal : (bool)
            show histogram horizontally
        char : (str)
            character to be used
        bold : (bool)
            use boldface
        title : (str)
            histogram title
        spacer : (str)
            horizontal spacer

    Example:
        ```python
        from vedo import print_histogram
        import numpy as np
        d = np.random.normal(size=1000)
        data = print_histogram(d, c='b', logscale=True, title='my scalars')
        data = print_histogram(d, c='o')
        print(np.mean(data)) # data here is same as d
        ```
        ![](https://vedo.embl.es/images/feats/print_histogram.png)
    """
    # credits: http://pyinsci.blogspot.com/2009/10/ascii-histograms.html
    # adapted for vedo by M.Musy, 2019

    if not horizontal:  # better aspect ratio
        bins *= 2

    try:
        data = vtk2numpy(data.dataset.GetPointData().GetScalars())
    except AttributeError:
        # already an array
        data = np.asarray(data)

    if isinstance(data, vtki.vtkImageData):
        dims = data.GetDimensions()
        nvx = min(100000, dims[0] * dims[1] * dims[2])
        idxs = np.random.randint(0, min(dims), size=(nvx, 3))
        data = []
        for ix, iy, iz in idxs:
            d = data.GetScalarComponentAsFloat(ix, iy, iz, 0)
            data.append(d)
        data = np.array(data)

    elif isinstance(data, vtki.vtkPolyData):
        arr = data.dataset.GetPointData().GetScalars()
        if not arr:
            arr = data.dataset.GetCellData().GetScalars()
            if not arr:
                return None
        data = vtk2numpy(arr)

    try:
        h = np.histogram(data, bins=bins)
    except TypeError as e:
        vedo.logger.error(f"cannot compute histogram: {e}")
        return ""

    if minbin:
        hi = h[0][minbin:-1]
    else:
        hi = h[0]

    if char == "\U00002589" and horizontal:
        char = "\U00002586"

    title = title.ljust(14) + ":"
    entrs = " entries=" + str(len(data))
    if logscale:
        h0 = np.log10(hi + 1)
        maxh0 = int(max(h0) * 100) / 100
        title = title + entrs + " (logscale)"
    else:
        h0 = hi
        maxh0 = max(h0)
        title = title + entrs

    def _v():
        his = ""
        if title:
            his += title + "\n"
        bars = h0 / maxh0 * height
        for l in reversed(range(1, height + 1)):
            line = ""
            if l == height:
                line = "%s " % maxh0
            else:
                line = "   |" + " " * (len(str(maxh0)) - 3)
            for c in bars:
                if c >= np.ceil(l):
                    line += char
                else:
                    line += " "
            line += "\n"
            his += line
        his += "%.2f" % h[1][0] + "." * (bins) + "%.2f" % h[1][-1] + "\n"
        return his

    def _h():
        his = ""
        if title:
            his += title + "\n"
        xl = ["%.2f" % n for n in h[1]]
        lxl = [len(l) for l in xl]
        bars = h0 / maxh0 * height
        his += spacer + " " * int(max(bars) + 2 + max(lxl)) + "%s\n" % maxh0
        for i, c in enumerate(bars):
            line = xl[i] + " " * int(max(lxl) - lxl[i]) + "| " + char * int(c) + "\n"
            his += spacer + line
        return his

    if horizontal:
        height *= 2
        vedo.printc(_h(), c=c, bold=bold)
    else:
        vedo.printc(_v(), c=c, bold=bold)
    return data


def print_table(*columns, headers=None, c="g"):
    """
    Print lists as tables.

    Example:
        ```python
        from vedo.utils import print_table
        list1 = ["A", "B", "C"]
        list2 = [142, 220, 330]
        list3 = [True, False, True]
        headers = ["First Column", "Second Column", "Third Column"]
        print_table(list1, list2, list3, headers=headers)
        ```

        ![](https://vedo.embl.es/images/feats/)
    """
    # If headers is not provided, use default header names
    corner = "─"
    if headers is None:
        headers = [f"Column {i}" for i in range(1, len(columns) + 1)]
    assert len(headers) == len(columns)

    # Find the maximum length of the elements in each column and header
    max_lens = [max(len(str(x)) for x in column) for column in columns]
    max_len_headers = [max(len(str(header)), max_len) for header, max_len in zip(headers, max_lens)]

    # Construct the table header
    header = (
        "│ "
        + " │ ".join(header.ljust(max_len) for header, max_len in zip(headers, max_len_headers))
        + " │"
    )

    # Construct the line separator
    line1 = "┌" + corner.join("─" * (max_len + 2) for max_len in max_len_headers) + "┐"
    line2 = "└" + corner.join("─" * (max_len + 2) for max_len in max_len_headers) + "┘"

    # Print the table header
    vedo.printc(line1, c=c)
    vedo.printc(header, c=c)
    vedo.printc(line2, c=c)

    # Print the data rows
    for row in zip(*columns):
        row = (
            "│ "
            + " │ ".join(str(col).ljust(max_len) for col, max_len in zip(row, max_len_headers))
            + " │"
        )
        vedo.printc(row, bold=False, c=c)

    # Print the line separator again to close the table
    vedo.printc(line2, c=c)

def print_inheritance_tree(C):
    """Prints the inheritance tree of class C."""
    # Adapted from: https://stackoverflow.com/questions/26568976/
    def class_tree(cls):
        subc = [class_tree(sub_class) for sub_class in cls.__subclasses__()]
        return {cls.__name__: subc}

    def print_tree(tree, indent=8, current_ind=0):
        for k, v in tree.items():
            if current_ind:
                before_dashes = current_ind - indent
                m = " " * before_dashes + "└" + "─" * (indent - 1) + " " + k
                vedo.printc(m)
            else:
                vedo.printc(k)
            for sub_tree in v:
                print_tree(sub_tree, indent=indent, current_ind=current_ind + indent)

    if str(C.__class__) != "<class 'type'>":
        C = C.__class__
    ct = class_tree(C)
    print_tree(ct)


def make_bands(inputlist, n):
    """
    Group values of a list into bands of equal value, where
    `n` is the number of bands, a positive integer > 2.

    Returns a binned list of the same length as the input.
    """
    if n < 2:
        return inputlist
    vmin = np.min(inputlist)
    vmax = np.max(inputlist)
    bb = np.linspace(vmin, vmax, n, endpoint=0)
    dr = bb[1] - bb[0]
    bb += dr / 2
    tol = dr / 2 * 1.001
    newlist = []
    for s in inputlist:
        for b in bb:
            if abs(s - b) < tol:
                newlist.append(b)
                break
    return np.array(newlist)


#################################################################
# Functions adapted from:
# https://github.com/sdorkenw/MeshParty/blob/master/meshparty/trimesh_vtk.py
def camera_from_quaternion(pos, quaternion, distance=10000, ngl_correct=True):
    """
    Define a `vtkCamera` with a particular orientation.

    Arguments:
        pos: (np.array, list, tuple)
            an iterator of length 3 containing the focus point of the camera
        quaternion: (np.array, list, tuple)
            a len(4) quaternion `(x,y,z,w)` describing the rotation of the camera
            such as returned by neuroglancer `x,y,z,w` all in `[0,1]` range
        distance: (float)
            the desired distance from pos to the camera (default = 10000 nm)

    Returns:
        `vtki.vtkCamera`, a vtk camera setup according to these rules.
    """
    camera = vtki.vtkCamera()
    # define the quaternion in vtk, note the swapped order
    # w,x,y,z instead of x,y,z,w
    quat_vtk = vtki.get_class("Quaternion")(
        quaternion[3], quaternion[0], quaternion[1], quaternion[2])
    # use this to define a rotation matrix in x,y,z
    # right handed units
    M = np.zeros((3, 3), dtype=np.float32)
    quat_vtk.ToMatrix3x3(M)
    # the default camera orientation is y up
    up = [0, 1, 0]
    # calculate default camera position is backed off in positive z
    pos = [0, 0, distance]

    # set the camera rototation by applying the rotation matrix
    camera.SetViewUp(*np.dot(M, up))
    # set the camera position by applying the rotation matrix
    camera.SetPosition(*np.dot(M, pos))
    if ngl_correct:
        # neuroglancer has positive y going down
        # so apply these azimuth and roll corrections
        # to fix orientatins
        camera.Azimuth(-180)
        camera.Roll(180)

    # shift the camera posiiton and focal position
    # to be centered on the desired location
    p = camera.GetPosition()
    p_new = np.array(p) + pos
    camera.SetPosition(*p_new)
    camera.SetFocalPoint(*pos)
    return camera


def camera_from_neuroglancer(state, zoom=300):
    """
    Define a `vtkCamera` from a neuroglancer state dictionary.

    Arguments:
        state: (dict)
            an neuroglancer state dictionary.
        zoom: (float)
            how much to multiply zoom by to get camera backoff distance
            default = 300 > ngl_zoom = 1 > 300 nm backoff distance.

    Returns:
        `vtki.vtkCamera`, a vtk camera setup that matches this state.
    """
    orient = state.get("perspectiveOrientation", [0.0, 0.0, 0.0, 1.0])
    pzoom = state.get("perspectiveZoom", 10.0)
    position = state["navigation"]["pose"]["position"]
    pos_nm = np.array(position["voxelCoordinates"]) * position["voxelSize"]
    return camera_from_quaternion(pos_nm, orient, pzoom * zoom, ngl_correct=True)


def oriented_camera(center=(0, 0, 0), up_vector=(0, 1, 0), backoff_vector=(0, 0, 1), backoff=1.0):
    """
    Generate a `vtkCamera` pointed at a specific location,
    oriented with a given up direction, set to a backoff.
    """
    vup = np.array(up_vector)
    vup = vup / np.linalg.norm(vup)
    pt_backoff = center - backoff * np.array(backoff_vector)
    camera = vtki.vtkCamera()
    camera.SetFocalPoint(center[0], center[1], center[2])
    camera.SetViewUp(vup[0], vup[1], vup[2])
    camera.SetPosition(pt_backoff[0], pt_backoff[1], pt_backoff[2])
    return camera


def camera_from_dict(camera, modify_inplace=None):
    """
    Generate a `vtkCamera` object from a python dictionary.

    Parameters of the camera are:
        - `position` or `pos` (3-tuple)
        - `focal_point` (3-tuple)
        - `viewup` (3-tuple)
        - `distance` (float)
        - `clipping_range` (2-tuple)
        - `parallel_scale` (float)
        - `thickness` (float)
        - `view_angle` (float)
        - `roll` (float)

    Exaplanation of the parameters can be found in the
    [vtkCamera documentation](https://vtk.org/doc/nightly/html/classvtkCamera.html).

    Arguments:
        camera: (dict)
            a python dictionary containing camera parameters.
        modify_inplace: (vtkCamera)
            an existing `vtkCamera` object to modify in place.

    Returns:
        `vtki.vtkCamera`, a vtk camera setup that matches this state.
    """
    if modify_inplace:
        vcam = modify_inplace
    else:
        vcam = vtki.vtkCamera()

    camera = dict(camera)  # make a copy so input is not emptied by pop()

    cm_pos         = camera.pop("position", camera.pop("pos", None))
    cm_focal_point = camera.pop("focal_point", camera.pop("focalPoint", None))
    cm_viewup      = camera.pop("viewup", None)
    cm_distance    = camera.pop("distance", None)
    cm_clipping_range = camera.pop("clipping_range", camera.pop("clippingRange", None))
    cm_parallel_scale = camera.pop("parallel_scale", camera.pop("parallelScale", None))
    cm_thickness   = camera.pop("thickness", None)
    cm_view_angle  = camera.pop("view_angle", camera.pop("viewAngle", None))
    cm_roll        = camera.pop("roll", None)

    if len(camera.keys()) > 0:
        vedo.logger.warning(f"in camera_from_dict, key(s) not recognized: {camera.keys()}")
    if cm_pos is not None:            vcam.SetPosition(cm_pos)
    if cm_focal_point is not None:    vcam.SetFocalPoint(cm_focal_point)
    if cm_viewup is not None:         vcam.SetViewUp(cm_viewup)
    if cm_distance is not None:       vcam.SetDistance(cm_distance)
    if cm_clipping_range is not None: vcam.SetClippingRange(cm_clipping_range)
    if cm_parallel_scale is not None: vcam.SetParallelScale(cm_parallel_scale)
    if cm_thickness is not None:      vcam.SetThickness(cm_thickness)
    if cm_view_angle is not None:     vcam.SetViewAngle(cm_view_angle)
    if cm_roll is not None:           vcam.SetRoll(cm_roll)
    return vcam

def camera_to_dict(vtkcam):
    """
    Convert a [vtkCamera](https://vtk.org/doc/nightly/html/classvtkCamera.html)
    object into a python dictionary.

    Parameters of the camera are:
        - `position` (3-tuple)
        - `focal_point` (3-tuple)
        - `viewup` (3-tuple)
        - `distance` (float)
        - `clipping_range` (2-tuple)
        - `parallel_scale` (float)
        - `thickness` (float)
        - `view_angle` (float)
        - `roll` (float)

    Arguments:
        vtkcam: (vtkCamera)
            a `vtkCamera` object to convert.
    """
    cam = dict()
    cam["position"] = np.array(vtkcam.GetPosition())
    cam["focal_point"] = np.array(vtkcam.GetFocalPoint())
    cam["viewup"] = np.array(vtkcam.GetViewUp())
    cam["distance"] = vtkcam.GetDistance()
    cam["clipping_range"] = np.array(vtkcam.GetClippingRange())
    cam["parallel_scale"] = vtkcam.GetParallelScale()
    cam["thickness"] = vtkcam.GetThickness()
    cam["view_angle"] = vtkcam.GetViewAngle()
    cam["roll"] = vtkcam.GetRoll()
    return cam


def vtkCameraToK3D(vtkcam):
    """
    Convert a `vtkCamera` object into a 9-element list to be used by the K3D backend.

    Output format is:
        `[posx,posy,posz, targetx,targety,targetz, upx,upy,upz]`.
    """
    cpos = np.array(vtkcam.GetPosition())
    kam = [cpos.tolist()]
    kam.append(vtkcam.GetFocalPoint())
    kam.append(vtkcam.GetViewUp())
    return np.array(kam).ravel()


def make_ticks(x0, x1, n=None, labels=None, digits=None, logscale=False, useformat=""):
    """
    Generate numeric labels for the `[x0, x1]` range.

    The format specifier could be expressed in the format:
        `:[[fill]align][sign][#][0][width][,][.precision][type]`

    where, the options are:
    ```
    fill        =  any character
    align       =  < | > | = | ^
    sign        =  + | - | " "
    width       =  integer
    precision   =  integer
    type        =  b | c | d | e | E | f | F | g | G | n | o | s | x | X | %
    ```

    E.g.: useformat=":.2f"
    """
    # Copyright M. Musy, 2021, license: MIT.
    #
    # useformat eg: ":.2f", check out:
    # https://mkaz.blog/code/python-string-format-cookbook/
    # https://www.programiz.com/python-programming/methods/built-in/format

    if x1 <= x0:
        # vedo.printc("Error in make_ticks(): x0 >= x1", x0,x1, c='r')
        return np.array([0.0, 1.0]), ["", ""]

    ticks_str, ticks_float = [], []
    baseline = (1, 2, 5, 10, 20, 50)

    if logscale:
        if x0 <= 0 or x1 <= 0:
            vedo.logger.error("make_ticks: zero or negative range with log scale.")
            raise RuntimeError
        if n is None:
            n = int(abs(np.log10(x1) - np.log10(x0))) + 1
        x0, x1 = np.log10([x0, x1])

    if not n:
        n = 5

    if labels is not None:
        # user is passing custom labels

        ticks_float.append(0)
        ticks_str.append("")
        for tp, ts in labels:
            if tp == x1:
                continue
            ticks_str.append(str(ts))
            tickn = lin_interpolate(tp, [x0, x1], [0, 1])
            ticks_float.append(tickn)

    else:
        # ..here comes one of the shortest and most painful pieces of code:
        # automatically choose the best natural axis subdivision based on multiples of 1,2,5
        dstep = (x1 - x0) / n  # desired step size, begin of the nightmare

        basestep = pow(10, np.floor(np.log10(dstep)))
        steps = np.array([basestep * i for i in baseline])
        idx = (np.abs(steps - dstep)).argmin()
        s = steps[idx]  # chosen step size

        low_bound, up_bound = 0, 0
        if x0 < 0:
            low_bound = -pow(10, np.ceil(np.log10(-x0)))
        if x1 > 0:
            up_bound = pow(10, np.ceil(np.log10(x1)))

        if low_bound < 0:
            if up_bound < 0:
                negaxis = np.arange(low_bound, int(up_bound / s) * s)
            else:
                if -low_bound / s > 1.0e06:
                    return np.array([0.0, 1.0]), ["", ""]
                negaxis = np.arange(low_bound, 0, s)
        else:
            negaxis = np.array([])

        if up_bound > 0:
            if low_bound > 0:
                posaxis = np.arange(int(low_bound / s) * s, up_bound, s)
            else:
                if up_bound / s > 1.0e06:
                    return np.array([0.0, 1.0]), ["", ""]
                posaxis = np.arange(0, up_bound, s)
        else:
            posaxis = np.array([])

        fulaxis = np.unique(np.clip(np.concatenate([negaxis, posaxis]), x0, x1))
        # end of the nightmare

        if useformat:
            sf = "{" + f"{useformat}" + "}"
            sas = ""
            for x in fulaxis:
                sas += sf.format(x) + " "
        elif digits is None:
            np.set_printoptions(suppress=True)  # avoid zero precision
            sas = str(fulaxis).replace("[", "").replace("]", "")
            sas = sas.replace(".e", "e").replace("e+0", "e+").replace("e-0", "e-")
            np.set_printoptions(suppress=None)  # set back to default
        else:
            sas = precision(fulaxis, digits, vrange=(x0, x1))
            sas = sas.replace("[", "").replace("]", "").replace(")", "").replace(",", "")

        sas2 = []
        for s in sas.split():
            if s.endswith("."):
                s = s[:-1]
            if s == "-0":
                s = "0"
            if digits is not None and "e" in s:
                s += " "  # add space to terminate modifiers
            sas2.append(s)

        for ts, tp in zip(sas2, fulaxis):
            if tp == x1:
                continue
            tickn = lin_interpolate(tp, [x0, x1], [0, 1])
            ticks_float.append(tickn)
            if logscale:
                val = np.power(10, tp)
                if useformat:
                    sf = "{" + f"{useformat}" + "}"
                    ticks_str.append(sf.format(val))
                else:
                    if val >= 10:
                        val = int(val + 0.5)
                    else:
                        val = round_to_digit(val, 2)
                    ticks_str.append(str(val))
            else:
                ticks_str.append(ts)

    ticks_str.append("")
    ticks_float.append(1)
    ticks_float = np.array(ticks_float)
    return ticks_float, ticks_str


def grid_corners(i, nm, size, margin=0, yflip=True):
    """
    Compute the 2 corners coordinates of the i-th box in a grid of shape n*m.
    The top-left square is square number 1.

    Arguments:
        i : (int)
            input index of the desired grid square (to be used in `show(..., at=...)`).
        nm : (list)
            grid shape as (n,m).
        size : (list)
            total size of the grid along x and y.
        margin : (float)
            keep a small margin between boxes.
        yflip : (bool)
            y-coordinate points downwards

    Returns:
        Two 2D points representing the bottom-left corner and the top-right corner
        of the `i`-nth box in the grid.

    Example:
        ```python
        from vedo import *
        acts=[]
        n,m = 5,7
        for i in range(1, n*m + 1):
            c1,c2 = utils.grid_corners(i, [n,m], [1,1], 0.01)
            t = Text3D(i, (c1+c2)/2, c='k', s=0.02, justify='center').z(0.01)
            r = Rectangle(c1, c2, c=i)
            acts += [t,r]
        show(acts, axes=1).close()
        ```
        ![](https://vedo.embl.es/images/feats/grid_corners.png)
    """
    i -= 1
    n, m = nm
    sx, sy = size
    dx, dy = sx / n, sy / m
    nx = i % n
    ny = int((i - nx) / n)
    if yflip:
        ny = n - ny
    c1 = (dx * nx + margin, dy * ny + margin)
    c2 = (dx * (nx + 1) - margin, dy * (ny + 1) - margin)
    return np.array(c1), np.array(c2)


############################################################################
# Trimesh support
#
# Install trimesh with:
#
#    sudo apt install python3-rtree
#    pip install rtree shapely
#    conda install trimesh
#
# Check the example gallery in: examples/other/trimesh>
###########################################################################
def vedo2trimesh(mesh):
    """
    Convert `vedo.mesh.Mesh` to `Trimesh.Mesh` object.
    """
    if is_sequence(mesh):
        tms = []
        for a in mesh:
            tms.append(vedo2trimesh(a))
        return tms

    try:
        from trimesh import Trimesh
    except ModuleNotFoundError:
        vedo.logger.error("Need trimesh to run:\npip install trimesh")
        return None

    tris = mesh.cells
    carr = mesh.celldata["CellIndividualColors"]
    ccols = carr

    points = mesh.vertices
    varr = mesh.pointdata["VertexColors"]
    vcols = varr

    if len(tris) == 0:
        tris = None

    return Trimesh(vertices=points, faces=tris, face_colors=ccols, vertex_colors=vcols)


def trimesh2vedo(inputobj):
    """
    Convert a `Trimesh` object to `vedo.Mesh` or `vedo.Assembly` object.
    """
    if is_sequence(inputobj):
        vms = []
        for ob in inputobj:
            vms.append(trimesh2vedo(ob))
        return vms

    inputobj_type = str(type(inputobj))

    if "Trimesh" in inputobj_type or "primitives" in inputobj_type:
        faces = inputobj.faces
        poly = buildPolyData(inputobj.vertices, faces)
        tact = vedo.Mesh(poly)
        if inputobj.visual.kind == "face":
            trim_c = inputobj.visual.face_colors
        elif inputobj.visual.kind == "texture":
            trim_c = inputobj.visual.to_color().vertex_colors
        else:
            trim_c = inputobj.visual.vertex_colors

        if is_sequence(trim_c):
            if is_sequence(trim_c[0]):
                same_color = len(np.unique(trim_c, axis=0)) < 2  # all vtxs have same color

                if same_color:
                    tact.c(trim_c[0, [0, 1, 2]]).alpha(trim_c[0, 3])
                else:
                    if inputobj.visual.kind == "face":
                        tact.cellcolors = trim_c
        return tact

    if "PointCloud" in inputobj_type:

        vdpts = vedo.shapes.Points(inputobj.vertices, r=8, c='k')
        if hasattr(inputobj, "vertices_color"):
            vcols = (inputobj.vertices_color * 1).astype(np.uint8)
            vdpts.pointcolors = vcols
        return vdpts

    if "path" in inputobj_type:

        lines = []
        for e in inputobj.entities:
            # print('trimesh entity', e.to_dict())
            l = vedo.shapes.Line(inputobj.vertices[e.points], c="k", lw=2)
            lines.append(l)
        return vedo.Assembly(lines)

    return None


def vedo2meshlab(vmesh):
    """Convert a `vedo.Mesh` to a Meshlab object."""
    try:
        import pymeshlab as mlab
    except ModuleNotFoundError:
        vedo.logger.error("Need pymeshlab to run:\npip install pymeshlab")

    vertex_matrix = vmesh.vertices.astype(np.float64)

    try:
        face_matrix = np.asarray(vmesh.cells, dtype=np.float64)
    except:
        print("WARNING: in vedo2meshlab(), need to triangulate mesh first!")
        face_matrix = np.array(vmesh.clone().triangulate().cells, dtype=np.float64)

    # v_normals_matrix = vmesh.normals(cells=False, recompute=False)
    v_normals_matrix = vmesh.vertex_normals
    if not v_normals_matrix.shape[0]:
        v_normals_matrix = np.empty((0, 3), dtype=np.float64)

    # f_normals_matrix = vmesh.normals(cells=True, recompute=False)
    f_normals_matrix = vmesh.cell_normals
    if not f_normals_matrix.shape[0]:
        f_normals_matrix = np.empty((0, 3), dtype=np.float64)

    v_color_matrix = vmesh.pointdata["RGBA"]
    if v_color_matrix is None:
        v_color_matrix = np.empty((0, 4), dtype=np.float64)
    else:
        v_color_matrix = v_color_matrix.astype(np.float64) / 255
        if v_color_matrix.shape[1] == 3:
            v_color_matrix = np.c_[
                v_color_matrix, np.ones(v_color_matrix.shape[0], dtype=np.float64)
            ]

    f_color_matrix = vmesh.celldata["RGBA"]
    if f_color_matrix is None:
        f_color_matrix = np.empty((0, 4), dtype=np.float64)
    else:
        f_color_matrix = f_color_matrix.astype(np.float64) / 255
        if f_color_matrix.shape[1] == 3:
            f_color_matrix = np.c_[
                f_color_matrix, np.ones(f_color_matrix.shape[0], dtype=np.float64)
            ]

    m = mlab.Mesh(
        vertex_matrix=vertex_matrix,
        face_matrix=face_matrix,
        v_normals_matrix=v_normals_matrix,
        f_normals_matrix=f_normals_matrix,
        v_color_matrix=v_color_matrix,
        f_color_matrix=f_color_matrix,
    )

    for k in vmesh.pointdata.keys():
        data = vmesh.pointdata[k]
        if data is not None:
            if data.ndim == 1:  # scalar
                m.add_vertex_custom_scalar_attribute(data.astype(np.float64), k)
            elif data.ndim == 2:  # vectorial data
                if "tcoord" not in k.lower() and k not in ["Normals", "TextureCoordinates"]:
                    m.add_vertex_custom_point_attribute(data.astype(np.float64), k)

    for k in vmesh.celldata.keys():
        data = vmesh.celldata[k]
        if data is not None:
            if data.ndim == 1:  # scalar
                m.add_face_custom_scalar_attribute(data.astype(np.float64), k)
            elif data.ndim == 2 and k != "Normals":  # vectorial data
                m.add_face_custom_point_attribute(data.astype(np.float64), k)

    m.update_bounding_box()
    return m


def meshlab2vedo(mmesh, pointdata_keys=(), celldata_keys=()):
    """Convert a Meshlab object to `vedo.Mesh`."""
    inputtype = str(type(mmesh))

    if "MeshSet" in inputtype:
        mmesh = mmesh.current_mesh()

    mpoints, mcells = mmesh.vertex_matrix(), mmesh.face_matrix()
    if len(mcells) > 0:
        polydata = buildPolyData(mpoints, mcells)
    else:
        polydata = buildPolyData(mpoints, None)

    if mmesh.has_vertex_scalar():
        parr = mmesh.vertex_scalar_array()
        parr_vtk = numpy_to_vtk(parr)
        parr_vtk.SetName("MeshLabScalars")
        polydata.GetPointData().AddArray(parr_vtk)
        polydata.GetPointData().SetActiveScalars("MeshLabScalars")

    if mmesh.has_face_scalar():
        carr = mmesh.face_scalar_array()
        carr_vtk = numpy_to_vtk(carr)
        carr_vtk.SetName("MeshLabScalars")
        x0, x1 = carr_vtk.GetRange()
        polydata.GetCellData().AddArray(carr_vtk)
        polydata.GetCellData().SetActiveScalars("MeshLabScalars")

    for k in pointdata_keys:
        parr = mmesh.vertex_custom_scalar_attribute_array(k)
        parr_vtk = numpy_to_vtk(parr)
        parr_vtk.SetName(k)
        polydata.GetPointData().AddArray(parr_vtk)
        polydata.GetPointData().SetActiveScalars(k)

    for k in celldata_keys:
        carr = mmesh.face_custom_scalar_attribute_array(k)
        carr_vtk = numpy_to_vtk(carr)
        carr_vtk.SetName(k)
        polydata.GetCellData().AddArray(carr_vtk)
        polydata.GetCellData().SetActiveScalars(k)

    pnorms = mmesh.vertex_normal_matrix()
    if len(pnorms) > 0:
        polydata.GetPointData().SetNormals(numpy2vtk(pnorms, name="Normals"))

    cnorms = mmesh.face_normal_matrix()
    if len(cnorms) > 0:
        polydata.GetCellData().SetNormals(numpy2vtk(cnorms, name="Normals"))
    return vedo.Mesh(polydata)


def open3d2vedo(o3d_mesh):
    """Convert `open3d.geometry.TriangleMesh` to a `vedo.Mesh`."""
    m = vedo.Mesh([np.array(o3d_mesh.vertices), np.array(o3d_mesh.triangles)])
    # TODO: could also check whether normals and color are present in
    # order to port with the above vertices/faces
    return m


def vedo2open3d(vedo_mesh):
    """
    Return an `open3d.geometry.TriangleMesh` version of the current mesh.
    """
    try:
        import open3d as o3d
    except RuntimeError:
        vedo.logger.error("Need open3d to run:\npip install open3d")

    # create from numpy arrays
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vedo_mesh.vertices),
        triangles=o3d.utility.Vector3iVector(vedo_mesh.cells),
    )
    # TODO: need to add some if check here in case color and normals
    #  info are not existing
    # o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vedo_mesh.pointdata["RGB"]/255)
    # o3d_mesh.vertex_normals= o3d.utility.Vector3dVector(vedo_mesh.pointdata["Normals"])
    return o3d_mesh

def vedo2madcad(vedo_mesh):
    """
    Convert a `vedo.Mesh` to a `madcad.Mesh`.
    """
    try:
        import madcad
        import numbers
    except ModuleNotFoundError:
        vedo.logger.error("Need madcad to run:\npip install pymadcad")

    points = [madcad.vec3(*pt) for pt in vedo_mesh.vertices]
    faces = [madcad.vec3(*fc) for fc in vedo_mesh.cells]

    options = {}
    for key, val in vedo_mesh.pointdata.items():
        vec_type = f"vec{val.shape[-1]}"
        is_float = np.issubdtype(val.dtype, np.floating)
        madcad_dtype = getattr(madcad, f"f{vec_type}" if is_float else vec_type)
        options[key] = [madcad_dtype(v) for v in val]

    madcad_mesh = madcad.Mesh(points=points, faces=faces, options=options)

    return madcad_mesh


def madcad2vedo(madcad_mesh):
    """
    Convert a `madcad.Mesh` to a `vedo.Mesh`.

    A pointdata or celldata array named "tracks" is added to the output mesh, indicating
    the mesh region each point belongs to.

    A metadata array named "madcad_groups" is added to the output mesh, indicating
    the mesh groups.

    See [pymadcad website](https://pymadcad.readthedocs.io/en/latest/index.html)
    for more info.
    """
    try:
        madcad_mesh = madcad_mesh["part"]
    except:
        pass

    madp = []
    for p in madcad_mesh.points:
        madp.append([float(p[0]), float(p[1]), float(p[2])])
    madp = np.array(madp)

    madf = []
    try:
        for f in madcad_mesh.faces:
            madf.append([int(f[0]), int(f[1]), int(f[2])])
        madf = np.array(madf).astype(np.uint16)
    except AttributeError:
        # print("no faces")
        pass

    made = []
    try:
        edges = madcad_mesh.edges
        for e in edges:
            made.append([int(e[0]), int(e[1])])
        made = np.array(made).astype(np.uint16)
    except (AttributeError, TypeError):
        # print("no edges")
        pass

    try:
        line = np.array(madcad_mesh.indices).astype(np.uint16)
        made.append(line)
    except AttributeError:
        # print("no indices")
        pass

    madt = []
    try:
        for t in madcad_mesh.tracks:
            madt.append(int(t))
        madt = np.array(madt).astype(np.uint16)
    except AttributeError:
        # print("no tracks")
        pass

    ###############################
    poly = vedo.utils.buildPolyData(madp, madf, made)
    if len(madf) == 0 and len(made) == 0:
        m = vedo.Points(poly)
    else:
        m = vedo.Mesh(poly)

    if len(madt) == len(madf):
        m.celldata["tracks"] = madt
        maxt = np.max(madt)
        m.mapper.SetScalarRange(0, np.max(madt))
        if maxt==0: m.mapper.SetScalarVisibility(0)
    elif len(madt) == len(madp):
        m.pointdata["tracks"] = madt
        maxt = np.max(madt)
        m.mapper.SetScalarRange(0, maxt)
        if maxt==0: m.mapper.SetScalarVisibility(0)

    try:
        m.info["madcad_groups"] = madcad_mesh.groups
    except AttributeError:
        # print("no groups")
        pass

    try:
        options = dict(madcad_mesh.options)
        if "display_wire" in options and options["display_wire"]:
            m.lw(1).lc(madcad_mesh.c())
        if "display_faces" in options and not options["display_faces"]:
            m.alpha(0.2)
        if "color" in options:
            m.c(options["color"])

        for key, val in options.items():
            m.pointdata[key] = val

    except AttributeError:
        # print("no options")
        pass

    return m


def vtk_version_at_least(major, minor=0, build=0):
    """
    Check the installed VTK version.

    Return `True` if the requested VTK version is greater or equal to the actual VTK version.

    Arguments:
        major : (int)
            Major version.
        minor : (int)
            Minor version.
        build : (int)
            Build version.
    """
    needed_version = 10000000000 * int(major) + 100000000 * int(minor) + int(build)
    try:
        vtk_version_number = vtki.VTK_VERSION_NUMBER
    except AttributeError:  # as error:
        ver = vtki.vtkVersion()
        vtk_version_number = (
            10000000000 * ver.GetVTKMajorVersion()
            + 100000000 * ver.GetVTKMinorVersion()
            + ver.GetVTKBuildVersion()
        )
    return vtk_version_number >= needed_version


def ctf2lut(vol, logscale=False):
    """Internal use."""
    # build LUT from a color transfer function for tmesh or volume

    ctf = vol.properties.GetRGBTransferFunction()
    otf = vol.properties.GetScalarOpacity()
    x0, x1 = vol.dataset.GetScalarRange()
    cols, alphas = [], []
    for x in np.linspace(x0, x1, 256):
        cols.append(ctf.GetColor(x))
        alphas.append(otf.GetValue(x))

    if logscale:
        lut = vtki.vtkLogLookupTable()
    else:
        lut = vtki.vtkLookupTable()

    lut.SetRange(x0, x1)
    lut.SetNumberOfTableValues(len(cols))
    for i, col in enumerate(cols):
        r, g, b = col
        lut.SetTableValue(i, r, g, b, alphas[i])
    lut.Build()
    return lut


def get_vtk_name_event(name):
    """
    Return the name of a VTK event.

    Frequently used events are:
    - KeyPress, KeyRelease: listen to keyboard events
    - LeftButtonPress, LeftButtonRelease: listen to mouse clicks
    - MiddleButtonPress, MiddleButtonRelease
    - RightButtonPress, RightButtonRelease
    - MouseMove: listen to mouse pointer changing position
    - MouseWheelForward, MouseWheelBackward
    - Enter, Leave: listen to mouse entering or leaving the window
    - Pick, StartPick, EndPick: listen to object picking
    - ResetCamera, ResetCameraClippingRange
    - Error, Warning, Char, Timer

    Check the complete list of events here:
    https://vtk.org/doc/nightly/html/classvtkCommand.html
    """
    # as vtk names are ugly and difficult to remember:
    ln = name.lower()
    if "click" in ln or "button" in ln:
        event_name = "LeftButtonPress"
        if "right" in ln:
            event_name = "RightButtonPress"
        elif "mid" in ln:
            event_name = "MiddleButtonPress"
        if "release" in ln:
            event_name = event_name.replace("Press", "Release")
    else:
        event_name = name
        if "key" in ln:
            if "release" in ln:
                event_name = "KeyRelease"
            else:
                event_name = "KeyPress"

    if ("mouse" in ln and "mov" in ln) or "over" in ln:
        event_name = "MouseMove"

    words = [
        "pick", "timer", "reset", "enter", "leave", "char",
        "error", "warning", "start", "end", "wheel", "clipping",
        "range", "camera", "render", "interaction", "modified",
    ]
    for w in words:
        if w in ln:
            event_name = event_name.replace(w, w.capitalize())

    event_name = event_name.replace("REnd ", "Rend")
    event_name = event_name.replace("the ", "")
    event_name = event_name.replace(" of ", "").replace(" ", "")

    if not event_name.endswith("Event"):
        event_name += "Event"

    if vtki.vtkCommand.GetEventIdFromString(event_name) == 0:
        vedo.printc(
            f"Error: '{name}' is not a valid event name.", c='r')
        vedo.printc("Check the list of events here:", c='r')
        vedo.printc("\thttps://vtk.org/doc/nightly/html/classvtkCommand.html", c='r')
        # raise RuntimeError

    return event_name

