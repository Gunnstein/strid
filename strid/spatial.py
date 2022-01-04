# -*- coding: utf-8 -*-
import abc
import copy
import itertools
import numpy as np
import matplotlib
import matplotlib.animation
import matplotlib.colors
import matplotlib.pyplot as plt


def isometric_projection(X, vertical_axis="z", view_from_octant="++"):
    """Isometric projection of coordinates from 3D to 2D

    Isometric projection of coordinate X=(x, y, z) in 3D space to 2D
    space X'=(x', y').

    Arguments
    ---------
    X : 2darray
        (nx3) array with x, y and z coordinates to be projected onto
        the isometric view.
    vertical_axis : {'x', 'y', 'z'}, optional
        Vertical axis in projected coordinates.
    view_from_octant : {'++', '+-', '-+', '--'}, optional
        Octant the coordinate system is viewed from. First character
        denotes the first axis (x,y,z) that is not the vertical axis.

    Returns
    -------
    x : 2darray
        (nx2) array with the x, y coordinates of the isometric
        viewplane.

    """
    c = np.cos
    s = np.sin
    Rx = lambda t: np.array([
        [1., 0., 0.],
        [0., c(t), -s(t)],
        [0., s(t), c(t)],
    ])
    Ry = lambda t: np.array([
        [c(t), 0., s(t)],
        [0., 1., 0.],
        [-s(t), 0., c(t)],
    ])
    Rz = lambda t: np.array([
        [c(t), -s(t), 0.],
        [s(t), c(t), 0.],
        [0., 0., 1.],
    ])

    a = np.arcsin(np.tan(np.pi/6.))
    b0 = np.pi / 4.
    b = {
        "z": {
            "++": b0+np.pi,
            "+-": b0+np.pi*3/2,
            "-+": b0-np.pi*3/2,
            "--": b0,},
        "y": {
            "++": b0+np.pi*3/2,
            "+-": b0+np.pi,
            "-+": b0,
            "--": b0-np.pi*3/2,},
        "x": {
            "++": b0+np.pi,
            "+-": b0+np.pi*3/2,
            "-+": b0-np.pi*3/2,
            "--": b0,},
    }[vertical_axis][view_from_octant]

    a_axis = {"x": "y", "y": "x", "z": "x"}[vertical_axis]
    Ra = {"x": Rx, "y": Ry, "z": Rz}[a_axis](a)
    Rb = {"x": Rx, "y": Ry, "z": Rz}[vertical_axis](b)

    p = {"x": [1., 0., 0.],
         "y": [0., 1., 0.],
         "z": [0., 0., 1.],}
    P = np.array([
        p[a_axis],
        p[vertical_axis]])

    return ((P @ Ra @ Rb) @ X.T).T


class AbstractVisualizer(abc.ABC):
    """Visualize spatial models and meshes with matplotlib.

    This class provides functionality to plot and animate modes with
    matplotlib, see `plot`, `plot_elements`, `plot_nodenumbers` and
    `animate` methods.

    The "view" or camera placement of the spatial model is defined
    by the `projection` method. Concrete subclasses must implement
    `projection` method.

    """
    def __init__(self, model, axes=None):
        """Create a visualizer of a mesh or a spatial model.

        Arguments
        ---------
        model : {strid.spatial.Mesh, strid.spatial.SpatialModel}
            Mesh or SpatialModel to visualize. Note, functionality of
            methods are limited when initialized with Mesh.
        axes : matplotlib.axes.Axes, optional
            Axes to visualize on.

        """
        self.model = model

        if axes is None:
            _, self.axes = plt.subplots()
            self.axes.axis('off')
            self.axes.set_aspect('equal')
        else:
            self.axes = axes

    @property
    def figure(self):
        return self.axes.figure

    @abc.abstractmethod
    def projection(self, X):
        """Projection from coordinates in 3D to 2D.

        This method projects coordinates
        :math:`X\\in\\mathbb{R}^{n\\times 3}` (3D space) to coordinates
        :math:`X'\\in\\mathbb{R}^{n\\times 2}` (2D space).

        Arguments
        ---------
        X : 2darray
            (nx3) array with x, y and z coordinates to
            be projected.

        Returns
        -------
        2darray
            (nx2) array with the projected x', y' coordinates.

        """
        pass

    def plot_elements(self, mode=None, amplitude=1.0, colormap=None,
                      **kwargs):
        """Plot the elements of the spatial model.

        Plot the elements of the spatial model, if mode is `None`, the
        undeformed model is plotted.

        Arguments
        ---------
        mode : strid.Mode, optional
            Plot the deformed spatial model as defined by the mode.
        amplitude : float or complex, optional
            Amplitude to be applied to the modeshape.
        colormap : str or matplotlib.colors.ColorMap, optional
            Colormap to be applied to the deformation vector. If
            keyword `colors` is given, `colormap` is ignored.
        kwargs : dict, optional
            All keyword arguments are passed to matplotlib.Axes.plot
            method.

        Returns
        -------
        list
            List of matplotlib.artist.Artist instances of the plotted
            elements.

        Raises
        ------
        NotImplementedError
            NotImplementedError is raised if the mesh consists of any
            other elements than LineElements.
        ValueError
            ValueError is raised if the `mode` keyword is used when
            class was initialized with a `Mesh`, must initialize with
            a consistent `SpatialModel` to be able to convert mode
            shape of Mode to node deformation array of mesh.

        """
        if mode is None:
            U = np.zeros_like(self.model.X0)
            if "color" not in kwargs:
                kwargs["color"] = 'k'
        else:
            if isinstance(self.model, Mesh):
                raise ValueError(
                    "Visualizer initialized with Mesh instance, cannot plot mode.")
            U = self.model.U(mode.v*np.exp(-1j*mode.mp))
        if "color" not in kwargs:
            cmap = matplotlib.cm.get_cmap(colormap)
        else:
            cmap = matplotlib.colors.ListedColormap([kwargs["color"]])
        uabs = np.linalg.norm(np.abs(amplitude*U), axis=1)
        umax, umin = uabs.max(), uabs.min()
        u = np.linalg.norm((amplitude*U).real, axis=1)
        u -= umin
        if umax != 0:
            u /= umax
        U = amplitude * U

        artists = []
        X = self.model.X0 + U.real
        for el_type, connectivity in self.model.topology:
            if not el_type == "line":
                raise NotImplementedError(
                    "Only plotting of LineElements implemented.")
            for node_ixs in connectivity:
                color = cmap(u[node_ixs].mean())
                kwargs["color"] = color
                Xc = X[node_ixs]
                x, y = self.projection(Xc).T
                ls, = self.axes.plot(x, y, **kwargs)
                artists.append(ls)
        return artists

    def plot_nodes(self, **kwargs):
        """Plot the nodes of the spatial model on the axes.

        Arguments
        ---------
        kwargs : dict, optional
            All keyword arguments are passed to matplotlib.Axes.plot
            method.

        """
        if "color" not in kwargs:
            kwargs["color"] = 'k'
        artists = []
        for i, Xi in enumerate(self.model.X0):
            x, y = self.projection(Xi).T
            ap, = self.axes.plot(x, y, **kwargs)
            artists.append(ap)
        return artists

    def plot_nodenumbers(self, **kwargs):
        """Plot the nodenumbers of the spatial model on the axes.

        Arguments
        ---------
        kwargs : dict, optional
            All keyword arguments are passed to matplotlib.Axes.text
            method.

        """
        artists = []
        for i, Xi in enumerate(self.model.X0):
            x, y = self.projection(Xi).T
            at = self.axes.text(x, y, f"{i}", **kwargs)
            artists.append(at)
        return artists

    def plot(self, kw_els={}, kw_nodes={}, kw_nodenums={}):
        """Plot the undeformed model.

        This method combines the `plot_elements`, `plot_nodes`
        and `plot_nodenumbers` methods.

        Arguments
        ---------
        kw_els, kw_nodes, kw_nodenums : dict, optional
            Keyword arguments passed to `plot_elements`, `plot_nodes`
            and `plot_nodenumbers` methods.

        """
        if ("lw" not in kw_els) and ("linewidth" not in kw_els):
            kw_els["lw"] = .3
        self.plot_elements(**kw_els)
        self.plot_nodes(**kw_nodes)
        self.plot_nodenumbers(**kw_nodenums)


    def animate(self, mode, amplitude=1.0, nframes=16, **kwargs):
        """Animate the mode shape

        Create an animation of the mode shape.

        Arguments
        ---------
        mode : strid.Mode
            Mode shape to create an animation of.
        amplitude : float or complex, optional
            Amplitude to be applied to the modeshape.
        nframes : int, optional
            Number of frames used to animate a full cycle of the modeshape.
        kwargs : dict, optional
            All keyword arguments are passed to matplotlib.Axes.plot
            method.

        Returns
        -------
        matplotlib.animation.Animation
            Instance of Animation class from matplotlib. Important
            methods of this instance are `pause`, `resume` and `save`,
            see matplotlib documentation.

        Raises
        ------
        ValueError
            ValueError is raised if the class was initialized with a
            `Mesh`, must initialize with a consistent `SpatialModel`
            to be able to convert mode shape of Mode to node
            deformation array of mesh and produce animation.

        """
        if isinstance(self.model, Mesh):
            raise ValueError(
                "Visualizer initialized with Mesh instance, cannot animate mode.")
        angle = 2*np.pi*np.arange(nframes)/nframes
        scales = amplitude * np.exp(1j*angle)
        artists = []
        for scale in scales:
            artists.append(
                self.plot_elements(
                    mode=mode, amplitude=scale, animated=True, **kwargs))
        animation = matplotlib.animation.ArtistAnimation(
            self.figure, artists, interval=1000./nframes)
        return animation


class VisualizerIsometric(AbstractVisualizer):
    """Visualize a mesh or spatial model with matplotlib

    This visualizer implements an isometric projection for converting
    the coordinates of the the 3D mesh or spatial model to 2D
    coordinates.

    """
    def projection(self, X):
        return isometric_projection(X)


class VisualizerXY(AbstractVisualizer):
    """Visualize a mesh or spatial model with matplotlib

    This visualizer projects the coordinates of the 3D mesh or spatial
    model to XY plane.

    """
    def projection(self, X):
        P = np.array([[1., 0., 0.], [0., 1., 0.]])
        return (P@X.T).T


class VisualizerXZ(AbstractVisualizer):
    """Visualize a mesh or spatial model with matplotlib

    This visualizer projects the coordinates of the 3D mesh or spatial
    model to XZ plane.

    """
    def projection(self, X):
        P = np.array([[1., 0., 0.], [0., 0., 1.]])
        return (P@X.T).T


class VisualizerYZ(AbstractVisualizer):
    """Visualize a mesh or spatial model with matplotlib

    This visualizer projects the coordinates of the 3D mesh or spatial
    model to YZ plane.

    """
    def projection(self, X):
        P = np.array([[0., 1., 0.], [0., 0., 1.]])
        return (P@X.T).T


class StridError(Exception):
    pass


class DOF:
    """Define a degree of freedom (DOF)

    A degree of freedom has a number that defines the direction
    in 3D space. The numbers are ordered according to the right
    hand rule.


              z, 2                     DOF numbering
                |   y, 1               -------------
                |  /                 - Right hand rule
                | /
                |/
                x--------
                        x, 0


    Typically, the DOF object is not instantiated directly by the
    user, the user instead defines a Node object that is responsible
    for defining the DOFs.

    """
    def __init__(self, number):
        self.number = number
        self.axis = {0:"x", 1:"y", 2:"z"}[number]
        self.vector = np.eye(3)[number]

    def __str__(self):
        return "DOF{axis}".format(**self.__dict__)


class Node:
    """Define a node

    A node has a coordinate and three degrees of freedom.

    This object defines the geometry of a mesh.

    Arguments
    ---------
    coordinate : 1darray
        Coordinate of the node.

    """
    def __init__(self, coordinate):
        self.coordinate = np.asfarray(coordinate)
        self.dofs = [DOF(n) for n in range(3)]


class Element:
    """Elements defines the connectivity of a mesh.

    A element can be a vertex, line, surface (area) or volume and is
    defined by an appropriate number of nodes (e.g. one node for a
    vertex, two for a line, etc).

    """
    def __init__(self, *args):
        self.nodes = args

    @property
    def vtk_type(self):
        return f"{self.__class__.__name__}".replace("Element", "").lower()


class VertexElement(Element):
    pass


class LineElement(Element):
    pass


class TriangleElement(Element):
    pass


class QuadElement(Element):
    pass


class Mesh:
    def __init__(self, nodes=None, elements=None, meshes=None):
        """Define a mesh from nodes, elements and submeshes.

        A mesh describes the discretization of a continous domain
        and is defined by nodes (aka points or vertices) and
        elements (aka cells or zones).

        The nodes define the geometry of the domain and the elements
        define the topology or connectivity between the nodes.

        A mesh can also consist of submeshes, by inheriting the
        nodes and elements from the submeshes.

        Arguments
        ---------
        nodes : Optional[list[Node]]
            List of Node instances. These nodes, together with
            nodes from submeshes, defines the geometry of the
            mesh.
        elements: Optional[list[Element]]
            List of Element instances. These elements, together with
            elements from submeshes, defines the topology of the
            mesh.
        meshes : Optional[list[Mesh]]
            List of Mesh instances. Submeshes, together with nodes
            and elements from this mesh, defines the discretization
            of a continous domain.

        Example
        -------
        This example shows how to create a mesh for a shear frame
        with 5 floors, each floor 3.0 units high and 5.0 units wide.

        >>> nodes = [Node((0., 0., i*3.0)) for i in range(6)]
        >>> elements = [LineElement(nodes[i], nodes[i+1]) for i in range(5)]
        >>> mesh_left_column = Mesh(nodes=nodes, elements=elements)

        We create the right column by creating a copy of the left
        column and translating it 5.0 units in the x-direction.

        >>> mesh_right_column = mesh_left_column.copy()
        >>> mesh_right_column.translate((5., 0., 0.))

        Finally, we create the mesh of the shear frame by using the
        meshes for the left and the right column and connecting the
        nodes from those submeshes with a line element.

        >>> floor_elements = [
        ... LineElement(node_left, node_right) for node_left, node_right in
        ... zip(mesh_left_column.nodes[1:], mesh_right_column.nodes[1:])
        ... ]
        >>> mesh_shear_frame = Mesh(
        ... elements=floor_elements,
        ... meshes=[mesh_left_column, mesh_right_column]
        ... )

        and now we have our mesh for the shear frame. Note that
        the mesh object has several useful methods to help build
        a mesh.

        We can for instance find a node by a coordinate:

        >>> node_at_origin = mesh_shear_frame.find_node_by_coordinate((0., 0., 0.))
        >>> node_at_origin is mesh_left_column.nodes[0]
        True

        or the node number in the mesh from a node object:

        >>> mesh_shear_frame.find_nodenumber_by_node(node_at_origin)
        0

        or the node number by a DOF:

        >>> dof1_at_origin = node_at_origin.dofs[0]
        >>> mesh_shear_frame.find_nodenumber_by_dof(dof1_at_origin)
        0

        """
        self._elements = elements or []
        self._nodes = nodes or []
        self._meshes = meshes or []

    @property
    def nodes(self):
        nodes = []
        for mesh in self.meshes:
            for node in mesh.nodes:
                if node not in nodes:
                    nodes.append(node)
        for element in self.elements:
            for node in element.nodes:
                if node not in nodes:
                    nodes.append(node)
        for node in self._nodes:
            if node not in nodes:
                nodes.append(node)
        return nodes

    @nodes.setter
    def nodes(self, v):
        raise StridError("Add node/nodes with the `add_node`/`add_nodes` method.")

    @property
    def dofs(self):
        dofs = []
        for node in self.nodes:
            dofs.extend(node.dofs)
        return dofs

    @property
    def elements(self):
        elements = []
        for mesh in self.meshes:
            for element in mesh.elements:
                if element not in elements:
                    elements.append(element)
        for element in self._elements:
            if element not in elements:
                elements.append(element)
        return elements

    @elements.setter
    def elements(self, v):
        raise StridError(
            "Add element/elements with the `add_element`/`add_elements` method.")

    @property
    def meshes(self):
        return self._meshes

    @meshes.setter
    def meshes(self, v):
        raise StridError("Add mesh/meshes with the `add_mesh`/`add_meshes` method.")

    def add_meshes(self, meshlist):
        self._meshes.extend(v)

    def add_mesh(self, mesh):
        self._meshes.append(mesh)

    def add_elements(self, v):
        self._elements.extend(v)

    def add_element(self, v):
        self._elements.append(v)

    def add_node(self, v):
        self._nodes.append(v)

    def add_nodes(self, v):
        self._nodes.extend(v)

    def translate(self, v):
        u = np.asfarray(v)
        for node in self.nodes:
            node.coordinate += u

    def find_node_by_number(self, number):
        for i, node in enumerate(self.nodes):
            if i == number:
                return node
        return None

    def copy(self):
        return copy.deepcopy(self)

    def find_node_by_coordinate(self, coordinate):
        c = np.asfarray(coordinate)
        nodes = [(node, np.linalg.norm(c-node.coordinate))
                 for node in self.nodes]
        nodes.sort(key=lambda x: x[1])
        return nodes[0][0]

    def find_node_by_dof(self, dof):
        for node in self.nodes:
            if dof in node.dofs:
                return node
        return None

    def find_nodenumber_by_node(self, node):
        return {node: i
                for i, node in enumerate(self.nodes)}[node]

    def find_nodenumber_by_dof(self, dof):
        return {dof: i
                for i, node in enumerate(self.nodes)
                for dof in node.dofs}[dof]

    @property
    def X0(self):
        return np.array([node.coordinate for node in self.nodes])

    @property
    def topology(self):
        nodenum_by_node = self.find_nodenumber_by_node
        element_types = set([element.vtk_type for element in self.elements])
        topology = [tuple(
            [element_type,[[nodenum_by_node(n) for n in element.nodes]
                           for element in self.elements
                           if element.vtk_type==element_type]])
            for element_type in element_types]
        return topology


class LinearConstraint:
    """Linear constraint equation between DOFs

    A linear constraint equation between dofs is defined by

    .. math::

        \\sum_{j=0}^{n-1} g_ju_j + c = 0

    where :math:`g_j` are the constraint coefficients, :math:`u_j` is
    the j'th dof of the n dof model and :math:`c` is the constant
    (non-homogeneous) term of the constraint.

    """
    def __init__(self, dofs, coefficients, constant=0.0):
        """Create a linear constraint equation between dofs.

        Arguments
        ---------
        dofs : list[strid.visualization.DOF]
            DOFs involved in the constraint.
        coefficients : 1darray
            Constraint coefficients, must be of same length as `dofs`.
        c : float, optional
            Constant in the constraint equation
        """
        self.dofs = dofs
        self.coefficients = coefficients
        self.constant = constant


class SpatialModel:
    def __init__(self, mesh, sensors, constraints=None):
        """Establish a spatial model of the identified system.

        System identification yields a mode vector `u` which
        defines the response at discrete points of a spatially
        continous system.

        The geometry of this continous system is in a computer
        defined by a mesh. The mesh is a discrete structure consisting
        of nodes and elements, which defines the geometry and topology
        (connectivity) of the mesh, respectively.

        An essential step is therefore to transform the mode vector
        `u` to a mesh deformation array `U` which defines the
        deformation vector of each node. (U[0] = [u, v, w] is the
        deformation vector of the 0th node.)

        This class provides methods `X0`, `U` and `X` to compute
        the initial geometry `X0`, the deformation `U(u)` and the
        final geometry X(u) of the mesh given a mode vector `u` and
        possibly, a set of constraints.

        Arguments
        ---------
        mesh : Mesh
            Mesh of the system.
        sensors : list[DOF]
            List of dofs where sensors are installed and used to
            measure response for system identification.
        constraints : list[LinearConstraint], optional
            List of LinearConstraint objects that defines the deformation
            of non-sensor dofs in terms of the sensor dofs or constrain
            them to a constant value (e.g. zero).

        """
        self.mesh = mesh
        self.sensors = sensors
        self.constraints = constraints or []
        self._eqnums = None

    def copy(self):
        return copy.deepcopy(self)

    def check_consistency(self):
        """Check the consistency of the mesh and constraints.

        Check whether the system is consistent. The dofs `d` of the mesh
        are either independent dofs `d_i`, dependent dofs `d_j` or free
        dofs `d_k`. The sensor dofs are the independent dofs of the system,
        other dofs which are part the constraints are dependent
        dofs and all other dofs are free dofs.

        The model must be properly constrained to be able to compute the
        deformation vectors and final coordinates for the nodes from the
        deformation vector of the sensor dofs. In other words, all dofs
        must either be sensor dofs, constrained to sensor dofs or constrained
        to a predefined value.

            n_independent_dofs + n_dependent_dofs = n_dofs

        Raises
        ------

        StridError
            There are two different cases where this error is raised:
                - Overconstrained case, when a dependent dof is included in
                  several constraints. Remove or change relevant constraint(s).
                - Underconstrained case, when a dof is not a sensor dof
                  (independent) nor a dependent dof. Add constraint with
                  relevant dof.
        """
        nodenum_by_node = {node: i
                           for i, node in enumerate(self.mesh.nodes)}
        node_by_dof = {dof: node
                       for node in self.mesh.nodes
                       for dof in node.dofs}

        independent_dofs = self.sensors
        err_str_indep = [
            "Constraint error(s), sensor dofs must be in constraint with non-sensor dofs."]
        for constraint in self.constraints:
            if all([dof in independent_dofs for dof in constraint.dofs]):
                for dof in constraint.dofs:
                    node = node_by_dof[dof]
                    nodenum = nodenum_by_node[node]
                    err_str_indep.append(
                        f"\tNode {nodenum}, DOF {dof.number} ({dof.axis}-dir) (sensor dof), constrained to other sensor dof or to predefined value.")

        dependent_dofs = []
        err_str = [
            "Constraint error(s), non-sensor dofs must be present in one and only one constraint equation."]
        for constraint in self.constraints:
            for dof in constraint.dofs:
                if dof in independent_dofs:
                    continue
                elif dof in dependent_dofs:
                    node = node_by_dof[dof]
                    nodenum = nodenum_by_node[node]
                    err_str.append(f"\tNode {nodenum}, DOF {dof.number} ({dof.axis}-dir) present in more than one constraint.")
                else:
                    dependent_dofs.append(dof)

        constraint_dofs = independent_dofs + dependent_dofs
        free_dofs = []
        for dof in self.mesh.dofs:
            if dof in constraint_dofs:
                continue
            free_dofs.append(dof)

        for dof in free_dofs:
            node = node_by_dof[dof]
            nodenum = nodenum_by_node[node]
            err_str.append(f"\tNode {nodenum}, DOF {dof.number} ({dof.axis}-dir) is not present in any constraint.")

        if (len(err_str_indep) > 1) and (len(err_str) > 1):
            errors = err_str_indep + err_str
        elif (len(err_str_indep) > 1) and (len(err_str) == 1):
            errors = err_str_indep
        elif (len(err_str_indep) == 1) and (len(err_str) > 1):
            errors = err_str
        else:
            errors = []
        if len(errors) > 0:
            raise StridError("\n".join(errors))

    def _assign_equation_numbers(self):
        eqnum = itertools.count()
        eqnums = {}
        for dof in self.sensors:
            eqnums[dof] = next(eqnum)
        for dof in self.mesh.dofs:
            if dof in self.sensors:
                continue
            eqnums[dof] = next(eqnum)
        self._eqnums = eqnums

    def _assemble_equations(self):
        self._assign_equation_numbers()
        n = len(self.mesh.dofs)
        m = len(self.constraints)
        o = len(self.sensors)
        G = np.zeros((m, n), dtype=float)
        c = np.zeros(m, dtype=float)
        for i, constraint in enumerate(self.constraints):
            c[i] = constraint.constant
            for gij, dof in zip(constraint.coefficients, constraint.dofs):
                j = self._eqnums[dof]
                G[i, j] = gij
        H = np.linalg.lstsq(G[:, o:], -np.c_[G[:, :o], c], rcond=None)[0]
        T = np.zeros((n, o), dtype=float)
        T[:o, :] = np.eye(o)
        T[o:, :] = H[:, :o]

        s = np.zeros(n, dtype=float)
        s[o:] = H[:, o:].flatten()

        return T, s

    @property
    def X0(self):
        """Undeformed (initial) geometry of the mesh.

        `X0` is the initial geometry of the mesh as defined by the nodes
        of the mesh. Row `i` of `X0` corresponds to the initial coordinate
        of node `i` of the mesh.

        Returns
        -------
        2darray
            n x 3 array with the coordinates of the nodes with an undeformed
            geometry.

        """
        return np.array([node.coordinate for node in self.mesh.nodes])

    def U(self, u):
        """Deformation matrix for deformation vector u.

        Computes the deformation matrix U (nx3) for all `n` dofs of the model
        given the deformation vector u for the `m` sensor dofs of the model.


        Arguments
        ---------
        u : 1darray
            Deformation vector (size m) for sensor dofs of the model.

        Returns
        -------
        U : 2darray
            Deformation array (nx3)

        """
        u = np.asarray(u)
        self._assign_equation_numbers()
        T, s = self._assemble_equations()
        v = T@u + s
        U = np.zeros((len(self.mesh.nodes), 3), dtype=u.dtype)
        nodenums = {node: i for i, node in enumerate(self.mesh.nodes)}
        for dof, eqnum in self._eqnums.items():
            i = nodenums[self.mesh.find_node_by_dof(dof)]
            j = dof.number
            U[i, j] = v[eqnum]
        return U

    def X(self, u):
        """Deformed (final) geometry of the mesh.

        `X` is the final geometry of the mesh given the deformation vector `u`
        for the sensor dofs (independent dofs).
        Row `i` of `X` corresponds to the final coordinate of node `i` of the
        mesh.

        Formally, the deformation matrix `X(u)` is defined as the sum of
        the undeformed geometry `X0` and the deformation `U(u)`

        .. math::

            X(u) = X0 + U(u)

        Arguments
        ---------
        u : 1darray
            Deformation vector (size m) for sensor dofs of the model.

        Returns
        -------
        U : 2darray
            Deformed geometry of (nx3)

        """
        return self.X0 + self.U(u)

    @property
    def topology(self):
        nodenum_by_node = self.mesh.find_nodenumber_by_node
        element_types = set([element.vtk_type for element in self.mesh.elements])
        topology = [tuple(
            [element_type,[[nodenum_by_node(n) for n in element.nodes]
                           for element in self.mesh.elements
                           if element.vtk_type==element_type]])
            for element_type in element_types]
        return topology
