# -*- coding: utf-8 -*-
import numpy as np
import unittest

from .. import *

pr = lambda s: print(repr(s))


class TestDOF(unittest.TestCase):
    def test_axis(self):
        assert spatial.DOF(0).axis == "x"
        assert spatial.DOF(1).axis == "y"
        assert spatial.DOF(2).axis == "z"

    def test_number(self):
        assert spatial.DOF(1).number == 1


class TestNode(unittest.TestCase):
    def setUp(self):
        self.coordinate = np.array([1., -1., .3])
        self.node = spatial.Node(self.coordinate)

    def test_coordinate(self):
        np.testing.assert_almost_equal(
            self.node.coordinate, (1., -1., .3))

    def test_dofnums(self):
        dof_numbers = range(3)
        assert all([dof.number==dof_number
                    for dof, dof_number in zip(self.node.dofs, dof_numbers)])


class TestLineElement(unittest.TestCase):
    def setUp(self):
        self.element = spatial.LineElement(spatial.Node((.0, .0, .0)),
                                           spatial.Node((3., .0, .0)))

    def test_type_vtk(self):
        assert self.element.vtk_type == "line"


class ShearFrameMesh:
    def setUp(self):
        # Left column
        n_floors = 10
        floor_height = 1.0
        floor_width = 5.0

        nodes = [spatial.Node((0., 0., i*floor_height))
                 for i in range(n_floors+1)]
        elements = [spatial.LineElement(nodes[i], nodes[i+1])
                    for i in range(n_floors)]
        mesh_lc = spatial.Mesh(nodes=nodes, elements=elements)
        mesh_rc = mesh_lc.copy()
        mesh_rc.translate((floor_width, 0., 0.))
        floor_elements = [
            spatial.LineElement(n1, n2)
            for n1, n2 in zip(mesh_lc.nodes[1:], mesh_rc.nodes[1:])]
        self.mesh = spatial.Mesh(elements=floor_elements,
                                 meshes=[mesh_lc, mesh_rc])
        self.X0 = np.array([
            [ 0.,  0.,  0.],
            [ 0.,  0.,  1.],
            [ 0.,  0.,  2.],
            [ 0.,  0.,  3.],
            [ 0.,  0.,  4.],
            [ 0.,  0.,  5.],
            [ 0.,  0.,  6.],
            [ 0.,  0.,  7.],
            [ 0.,  0.,  8.],
            [ 0.,  0.,  9.],
            [ 0.,  0., 10.],
            [ 5.,  0.,  0.],
            [ 5.,  0.,  1.],
            [ 5.,  0.,  2.],
            [ 5.,  0.,  3.],
            [ 5.,  0.,  4.],
            [ 5.,  0.,  5.],
            [ 5.,  0.,  6.],
            [ 5.,  0.,  7.],
            [ 5.,  0.,  8.],
            [ 5.,  0.,  9.],
            [ 5.,  0., 10.],])

class TestMesh(ShearFrameMesh, unittest.TestCase):
    def test_X0(self):
        np.testing.assert_almost_equal(self.mesh.X0, self.X0)

    def test_topology(self):
        topology = [['line',
                     [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                      [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                      [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                      [16, 17], [17, 18], [18, 19], [19, 20], [20, 21],
                      [1, 12], [2, 13], [3, 14], [4, 15], [5, 16],
                      [6, 17], [7, 18], [8, 19], [9, 20], [10, 21]]]]

        assert topology[0][0] == self.mesh.topology[0][0]
        assert len(topology[0][1]) == len(self.mesh.topology[0][1])
        assert all([c1==c2
                    for c1, c2 in
                    zip(topology[0][1], self.mesh.topology[0][1])])

    def test_copy(self):
        mesh = self.mesh.copy()
        for n1, n2 in zip(mesh.nodes, self.mesh.nodes):
            np.testing.assert_almost_equal(n1.coordinate, n2.coordinate)
            assert n1 is not n2

        assert mesh.topology[0][0] == self.mesh.topology[0][0]
        assert len(mesh.topology[0][1]) == len(self.mesh.topology[0][1])
        assert all([c1==c2
                    for c1, c2 in
                    zip(mesh.topology[0][1], self.mesh.topology[0][1])])

    def test_translate(self):
        nodes = [spatial.Node((0., 0., 0.)),]
        mesh = spatial.Mesh(nodes=nodes)
        mesh.translate((1., 1., 1.))
        np.testing.assert_almost_equal(
            nodes[0].coordinate, np.array([1., 1., 1.]))

    def test_find_node_by_coordinate(self):
        node0 = self.mesh.nodes[11]
        node = self.mesh.find_node_by_coordinate(node0.coordinate)
        assert node is node0

    def test_find_node_by_dof(self):
        node0 = self.mesh.nodes[3]
        node = self.mesh.find_node_by_dof(node0.dofs[0])
        assert node is node0

    def test_find_node_by_number(self):
        nodenum = 15
        node0 = self.mesh.nodes[nodenum]
        node = self.mesh.find_node_by_number(nodenum)
        assert node is node0

    def test_find_nodenumber_by_node(self):
        nodenum0 = 13
        node0 = self.mesh.nodes[nodenum0]
        nodenum = self.mesh.find_nodenumber_by_node(node0)
        assert nodenum0 == nodenum

    def test_find_nodenumber_by_dof(self):
        nodenum0 = 8
        dof = self.mesh.nodes[nodenum0].dofs[2]
        nodenum = self.mesh.find_nodenumber_by_dof(dof)
        assert nodenum0 == nodenum


class TestLinearConstraint(unittest.TestCase):
    def setUp(self):
        self.dofs = [spatial.DOF(1),]
        self.coefficients = (1.,)
        self.constant = -3.0
        self.constraint = spatial.LinearConstraint(
            self.dofs, self.coefficients, constant=self.constant)

    def test_dofs(self):
        assert self.constraint.dofs[0] == self.dofs[0]

    def test_constant(self):
        assert self.constraint.constant == self.constant

    def test_coefficients(self):
        assert all(
            [g==gt
             for g, gt in zip(
                     self.constraint.coefficients, self.coefficients)])


class TestSpatialModel(ShearFrameMesh, unittest.TestCase):
    def setUp(self):
        super().setUp()

        nodes_lc = self.mesh.meshes[0].nodes
        nodes_rc = self.mesh.meshes[1].nodes
        # Left column dofs in xdir above z=0.0 are sensor dofs
        sensors = [node.dofs[0] for node in nodes_lc[1:]]
        self.sensors = sensors

        constraints_y = []
        for node in self.mesh.nodes:
            constraint = spatial.LinearConstraint(
                [node.dofs[1],], [1.,], constant=0.)
            constraints_y.append(constraint)

        constraints_z = []
        for node in self.mesh.nodes:
            constraint = spatial.LinearConstraint(
                [node.dofs[2],], [1.,], constant=0.)
            constraints_z.append(constraint)

        constraints_x = []
        # BC x-direction
        for dof in (nodes_lc[0].dofs[0], nodes_rc[0].dofs[0]):
            constraint = spatial.LinearConstraint(
                [dof,], [1.], constant=0.)
            constraints_x.append(constraint)
        # Constrain right column dofs in xdir to corresponding left
        # column dofs in xdir (sensor dofs).
        for node_left, node_right in zip(nodes_lc[1:], nodes_rc[1:]):
            constraint = spatial.LinearConstraint(
                [node_left.dofs[0], node_right.dofs[0]],
                [1., -1.],
                constant=0)
            constraints_x.append(constraint)
        constraints = constraints_x + constraints_y + constraints_z
        self.constraints = constraints
        self.model = spatial.SpatialModel(self.mesh, sensors, constraints)
        self.model.check_consistency()

    def test_check_consistency(self):
        self.model.check_consistency()

    def test_check_consistency_underconstrained(self):
        model = self.model.copy()
        model.constraints = model.constraints[:-1]
        try:
            model.check_consistency()
        except spatial.StridError:
            pass
        except Exception:
            self.fail("Unexpected exception raised by consistency check")
        else:
            self.fail("Expected error not raised, StridError expected.")

    def test_check_consistency_overconstrained(self):
        model = self.model.copy()
        dof = model.mesh.nodes[0].dofs[0]
        constraint = spatial.LinearConstraint(
            [dof,], [1.,], constant=0.)
        model.constraints.append(constraint)
        try:
            model.check_consistency()
        except spatial.StridError:
            pass
        except Exception:
            self.fail("Unexpected exception raised by consistency check")
        else:
            self.fail("Expected error not raised, StridError expected.")


    def test_check_consistency_sensordof_constrained(self):
        model = self.model.copy()
        dof = model.mesh.nodes[1].dofs[0]
        constraint = spatial.LinearConstraint(
            [dof,], [1.,], constant=0.)
        model.constraints.append(constraint)
        try:
            model.check_consistency()
        except spatial.StridError:
            pass
        except Exception as e:
            self.fail(f"Unexpected exception raised by consistency check: {e}")
        else:
            self.fail("Expected error not raised, StridError expected.")

    def test_X0(self):
        np.testing.assert_almost_equal(self.model.X0, self.X0)

    def test_U(self):
        u = np.arange(1, 11.)
        U = np.zeros_like(self.X0)
        U[1:u.size+1, 0] = u
        U[u.size+2:, 0] = u
        np.testing.assert_almost_equal(self.model.U(u), U)

    def test_X(self):
        u = np.arange(1., 11.)
        X0 = self.model.X0
        U = np.zeros_like(X0)
        U[1:u.size+1, 0] = u
        U[u.size+2:, 0] = u
        X = X0 + U
        np.testing.assert_almost_equal(self.model.X(u), X)

    def test_topology(self):
        topology = [['line',
                     [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                      [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                      [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                      [16, 17], [17, 18], [18, 19], [19, 20], [20, 21],
                      [1, 12], [2, 13], [3, 14], [4, 15], [5, 16],
                      [6, 17], [7, 18], [8, 19], [9, 20], [10, 21]]]]

        assert topology[0][0] == self.model.mesh.topology[0][0]
        assert len(topology[0][1]) == len(self.model.mesh.topology[0][1])
        assert all([c1==c2
                    for c1, c2 in
                    zip(topology[0][1], self.model.mesh.topology[0][1])])

if __name__ == "__main__":
    unittest.main()
