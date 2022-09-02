# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""
Custom Operations
=================

**Module name:** :mod:`braket.pennylane_braket.ops`

.. currentmodule:: braket.pennylane_braket.ops

Contains some additional PennyLane qubit operations.

These operations can be imported via

.. code-block:: python

    from braket.pennylane_plugin import (
        ECR,
        PSWAP,
        CPhaseShift00,
        CPhaseShift01,
        CPhaseShift10,
        GPi,
        GPi2,
    )

Operations
----------

.. autosummary::
    CPhaseShift00
    CPhaseShift01
    CPhaseShift10
    ECR
    PSWAP
    GPi
    GPi2

Code details
~~~~~~~~~~~~
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation


class CPhaseShift00(Operation):
    r""" CPhaseShift00(phi, wires)

    Controlled phase shift gate phasing the :math:`| 00 \rangle` state.

    .. math:: \mathtt{CPhaseShift00}(\phi) = \begin{bmatrix}
            e^{i \phi} & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} \mathtt{CPhaseShift00}(\phi)
        = \frac{1}{2} \left[ \mathtt{CPhaseShift00}(\phi + \pi / 2)
            - \mathtt{CPhaseShift00}(\phi - \pi / 2) \right]

    Args:
        phi (float): the controlled phase angle
        wires (int): the subsystem the gate acts on
        do_queue (bool, optional): Indicates whether the operator should be
            immediately pushed into the Operator queue. Default: None
        id (str, optional): String representing the operation. Default: None

    """
    num_params = 1
    num_wires = 2
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Projector(np.array([0, 0]), wires=self.wires)

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        return [
            qml.PauliX(wires[0]),
            qml.PauliX(wires[1]),
            qml.PhaseShift(phi / 2, wires=[wires[0]]),
            qml.PhaseShift(phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(-phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            qml.PauliX(wires[1]),
            qml.PauliX(wires[0]),
        ]

    @staticmethod
    def compute_matrix(phi):
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        return qml.math.diag([qml.math.exp(1j * phi), 1, 1, 1])

    def adjoint(self):
        (phi,) = self.parameters
        return CPhaseShift00(-phi, wires=self.wires)


class CPhaseShift01(Operation):
    r""" CPhaseShift01(phi, wires)

    Controlled phase shift gate phasing the :math:`| 01 \rangle` state.

    .. math:: \mathtt{CPhaseShift01}(\phi) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & e^{i \phi} & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} \mathtt{CPhaseShift01}(\phi)
        = \frac{1}{2} \left[ \mathtt{CPhaseShift01}(\phi + \pi / 2)
            - \mathtt{CPhaseShift01}(\phi - \pi / 2) \right]

    Args:
        phi (float): the controlled phase angle
        wires (int): the subsystem the gate acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 2
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Projector(np.array([0, 1]), wires=self.wires)

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        return [
            qml.PauliX(wires[0]),
            qml.PhaseShift(phi / 2, wires=[wires[0]]),
            qml.PhaseShift(phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(-phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            qml.PauliX(wires[0]),
        ]

    @staticmethod
    def compute_matrix(phi):
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        return qml.math.diag([1, qml.math.exp(1j * phi), 1, 1])

    def adjoint(self):
        (phi,) = self.parameters
        return CPhaseShift01(-phi, wires=self.wires)


class CPhaseShift10(Operation):
    r""" CPhaseShift10(phi, wires)

    Controlled phase shift gate phasing the :math:`| 10 \rangle` state.

    .. math:: \mathtt{CPhaseShift10}(\phi) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & e^{i \phi} & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} \mathtt{CPhaseShift10}(\phi)
        = \frac{1}{2} \left[ \mathtt{CPhaseShift10}(\phi + \pi / 2)
            - \mathtt{CPhaseShift10}(\phi - \pi / 2) \right]

    Args:
        phi (float): the controlled phase angle
        wires (int): the subsystem the gate acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 2
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Projector(np.array([1, 0]), wires=self.wires)

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        return [
            qml.PauliX(wires[1]),
            qml.PhaseShift(phi / 2, wires=[wires[0]]),
            qml.PhaseShift(phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(-phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            qml.PauliX(wires[1]),
        ]

    @staticmethod
    def compute_matrix(phi):
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        return qml.math.diag([1, 1, qml.math.exp(1j * phi), 1])

    def adjoint(self):
        (phi,) = self.parameters
        return CPhaseShift10(-phi, wires=self.wires)


class PSWAP(Operation):
    r""" PSWAP(phi, wires)

    Phase-SWAP gate.

    .. math:: \mathtt{PSWAP}(\phi) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & e^{i \phi} & 0 \\
            0 & e^{i \phi} & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} \mathtt{PSWAP}(\phi)
        = \frac{1}{2} \left[ \mathtt{PSWAP}(\phi + \pi / 2) - \mathtt{PSWAP}(\phi - \pi / 2) \right]

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 2
    grad_method = "A"
    grad_recipe = ([[0.5, 1, np.pi / 2], [-0.5, 1, -np.pi / 2]],)

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        return [
            qml.SWAP(wires=wires),
            qml.CNOT(wires=wires),
            qml.PhaseShift(phi, wires=[wires[1]]),
            qml.CNOT(wires=wires),
        ]

    @staticmethod
    def compute_matrix(phi):
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        return qml.math.diag([1, np.exp(1j * phi), np.exp(1j * phi), 1])[[0, 2, 1, 3]]

    def adjoint(self):
        (phi,) = self.parameters
        return PSWAP(-phi, wires=self.wires)


class GPi(Operation):
    r""" GPi(phi, wires)

    IonQ native GPi gate.

    .. math:: \mathtt{GPi}(\phi) = \begin{bmatrix}
            0 & e^{-i \phi} \\
            e^{i \phi} & 0
        \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 1
    grad_method = "F"

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(phi):
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        return np.array(
            [
                [0, np.exp(-1j * phi)],
                [np.exp(1j * phi), 0],
            ]
        )

    def adjoint(self):
        (phi,) = self.parameters
        return GPi(phi, wires=self.wires)


class GPi2(Operation):
    r""" GPi2(phi, wires)

    IonQ native GPi2 gate.

    .. math:: \mathtt{GPi2}(\phi) = \frac{1}{\sqrt{2}} \begin{bmatrix}
            1 & -ie^{-i \phi} \\
            -ie^{i \phi} & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 1
    grad_method = "F"

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(phi):
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        return np.array(
            [
                [1, -1j * np.exp(-1j * phi)],
                [-1j * np.exp(1j * phi), 1],
            ]
        ) / np.sqrt(2)

    def adjoint(self):
        (phi,) = self.parameters
        return GPi2(phi + np.pi, wires=self.wires)


class MS(Operation):
    r""" MS(phi_0, phi_1, wires)

    IonQ native Mølmer-Sørenson gate.


    .. math:: \mathtt{MS}(\phi_0, \phi_1) = \begin{bmatrix}
            1 & 0 & 0 & -ie^{-i (\phi_0 + \phi_1)} \\
            0 & 1 & -ie^{-i (\phi_0 - \phi_1)} & 0 \\
            0 & -ie^{i (\phi_0 - \phi_1)} & 1 & 0 \\
            -ie^{i (\phi_0 + \phi_1)} & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 2

    Args:
        phi_0 (float): the first phase angle
        phi_1 (float): the second phase angle
        wires (int): the subsystem the gate acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_params = 2
    num_wires = 2
    grad_method = "F"

    def __init__(self, phi_0, phi_1, wires, do_queue=True, id=None):
        super().__init__(phi_0, phi_1, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(phi_0, phi_1):
        if qml.math.get_interface(phi_0) == "tensorflow":
            phi_0 = qml.math.cast_like(phi_0, 1j)
        if qml.math.get_interface(phi_1) == "tensorflow":
            phi_1 = qml.math.cast_like(phi_1, 1j)

        return np.array(
            [
                [1, 0, 0, -1j * np.exp(-1j * (phi_0 + phi_1))],
                [0, 1, -1j * np.exp(-1j * (phi_0 - phi_1)), 0],
                [0, -1j * np.exp(1j * (phi_0 - phi_1)), 1, 0],
                [-1j * np.exp(1j * (phi_0 + phi_1)), 0, 0, 1],
            ]
        ) / np.sqrt(2)

    def adjoint(self):
        (phi_0, phi_1) = self.parameters
        return MS(phi_0 + np.pi, phi_1, wires=self.wires)
