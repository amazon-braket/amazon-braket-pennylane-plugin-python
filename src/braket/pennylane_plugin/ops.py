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
        XY,
        CPhaseShift00,
        CPhaseShift01,
        CPhaseShift10,
    )

Operations
----------

.. autosummary::
    CPhaseShift00
    CPhaseShift01
    CPhaseShift10
    ECR
    PSWAP
    XY

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


class ECR(Operation):
    r""" ECR(wires)

    An echoed RZX(pi/2) gate.

    .. math:: \mathtt{ECR} = {1/\sqrt{2}} \begin{bmatrix}
            0 & 0 & 1 & i \\
            0 & 0 & i & 1 \\
            1 & -i & 0 & 0 \\
            -i & 1 & 0 & 0
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """

    num_params = 0
    num_wires = 2

    def __init__(self, wires, do_queue=True, id=None):
        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(wires):
        pi = np.pi
        return [
            qml.PauliZ(wires=[wires[0]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.SX(wires=[wires[1]]),
            qml.RX(pi / 2, wires=[wires[0]]),
            qml.RY(pi / 2, wires=[wires[0]]),
            qml.RX(pi / 2, wires=[wires[0]]),
        ]

    @staticmethod
    def compute_matrix():
        return (
            1
            / np.sqrt(2)
            * np.array(
                [[0, 0, 1, 1.0j], [0, 0, 1.0j, 1], [1, -1.0j, 0, 0], [-1.0j, 1, 0, 0]],
                dtype=complex,
            )
        )

    def adjoint(self):
        return ECR(wires=self.wires)


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


class XY(Operation):
    r""" XY(phi, wires)

    Parameterized ISWAP gate: https://arxiv.org/abs/1912.04424v1

    .. math:: \mathtt{XY}(\phi) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & \cos(\phi / 2) & i \sin(\phi / 2) & 0 \\
            0 & i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: The XY operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695):

      .. math::
          \frac{d}{d \phi} f(XY(\phi))
          = c_+ \left[ f(XY(\phi + a)) - f(XY(\phi - a)) \right]
          - c_- \left[ f(XY(\phi + b)) - f(XY(\phi - b)) \right]

      where :math:`f` is an expectation value depending on :math:`XY(\phi)`, and

      - :math:`a = \pi / 2`
      - :math:`b = 3 \pi / 2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4 \sqrt{2}}`

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0)]

    # TODO: Modify generator function for this class, add scalar multiplication for hamiltonians
    #  back when this issue is fixed: https://github.com/PennyLaneAI/pennylane/issues/2361
    def generator(self):
        return qml.Hamiltonian(
            [0.25, 0.25],
            [
                qml.PauliX(wires=self.wires[0]) @ qml.PauliX(wires=self.wires[1]),
                qml.PauliY(wires=self.wires[0]) @ qml.PauliY(wires=self.wires[1]),
            ],
        )

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        return [
            qml.Hadamard(wires=[wires[0]]),
            qml.CY(wires=wires),
            qml.RY(phi / 2, wires=[wires[0]]),
            qml.RX(-phi / 2, wires=[wires[1]]),
            qml.CY(wires=wires),
            qml.Hadamard(wires=[wires[0]]),
        ]

    @staticmethod
    def compute_matrix(phi):
        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)
        off_diag = qml.math.convert_like(np.diag([0, 1, 1, 0])[::-1].copy(), phi)

        if qml.math.get_interface(phi) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
            off_diag = qml.math.cast_like(off_diag, 1j)

        return qml.math.diag([1, c, c, 1]) + 1j * s * off_diag

    def adjoint(self):
        (phi,) = self.parameters
        return XY(-phi, wires=self.wires)
