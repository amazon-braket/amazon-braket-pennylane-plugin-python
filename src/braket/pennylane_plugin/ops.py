# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

    from braket.pennylane_braket.ops import CPHASE, ISWAP, PSWAP

Operations
----------

.. autosummary::
    V
    CY
    CPhaseShift
    CPhaseShift00
    CPhaseShift01
    CPhaseShift10
    ISWAP
    PSWAP
    XY
    XX
    YY
    ZZ

Code details
~~~~~~~~~~~~
"""
import numpy as np
import pennylane as qml
from pennylane.operation import Operation


class V(Operation):
    r""" V(wires)

    Square root of the Pauli X (NOT) gate.

    .. math:: \mathtt{V} = \frac{1}{2} \begin{bmatrix}
            1 + i & 1 - i \\
            1 - i & 1 + i
        \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None

    @classmethod
    def _matrix(cls, *params):
        return np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)


class CY(Operation):
    r""" CY(wires)

    The controlled-Y operator.

    .. math:: \mathtt{CY} = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & -i \\
            0 & 0 & i & 0
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 2
    par_domain = None

    @classmethod
    def _matrix(cls, *params):
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0j],
                [0.0, 0.0, +1.0j, 0.0],
            ],
            dtype=complex,
        )


class CPhaseShift(Operation):
    r""" CPhaseShift(phi, wires)

    Controlled phase shift gate phasing the :math:`| 11 \rangle` state.

    .. math::

      \mathtt{CPhaseShift}(\phi) = \begin{bmatrix}
          1 & 0 & 0 & 0 \\
          0 & 1 & 0 & 0 \\
          0 & 0 & 1 & 0 \\
          0 & 0 & 0 & e^{i \phi}
      \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe:

    .. math::

        \frac{d}{d \phi} \mathtt{CPhaseShift}(\phi)
        = \frac{1}{2} \left[ \mathtt{CPhaseShift}(\phi + \pi / 2)
            + \mathtt{CPhaseShift}(\phi - \pi / 2) \right]

    Args:
        phi (float): the controlled phase angle
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    @staticmethod
    def decomposition(phi, wires):
        return [
            qml.PhaseShift(phi / 2, wires=[wires[0]]),
            qml.PhaseShift(phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(-phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
        ]

    @classmethod
    def _matrix(cls, *params):
        return np.diag([1.0, 1.0, 1.0, np.exp(1.0j * params[0])])


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
            + \mathtt{CPhaseShift00}(\phi - \pi / 2) \right]

    Args:
        phi (float): the controlled phase angle
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    @staticmethod
    def decomposition(phi, wires):
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

    @classmethod
    def _matrix(cls, *params):
        return np.diag([np.exp(1.0j * params[0]), 1.0, 1.0, 1.0])


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
            + \mathtt{CPhaseShift01}(\phi - \pi / 2) \right]

    Args:
        phi (float): the controlled phase angle
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    @staticmethod
    def decomposition(phi, wires):
        return [
            qml.PauliX(wires[0]),
            qml.PhaseShift(phi / 2, wires=[wires[0]]),
            qml.PhaseShift(phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(-phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            qml.PauliX(wires[0]),
        ]

    @classmethod
    def _matrix(cls, *params):
        return np.diag([1.0, np.exp(1.0j * params[0]), 1.0, 1.0])


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
            + \mathtt{CPhaseShift10}(\phi - \pi / 2) \right]

    Args:
        phi (float): the controlled phase angle
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    @staticmethod
    def decomposition(phi, wires):
        return [
            qml.PauliX(wires[1]),
            qml.PhaseShift(phi / 2, wires=[wires[0]]),
            qml.PhaseShift(phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            qml.PhaseShift(-phi / 2, wires=[wires[1]]),
            qml.CNOT(wires=wires),
            qml.PauliX(wires[1]),
        ]

    @classmethod
    def _matrix(cls, *params):
        return np.diag([1.0, 1.0, np.exp(1.0j * params[0]), 1.0])


class ISWAP(Operation):
    r""" ISWAP(wires)

    ISWAP gate.

    .. math:: \mathtt{ISWAP} = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & i & 0 \\
            0 & i & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 2
    par_domain = None

    @staticmethod
    def decomposition(wires):
        return [
            qml.SWAP(wires=wires),
            qml.S(wires=[wires[0]]),
            qml.S(wires=[wires[1]]),
            qml.CZ(wires=wires),
        ]

    @classmethod
    def _matrix(cls, *params):
        return np.diag([1, 1j, 1j, 1])[[0, 2, 1, 3]]


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
        = \frac{1}{2} \left[ \mathtt{PSWAP}(\phi + \pi / 2) + \mathtt{PSWAP}(\phi - \pi / 2) \right]

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    @staticmethod
    def decomposition(phi, wires):
        return [
            qml.SWAP(wires=wires),
            qml.CNOT(wires=wires),
            qml.PhaseShift(phi, wires=[wires[1]]),
            qml.CNOT(wires=wires),
        ]

    @classmethod
    def _matrix(cls, *params):
        phi = params[0]
        return np.diag([1, np.exp(1j * phi), np.exp(1j * phi), 1])[[0, 2, 1, 3]]


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
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} \mathtt{XY}(\phi)
        = \frac{1}{2} \left[ \mathtt{XY}(\phi + \pi / 2) + \mathtt{XY}(\phi - \pi / 2) \right]

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    @classmethod
    def _matrix(cls, *params):
        phi = params[0]
        cos = np.cos(phi / 2)
        isin = 1.0j * np.sin(phi / 2)
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, isin, 0.0],
                [0.0, isin, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )


class XX(Operation):
    r""" XX(phi, wires)

    Ising XX coupling gate: https://arxiv.org/abs/1707.06356

    .. math:: \mathtt{XX}(\phi) = \begin{bmatrix}
            \cos(\phi / 2) & 0 & 0 & -i \sin(\phi / 2) \\
            0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
            0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
            -i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} \mathtt{XX}(\phi)
        = \frac{1}{2} \left[ \mathtt{XX}(\phi + \pi / 2) + \mathtt{XX}(\phi - \pi / 2) \right]

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    @classmethod
    def _matrix(cls, *params):
        phi = params[0]
        cos = np.cos(phi / 2)
        isin = 1.0j * np.sin(phi / 2)
        return np.array(
            [
                [cos, 0.0, 0.0, -isin],
                [0.0, cos, -isin, 0.0],
                [0.0, -isin, cos, 0.0],
                [-isin, 0.0, 0.0, cos],
            ],
            dtype=complex,
        )


class YY(Operation):
    r""" YY(phi, wires)

    Ising YY coupling gate: https://arxiv.org/abs/1707.06356

    .. math:: \mathtt{YY}(\phi) = \begin{bmatrix}
            \cos(\phi / 2) & 0 & 0 & i \sin(\phi / 2) \\
            0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
            0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
            i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} \mathtt{YY}(\phi)
        = \frac{1}{2} \left[ \mathtt{YY}(\phi + \pi / 2) + \mathtt{YY}(\phi - \pi / 2) \right]

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    @classmethod
    def _matrix(cls, *params):
        phi = params[0]
        cos = np.cos(phi / 2)
        isin = 1.0j * np.sin(phi / 2)
        return np.array(
            [
                [cos, 0.0, 0.0, isin],
                [0.0, cos, -isin, 0.0],
                [0.0, -isin, cos, 0.0],
                [isin, 0.0, 0.0, cos],
            ],
            dtype=complex,
        )


class ZZ(Operation):
    r""" ZZ(phi, wires)

    Ising ZZ coupling gate: https://arxiv.org/abs/1707.06356

    .. math:: \mathtt{ZZ}(\phi) = \begin{bmatrix}
            e^{-i \phi / 2} & 0 & 0 & 0 \\
            0 & e^{i \phi / 2} & 0 & 0 \\
            0 & 0 & e^{i \phi / 2} & 0 \\
            0 & 0 & 0 & e^{-i \phi / 2}
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe:

    .. math::
        \frac{d}{d \phi} \mathtt{ZZ}(\phi)
        = \frac{1}{2} \left[ \mathtt{ZZ}(\phi + \pi / 2) + \mathtt{ZZ}(\phi - \pi / 2) \right]

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    @classmethod
    def _matrix(cls, *params):
        phi = params[0]
        pos_phase = np.exp(1.0j * phi / 2)
        neg_phase = np.exp(-1.0j * phi / 2)
        return np.diag([neg_phase, pos_phase, pos_phase, neg_phase])
