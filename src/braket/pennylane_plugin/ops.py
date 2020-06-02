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
    CPHASE
    ISWAP
    PSWAP


Code details
~~~~~~~~~~~~
"""
import numpy as np
import pennylane as qml
from pennylane.operation import Operation


class CPHASE(Operation):
    r"""CPHASE(phi, q, wires)
    Controlled-phase gate.

    .. math::

        CPHASE_{ij}(phi, q) = \begin{cases}
            0, & i\neq j\\
            1, & i=j, i\neq q\\
            e^{i\phi}, & i=j=q
        \end{cases}\in\mathbb{C}^{4\times 4}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 2
    * Gradient recipe:
    .. math::
        \frac{d}{d\phi}CPHASE(\phi) = \frac{1}{2}\left[CPHASE(\phi+\pi/2)+CPHASE(\phi-\pi/2)\right]

    Note:
        The gradient recipe only applies to parameter :math:`\phi`.
        Parameter :math:`q\in\mathbb{N}_0` and thus ``CPHASE`` can not be differentiated
        with respect to :math:`q`.

    Args:
        phi (float): the controlled phase angle
        q (int): an integer between 0 and 3 that corresponds to a state
            :math:`\{00, 01, 10, 11\}` on which the conditional phase
            gets applied
        wires (int): the subsystem the gate acts on
    """
    num_params = 2
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    @staticmethod
    def decomposition(phi, q, wires):
        if q == 0:
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

        elif q == 1:
            return [
                qml.PauliX(wires[0]),
                qml.PhaseShift(phi / 2, wires=[wires[0]]),
                qml.PhaseShift(phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PhaseShift(-phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PauliX(wires[0]),
            ]

        elif q == 2:
            return [
                qml.PauliX(wires[1]),
                qml.PhaseShift(phi / 2, wires=[wires[0]]),
                qml.PhaseShift(phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PhaseShift(-phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PauliX(wires[1]),
            ]

        elif q == 3:
            return [
                qml.PhaseShift(phi / 2, wires=[wires[0]]),
                qml.PhaseShift(phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
                qml.PhaseShift(-phi / 2, wires=[wires[1]]),
                qml.CNOT(wires=wires),
            ]

    @staticmethod
    def _matrix(*params):
        phi, q = params
        mat = np.identity(4)
        mat[q, q] = np.exp(1j * phi)
        return mat


class ISWAP(Operation):
    r"""ISWAP(wires)
    iSWAP gate.

    .. math:: ISWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & i & 0\\
            0 & i & 0 & 0\\
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

    @staticmethod
    def _matrix(*params):
        return np.diag([1, 1j, 1j, 1])[[0, 2, 1, 3]]


class PSWAP(Operation):
    r"""PSWAP(phi, wires)
    Phase-SWAP gate.

    .. math:: PSWAP(\phi) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & e^{i\phi} & 0\\
            0 & e^{i\phi} & 0 & 0\\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe:
    .. math::
        \frac{d}{d\phi}PSWAP(\phi) = \frac{1}{2}\left[PSWAP(\phi+\pi/2)+PSWAP(\phi-\pi/2)\right]

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

    @staticmethod
    def _matrix(*params):
        phi = params[0]
        return np.diag([1, np.exp(1j * phi), np.exp(1j * phi), 1])[[0, 2, 1, 3]]
