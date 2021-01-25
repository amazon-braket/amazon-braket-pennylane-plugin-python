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

from functools import singledispatch
from typing import FrozenSet, Type

import numpy as np
import pennylane as qml
from braket.circuits import Gate, ResultType, gates, observables
from braket.circuits.result_types import Expectation, Probability, Sample, Variance
from pennylane.operation import Observable, ObservableReturnTypes, Operation

_OPERATION_MAP = {
    # pennylane operations
    "Hadamard": gates.H,
    "Hadamard.inv": gates.H,
    "PauliX": gates.X,
    "PauliX.inv": gates.X,
    "PauliY": gates.Y,
    "PauliY.inv": gates.Y,
    "PauliZ": gates.Z,
    "PauliZ.inv": gates.Z,
    "S": gates.S,
    "S.inv": gates.Si,
    "T": gates.T,
    "T.inv": gates.Ti,
    "SX": gates.V,
    "SX.inv": gates.Vi,
    "CNOT": gates.CNot,
    "CNOT.inv": gates.CNot,
    "CZ": gates.CZ,
    "CZ.inv": gates.CZ,
    "CY": gates.CY,
    "CY.inv": gates.CY,
    "SWAP": gates.Swap,
    "SWAP.inv": gates.Swap,
    "CSWAP": gates.CSwap,
    "CSWAP.inv": gates.CSwap,
    "Toffoli": gates.CCNot,
    "Toffoli.inv": gates.CCNot,
    "RX": gates.Rx,
    "RX.inv": gates.Rx,
    "RY": gates.Ry,
    "RY.inv": gates.Ry,
    "RZ": gates.Rz,
    "RZ.inv": gates.Rz,
    "PhaseShift": gates.PhaseShift,
    "PhaseShift.inv": gates.PhaseShift,
    "QubitUnitary": gates.Unitary,
    "QubitUnitary.inv": gates.Unitary,
    # plugin operations
    "CPhaseShift": gates.CPhaseShift,
    "CPhaseShift.inv": gates.CPhaseShift,
    "CPhaseShift00": gates.CPhaseShift00,
    "CPhaseShift00.inv": gates.CPhaseShift00,
    "CPhaseShift01": gates.CPhaseShift01,
    "CPhaseShift01.inv": gates.CPhaseShift01,
    "CPhaseShift10": gates.CPhaseShift10,
    "CPhaseShift10.inv": gates.CPhaseShift10,
    "ISWAP": gates.ISwap,
    "ISWAP.inv": gates.PSwap,
    "PSWAP": gates.PSwap,
    "PSWAP.inv": gates.PSwap,
    "XY": gates.XY,
    "XY.inv": gates.XY,
    "XX": gates.XX,
    "XX.inv": gates.XX,
    "YY": gates.YY,
    "YY.inv": gates.YY,
    "ZZ": gates.ZZ,
    "ZZ.inv": gates.ZZ,
}


def supported_operations() -> FrozenSet[str]:
    """Returns the operations supported by the plugin.

    Returns:
        FrozenSet[str]: The names of the supported operations
    """
    return frozenset(_OPERATION_MAP.keys())


def translate_operation(operation: Operation) -> Type[Gate]:
    """Translates a PennyLane ``Operation`` into the corresponding Braket gate.

    Args:
        operation (Operation): The Pennylane ``Operation`` to translate

    Returns:
        Type[Gate]: The `Gate` class corresponding to the given operation
    """
    return _OPERATION_MAP[operation.name]


def translate_parameters(parameters, operation: Operation):
    """Modifies the parameters of the gate, particularly to translate between
    Pennylane and Braket inverse operations.

    Args:
        parameters (List[int]): The list of parameters for the given operation
        operation (Operation): The Pennylane ``Operation`` to which the parameter list belongs

    Returns:
        List[int]: The translated list of parameters
    """
    if not operation.inverse:
        return parameters

    # these gates are inverted by flipping the sign of the angle
    rotation_gates = [
        "RX",
        "RY",
        "RZ",
        "PhaseShift",
        "CPhaseShift",
        "CPhaseShift00",
        "CPhaseShift01",
        "CPhaseShift10",
        "PSWAP",
        "XY",
        "XX",
        "YY",
        "ZZ",
    ]

    if operation.base_name in rotation_gates:
        phi = parameters[0]
        return [-phi]

    # these gates are inverted by inverting the unitary matrix
    matrix_gates = [
        "QubitUnitary",
    ]

    if operation.base_name in matrix_gates:
        U = np.asarray(parameters[0])
        return [U.T.conj()]

    # ISWAP gets inverted to PSWAP(3*pi/2)
    if operation.base_name == "ISWAP":
        return [3 * np.pi / 2]

    return parameters


def translate_result_type(observable: Observable) -> ResultType:
    """Translates a PennyLane ``Observable`` into the corresponding Braket ``ResultType``.

    Args:
        observable (Observable): The PennyLane ``Observable`` to translate

    Returns:
        ResultType: The Braket result type corresponding to the given observable
    """
    braket_observable = _translate_observable(observable)
    return_type = observable.return_type
    targets = observable.wires.tolist()

    if return_type is ObservableReturnTypes.Expectation:
        return Expectation(braket_observable, targets)
    elif return_type is ObservableReturnTypes.Variance:
        return Variance(braket_observable, targets)
    elif return_type is ObservableReturnTypes.Sample:
        return Sample(braket_observable, targets)
    elif return_type is ObservableReturnTypes.Probability:
        return Probability(targets)
    else:
        raise NotImplementedError(f"Unsupported return type: {return_type}")


@singledispatch
def _translate_observable(observable):
    raise TypeError(f"Unsupported observable: {observable}")


@_translate_observable.register
def _(x: qml.PauliX):
    return observables.X()


@_translate_observable.register
def _(y: qml.PauliY):
    return observables.Y()


@_translate_observable.register
def _(z: qml.PauliZ):
    return observables.Z()


@_translate_observable.register
def _(h: qml.Hadamard):
    return observables.H()


@_translate_observable.register
def _(i: qml.Identity):
    return observables.I()


@_translate_observable.register
def _(h: qml.Hermitian):
    return observables.Hermitian(h.matrix)


@_translate_observable.register
def _(t: qml.operation.Tensor):
    return observables.TensorProduct([_translate_observable(factor) for factor in t.obs])
