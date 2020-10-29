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

import pennylane as qml
from braket.circuits import Gate, ResultType, gates, observables
from braket.circuits.result_types import Expectation, Probability, Sample, Variance
from pennylane.operation import Observable, ObservableReturnTypes, Operation

_OPERATION_MAP = {
    "Hadamard": gates.H,
    "PauliX": gates.X,
    "PauliY": gates.Y,
    "PauliZ": gates.Z,
    "CNOT": gates.CNot,
    "CY": gates.CY,
    "CZ": gates.CZ,
    "S": gates.S,
    "T": gates.T,
    "V": gates.V,
    "PhaseShift": gates.PhaseShift,
    "CPhaseShift": gates.CPhaseShift,
    "CPhaseShift00": gates.CPhaseShift00,
    "CPhaseShift01": gates.CPhaseShift01,
    "CPhaseShift10": gates.CPhaseShift10,
    "RX": gates.Rx,
    "RY": gates.Ry,
    "RZ": gates.Rz,
    "SWAP": gates.Swap,
    "CSWAP": gates.CSwap,
    "ISWAP": gates.ISwap,
    "PSWAP": gates.PSwap,
    "XY": gates.XY,
    "XX": gates.XX,
    "YY": gates.YY,
    "ZZ": gates.ZZ,
    "Toffoli": gates.CCNot,
    "QubitUnitary": gates.Unitary,
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
