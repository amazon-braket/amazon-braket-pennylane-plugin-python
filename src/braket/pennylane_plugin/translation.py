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
from typing import FrozenSet

import numpy as np
import pennylane as qml
from braket.circuits import Gate, ResultType, gates, observables
from braket.circuits.result_types import Expectation, Probability, Sample, Variance
from pennylane.operation import Observable, ObservableReturnTypes, Operation

from braket.pennylane_plugin.ops import (
    ISWAP,
    PSWAP,
    XX,
    XY,
    YY,
    ZZ,
    CPhaseShift,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
)

_OPERATION_MAP = {
    # pennylane operations
    "Hadamard": gates.H,
    "PauliX": gates.X,
    "PauliY": gates.Y,
    "PauliZ": gates.Z,
    "S": gates.S,
    "T": gates.T,
    "SX": gates.V,
    "CNOT": gates.CNot,
    "CZ": gates.CZ,
    "CY": gates.CY,
    "SWAP": gates.Swap,
    "CSWAP": gates.CSwap,
    "Toffoli": gates.CCNot,
    "RX": gates.Rx,
    "RY": gates.Ry,
    "RZ": gates.Rz,
    "PhaseShift": gates.PhaseShift,
    "QubitUnitary": gates.Unitary,
    # plugin operations
    "CPhaseShift": gates.CPhaseShift,
    "CPhaseShift00": gates.CPhaseShift00,
    "CPhaseShift01": gates.CPhaseShift01,
    "CPhaseShift10": gates.CPhaseShift10,
    "ISWAP": gates.ISwap,
    "PSWAP": gates.PSwap,
    "XY": gates.XY,
    "XX": gates.XX,
    "YY": gates.YY,
    "ZZ": gates.ZZ,
}


def supported_operations() -> FrozenSet[str]:
    """Returns the operations supported by the plugin.

    Returns:
        FrozenSet[str]: The names of the supported operations
    """
    return frozenset(_OPERATION_MAP.keys())


@singledispatch
def translate_operation(operation: Operation, parameters) -> Gate:
    """Translates a PennyLane ``Operation`` into the corresponding Braket ``Gate``.

    Args:
        operation (Operation): The PennyLane ``Operation`` to translate
        parameters: The parameters of the operation

    Returns:
        Gate: The Braket gate corresponding to the given operation
    """
    raise NotImplementedError(
        f"Braket PennyLane plugin does not support operation {operation.name}."
    )


@translate_operation.register
def _(_h: qml.Hadamard, _parameters):
    return gates.H()


@translate_operation.register
def _(_x: qml.PauliX, _parameters):
    return gates.X()


@translate_operation.register
def _(_y: qml.PauliY, _parameters):
    return gates.Y()


@translate_operation.register
def _(_z: qml.PauliZ, _parameters):
    return gates.Z()


@translate_operation.register
def _(s: qml.S, _parameters):
    return gates.Si() if s.inverse else gates.S()


@translate_operation.register
def _(t: qml.T, _parameters):
    return gates.Ti() if t.inverse else gates.T()


@translate_operation.register
def _(sx: qml.SX, _parameters):
    return gates.Vi() if sx.inverse else gates.V()


@translate_operation.register
def _(_cnot: qml.CNOT, _parameters):
    return gates.CNot()


@translate_operation.register
def _(_cz: qml.CZ, _parameters):
    return gates.CZ()


@translate_operation.register
def _(_cy: qml.CY, _parameters):
    return gates.CY()


@translate_operation.register
def _(_swap: qml.SWAP, _parameters):
    return gates.Swap()


@translate_operation.register
def _(_cswap: qml.CSWAP, _parameters):
    return gates.CSwap()


@translate_operation.register
def _(_toffoli: qml.Toffoli, _parameters):
    return gates.CCNot()


@translate_operation.register
def _(rx: qml.RX, parameters):
    phi = parameters[0]
    return gates.Rx(-phi) if rx.inverse else gates.Rx(phi)


@translate_operation.register
def _(ry: qml.RY, parameters):
    phi = parameters[0]
    return gates.Ry(-phi) if ry.inverse else gates.Ry(phi)


@translate_operation.register
def _(rz: qml.RZ, parameters):
    phi = parameters[0]
    return gates.Rz(-phi) if rz.inverse else gates.Rz(phi)


@translate_operation.register
def _(phase_shift: qml.PhaseShift, parameters):
    phi = parameters[0]
    return gates.PhaseShift(-phi) if phase_shift.inverse else gates.PhaseShift(phi)


@translate_operation.register
def _(qubit_unitary: qml.QubitUnitary, parameters):
    U = np.asarray(parameters[0])
    return gates.Unitary(U.T.conj()) if qubit_unitary.inverse else gates.Unitary(U)


@translate_operation.register
def _(c_phase_shift: CPhaseShift, parameters):
    phi = parameters[0]
    return gates.CPhaseShift(-phi) if c_phase_shift.inverse else gates.CPhaseShift(phi)


@translate_operation.register
def _(c_phase_shift: CPhaseShift00, parameters):
    phi = parameters[0]
    return gates.CPhaseShift00(-phi) if c_phase_shift.inverse else gates.CPhaseShift00(phi)


@translate_operation.register
def _(c_phase_shift: CPhaseShift01, parameters):
    phi = parameters[0]
    return gates.CPhaseShift01(-phi) if c_phase_shift.inverse else gates.CPhaseShift01(phi)


@translate_operation.register
def _(c_phase_shift: CPhaseShift10, parameters):
    phi = parameters[0]
    return gates.CPhaseShift10(-phi) if c_phase_shift.inverse else gates.CPhaseShift10(phi)


@translate_operation.register
def _(iswap: ISWAP, _parameters):
    return gates.PSwap(3 * np.pi / 2) if iswap.inverse else gates.ISwap()


@translate_operation.register
def _(pswap: PSWAP, parameters):
    phi = parameters[0]
    return gates.PSwap(-phi) if pswap.inverse else gates.PSwap(phi)


@translate_operation.register
def _(xy: XY, parameters):
    phi = parameters[0]
    return gates.XY(-phi) if xy.inverse else gates.XY(phi)


@translate_operation.register
def _(xx: XX, parameters):
    phi = parameters[0]
    return gates.XX(-phi) if xx.inverse else gates.XX(phi)


@translate_operation.register
def _(yy: YY, parameters):
    phi = parameters[0]
    return gates.YY(-phi) if yy.inverse else gates.YY(phi)


@translate_operation.register
def _(zz: ZZ, parameters):
    phi = parameters[0]
    return gates.ZZ(-phi) if zz.inverse else gates.ZZ(phi)


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
