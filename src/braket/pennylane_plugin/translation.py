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

from functools import reduce, singledispatch
from typing import Any, FrozenSet, List, Tuple, Union

import pennylane as qml
from braket.circuits import Gate, ResultType, gates, noises, observables
from braket.circuits.result_types import (
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)
from braket.devices import Device
from braket.tasks import GateModelQuantumTaskResult
from pennylane import numpy as np
from pennylane.operation import Observable, ObservableReturnTypes, Operation

from braket.pennylane_plugin.ops import PSWAP, XY, CPhaseShift00, CPhaseShift01, CPhaseShift10

_BRAKET_TO_PENNYLANE_OPERATIONS = {
    "x": "PauliX",
    "y": "PauliY",
    "z": "PauliZ",
    "h": "Hadamard",
    "ry": "RY",
    "rx": "RX",
    "rz": "RZ",
    "s": "S",
    "si": "S.inv",
    "v": "SX",
    "vi": "SX.inv",
    "t": "T",
    "ti": "T.inv",
    "cnot": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "swap": "SWAP",
    "cswap": "CSWAP",
    "ccnot": "Toffoli",
    "phaseshift": "PhaseShift",
    "unitary": "QubitUnitary",
    "amplitude_damping": "AmplitudeDamping",
    "generalized_amplitude_damping": "GeneralizedAmplitudeDamping",
    "phase_damping": "PhaseDamping",
    "depolarizing": "DepolarizingChannel",
    "bit_flip": "BitFlip",
    "phase_flip": "PhaseFlip",
    "kraus": "QubitChannel",
    "cphaseshift": "ControlledPhaseShift",
    "cphaseshift00": "CPhaseShift00",
    "cphaseshift01": "CPhaseShift01",
    "cphaseshift10": "CPhaseShift10",
    "iswap": "ISWAP",
    "pswap": "PSWAP",
    "xy": "XY",
    "xx": "IsingXX",
    "yy": "IsingYY",
    "zz": "IsingZZ",
}


def supported_operations(device: Device) -> FrozenSet[str]:
    """Returns the operations supported by the plugin based upon the device.

    Args:
        device (Device): The device to obtain the supported operations for

    Returns:
        FrozenSet[str]: The names of the supported operations
    """
    try:
        properties = device.properties.action["braket.ir.jaqcd.program"]
    except AttributeError:
        raise AttributeError("Device needs to have properties defined.")
    supported_ops = frozenset(op.lower() for op in properties.supportedOperations)
    return frozenset(
        _BRAKET_TO_PENNYLANE_OPERATIONS[op]
        for op in _BRAKET_TO_PENNYLANE_OPERATIONS
        if op.lower() in supported_ops
    )


def translate_operation(operation: Operation, *args, **kwargs) -> Gate:
    """Translates a PennyLane ``Operation`` into the corresponding Braket ``Gate``.

    Args:
        operation (Operation): The PennyLane ``Operation`` to translate

    Returns:
        Gate: The Braket gate corresponding to the given operation
    """
    parameters = [p.numpy() if isinstance(p, qml.numpy.tensor) else p for p in operation.parameters]
    return _translate_operation(operation, parameters)


@singledispatch
def _translate_operation(operation: Operation, _parameters) -> Gate:
    raise NotImplementedError(
        f"Braket PennyLane plugin does not support operation {operation.name}."
    )


@_translate_operation.register
def _(h: qml.Hadamard, _parameters):
    return gates.H()


@_translate_operation.register
def _(x: qml.PauliX, _parameters):
    return gates.X()


@_translate_operation.register
def _(y: qml.PauliY, _parameters):
    return gates.Y()


@_translate_operation.register
def _(z: qml.PauliZ, _parameters):
    return gates.Z()


@_translate_operation.register
def _(s: qml.S, _parameters):
    return gates.Si() if s.inverse else gates.S()


@_translate_operation.register
def _(sx: qml.SX, _parameters):
    return gates.Vi() if sx.inverse else gates.V()


@_translate_operation.register
def _(t: qml.T, _parameters):
    return gates.Ti() if t.inverse else gates.T()


@_translate_operation.register
def _(cnot: qml.CNOT, _parameters):
    return gates.CNot()


@_translate_operation.register
def _(cy: qml.CY, _parameters):
    return gates.CY()


@_translate_operation.register
def _(cz: qml.CZ, _parameters):
    return gates.CZ()


@_translate_operation.register
def _(swap: qml.SWAP, _parameters):
    return gates.Swap()


@_translate_operation.register
def _(cswap: qml.CSWAP, _parameters):
    return gates.CSwap()


@_translate_operation.register
def _(toffoli: qml.Toffoli, _parameters):
    return gates.CCNot()


@_translate_operation.register
def _(rx: qml.RX, parameters):
    phi = parameters[0]
    return gates.Rx(-phi) if rx.inverse else gates.Rx(phi)


@_translate_operation.register
def _(ry: qml.RY, parameters):
    phi = parameters[0]
    return gates.Ry(-phi) if ry.inverse else gates.Ry(phi)


@_translate_operation.register
def _(rz: qml.RZ, parameters):
    phi = parameters[0]
    return gates.Rz(-phi) if rz.inverse else gates.Rz(phi)


@_translate_operation.register
def _(phase_shift: qml.PhaseShift, parameters):
    phi = parameters[0]
    return gates.PhaseShift(-phi) if phase_shift.inverse else gates.PhaseShift(phi)


@_translate_operation.register
def _(qubit_unitary: qml.QubitUnitary, parameters):
    U = np.asarray(parameters[0])
    return gates.Unitary(U.conj().T) if qubit_unitary.inverse else gates.Unitary(U)


@_translate_operation.register
def _(amplitude_damping: qml.AmplitudeDamping, parameters):
    gamma = parameters[0]
    return noises.AmplitudeDamping(gamma)


@_translate_operation.register
def _(generalized_amplitude_damping: qml.GeneralizedAmplitudeDamping, parameters):
    gamma = parameters[0]
    probability = parameters[1]
    return noises.GeneralizedAmplitudeDamping(probability=probability, gamma=gamma)


@_translate_operation.register
def _(phase_damping: qml.PhaseDamping, parameters):
    gamma = parameters[0]
    return noises.PhaseDamping(gamma)


@_translate_operation.register
def _(depolarizing_channel: qml.DepolarizingChannel, parameters):
    probability = parameters[0]
    return noises.Depolarizing(probability)


@_translate_operation.register
def _(bit_flip: qml.BitFlip, parameters):
    probability = parameters[0]
    return noises.BitFlip(probability)


@_translate_operation.register
def _(phase_flip: qml.PhaseFlip, parameters):
    probability = parameters[0]
    return noises.PhaseFlip(probability)


@_translate_operation.register
def _(qubit_channel: qml.QubitChannel, parameters):
    K_list = [np.asarray(matrix) for matrix in parameters[0]]
    return noises.Kraus(K_list)


@_translate_operation.register
def _(c_phase_shift: qml.ControlledPhaseShift, parameters):
    phi = parameters[0]
    return gates.CPhaseShift(-phi) if c_phase_shift.inverse else gates.CPhaseShift(phi)


@_translate_operation.register
def _(c_phase_shift_00: CPhaseShift00, parameters):
    phi = parameters[0]
    return gates.CPhaseShift00(-phi) if c_phase_shift_00.inverse else gates.CPhaseShift00(phi)


@_translate_operation.register
def _(c_phase_shift_01: CPhaseShift01, parameters):
    phi = parameters[0]
    return gates.CPhaseShift01(-phi) if c_phase_shift_01.inverse else gates.CPhaseShift01(phi)


@_translate_operation.register
def _(c_phase_shift_10: CPhaseShift10, parameters):
    phi = parameters[0]
    return gates.CPhaseShift10(-phi) if c_phase_shift_10.inverse else gates.CPhaseShift10(phi)


@_translate_operation.register
def _(iswap: qml.ISWAP, _parameters):
    return gates.PSwap(3 * np.pi / 2) if iswap.inverse else gates.ISwap()


@_translate_operation.register
def _(pswap: PSWAP, parameters):
    phi = parameters[0]
    return gates.PSwap(-phi) if pswap.inverse else gates.PSwap(phi)


@_translate_operation.register
def _(xy: XY, parameters):
    phi = parameters[0]
    return gates.XY(-phi) if xy.inverse else gates.XY(phi)


@_translate_operation.register
def _(xx: qml.IsingXX, parameters):
    phi = parameters[0]
    return gates.XX(-phi) if xx.inverse else gates.XX(phi)


@_translate_operation.register
def _(yy: qml.IsingYY, parameters):
    phi = parameters[0]
    return gates.YY(-phi) if yy.inverse else gates.YY(phi)


@_translate_operation.register
def _(zz: qml.IsingZZ, parameters):
    phi = parameters[0]
    return gates.ZZ(-phi) if zz.inverse else gates.ZZ(phi)


def translate_result_type(
    observable: Observable, targets: List[int], supported_result_types: FrozenSet[str]
) -> Union[ResultType, Tuple[ResultType, ...]]:
    """Translates a PennyLane ``Observable`` into the corresponding Braket ``ResultType``.

    Args:
        observable (Observable): The PennyLane ``Observable`` to translate
        targets (List[int]): The target wires of the observable using a consecutive integer wire
            ordering
        supported_result_types (FrozenSet[str]): Braket result types supported by the Braket device

    Returns:
        Union[ResultType, Tuple[ResultType]]: The Braket result type corresponding to
        the given observable; if the observable type has multiple terms, for example a Hamiltonian,
        then this will return a result type for each term.
    """
    return_type = observable.return_type

    if return_type is ObservableReturnTypes.Probability:
        return Probability(targets)

    if return_type is ObservableReturnTypes.State:
        if not targets and "StateVector" in supported_result_types:
            return StateVector()
        elif "DensityMatrix" in supported_result_types:
            return DensityMatrix(targets)
        raise NotImplementedError(f"Unsupported return type: {return_type}")

    if isinstance(observable, qml.Hamiltonian):
        if return_type is ObservableReturnTypes.Expectation:
            return tuple(
                Expectation(_translate_observable(term), term.wires) for term in observable.ops
            )
        raise NotImplementedError(f"Return type {return_type} unsupported for Hamiltonian")

    braket_observable = _translate_observable(observable)
    if return_type is ObservableReturnTypes.Expectation:
        return Expectation(braket_observable, targets)
    elif return_type is ObservableReturnTypes.Variance:
        return Variance(braket_observable, targets)
    elif return_type is ObservableReturnTypes.Sample:
        return Sample(braket_observable, targets)
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


_zero = np.array([[1, 0], [0, 0]])
_one = np.array([[0, 0], [0, 1]])


@_translate_observable.register
def _(p: qml.Projector):
    bitstring = p.parameters[0]

    products = [_one if b else _zero for b in bitstring]
    hermitians = [observables.Hermitian(p) for p in products]
    return observables.TensorProduct(hermitians)


@_translate_observable.register
def _(t: qml.operation.Tensor):
    return reduce(lambda x, y: x @ y, [_translate_observable(factor) for factor in t.obs])


def translate_result(
    braket_result: GateModelQuantumTaskResult,
    observable: Observable,
    targets: List[int],
    supported_result_types: FrozenSet[str],
) -> Any:
    """Translates a Braket result into the corresponding PennyLane return type value.

    Args:
        braket_result (GateModelQuantumTaskResult): The Braket result to translate.
        observable (Observable): The PennyLane observable associated with the result.
        targets (List[int]): The qubits in the result.
        supported_result_types (FrozenSet[str]): The result types supported by the device.

    Returns:
        Any: The translated return value.

    Note:
        Hamiltonian results will be summed over all terms.
    """
    translated = translate_result_type(observable, targets, supported_result_types)
    if isinstance(observable, qml.Hamiltonian):
        coeffs, _ = observable.terms
        return sum(
            coeff * braket_result.get_value_by_result_type(result_type)
            for coeff, result_type in zip(coeffs, translated)
        )
    else:
        return braket_result.get_value_by_result_type(translated)
