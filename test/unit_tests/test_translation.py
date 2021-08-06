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

import numpy as np
import pennylane as qml
import pytest
from braket.circuits import gates, noises, observables
from braket.circuits.result_types import (
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)
from pennylane.measure import MeasurementProcess
from pennylane.operation import ObservableReturnTypes
from pennylane.wires import Wires

from braket.pennylane_plugin import PSWAP, XY, YY, CPhaseShift00, CPhaseShift01, CPhaseShift10
from braket.pennylane_plugin.translation import (
    _BRAKET_TO_PENNYLANE_OPERATIONS,
    translate_operation,
    translate_result_type,
)

testdata = [
    (qml.Hadamard, gates.H, [0], []),
    (qml.PauliX, gates.X, [0], []),
    (qml.PauliY, gates.Y, [0], []),
    (qml.PauliZ, gates.Z, [0], []),
    (qml.S, gates.S, [0], []),
    (qml.T, gates.T, [0], []),
    (qml.CNOT, gates.CNot, [0, 1], []),
    (qml.CZ, gates.CZ, [0, 1], []),
    (qml.PhaseShift, gates.PhaseShift, [0], [np.pi]),
    (qml.RX, gates.Rx, [0], [np.pi]),
    (qml.RY, gates.Ry, [0], [np.pi]),
    (qml.RZ, gates.Rz, [0], [np.pi]),
    (qml.SWAP, gates.Swap, [0, 1], []),
    (qml.CSWAP, gates.CSwap, [0, 1, 2], []),
    (qml.Toffoli, gates.CCNot, [0, 1, 2], []),
    (qml.QubitUnitary, gates.Unitary, [0], [np.array([[0, 1], [1, 0]])]),
    (qml.SX, gates.V, [0], []),
    (qml.CY, gates.CY, [0, 1], []),
    (qml.ControlledPhaseShift, gates.CPhaseShift, [0, 1], [np.pi]),
    (CPhaseShift00, gates.CPhaseShift00, [0, 1], [np.pi]),
    (CPhaseShift01, gates.CPhaseShift01, [0, 1], [np.pi]),
    (CPhaseShift10, gates.CPhaseShift10, [0, 1], [np.pi]),
    (qml.ISWAP, gates.ISwap, [0, 1], []),
    (PSWAP, gates.PSwap, [0, 1], [np.pi]),
    (XY, gates.XY, [0, 1], [np.pi]),
    (qml.IsingXX, gates.XX, [0, 1], [np.pi]),
    (YY, gates.YY, [0, 1], [np.pi]),
    (qml.IsingZZ, gates.ZZ, [0, 1], [np.pi]),
    (qml.AmplitudeDamping, noises.AmplitudeDamping, [0], [0.1]),
    (qml.GeneralizedAmplitudeDamping, noises.GeneralizedAmplitudeDamping, [0], [0.1, 0.15]),
    (qml.PhaseDamping, noises.PhaseDamping, [0], [0.1]),
    (qml.DepolarizingChannel, noises.Depolarizing, [0], [0.1]),
    (qml.BitFlip, noises.BitFlip, [0], [0.1]),
    (qml.PhaseFlip, noises.PhaseFlip, [0], [0.1]),
    (
        qml.QubitChannel,
        noises.Kraus,
        [0],
        [[np.array([[0, 0.8], [0.8, 0]]), np.array([[0.6, 0], [0, 0.6]])]],
    ),
]

testdata_inverses = [
    (qml.Hadamard, gates.H, [0], [], []),
    (qml.PauliX, gates.X, [0], [], []),
    (qml.PauliY, gates.Y, [0], [], []),
    (qml.PauliZ, gates.Z, [0], [], []),
    (qml.Hadamard, gates.H, [0], [], []),
    (qml.CNOT, gates.CNot, [0, 1], [], []),
    (qml.CZ, gates.CZ, [0, 1], [], []),
    (qml.CY, gates.CY, [0, 1], [], []),
    (qml.SWAP, gates.Swap, [0, 1], [], []),
    (qml.CSWAP, gates.CSwap, [0, 1, 2], [], []),
    (qml.Toffoli, gates.CCNot, [0, 1, 2], [], []),
    (qml.RX, gates.Rx, [0], [0.15], [-0.15]),
    (qml.RY, gates.Ry, [0], [0.15], [-0.15]),
    (qml.RZ, gates.Rz, [0], [0.15], [-0.15]),
    (qml.PhaseShift, gates.PhaseShift, [0], [0.15], [-0.15]),
    (
        qml.QubitUnitary,
        gates.Unitary,
        [0, 1],
        [
            1
            / np.sqrt(2)
            * np.array(
                [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]], dtype=complex
            )
        ],
        [
            1
            / np.sqrt(2)
            * np.array(
                [[1, 0, 0, 1], [0, -1j, -1j, 0], [0, 1, -1, 0], [-1j, 0, 0, 1j]], dtype=complex
            )
        ],
    ),
    (qml.ControlledPhaseShift, gates.CPhaseShift, [0, 1], [0.15], [-0.15]),
    (CPhaseShift00, gates.CPhaseShift00, [0, 1], [0.15], [-0.15]),
    (CPhaseShift01, gates.CPhaseShift01, [0, 1], [0.15], [-0.15]),
    (CPhaseShift10, gates.CPhaseShift10, [0, 1], [0.15], [-0.15]),
    (PSWAP, gates.PSwap, [0, 1], [0.15], [-0.15]),
    (qml.IsingXX, gates.XX, [0, 1], [0.15], [-0.15]),
    (XY, gates.XY, [0, 1], [0.15], [-0.15]),
    (YY, gates.YY, [0, 1], [0.15], [-0.15]),
    (qml.IsingZZ, gates.ZZ, [0, 1], [0.15], [-0.15]),
]

testdata_named_inverses = [
    (qml.S, gates.Si, 0),
    (qml.T, gates.Ti, 0),
    (qml.SX, gates.Vi, 0),
]

_braket_to_pl = {
    op.lower().replace("_", ""): _BRAKET_TO_PENNYLANE_OPERATIONS[op]
    for op in _BRAKET_TO_PENNYLANE_OPERATIONS
}

pl_return_types = [
    ObservableReturnTypes.Expectation,
    ObservableReturnTypes.Variance,
    ObservableReturnTypes.Sample,
]

braket_results = [
    Expectation(observables.H(), [0]),
    Variance(observables.H(), [0]),
    Sample(observables.H(), [0]),
]


@pytest.mark.parametrize("pl_cls, braket_cls, qubits, params", testdata)
def test_translate_operation(pl_cls, braket_cls, qubits, params):
    """Tests that Braket operations are translated correctly"""
    pl_op = pl_cls(*params, wires=qubits)
    braket_gate = braket_cls(*params)
    assert translate_operation(pl_op) == braket_gate
    assert (
        _braket_to_pl[braket_gate.to_ir(qubits).__class__.__name__.lower().replace("_", "")]
        == pl_op.name
    )


@pytest.mark.parametrize("pl_cls, braket_cls, qubits, params, inv_params", testdata_inverses)
def test_translate_operation_inverse(pl_cls, braket_cls, qubits, params, inv_params):
    """Tests that inverse gates are translated correctly"""
    pl_op = pl_cls(*params, wires=qubits).inv()
    braket_gate = braket_cls(*inv_params)
    assert translate_operation(pl_op) == braket_gate
    assert _braket_to_pl[
        braket_gate.to_ir(qubits).__class__.__name__.lower().replace("_", "")
    ] == pl_op.name.replace(".inv", "")


@pytest.mark.parametrize("pl_cls, braket_cls, qubit", testdata_named_inverses)
def test_translate_operation_named_inverse(pl_cls, braket_cls, qubit):
    pl_op = pl_cls(wires=[qubit]).inv()
    braket_gate = braket_cls()
    assert translate_operation(pl_op) == braket_gate
    assert (
        _braket_to_pl[braket_gate.to_ir([qubit]).__class__.__name__.lower().replace("_", "")]
        == pl_op.name
    )


def test_translate_iswap_inverse():
    qubits = [0, 1]
    pl_op = qml.ISWAP(wires=qubits).inv()
    braket_gate = gates.PSwap(3 * np.pi / 2)
    assert translate_operation(pl_op) == braket_gate


@pytest.mark.parametrize("return_type, braket_result", zip(pl_return_types, braket_results))
def test_translate_result_type_observable(return_type, braket_result):
    """Tests if a PennyLane return type that involves an observable is successfully converted into a
    Braket result using translate_result_type"""
    obs = qml.Hadamard(0)
    obs.return_type = return_type
    braket_result_calculated = translate_result_type(obs, [0], frozenset())

    assert braket_result == braket_result_calculated


def test_translate_result_type_probs():
    """Tests if a PennyLane probability return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.Probability, wires=Wires([0]))
    braket_result_calculated = translate_result_type(mp, [0], frozenset())

    braket_result = Probability([0])

    assert braket_result == braket_result_calculated


def test_translate_result_type_state_vector():
    """Tests if a PennyLane state vector return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.State)
    braket_result_calculated = translate_result_type(
        mp, [], frozenset(["StateVector", "DensityMatrix"])
    )

    braket_result = StateVector()

    assert braket_result == braket_result_calculated


def test_translate_result_type_density_matrix():
    """Tests if a PennyLane density matrix return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.State)
    braket_result_calculated = translate_result_type(mp, [], frozenset(["DensityMatrix"]))

    braket_result = DensityMatrix()

    assert braket_result == braket_result_calculated


def test_translate_result_type_density_matrix_partial():
    """Tests if a PennyLane partial density matrix return type is successfully converted into a
    Braket result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.State, wires=[0])
    braket_result_calculated = translate_result_type(
        mp, [0], frozenset(["StateVector", "DensityMatrix"])
    )

    braket_result = DensityMatrix([0])

    assert braket_result == braket_result_calculated


def test_translate_result_type_state_unimplemented():
    """Tests if a NotImplementedError is raised by translate_result_type when a PennyLane state
    return type is converted while not supported by the device"""
    mp = MeasurementProcess(ObservableReturnTypes.State)
    with pytest.raises(NotImplementedError, match="Unsupported return type"):
        translate_result_type(mp, [0], frozenset())


def test_translate_result_type_unsupported_return():
    """Tests if a NotImplementedError is raised by translate_result_type for an unknown
    return_type"""
    obs = qml.Hadamard(0)
    obs.return_type = None

    with pytest.raises(NotImplementedError, match="Unsupported return type"):
        translate_result_type(obs, [0], frozenset())


def test_translate_result_type_unsupported_obs():
    """Tests if a TypeError is raised by translate_result_type for an unknown observable"""
    obs = qml.S(wires=0)
    obs.return_type = None

    with pytest.raises(TypeError, match="Unsupported observable"):
        translate_result_type(obs, [0], frozenset())
