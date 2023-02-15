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

import json
import re

import numpy as np
import pennylane as qml
import pytest
from braket.circuits import FreeParameter, gates, noises, observables
from braket.circuits.result_types import (
    AdjointGradient,
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)
from braket.circuits.serialization import IRType
from braket.tasks import GateModelQuantumTaskResult
from pennylane import numpy as pnp
from pennylane.measurements import MeasurementProcess, ObservableReturnTypes
from pennylane.wires import Wires

from braket.pennylane_plugin import PSWAP, CPhaseShift00, CPhaseShift01, CPhaseShift10
from braket.pennylane_plugin.ops import MS, GPi, GPi2
from braket.pennylane_plugin.translation import (
    _BRAKET_TO_PENNYLANE_OPERATIONS,
    _translate_observable,
    get_adjoint_gradient_result_type,
    translate_operation,
    translate_result,
    translate_result_type,
)

testdata = [
    (qml.Identity, gates.I, [0], []),
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
    (GPi, gates.GPi, [0], [2]),
    (GPi2, gates.GPi2, [0], [2]),
    (MS, gates.MS, [0, 1], [2, 3]),
    (qml.ECR, gates.ECR, [0, 1], []),
    (qml.ISWAP, gates.ISwap, [0, 1], []),
    (PSWAP, gates.PSwap, [0, 1], [np.pi]),
    (qml.IsingXY, gates.XY, [0, 1], [np.pi]),
    (qml.IsingXX, gates.XX, [0, 1], [np.pi]),
    (qml.IsingYY, gates.YY, [0, 1], [np.pi]),
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
    (qml.Identity, gates.I, [0], [], []),
    (qml.Hadamard, gates.H, [0], [], []),
    (qml.PauliX, gates.X, [0], [], []),
    (qml.PauliY, gates.Y, [0], [], []),
    (qml.PauliZ, gates.Z, [0], [], []),
    (qml.Hadamard, gates.H, [0], [], []),
    (qml.CNOT, gates.CNot, [0, 1], [], []),
    (qml.CZ, gates.CZ, [0, 1], [], []),
    (qml.CY, gates.CY, [0, 1], [], []),
    (qml.SWAP, gates.Swap, [0, 1], [], []),
    (qml.ECR, gates.ECR, [0, 1], [], []),
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
    (GPi, gates.GPi, [0], [2], [2]),
    (GPi2, gates.GPi2, [0], [2], [2 + np.pi]),
    (MS, gates.MS, [0, 1], [2, 3], [2 + np.pi, 3]),
    (PSWAP, gates.PSwap, [0, 1], [0.15], [-0.15]),
    (qml.IsingXX, gates.XX, [0, 1], [0.15], [-0.15]),
    (qml.IsingXY, gates.XY, [0, 1], [0.15], [-0.15]),
    (qml.IsingYY, gates.YY, [0, 1], [0.15], [-0.15]),
    (qml.IsingZZ, gates.ZZ, [0, 1], [0.15], [-0.15]),
]

testdata_named_inverses = [
    (qml.S, gates.Si, 0),
    (qml.T, gates.Ti, 0),
    (qml.SX, gates.Vi, 0),
]

testdata_with_params = [
    (qml.Identity, gates.I, [0], [], [], []),
    (qml.Hadamard, gates.H, [0], [], [], []),
    (qml.PauliX, gates.X, [0], [], [], []),
    (qml.PauliY, gates.Y, [0], [], [], []),
    (qml.PauliZ, gates.Z, [0], [], [], []),
    (qml.Hadamard, gates.H, [0], [], [], []),
    (qml.CNOT, gates.CNot, [0, 1], [], [], []),
    (qml.CZ, gates.CZ, [0, 1], [], [], []),
    (qml.CY, gates.CY, [0, 1], [], [], []),
    (qml.SWAP, gates.Swap, [0, 1], [], [], []),
    (qml.ECR, gates.ECR, [0, 1], [], [], []),
    (qml.CSWAP, gates.CSwap, [0, 1, 2], [], [], []),
    (qml.Toffoli, gates.CCNot, [0, 1, 2], [], [], []),
    (qml.PhaseShift, gates.PhaseShift, [0], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.RX, gates.Rx, [0], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.RY, gates.Ry, [0], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.RZ, gates.Rz, [0], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.SWAP, gates.Swap, [0, 1], [], [], []),
    (qml.CSWAP, gates.CSwap, [0, 1, 2], [], [], []),
    (qml.Toffoli, gates.CCNot, [0, 1, 2], [], [], []),
    (qml.ControlledPhaseShift, gates.CPhaseShift, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (CPhaseShift00, gates.CPhaseShift00, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (CPhaseShift01, gates.CPhaseShift01, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (CPhaseShift10, gates.CPhaseShift10, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (GPi, gates.GPi, [0], [2], ["a"], [FreeParameter("a")]),
    (GPi2, gates.GPi2, [0], [2], ["a"], [FreeParameter("a")]),
    (MS, gates.MS, [0, 1], [2, 3], ["a", "b"], [FreeParameter("a"), FreeParameter("b")]),
    (PSWAP, gates.PSwap, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.ECR, gates.ECR, [0, 1], [], [], []),
    (qml.ISWAP, gates.ISwap, [0, 1], [], [], []),
    (qml.IsingXY, gates.XY, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.IsingXX, gates.XX, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.IsingYY, gates.YY, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.IsingZZ, gates.ZZ, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (
        qml.AmplitudeDamping,
        noises.AmplitudeDamping,
        [0],
        [0.1],
        ["alpha"],
        [FreeParameter("alpha")],
    ),
    (
        qml.GeneralizedAmplitudeDamping,
        noises.GeneralizedAmplitudeDamping,
        [0],
        [0.1, 0.15],
        ["p_000", "p_001"],
        [FreeParameter("p_000"), FreeParameter("p_001")],
    ),
    (qml.PhaseDamping, noises.PhaseDamping, [0], [0.1], ["a"], [FreeParameter("a")]),
    (qml.DepolarizingChannel, noises.Depolarizing, [0], [0.1], ["a"], [FreeParameter("a")]),
    (qml.BitFlip, noises.BitFlip, [0], [0.1], ["a"], [FreeParameter("a")]),
    (qml.PhaseFlip, noises.PhaseFlip, [0], [0.1], ["a"], [FreeParameter("a")]),
    (
        qml.QubitUnitary,
        gates.Unitary,
        [0],
        [np.array([[0, 1], [1, 0]])],
        [],
        [np.array([[0, 1], [1, 0]])],
    ),
    (
        qml.QubitChannel,
        noises.Kraus,
        [0],
        [[np.array([[0, 0.8], [0.8, 0]]), np.array([[0.6, 0], [0, 0.6]])]],
        [],
        [[np.array([[0, 0.8], [0.8, 0]]), np.array([[0.6, 0], [0, 0.6]])]],
    ),
    (
        qml.QubitChannel,
        noises.Kraus,
        [0],
        [pnp.tensor([np.array([[0, 0.8], [0.8, 0]]), np.array([[0.6, 0], [0, 0.6]])])],
        [],
        [pnp.tensor([np.array([[0, 0.8], [0.8, 0]]), np.array([[0.6, 0], [0, 0.6]])])],
    ),
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

braket_result_types = [
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
    if isinstance(pl_op, (GPi, GPi2, MS)):
        assert (
            _braket_to_pl[
                re.match("^[a-z0-2]+", braket_gate.to_ir(qubits, ir_type=IRType.OPENQASM)).group(0)
            ]
            == pl_op.name
        )
    else:
        assert (
            _braket_to_pl[braket_gate.to_ir(qubits).__class__.__name__.lower().replace("_", "")]
            == pl_op.name
        )


@pytest.mark.parametrize(
    "pl_gate_fn, braket_gate_fn, qubits, pl_params, pl_param_names, expected_params",
    testdata_with_params,
)
def test_translate_operation_with_unique_params(
    pl_gate_fn, braket_gate_fn, qubits, pl_params, pl_param_names, expected_params
):
    """Tests that Braket operations are translated correctly"""
    pl_op = pl_gate_fn(*pl_params, wires=qubits)
    braket_gate = braket_gate_fn(*expected_params)
    assert (
        translate_operation(pl_op, use_unique_params=True, param_names=pl_param_names)
        == braket_gate
    )
    if isinstance(pl_op, (GPi, GPi2, MS)):
        assert (
            _braket_to_pl[
                re.match("^[a-z0-2]+", braket_gate.to_ir(qubits, ir_type=IRType.OPENQASM)).group(0)
            ]
            == pl_op.name
        )
    else:
        assert (
            _braket_to_pl[braket_gate.to_ir(qubits).__class__.__name__.lower().replace("_", "")]
            == pl_op.name
        )


@pytest.mark.parametrize("pl_cls, braket_cls, qubits, params, inv_params", testdata_inverses)
def test_translate_operation_inverse(pl_cls, braket_cls, qubits, params, inv_params):
    """Tests that inverse gates are translated correctly"""
    pl_op = qml.adjoint(pl_cls(*params, wires=qubits))
    braket_gate = braket_cls(*inv_params)
    assert translate_operation(pl_op) == braket_gate
    if isinstance(pl_op.base, (GPi, GPi2, MS)):
        op_name = _braket_to_pl[
            re.match(
                "^[a-z0-2]+",
                braket_gate.to_ir(qubits, ir_type=IRType.OPENQASM),
            )[0]
        ]
    else:
        op_name = _braket_to_pl[
            braket_gate.to_ir(qubits).__class__.__name__.lower().replace("_", "")
        ]

    assert f"Adjoint({op_name})" == pl_op.name


@pytest.mark.parametrize("pl_cls, braket_cls, qubit", testdata_named_inverses)
def test_translate_operation_named_inverse(pl_cls, braket_cls, qubit):
    """Tests that operations whose inverses are named Braket gates are inverted correctly"""
    pl_op = qml.adjoint(pl_cls(wires=[qubit]))
    braket_gate = braket_cls()
    assert translate_operation(pl_op) == braket_gate
    assert (
        _braket_to_pl[braket_gate.to_ir([qubit]).__class__.__name__.lower().replace("_", "")]
        == pl_op.name
    )


def test_translate_operation_iswap_inverse():
    """Tests that the iSwap gate is inverted correctly"""
    assert translate_operation(qml.adjoint(qml.ISWAP(wires=[0, 1]))) == gates.PSwap(3 * np.pi / 2)


@pytest.mark.parametrize(
    "return_type, braket_result_type", zip(pl_return_types, braket_result_types)
)
def test_translate_result_type_observable(return_type, braket_result_type):
    """Tests if a PennyLane return type that involves an observable is successfully converted into a
    Braket result using translate_result_type"""
    obs = qml.Hadamard(0)
    obs.return_type = return_type
    braket_result_type_calculated = translate_result_type(obs, [0], frozenset())

    assert braket_result_type == braket_result_type_calculated


@pytest.mark.parametrize(
    "pl_obs, braket_obs, targets, param_names",
    [
        (qml.Hadamard(0), observables.H(), [0], []),
        (qml.PauliX(0), observables.X(), [0], ["p_000"]),
        (
            qml.PauliX(0) @ qml.PauliY(1),
            observables.X() @ observables.Y(),
            [0, 1],
            ["p_000", "p_001", "p_003"],
        ),
    ],
)
def test_get_adjoint_gradient_result_type(pl_obs, braket_obs, targets, param_names):
    """Tests that an AdjointGradient result type is returned correctly"""
    braket_result_type_calculated = get_adjoint_gradient_result_type(
        pl_obs,
        targets,
        frozenset(["AdjointGradient"]),
        param_names,
    )
    braket_result_type = AdjointGradient(
        observable=braket_obs, target=targets, parameters=param_names
    )
    assert braket_result_type == braket_result_type_calculated


def test_get_adjoint_gradient_result_type_unsupported():
    """Tests if a NotImplementedError is raised by translate_result_type when a PennyLane state
    return type is converted while not supported by the device"""
    pl_obs = qml.Hadamard(0)
    targets = [0]
    param_names = ["p_000", "p_001"]
    with pytest.raises(NotImplementedError, match="Unsupported return type"):
        get_adjoint_gradient_result_type(pl_obs, targets, frozenset(), param_names)


def test_translate_result_type_hamiltonian_expectation():
    """Tests that a Hamiltonian is translated correctly"""
    obs = qml.Hamiltonian((2, 3), (qml.PauliX(wires=0), qml.PauliY(wires=1)))
    obs.return_type = ObservableReturnTypes.Expectation
    braket_result_type_calculated = translate_result_type(obs, [0], frozenset())
    braket_result_type = (Expectation(observables.X(), [0]), Expectation(observables.Y(), [1]))
    assert braket_result_type == braket_result_type_calculated


@pytest.mark.parametrize(
    "return_type", [ObservableReturnTypes.Variance, ObservableReturnTypes.Sample]
)
def test_translate_result_type_hamiltonian_unsupported_return(return_type):
    """Tests if a NotImplementedError is raised by translate_result_type
    with Hamiltonian observable and non-Expectation return type"""
    obs = qml.Hamiltonian((2, 3), (qml.PauliX(wires=0), qml.PauliY(wires=1)))
    obs.return_type = return_type
    with pytest.raises(NotImplementedError, match="unsupported for Hamiltonian"):
        translate_result_type(obs, [0], frozenset())


def test_translate_result_type_probs():
    """Tests if a PennyLane probability return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.Probability, wires=Wires([0]))
    braket_result_type_calculated = translate_result_type(mp, [0], frozenset())

    braket_result_type = Probability([0])

    assert braket_result_type == braket_result_type_calculated


def test_translate_result_type_state_vector():
    """Tests if a PennyLane state vector return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.State)
    braket_result_type_calculated = translate_result_type(
        mp, [], frozenset(["StateVector", "DensityMatrix"])
    )

    braket_result_type = StateVector()

    assert braket_result_type == braket_result_type_calculated


def test_translate_result_type_density_matrix():
    """Tests if a PennyLane density matrix return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.State)
    braket_result_type_calculated = translate_result_type(mp, [], frozenset(["DensityMatrix"]))

    braket_result_type = DensityMatrix()

    assert braket_result_type == braket_result_type_calculated


def test_translate_result_type_density_matrix_partial():
    """Tests if a PennyLane partial density matrix return type is successfully converted into a
    Braket result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.State, wires=[0])
    braket_result_type_calculated = translate_result_type(
        mp, [0], frozenset(["StateVector", "DensityMatrix"])
    )

    braket_result_type = DensityMatrix([0])

    assert braket_result_type == braket_result_type_calculated


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


def test_translate_result():
    result_dict = _result_meta()
    result_dict["resultTypes"] = [
        {"type": {"targets": [0], "type": "probability"}, "value": [0.5, 0.5]}
    ]
    targets = [0]
    result_dict["measuredQubits"]: targets
    result = GateModelQuantumTaskResult.from_string(json.dumps(result_dict))
    mp = MeasurementProcess(ObservableReturnTypes.Probability, wires=Wires([0]))
    translated = translate_result(result, mp, targets, frozenset())
    assert (translated == result.result_types[0].value).all()


def test_translate_result_hamiltonian():
    result_dict = _result_meta()
    result_dict["resultTypes"] = [
        {
            "type": {"observable": ["x", "y"], "targets": [0, 1], "type": "expectation"},
            "value": 2.0,
        },
        {
            "type": {"observable": ["x"], "targets": [1], "type": "expectation"},
            "value": 3.0,
        },
    ]
    targets = [0, 1]
    result_dict["measuredQubits"]: targets
    result = GateModelQuantumTaskResult.from_string(json.dumps(result_dict))
    ham = qml.Hamiltonian((2, 1), (qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(1)))
    ham.return_type = ObservableReturnTypes.Expectation
    translated = translate_result(result, ham, targets, frozenset())
    expected = 2 * result.result_types[0].value + result.result_types[1].value
    assert translated == expected


def _result_meta() -> dict:
    return {
        "braketSchemaHeader": {
            "name": "braket.task_result.gate_model_task_result",
            "version": "1",
        },
        "taskMetadata": {
            "braketSchemaHeader": {"name": "braket.task_result.task_metadata", "version": "1"},
            "id": "task_arn",
            "shots": 0,
            "deviceId": "default",
        },
        "additionalMetadata": {
            "action": {
                "braketSchemaHeader": {"name": "braket.ir.jaqcd.program", "version": "1"},
                "instructions": [{"control": 0, "target": 1, "type": "cnot"}],
            },
        },
    }


@pytest.mark.parametrize(
    "expected_braket_H, pl_H",
    [
        (
            2 * observables.X() @ observables.Y() @ observables.Z(),
            2 * qml.PauliX(wires=0) @ qml.PauliY(wires=1) @ qml.PauliZ(wires=2),
        ),
        (
            2 * (observables.X() @ observables.Y() @ observables.Z()),
            2 * (qml.PauliX(wires=0) @ qml.PauliY(wires=1) @ qml.PauliZ(wires=2)),
        ),
        (
            2 * observables.X() @ observables.Y() @ observables.Z() + 0.75 * observables.X(),
            2 * qml.PauliX(wires=0) @ qml.PauliY(wires=1) @ qml.PauliZ(wires=2)
            + 0.75 * qml.PauliX(0),
        ),
        (1.25 * observables.H(), 1.25 * qml.Hadamard(wires=0)),
    ],
)
def test_translate_hamiltonian_observable(expected_braket_H, pl_H):
    translated_braket_H = _translate_observable(pl_H)
    assert expected_braket_H == translated_braket_H


def test_translate_result_type_adjoint_gradient():
    print("not implemented yet")
