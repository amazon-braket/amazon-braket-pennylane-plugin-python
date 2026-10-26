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

"""Compare direct serialization with the SDK and execute the resulting programs."""

import re
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pennylane as qp
import pytest
from openqasm_helpers import sorted_declarations
from pennylane.tape import QuantumScript

from braket.circuits import (
    Circuit,
    FreeParameter,
    observables,
)
from braket.pennylane_plugin import BraketLocalQubitDevice
from braket.pennylane_plugin.braket_device import _builtin_number
from braket.pennylane_plugin.ops import (
    AAMS,
    MS,
    PSWAP,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
    GPi,
    GPi2,
    PRx,
)
from braket.pennylane_plugin.translation import _PENNYLANE_TO_OPENQASM_OPERATIONS, to_openqasm


def _assert_matches_sdk(dev, tape, **kwargs):
    """The program, or the error, matches the SDK's for the equivalent circuit."""
    try:
        expected = sorted_declarations(dev._pl_to_braket_circuit(tape, **kwargs).to_ir())
    except Exception as error:  # noqa: BLE001
        with pytest.raises(type(error), match=re.escape(str(error))):
            dev._pl_to_openqasm(tape, **kwargs)
    else:
        assert dev._pl_to_openqasm(tape, **kwargs) == expected


# An instance of every operation that is serialized directly
_OPERATIONS = [
    qp.Identity(0),
    qp.Hadamard(0),
    qp.PauliX(0),
    qp.PauliY(0),
    qp.PauliZ(0),
    qp.ECR([0, 1]),
    qp.S(0),
    qp.adjoint(qp.S(0)),
    qp.SX(0),
    qp.adjoint(qp.SX(0)),
    qp.T(0),
    qp.adjoint(qp.T(0)),
    qp.CNOT([1, 0]),
    qp.CY([0, 1]),
    qp.CZ([0, 1]),
    qp.SWAP([0, 1]),
    qp.CSWAP([0, 1, 2]),
    qp.Toffoli([2, 0, 1]),
    qp.RX(0.1, 0),
    qp.RY(qp.numpy.tensor(0.2), 0),
    qp.RZ(np.float32(0.3), 0),
    qp.PhaseShift(0.4, 1),
    qp.QubitUnitary(np.array([[1, 0], [0, 1j]]), 0),
    qp.QubitUnitary(np.kron([[0, 1], [1, 0]], [[1, 1], [1, -1]]) / np.sqrt(2), [1, 0]),
    qp.AmplitudeDamping(0.1, 0),
    qp.GeneralizedAmplitudeDamping(0.1, qp.numpy.tensor(0.2), 0),
    qp.PhaseDamping(np.float64(0.1), 0),
    qp.DepolarizingChannel(0.1, 0),
    qp.BitFlip(qp.numpy.tensor(0.1), 0),
    qp.PhaseFlip(0.1, 0),
    qp.QubitChannel([np.sqrt(0.9) * np.eye(2), np.sqrt(0.1) * np.array([[0, 1], [1, 0]])], 0),
    qp.ControlledPhaseShift(0.1, [0, 1]),
    CPhaseShift00(0.1, [0, 1]),
    CPhaseShift01(0.1, [0, 1]),
    CPhaseShift10(0.1, [0, 1]),
    qp.ISWAP([0, 1]),
    PSWAP(0.1, [0, 1]),
    qp.IsingXY(0.1, [0, 1]),
    qp.IsingXX(0.1, [0, 1]),
    qp.IsingYY(0.1, [0, 1]),
    qp.IsingZZ(0.1, [0, 1]),
    PRx(0.1, 0.2, 0),
    GPi(0.1, 0),
    GPi2(0.1, 0),
    MS(0.1, 0.2, [0, 1]),
    AAMS(0.1, 0.2, 0.3, [0, 1]),
]


def test_every_serialized_operation_is_tested():
    assert {operation.name for operation in _OPERATIONS} == set(_PENNYLANE_TO_OPENQASM_OPERATIONS)


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("operation", _OPERATIONS, ids=lambda operation: operation.name)
def test_operations_match_sdk_serialization(operation, trainable):
    """Each operation serializes as the SDK serializes the Braket operator it translates to."""
    dev = BraketLocalQubitDevice(wires=3)
    tape = QuantumScript([operation], [qp.expval(qp.Z(0))])
    # Only scalar parameters are trainable
    indices = frozenset(
        index
        for index, parameter in enumerate(operation.parameters)
        if trainable and np.ndim(parameter) == 0
    )
    _assert_matches_sdk(dev, tape, trainable_indices=indices)


def test_unsupported_operations_are_rejected():
    dev = BraketLocalQubitDevice(wires=1)
    tape = QuantumScript([qp.adjoint(qp.RX(0.1, 0))], [qp.expval(qp.Z(0))])
    with pytest.raises(NotImplementedError, match=r"does not support operation Adjoint\(RX\)"):
        dev._pl_to_openqasm(tape)


_RESULT_TYPES = frozenset(
    {"Expectation", "Variance", "Sample", "Probability", "StateVector", "DensityMatrix"}
)

# Measurements of every kind that is serialized directly, including unsupported ones
_MEASUREMENTS = [
    qp.probs(wires=[1, 0]),
    qp.probs(),
    qp.density_matrix(wires=[1]),
    qp.state(),
    qp.counts(),
    qp.counts(wires=[1, 0]),
    qp.counts(all_outcomes=True),
    qp.counts(qp.Z(0), all_outcomes=True),
    qp.mutual_info(wires0=[0], wires1=[1]),
    qp.expval(qp.Z(0)),
    qp.var(qp.X(1)),
    qp.sample(qp.Y(0)),
    qp.counts(qp.Hadamard(1)),
    qp.expval(qp.Identity([0, 1])),
    qp.expval(qp.Hermitian(np.array([[0, 1j], [-1j, 0]]), wires=1)),
    qp.expval(qp.Projector([1, 0], wires=[0, 1])),
    qp.expval(qp.Projector(np.array([1, 0, 0, 1]) / np.sqrt(2), wires=[0, 1])),
    qp.expval(qp.Z(1) @ qp.Hermitian(np.diag([1.0, -1.0]), wires=0)),
    qp.expval(2 * qp.Z(0)),
    qp.expval(qp.Hamiltonian([0.5, -1.5], [qp.Z(0) @ qp.X(1), qp.Y(1)])),
    qp.var(qp.Hamiltonian([0.5, -1.5], [qp.Z(0), qp.Y(1)])),
    qp.expval(qp.CNOT([0, 1])),
]


@pytest.mark.parametrize(
    "supported",
    [
        _RESULT_TYPES,
        _RESULT_TYPES - {"StateVector"},
        _RESULT_TYPES - {"StateVector", "DensityMatrix"},
    ],
    ids=["all", "density matrix", "neither"],
)
@pytest.mark.parametrize("measurement", _MEASUREMENTS, ids=repr)
def test_measurements_match_sdk_serialization(measurement, supported):
    # Wires that map to other device qubits
    dev = BraketLocalQubitDevice(wires=[1, 0])
    dev._braket_result_types = supported
    tape = QuantumScript([qp.Hadamard(0), qp.CNOT([0, 1])], [measurement])
    _assert_matches_sdk(dev, tape)


def test_noncommuting_measurement_rotations_match_sdk():
    dev = BraketLocalQubitDevice(wires=1)
    tape = QuantumScript([qp.Hadamard(0)], [qp.expval(qp.X(0)), qp.expval(qp.Z(0))])
    _assert_matches_sdk(dev, tape, add_observables=False)


@pytest.mark.parametrize(
    "measurements",
    [
        [qp.expval(qp.Z(0))],
        [qp.expval(2 * qp.Z(0))],
        [qp.expval(qp.X(0) @ qp.Y(1))],
        [qp.expval(qp.Hamiltonian([2.0, -0.75, 1], [qp.X(0) @ qp.Y(1), qp.Y(0), qp.X(1)]))],
        [qp.expval(qp.Hamiltonian(qp.numpy.tensor([0.5, 0.25]), [qp.Z(0), qp.X(1)]))],
        [qp.expval(qp.Hermitian(np.diag([1.0, -1.0]), wires=1))],
        [qp.expval(3 * qp.Hermitian(np.diag([1.0, -1.0]), wires=1))],
        [qp.expval(qp.Projector([1], wires=[0]))],
        [qp.expval(qp.Identity([0, 1]))],
        [qp.expval(qp.Z(0)), qp.expval(qp.X(1))],
        [qp.var(qp.Z(0))],
    ],
    ids=repr,
)
@pytest.mark.parametrize("trainable_params", [[0, 1], []])
@pytest.mark.parametrize("supported", [True, False])
def test_gradients_match_sdk_serialization(measurements, trainable_params, supported):
    dev = BraketLocalQubitDevice(wires=[1, 0])
    dev._braket_result_types = _RESULT_TYPES | ({"AdjointGradient"} if supported else set())
    tape = QuantumScript(
        [qp.RX(0.1, 0), qp.RY(0.2, 1)], measurements, trainable_params=trainable_params
    )
    _assert_matches_sdk(
        dev, tape, compute_gradient=True, trainable_indices=frozenset(trainable_params)
    )


def test_numeric_parameters_are_not_truncated():
    """Braket validates program inputs as built-in numbers, which truncates other types."""
    dev = BraketLocalQubitDevice(wires=1)
    tape = QuantumScript([qp.RX(np.float32(0.5), wires=0)], [qp.expval(qp.Z(0))])
    assert np.isclose(dev.execute(tape), np.cos(0.5), atol=1e-3)


@pytest.mark.parametrize(
    "value, expected",
    [(np.int64(2), 2), (Fraction(1, 2), 0.5), (np.longdouble(0.5), 0.5), (np.complex64(1j), 1j)],
)
def test_inputs_are_builtin_numbers(value, expected):
    converted = _builtin_number(value)
    assert type(converted) is type(expected)
    assert converted == expected


def test_execution_does_not_construct_sdk_circuits():
    dev = BraketLocalQubitDevice(wires=["first", "second"])
    tape = QuantumScript(
        [qp.Hadamard("first"), qp.CNOT(["first", "second"])],
        [qp.expval(qp.Z("first") @ qp.Z("second"))],
    )
    with (
        patch.object(Circuit, "__init__", side_effect=AssertionError("Circuit constructed")),
        patch.object(Circuit, "to_ir", side_effect=AssertionError("Circuit serialized")),
    ):
        assert np.isclose(dev.execute(tape), 1)
    expected = Circuit().h(0).cnot(0, 1).expectation(observables.Z(0) @ observables.Z(1))
    assert dev.circuit == expected.to_ir()


def test_empty_program_is_rejected():
    with pytest.raises(ValueError, match="at least one non-zero-qubit gate"):
        to_openqasm([], wire_map={})


def test_unbound_parameters_fail_before_submission():
    dev = BraketLocalQubitDevice(wires=1)
    tape = QuantumScript([qp.RX(FreeParameter("theta"), wires=0)], [qp.expval(qp.Z(0))])
    with (
        patch.object(dev._device, "run") as run,
        pytest.raises(ValueError, match=r"unbound parameters: \{'theta'\}"),
    ):
        dev.execute(tape)
    run.assert_not_called()
