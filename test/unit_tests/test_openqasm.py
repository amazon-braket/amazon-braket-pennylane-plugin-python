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
from test_translation import testdata, testdata_named_inverses

from braket.circuits import (
    Circuit,
    FreeParameter,
    observables,
)
from braket.pennylane_plugin import BraketLocalQubitDevice
from braket.pennylane_plugin.braket_device import _builtin_number
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


# Reuse the SDK translation cases, with additional parameter types and complex matrices.
_OPERATIONS = (
    [cls(*params, wires=wires) for cls, _, wires, params in testdata]
    + [qp.adjoint(cls(wire)) for cls, _, wire in testdata_named_inverses]
    + [
        qp.RY(qp.numpy.tensor(0.2), 0),
        qp.RZ(np.float32(0.3), 0),
        qp.QubitUnitary(np.array([[1, 0], [0, 1j]]), 0),
        qp.QubitUnitary(np.kron([[0, 1], [1, 0]], [[1, 1], [1, -1]]) / np.sqrt(2), [1, 0]),
        qp.GeneralizedAmplitudeDamping(0.1, qp.numpy.tensor(0.2), 0),
    ]
)


def test_every_serialized_operation_is_tested():
    assert {operation.name for operation in _OPERATIONS} == set(_PENNYLANE_TO_OPENQASM_OPERATIONS)


@pytest.mark.parametrize("operation", _OPERATIONS, ids=lambda operation: operation.name)
def test_operations_match_sdk_serialization(operation):
    """Each operation serializes as the SDK serializes the Braket operator it translates to."""
    dev = BraketLocalQubitDevice(wires=3)
    tape = QuantumScript([operation], [qp.expval(qp.Z(0))])
    _assert_matches_sdk(dev, tape)
    if operation.parameters and not isinstance(operation, qp.operation.Channel):
        indices = frozenset(
            index for index, parameter in enumerate(operation.parameters) if np.ndim(parameter) == 0
        )
        if indices:
            _assert_matches_sdk(dev, tape, trainable_indices=indices)


def test_unsupported_operations_are_rejected():
    dev = BraketLocalQubitDevice(wires=1)
    tape = QuantumScript([qp.adjoint(qp.RX(0.1, 0))], [qp.expval(qp.Z(0))])
    with pytest.raises(NotImplementedError, match=r"does not support operation Adjoint\(RX\)"):
        dev._pl_to_openqasm(tape)


# Other measurements and observables are checked by the existing device tests and
# openqasm_matches_sdk. Only state results depend on the device's result-type capabilities.
@pytest.mark.parametrize(
    "measurement, supported",
    [
        (qp.state(), {"StateVector", "DensityMatrix"}),
        (qp.state(), {"DensityMatrix"}),
        (qp.state(), set()),
        (qp.density_matrix(wires=[1]), {"DensityMatrix"}),
        (qp.density_matrix(wires=[1]), set()),
        (qp.sample(qp.Y(0)), set()),
        (qp.expval(qp.Hamiltonian([0.5, -1.5], [qp.Z(0) @ qp.X(1), qp.Y(1)])), set()),
        (qp.var(qp.Hamiltonian([0.5, -1.5], [qp.Z(0), qp.Y(1)])), set()),
        (qp.counts(qp.Z(0), all_outcomes=True), set()),
        (qp.expval(qp.CNOT([0, 1])), set()),
    ],
)
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
    "measurements, trainable_params, supported",
    [
        ([qp.expval(3 * qp.Hermitian(np.diag([1.0, -1.0]), wires=1))], [0, 1], True),
        ([qp.expval(qp.Z(0))], [], True),
        ([qp.expval(qp.Z(0))], [0, 1], False),
        ([qp.expval(qp.Z(0)), qp.expval(qp.X(1))], [0, 1], True),
        ([qp.var(qp.Z(0))], [0, 1], True),
    ],
)
def test_gradients_match_sdk_serialization(measurements, trainable_params, supported):
    dev = BraketLocalQubitDevice(wires=[1, 0])
    dev._braket_result_types = {"AdjointGradient"} if supported else set()
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


def test_multi_wire_identity_acts_on_each_qubit():
    program = to_openqasm([(qp.Identity([0, 1]), [])], wire_map={0: 0, 1: 1})
    assert "i q[0];\ni q[1];" in program.source


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
