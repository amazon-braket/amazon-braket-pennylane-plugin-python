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

"""Tests for OpenQASM generation."""

from braket.circuits import Circuit, FreeParameter, Instruction, gates, noises, observables
from braket.circuits.result_types import Expectation, Probability
from braket.circuits.serialization import IRType
from braket.pennylane_plugin.openqasm_translation import to_openqasm


def test_program_matches_circuit_serialization():
    """A program is serialized exactly as the Braket SDK serializes the equivalent circuit."""
    circuit = (
        Circuit()
        .h(0)
        .cnot(0, 1)
        .i(2)
        .probability(target=[0])
        .expectation(observable=observables.Z(1))
    )

    program = to_openqasm(
        [
            Instruction(gates.H(), 0),
            Instruction(gates.CNot(), [0, 1]),
            Instruction(gates.I(), 2),
        ],
        qubit_count=3,
        result_types=[Probability([0]), Expectation(observables.Z(1))],
    )

    assert program.source == circuit.to_ir(ir_type=IRType.OPENQASM).source


def test_terminal_measurements_match_circuit_serialization():
    """A program with no result types measures every qubit, as the Braket SDK does."""
    program = to_openqasm(
        [Instruction(gates.H(), 0), Instruction(gates.CNot(), [0, 1])],
        qubit_count=2,
        measure_all_qubits=True,
    )

    expected = Circuit().h(0).cnot(0, 1)
    assert program.source == expected.to_ir(ir_type=IRType.OPENQASM).source
    assert program.source.split("\n")[-2:] == ["b[0] = measure q[0];", "b[1] = measure q[1];"]


def test_noise_serialization():
    """Noise is serialized as a Braket pragma."""
    program = to_openqasm([Instruction(noises.BitFlip(0.1), 0)], qubit_count=1)
    assert "#pragma braket noise bit_flip(0.1) q[0]" in program.source


def test_free_parameters_are_declared_in_order_of_use():
    program = to_openqasm(
        [
            Instruction(gates.Rx(FreeParameter("p_2")), 0),
            Instruction(gates.Ry(FreeParameter("p_0")), 1),
            # A parameter used twice is only declared once
            Instruction(gates.Rz(FreeParameter("p_2")), 2),
        ],
        qubit_count=3,
    )
    assert program.source.index("input float p_2;") < program.source.index("input float p_0;")
    assert program.source.count("input float p_2;") == 1


def test_verbatim_program():
    """Verbatim programs wrap their instructions in a box and refer to qubits physically."""
    program = to_openqasm(
        [Instruction(gates.H(), 0), Instruction(gates.CNot(), [0, 1])],
        qubit_count=2,
        measure_all_qubits=True,
        verbatim=True,
    )

    expected = Circuit().add_verbatim_box(Circuit().h(0).cnot(0, 1))
    assert program.source == expected.to_ir(ir_type=IRType.OPENQASM).source


def test_physical_qubits():
    """Physical qubit references skip the qubit declaration, as the Braket SDK does."""
    program = to_openqasm(
        [Instruction(gates.H(), 0)],
        qubit_count=1,
        measure_all_qubits=True,
        physical_qubits=True,
    )
    assert "qubit[" not in program.source
    assert "h $0;" in program.source
    assert "b[0] = measure $0;" in program.source


def test_free_parameter_declarations_are_deterministic():
    """Unlike a Braket circuit, whose parameters are held in a set, the declaration order here
    does not vary between runs."""
    sources = {
        to_openqasm(
            [
                Instruction(gates.Rx(FreeParameter("p_2")), 0),
                Instruction(gates.Ry(FreeParameter("p_0")), 1),
                Instruction(gates.Rz(FreeParameter("p_1")), 2),
            ],
            qubit_count=3,
        ).source
        for _ in range(20)
    }
    assert len(sources) == 1
