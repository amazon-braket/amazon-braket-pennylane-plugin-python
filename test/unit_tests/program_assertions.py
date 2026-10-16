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

"""Assertions for the OpenQASM programs that the plugin submits.

The plugin serializes circuits to OpenQASM rather than to ``braket.circuits.Circuit``. These
helpers let tests keep expressing the expected program as a readable Braket circuit.
"""

from braket.circuits import Circuit
from braket.circuits.serialization import IRType
from braket.program_sets import ProgramSet


def _canonical_source(program_or_circuit):
    # The Braket SDK holds a circuit's free parameters in a set, so the order it declares them in
    # is not stable; the plugin declares them in the order they are first used. Sorting the
    # declarations lets the two be compared.
    source = (
        program_or_circuit.to_ir(ir_type=IRType.OPENQASM).source
        if isinstance(program_or_circuit, Circuit)
        else program_or_circuit.source
    )
    lines = source.split("\n")
    declarations = sorted(line for line in lines if line.startswith("input "))
    others = [line for line in lines if not line.startswith("input ")]
    return "\n".join(others[:1] + declarations + others[1:])


def assert_same_program(program, circuit):
    """Asserts that an OpenQASM program is the one the given Braket circuit serializes to."""
    assert _canonical_source(program) == _canonical_source(circuit)


def assert_program_called_with(mock, circuit, **kwargs):
    """Asserts that the mock was called with the program for the given Braket circuit."""
    (program,), call_kwargs = mock.call_args
    assert_same_program(program, circuit)
    assert call_kwargs == kwargs


def assert_programs_called_with(mock, circuits, **kwargs):
    """Asserts that the mock was called with the programs for the given Braket circuits."""
    (programs,), call_kwargs = mock.call_args
    assert len(programs) == len(circuits)
    for program, circuit in zip(programs, circuits):
        assert_same_program(program, circuit)
    assert call_kwargs == kwargs


def assert_program_set_called_with(mock, circuits, inputs=None, **kwargs):
    """Asserts that the mock was called with the program set for the given Braket circuits, each
    program carrying the corresponding entry of ``inputs``, or no values if none are given."""
    (program_set,), call_kwargs = mock.call_args
    assert isinstance(program_set, ProgramSet)
    assert len(program_set.entries) == len(circuits)
    for program, circuit in zip(program_set.entries, circuits):
        assert_same_program(program, circuit)
    assert [program.inputs for program in program_set.entries] == (
        inputs if inputs is not None else [{}] * len(circuits)
    )
    assert call_kwargs == kwargs
