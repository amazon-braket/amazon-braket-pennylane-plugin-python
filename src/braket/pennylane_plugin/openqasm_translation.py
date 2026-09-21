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

"""
===================
OpenQASM generation
===================

**Module name:** :mod:`braket.pennylane_plugin.openqasm_translation`

.. currentmodule:: braket.pennylane_plugin.openqasm_translation

Serializes PennyLane operations into OpenQASM 3 programs for Amazon Braket devices.

Functions
---------

.. autosummary::
   to_openqasm

Code details
~~~~~~~~~~~~
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from sympy import Expr

from braket.circuits import Instruction, ResultType
from braket.circuits.free_parameter_expression import FreeParameterExpression
from braket.circuits.parameterizable import Parameterizable
from braket.circuits.serialization import (
    IRType,
    OpenQASMSerializationProperties,
    QubitReferenceType,
)
from braket.ir.openqasm import Program as OpenQASMProgram

_MEASUREMENT_REGISTER = "b"


def to_openqasm(
    instructions: Sequence[Instruction],
    *,
    qubit_count: int,
    result_types: Sequence[ResultType] = (),
    measure_all_qubits: bool = False,
    verbatim: bool = False,
    physical_qubits: bool = False,
) -> OpenQASMProgram:
    """Serializes Braket instructions and result types into an OpenQASM 3 program.

    Each instruction and result type is serialized with the Braket SDK's own OpenQASM
    serialization, so the gate, noise and result type syntax is always in sync with the SDK, and
    the program is the same as the one the SDK produces for the equivalent circuit.

    Pulse gates are not supported: they need a header declaring the frames and waveforms they
    share, which only the Braket SDK generates, and only from a circuit.

    Args:
        instructions (Sequence[Instruction]): The instructions to apply.
        qubit_count (int): The number of qubits to declare.
        result_types (Sequence[ResultType]): The result types to request. Default: no result types.
        measure_all_qubits (bool): Whether to measure every qubit at the end of the program, so
            that the results can be computed from samples. Default: False.
        verbatim (bool): Whether to wrap the instructions in a verbatim box. Verbatim boxes
            require physical qubit references. Default: False.
        physical_qubits (bool): Whether to refer to qubits by their physical index, which stops
            the device from rewiring them. Default: False.

    Returns:
        OpenQASMProgram: The OpenQASM program.
    """
    properties = _serialization_properties(verbatim or physical_qubits)
    body = [_serialize(instruction, properties) for instruction in instructions]
    if verbatim:
        body = ["#pragma braket verbatim", "box{", *body, "}"]

    source = [
        "OPENQASM 3.0;",
        *(f"input float {parameter};" for parameter in _free_parameters(instructions)),
        *([f"bit[{qubit_count}] {_MEASUREMENT_REGISTER};"] if measure_all_qubits else []),
        *([] if verbatim or physical_qubits else [f"qubit[{qubit_count}] q;"]),
        *body,
        *(_serialize(result_type, properties) for result_type in result_types),
        *(
            f"{_MEASUREMENT_REGISTER}[{qubit}] = measure {properties.format_target(qubit)};"
            for qubit in (range(qubit_count) if measure_all_qubits else ())
        ),
    ]
    # The Braket SDK attaches an empty input map to the programs it serializes circuits to
    return OpenQASMProgram(source="\n".join(source), inputs={})


def _serialization_properties(physical_qubits: bool) -> OpenQASMSerializationProperties:
    return OpenQASMSerializationProperties(
        qubit_reference_type=(
            QubitReferenceType.PHYSICAL if physical_qubits else QubitReferenceType.VIRTUAL
        )
    )


def _serialize(
    serializable: Instruction | ResultType, properties: OpenQASMSerializationProperties
) -> str:
    return serializable.to_ir(ir_type=IRType.OPENQASM, serialization_properties=properties)


def _free_parameters(instructions: Iterable[Instruction]) -> list[str]:
    names = (
        symbol.name
        for instruction in instructions
        if isinstance(instruction.operator, Parameterizable)
        for parameter in instruction.operator.parameters
        if isinstance(parameter, FreeParameterExpression) and isinstance(parameter.expression, Expr)
        for symbol in sorted(parameter.expression.free_symbols, key=str)
    )
    return list(dict.fromkeys(names))
