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

"""SDK reference serialization for execution tests."""

from braket.circuits import Circuit
from braket.ir.openqasm import Program
from braket.program_sets import ProgramSet


def sorted_declarations(program):
    """The SDK declares free parameters in set order; direct serialization sorts them."""
    lines = program.source.splitlines()
    declarations = sorted(line for line in lines if line.startswith("input "))
    statements = [line for line in lines if not line.startswith("input ")]
    return Program(
        source="\n".join(statements[:1] + declarations + statements[1:]), inputs=program.inputs
    )


def as_openqasm(specification):
    if isinstance(specification, Circuit):
        return sorted_declarations(specification.to_ir())
    if isinstance(specification, ProgramSet):
        return ProgramSet(
            [sorted_declarations(program) for program in specification.to_ir().programs],
            shots_per_executable=specification.shots_per_executable,
        )
    if isinstance(specification, list):
        return [as_openqasm(item) for item in specification]
    return specification


def openqasm_source(specification):
    return as_openqasm(specification).source
