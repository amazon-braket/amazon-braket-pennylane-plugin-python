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

from unittest.mock import patch

import pytest
from openqasm_helpers import sorted_declarations

from braket.pennylane_plugin.braket_device import BraketQubitDevice

_PL_TO_OPENQASM = BraketQubitDevice._pl_to_openqasm
_OPERATIONS_TO_OPENQASM = BraketQubitDevice._operations_to_openqasm


class _SerializationRecorder:
    """Records each program serialized directly from PennyLane operations, with the device
    method that builds the equivalent Braket circuit from the same arguments."""

    def __init__(self):
        self.serialized = []

    def pl_to_openqasm(self, device, *args, **kwargs):
        program = _PL_TO_OPENQASM(device, *args, **kwargs)
        self.serialized.append((device._pl_to_braket_circuit, args, kwargs, program))
        return program

    def operations_to_openqasm(self, device, *args, **kwargs):
        program = _OPERATIONS_TO_OPENQASM(device, *args, **kwargs)
        self.serialized.append((device.apply, args, kwargs, program))
        return program


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item):
    report = yield
    if report.when == "call":
        item.call_report = report
    return report


@pytest.fixture(autouse=True)
def openqasm_matches_sdk(request):
    """Every program serialized without a Braket circuit must match the SDK's serialization of
    the equivalent circuit, including in tests that only check execution results."""
    recorder = _SerializationRecorder()
    with (
        patch.object(
            BraketQubitDevice,
            "_pl_to_openqasm",
            autospec=True,
            side_effect=recorder.pl_to_openqasm,
        ),
        patch.object(
            BraketQubitDevice,
            "_operations_to_openqasm",
            autospec=True,
            side_effect=recorder.operations_to_openqasm,
        ),
    ):
        yield
    # A test that did not pass already reports its own error
    report = getattr(request.node, "call_report", None)
    if report is None or not report.passed:
        return
    # Compared after the test, once its own patches of the SDK are undone
    for build_circuit, args, kwargs, program in recorder.serialized:
        assert program == sorted_declarations(build_circuit(*args, **kwargs).to_ir())
