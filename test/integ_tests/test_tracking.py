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

"""Tests that device resource tracking are correctly computed in the plugin device"""

import numpy as np
import pennylane as qml
import pytest

from braket.pennylane_plugin.braket_device import MIN_SIMULATOR_BILLED_MS, BraketAwsQubitDevice


@pytest.mark.parametrize("shots", [100])
class TestDeviceTracking:
    """Tests for the device resource tracking"""

    def test_tracking(self, device, shots, tol):
        """Tests that a Device Tracker example correctly records resource usage"""

        dev = device(1)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.1)

        with qml.Tracker(circuit.device) as tracker:
            qml.grad(circuit)(x)

        print(tracker.history)
        print(tracker.totals)

        expected_totals = {"executions": 3, "shots": 300, "batches": 2, "batch_len": 3}
        expected_history = {
            "executions": [1, 1, 1],
            "shots": [100, 100, 100],
            "batches": [1, 1],
            "batch_len": [1, 2],
        }
        expected_latest = {"batches": 1, "batch_len": 2}

        for key, total in expected_totals.items():
            assert tracker.totals[key] == total
        for key, history in expected_history.items():
            assert tracker.history[key] == history
        assert tracker.latest == expected_latest

        assert len(tracker.history["braket_task_id"]) == 3

        if type(dev) == BraketAwsQubitDevice:
            durations = tracker.history["braket_simulator_ms"]
            billed_durations = tracker.history["braket_simulator_billed_ms"]
            assert len(durations) == 3
            assert len(billed_durations) == 3
            for duration, billed in zip(durations, billed_durations):
                assert (
                    duration < MIN_SIMULATOR_BILLED_MS and billed == MIN_SIMULATOR_BILLED_MS
                ) or duration == billed
