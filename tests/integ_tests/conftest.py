# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import pytest

from pennylane_braket import AWSSimulatorDevice

np.random.seed(42)

# ==========================================================
# Some useful global variables

# single qubit unitary matrix
U = np.array([[0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
              [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j]])

# two qubit unitary matrix
U2 = np.array([[0, 1, 1, 1],
               [1, 0, 1, -1],
               [1, -1, 0, 1],
               [1, 1, -1, 0]]) / np.sqrt(3)

# single qubit Hermitian observable
A = np.array([[1.02789352, 1.61296440 - 0.3498192j],
              [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


# ==========================================================
# PennyLane devices

# List of all devices that support analytic expectation value
# computation. This generally includes statevector/wavefunction simulators.
analytic_devices = []

# List of all devices that do *not* support analytic expectation
# value computation. This generally includes hardware devices
# and hardware simulators.
hw_devices = [AWSSimulatorDevice]

# List of all device shortnames
shortnames = [d.short_name for d in analytic_devices + hw_devices]


# ==========================================================
# pytest options

def pytest_addoption(parser):
    parser.addoption("--bucket", action="store")
    parser.addoption("--prefix", action="store")


# ==========================================================
# pytest fixtures

@pytest.fixture
def s3(request):
    """
    S3 bucket and prefix, supplied as pytest arguments
    """
    config = request.config
    return config.getoption("--bucket"), config.getoption("--prefix")


@pytest.fixture
def tol(shots):
    """Numerical tolerance to be used in tests."""
    if shots == 0:
        # analytic expectation values can be computed,
        # so we can generally use a smaller tolerance
        return {"atol": 0.01, "rtol": 0}

    # for non-zero shots, there will be additional
    # noise and stochastic effects; will need to increase
    # the tolerance
    return {"atol": 0.05, "rtol": 0.1}


@pytest.fixture
def init_state(scope="session"):
    """Fixture to create an n-qubit initial state"""
    def _init_state(n):
        state = np.random.random([2 ** n]) + np.random.random([2 ** n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


@pytest.fixture(params=analytic_devices+hw_devices)
def device(request, shots, s3):
    """Fixture to initialize and return a PennyLane device"""
    device = request.param

    if device not in analytic_devices and shots == 0:
        pytest.skip("Hardware simulators do not support analytic mode")

    def _device(n):
        return device(
            wires=n,
            shots=shots,
            s3_destination_folder=s3,
        )

    return _device


def rotations(ops):
    """Returns the gates that diagonalize the measured wires such that they
    are in the eigenbasis of the circuit observables.

    Returns:
        List[~.Operation]: the operations that diagonalize the observables
    """
    rotation_gates = []

    for observable in ops:
        rotation_gates.extend(observable.diagonalizing_gates())

    return rotation_gates
