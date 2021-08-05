# Copyright 2019-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import inspect
import os

import boto3
import numpy as np
import pytest
from botocore.exceptions import ClientError

from braket.pennylane_plugin import BraketAwsQubitDevice, BraketLocalQubitDevice

DEVICE_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"

np.random.seed(42)

# ==========================================================
# Some useful global variables

# single qubit unitary matrix
U = np.array(([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]]))

# two qubit unitary matrix
U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)

# single qubit Hermitian observable
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

# single qubit Kraus operator
K = [
    np.array([[0.4 - 0.4j, 0.4 + 0.4j], [0.4 + 0.4j, 0.4 - 0.4j]]),
    np.array([[0, 0.6j], [-0.6j, 0]]),
]

# two qubit Kraus operator
K2 = [np.kron(mat1, mat2) for mat1 in K for mat2 in K]

# ==========================================================
# PennyLane devices

# List of all devices.
sv_devices = [(BraketAwsQubitDevice, DEVICE_ARN), (BraketLocalQubitDevice, "braket_sv")]
dm_devices = [(BraketLocalQubitDevice, "braket_dm")]
devices = sv_devices + dm_devices

# List of all device shortnames
shortname_and_backends = [(d.short_name, backend) for (d, backend) in devices]

# List of local devices
local_devices = [(BraketLocalQubitDevice, "braket_sv"), (BraketLocalQubitDevice, "braket_sv")]

# ==========================================================
# AWS resources

session = boto3.session.Session(profile_name=os.environ["AWS_PROFILE"])
account_id = session.client("sts").get_caller_identity()["Account"]
bucket_name = f"amazon-braket-pennylane-plugin-integ-tests-{account_id}"
s3_bucket = session.resource("s3").Bucket(bucket_name)
s3_client = session.client("s3")

# Create bucket if it doesn't exist
try:
    # Determine if bucket exists
    s3_client.head_bucket(Bucket=bucket_name)
except ClientError as e:
    error_code = e.response["Error"]["Code"]
    if error_code == "404":
        s3_bucket.create(
            ACL="private", CreateBucketConfiguration={"LocationConstraint": session.region_name}
        )


# ==========================================================
# pytest fixtures


@pytest.fixture
def s3():
    """
    S3 bucket and prefix, supplied as pytest arguments
    """
    current_test_path = os.environ.get("PYTEST_CURRENT_TEST")
    s3_prefix = current_test_path.rsplit(".py")[0].replace("test/", "")
    return bucket_name, s3_prefix


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


@pytest.fixture(params=devices)
def device(request, shots, extra_kwargs):
    """Fixture to initialize and return a PennyLane device"""
    device, backend = request.param

    def _device(n):
        return device(wires=n, shots=shots, **extra_kwargs(device, backend))

    return _device


@pytest.fixture(params=sv_devices)
def sv_device(request, shots, extra_kwargs):
    """Fixture to initialize and return a PennyLane device"""
    device, backend = request.param

    def _device(n):
        return device(wires=n, shots=shots, **extra_kwargs(device, backend))

    return _device


@pytest.fixture(params=dm_devices)
def dm_device(request, shots, extra_kwargs):
    """Fixture to initialize and return a PennyLane device"""
    device, backend = request.param

    def _device(n):
        return device(wires=n, shots=shots, **extra_kwargs(device, backend))

    return _device


@pytest.fixture(params=local_devices)
def local_device(request, shots, extra_kwargs):
    """Fixture to initialize and return a PennyLane device"""
    device, backend = request.param

    def _device(n):
        return device(wires=n, shots=shots, **extra_kwargs(device, backend))

    return _device


@pytest.fixture
def extra_kwargs(s3):
    """Fixture to determine extra kwargs for devices"""

    def _extra_kwargs(device_class, backend):
        signature = inspect.signature(device_class).parameters
        kwargs = {}
        if "device_arn" in signature:
            kwargs["device_arn"] = backend
        else:
            kwargs["backend"] = backend

        if "s3_destination_folder" in signature:
            kwargs["s3_destination_folder"] = s3
        return kwargs

    return _extra_kwargs
