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

from setuptools import find_namespace_packages, setup

with open("src/braket/pennylane_plugin/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

setup(
    name="amazon-braket-pennyLane-plugin-python",
    version=version,
    license="Apache License 2.0",
    python_requires=">= 3.7.2",
    packages=find_namespace_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    install_requires=[
        "braket-sdk @ git+https://github.com/aws/braket-python-sdk.git",
        "pennylane>=0.8",
    ],
    entry_points={
        "pennylane.plugins": [
            # List the short name of each device provided by
            # the plugin, as well as the path to the Device class
            # it corresponds to in the plugin. This allows
            # the device to be imported automatically via the
            # `pennylane.device` device loader.
            "braket.simulator = braket.pennylane_plugin:AWSSimulatorDevice",
            "braket.ionq = braket.pennylane_plugin:AWSIonQDevice",
            "braket.rigetti = braket.pennylane_plugin:AWSRigettiDevice",
        ]
    },
    extras_require={
        "test": [
            "black",
            "flake8",
            "isort",
            "pre-commit",
            "pylint",
            "pytest",
            "pytest-cov",
            "pytest-rerunfailures",
            "pytest-xdist",
            "sphinx",
            "sphinx-rtd-theme",
            "tox",
        ]
    },
)
