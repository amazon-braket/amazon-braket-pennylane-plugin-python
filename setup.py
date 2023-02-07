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

from setuptools import find_namespace_packages, setup

with open("README.rst", "r") as fh:
    long_description = fh.read()

with open("src/braket/pennylane_plugin/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

setup(
    name="amazon-braket-pennylane-plugin",
    version=version,
    license="Apache License 2.0",
    python_requires=">= 3.7.2",
    packages=find_namespace_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    install_requires=[
        "amazon-braket-sdk>=1.35.0",
        "pennylane>=0.25.1,<=0.27.0",
    ],
    entry_points={
        "pennylane.plugins": [
            # List the short name of each device provided by
            # the plugin, as well as the path to the Device class
            # it corresponds to in the plugin. This allows
            # the device to be imported automatically via the
            # `pennylane.device` device loader.
            "braket.aws.qubit = braket.pennylane_plugin:BraketAwsQubitDevice",
            "braket.local.qubit = braket.pennylane_plugin:BraketLocalQubitDevice",
        ]
    },
    extras_require={
        "test": [
            "black",
            "docutils<0.16,>=0.10",
            "flake8",
            "isort",
            "pre-commit",
            "pylint",
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "pytest-rerunfailures",
            "pytest-xdist",
            "sphinx",
            "sphinx-automodapi",
            "sphinx-rtd-theme",
            "sphinxcontrib-apidoc",
            "tox",
            "tensorflow>=2.6.0",
            "torch>=1.11",
        ]
    },
    url="https://github.com/aws/amazon-braket-pennylane-plugin-python",
    author="Amazon Web Services",
    description=(
        "An open source framework for using Amazon Braket devices with the PennyLane"
        " quantum machine learning library"
    ),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    keywords="Amazon AWS Quantum",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
