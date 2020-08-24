**This prerelease documentation is confidential and is provided under the terms of your nondisclosure agreement with Amazon Web Services (AWS) or other agreement governing your receipt of AWS confidential information.**

# PennyLane Braket plugin

[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/psf/black)

The Amazon Braket PennyLane plugin allows three AWS quantum devices to work with PennyLane:
the AWS quantum simulator, as well as Rigetti and IonQ Quantum Processing Units (QPUs).

The [Amazon Braket Python SDK](https://github.com/aws/amazon-braket-sdk-python) is an open source
library that provides a framework that you can use to interact with quantum computing hardware
devices through Amazon Braket.

[PennyLane](https://pennylane.readthedocs.io) is a machine learning library for optimization and automatic differentiation of hybrid quantum-classical computations.

**Providing Feedback and Getting Help**

To provide feedback or request support, please contact the Amazon Braket team at [amazon-braket-preview-support@amazon.com](mailto:amazon-braket-preview-support@amazon.com?subject=Add%20a%20brief%20description%20of%20the%20issue).

**Important**

If you **Star**, **Watch**, or submit a pull request for this repository, other users that have access to this repository are able to see your user name in the list of watchers. If you want to remain anonymous, you should not Watch or Star this repository, nor post any comments or submit a pull request.


## Features

* Provides three devices to be used with PennyLane: `braket.simulator`, `braket.ionq`,
  and `braket.rigetti`. These devices provide access to the AWS Quantum Simulator, IonQ QPUs, and
  Rigetti QPUs respectively.

* All provided devices support most core qubit PennyLane operations.

* All PennyLane observables are supported with the exception of `qml.Hermitian`.

* Provides custom PennyLane operations to cover additional Braket operations: `ISWAP`, `PSWAP`, and many more. Every custom operation supports analytic
  differentiation.

* Combine Amazon Braket with PennyLane's automatic differentiation and optimization.


## Installation

The PennyLane-Braket plugin requires both [PennyLane](https://pennylane.readthedocs.io) and the [Amazon Braket Python SDK](https://github.com/aws/amazon-braket-sdk-python/). Instructions for installing the Amazon Braket SDK are included in the Readme file for the repo.

After you install the Amazon Braket SDK, either clone or download the amazon-braket-pennylane-plugin-python repo to your local environment. You must clone or download the repo into a folder in the same virtual environment where you are using the Amazon Braket SDK.

Use the following command to clone the repo.

```bash
git clone https://github.com/aws/amazon-braket-pennylane-plugin-python.git
```

Note that you must have a valid SSH key created in your local environment that has been added to your GitHub account to clone the repo.

You can also download the repo as a .zip file by using the **Clone or download** button. 

After you add the repo to your local environment, install the plugin with the following `pip` command:

```bash
pip install -e amazon-braket-pennylane-plugin-python
```

## Documentation

To download the documentation for the plugin, use the following command:
```bash
aws s3 cp s3://braket-external-assets-prod-us-west-2/sdk-docs/amazon-braket-pennylane-plugin-python-docs.zip amazon-braket-pennylane-plugin-python-docs.zip
``` 

Extract the downloaded .zip file, and then open ..\html\index.html in a browser.

You can also generate the documentation for the plugin with the following command:

```bash
tox -e docs
```

To view the generated documentation, open the following file in a browser: `PLUGIN_ROOT/build/documentation/html/index.html`

## Getting started

Once the plugin is installed, you can access the three supported Amazon Braket devices in PennyLane.

You can instantiate these devices for PennyLane as follows:

```python
import pennylane as qml

s3 = ("my-bucket", "my-prefix")
dev_qs1 = qml.device("braket.simulator", s3_destination_folder=s3, wires=2)
dev_rigetti = qml.device("braket.rigetti", s3_destination_folder=s3, shots=1000, wires=3)
dev_ionq = qml.device("braket.ionq", s3_destination_folder=s3, poll_timeout_seconds=3600, shots=1000, wires=3)
```

You can use these devices just like other devices for the definition and evaluation of
QNodes within PennyLane. For more details, refer to the PennyLane documentation.


## Testing

Make sure to install test dependencies first:
```bash
pip install -e "amazon-braket-pennyLane-plugin-python[test]"
```

### Unit Tests
```bash
tox -e unit-tests
```

To run an individual test
```
tox -e unit-tests -- -k 'your_test'
```

To run linters and doc generators and unit tests
```bash
tox
```

### Integration Tests
Set the `AWS_PROFILE`, similar to in the amazon-braket-sdk-python [README](https://github.com/aws/amazon-braket-sdk-python/blob/main/README.md).
```bash
export AWS_PROFILE=Your_Profile_Name
```

Running the integration tests will create an S3 bucket in the same account as the `AWS_PROFILE` with the following naming convention `braket-pennylane-plugin-integ-tests-{account_id}`.

Run the tests
```bash
tox -e integ-tests
```

To run an individual test
```bash
tox -e integ-tests -- -k 'your_test'
```

## License

This project is licensed under the Apache-2.0 License.
