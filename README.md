**This prerelease documentation is confidential and is provided under the terms of your nondisclosure agreement with Amazon Web Services (AWS) or other agreement governing your receipt of AWS confidential information.**

# PennyLane Braket plugin

[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/psf/black)

The Amazon Braket PennyLane plugin offers two Amazon Braket quantum devices to work with PennyLane:

* `braket.aws.qubit` for running with the Amazon Braket service's quantum devices, both QPUs and simulators
* `braket.local.qubit` for running with the Amazon Braket SDK's local simulator

The [Amazon Braket Python SDK](https://github.com/aws/amazon-braket-sdk-python) is an open source
library that provides a framework that you can use to interact with quantum computing hardware
devices through Amazon Braket.

[PennyLane](https://pennylane.readthedocs.io) is a machine learning library for optimization and automatic differentiation of hybrid quantum-classical computations.

**Providing Feedback and Getting Help**

To provide feedback or request support, please contact the Amazon Braket team at [amazon-braket-preview-support@amazon.com](mailto:amazon-braket-preview-support@amazon.com?subject=Add%20a%20brief%20description%20of%20the%20issue).

**Important**

If you **Star**, **Watch**, or submit a pull request for this repository, other users that have access to this repository are able to see your user name in the list of watchers. If you want to remain anonymous, you should not Watch or Star this repository, nor post any comments or submit a pull request.


## Features

* Provides two devices to be used with PennyLane: `braket.aws.qubit` for running on the Braket service, and `braket.local.qubit` for running on Braket's local simulator.

* Both devices support most core qubit PennyLane operations.

* All PennyLane observables are supported.

* Provides custom PennyLane operations to cover additional Braket operations: `ISWAP`, `PSWAP`, and many more. Every custom operation supports analytic
  differentiation.

* Combine Amazon Braket with PennyLane's automatic differentiation and optimization.


## Installation

The PennyLane-Braket plugin requires both [PennyLane](https://pennylane.readthedocs.io) and the [Amazon Braket Python SDK](https://github.com/aws/amazon-braket-sdk-python/). Instructions for installing the Amazon Braket SDK are included in the README file for the repo.

After you install the Amazon Braket SDK, either clone or download the amazon-braket-pennylane-plugin-python repo to your local environment. You must clone or download the repo into a folder in the same virtual environment where you are using the Amazon Braket SDK.

Use the following command to clone the repo.

```bash
git clone https://github.com/aws/amazon-braket-pennylane-plugin-python.git
```

Note that, to clone the repo, you must have a valid SSH key created in your local environment and added to your GitHub account.

You can also download the repo as a .zip file by using the **Clone or download** button. 

After you add the repo to your local environment, install the plugin with the following `pip` command:

```bash
pip install -e amazon-braket-pennylane-plugin
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

After the Braket SDK and the plugin are installed, you have access to the devices immediately in PennyLane.

To instantiate an AWS device that communicates with the Braket service:

```python
import pennylane as qml
s3 = ("my-bucket", "my-prefix")
sv1 = qml.device("braket.aws.qubit", device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", s3_destination_folder=s3, wires=2)
```

This example utilizes the string `arn:aws:braket:::device/quantum-simulator/amazon/sv1` as the ARN to identify the SV1 device. You can find other supported devices and their ARNs in the [Amazon Braket Developer Guide](https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html). Note that the plugin works with digital (qubit) circuit-based devices only.

To instantiate the Braket simulator that runs locally:

```python
import pennylane as qml
local = qml.device("braket.local.qubit", wires=2)
```

For both devices, you can set the `shots` argument to 0 (simulators only for `braket.aws.qubit`) to get exact analytic results instead of samples.

You can then use the device just as you would other devices to define and evaluate QNodes within PennyLane. For more details, refer to the PennyLane documentation.


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

Running the integration tests creates an S3 bucket in the same account as the `AWS_PROFILE` with the following naming convention `braket-pennylane-plugin-integ-tests-{account_id}`.

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
