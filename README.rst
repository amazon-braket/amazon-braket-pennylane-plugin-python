PennyLane Braket plugin
#######################

[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/psf/black)

.. header-start-inclusion-marker-do-not-remove

The Amazon Braket PennyLane plugin offers two Amazon Braket quantum devices to work with PennyLane:

* `braket.aws.qubit` for running with the Amazon Braket service's quantum devices, both QPUs and simulators
* `braket.local.qubit` for running with the Amazon Braket SDK's local simulator

The [Amazon Braket Python SDK](https://github.com/aws/amazon-braket-sdk-python) is an open source
library that provides a framework that you can use to interact with quantum computing hardware
devices through Amazon Braket.

[PennyLane](https://pennylane.readthedocs.io) is a machine learning library for optimization and automatic 
differentiation of hybrid quantum-classical computations.

.. header-end-inclusion-marker-do-not-remove

The plugin documentation can be found here: `<https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/>`__.

Features
========

* Provides two devices to be used with PennyLane: `braket.aws.qubit` for running on the Braket service, 
and `braket.local.qubit` for running on Braket's local simulator.

* Both devices support most core qubit PennyLane operations.

* All PennyLane observables are supported.

* Provides custom PennyLane operations to cover additional Braket operations: `ISWAP`, `PSWAP`, and many more. 
Every custom operation supports analytic
  differentiation.

* Combines Amazon Braket with PennyLane's automatic differentiation and optimization.


Installation
============

Before you begin working with the Amazon Braket PennyLane Plugin, make sure 
that you installed or configured the following prerequisites.

Python 3.7.2 or greater
-----------------------

Download and install Python 3.7.2 or greater from [Python.org](https://www.python.org/downloads/).
If you are using Windows, choose **Add Python to environment variables** before you begin the installation.

Amazon Braket SDK
-----------------

Make sure that your AWS account is onboarded to Amazon Braket, 
as per the instructions in the [README](https://github.com/aws/amazon-braket-sdk-python#prerequisites).

PennyLane
---------

Download and install [PennyLane](https://pennylane.ai/install.html):
```bash
pip install pennylane
```

Install the Amazon Braket PennyLane Plugin
------------------------------------------

You can install the latest release of the PennyLane-Braket plugin as follows:

XXX check!

```bash
pip install pennylane-braket
```

You can also install the development version from source by cloning this repository and running a 
pip install command in the root directory of the repository:

```bash
git clone https://github.com/aws/amazon-braket-pennylane-plugin-python.git
cd amazon-braket-pennylane-plugin-python
pip install .
```

You can check your currently installed version of `amazon-braket-pennylane-plugin` with `pip show`:

```bash
pip show amazon-braket-pennylane-plugin
```

or alternatively from within Python:

```
>>> from braket import pennylane_plugin
>>> pennylane_plugin.__version__
```


Documentation
-------------

**To generate the API Reference HTML in your local environment**

First, you must have tox installed.

```bash
pip install tox
```

Then, you can run the following command with tox to generate the documentation:

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
pip install -e "amazon-braket-pennylane-plugin-python[test]"
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
Set the `AWS_PROFILE`, as instructed in the amazon-braket-sdk-python [README](https://github.com/aws/amazon-braket-sdk-python/blob/main/README.md).
```bash
export AWS_PROFILE=Your_Profile_Name
```

Running the integration tests creates an S3 bucket in the same account as the `AWS_PROFILE` with the following naming convention `amazon-braket-pennylane-plugin-integ-tests-{account_id}`.

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
