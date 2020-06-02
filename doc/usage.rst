.. _usage:

Plugin usage
############

This plugin provides three Braket devices for PennyLane:

* :class:`braket.pennylane_braket.AWSSimulatorDevice <~AWSSimulatorDevice>`: provides an PennyLane device for the Braket simulators

* :class:`braket.pennylane_braket.AWSIonQDevice <~AWSIonQDevice>`: provides an PennyLane device for the Braket IonQ QPU

* :class:`braket.pennylane_braket.AWSRigettiDevice <~AWSRigettiDevice>`: provides an PennyLane device for the Braket Rigetti QPU


Using the devices
=================

Once the Braket SDK and the plugin are installed, the devices
can be accessed straight away in PennyLane.

You can instantiate these devices in PennyLane as follows:

>>> import pennylane as qml
>>> dev_qs1 = qml.device("braket.simulator", s3_destination_folder=("my-bucket", "my-prefix"), backend="QS1", wires=2)
>>> dev_rigetti = qml.device("braket.rigetti", s3_destination_folder=("my-bucket", "my-prefix"), shots=1000, wires=3)
>>> dev_ionq= qml.device("braket.ionq", s3_destination_folder=("my-bucket", "my-prefix"), poll_timeout_seconds=3600, shots=1000, wires=3)

These devices can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.


Device options
==============

The Braket devices accept additional arguments beyond the PennyLane default device arguments:

* **s3_destination_folder** (*Tuple[str, str]*) -- A tuple of the S3 bucket and prefix where the results of the run will be stored. This must be provided.

* **poll_timeout_seconds** (*int*) -- Time in seconds to poll for results before timing out. Defaults to 120 for simulators and 86400 (1 day) for QPUs.

Additionally, ``AWSSimulatorDevice`` accepts

* **backend** (*str*) -- The simulator backend to target; can be one of "QS1", "QS2" or "QS3". Defaults to "QS3".

Supported operations
====================

All devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html#qubit-operations>`_, with the exception of the PennyLane ``QubitUnitary`` and ``Rot`` gates and ``Hermitian`` observable.

In addition, the plugin provides the following framework-specific operations for PennyLane. These are all importable from :mod:`braket.pennylane_braket.ops <.ops>`.

These operations include:

.. autosummary::
    braket.pennylane_braket.CPHASE
    braket.pennylane_braket.ISWAP
    braket.pennylane_braket.PSWAP
