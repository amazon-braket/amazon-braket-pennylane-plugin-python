.. _usage:

Plugin usage
############

This plugin provides two Braket devices to use with PennyLane:

* :class:`braket.pennylane_plugin.BraketAwsQubitDevice <~BraketAwsQubitDevice>`: provides a PennyLane device for running circuits on the Amazon Braket service
* :class:`braket.pennylane_plugin.BraketLocalQubitDevice <~BraketLocalQubitDevice>`: provides a PennyLane device for running circuits on the Braket SDK's local simulator

Using the devices
=================

Once the Braket SDK and the plugin are installed, the devices can be accessed straight away in PennyLane.

To instantiate an AWS device that communicates with the Braket service:

>>> import pennylane as qml
>>> s3 = ("my-bucket", "my-prefix")
>>> sv1 = qml.device("braket.aws.qubit", device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", s3_destination_folder=s3, wires=2)

In this example, the string ``arn:aws:braket:::device/quantum-simulator/amazon/sv1`` is the ARN used to identify the SV1 device. Other supported devices and their ARNs can be found in the `Amazon Braket Developer Guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`_. Note that the plugin only works with digital (qubit) circuit-based devices.

To instantiate the Braket simulator that runs locally:

>>> import pennylane as qml
>>> local = qml.device("braket.local.qubit", wires=2)

These devices can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.

Device options
==============

For both devices, ``shots`` can be set to 0 to get exact results instead of results calculated from samples. Note that for ``BraketAwsQubitDevice``, this only works for simulators.

The ``BraketAwsQubitDevice`` device accepts additional arguments beyond the PennyLane default device arguments:

* **device_arn** (*Tuple[str, str]*) -- A tuple of the S3 bucket and prefix where the results of the run will be stored. This must be provided.

* **s3_destination_folder** (*Tuple[str, str]*) -- A tuple of the S3 bucket and prefix where the results of the run will be stored. This must be provided.

* **poll_timeout_seconds** (*int*) -- Time in seconds to poll for results before timing out. Defaults to 432000 (5 days).

Supported operations
====================

All devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html#qubit-operations>`_, with the exception of the PennyLane ``QubitUnitary`` and ``Rot`` gates and ``Hermitian`` observable.

In addition, the plugin provides the following framework-specific operations for PennyLane. These are all importable from :mod:`braket.pennylane_plugin.ops <.ops>`.

These operations include:

.. autosummary::
    braket.pennylane_plugin.V
    braket.pennylane_plugin.CY
    braket.pennylane_plugin.CPhaseShift
    braket.pennylane_plugin.CPhaseShift00
    braket.pennylane_plugin.CPhaseShift01
    braket.pennylane_plugin.CPhaseShift10
    braket.pennylane_plugin.ISWAP
    braket.pennylane_plugin.PSWAP
    braket.pennylane_plugin.XY
    braket.pennylane_plugin.XX
    braket.pennylane_plugin.YY
    braket.pennylane_plugin.ZZ
