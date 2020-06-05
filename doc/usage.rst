.. _usage:

Plugin usage
############

This plugin provides three Braket devices for PennyLane:

* :class:`braket.pennylane_plugin.AWSSimulatorDevice <~AWSSimulatorDevice>`: provides a PennyLane device for the Braket simulators

* :class:`braket.pennylane_plugin.AWSIonQDevice <~AWSIonQDevice>`: provides a PennyLane device for the Braket IonQ QPU

* :class:`braket.pennylane_plugin.AWSRigettiDevice <~AWSRigettiDevice>`: provides a PennyLane device for the Braket Rigetti QPU


Using the devices
=================

Once the Braket SDK and the plugin are installed, the devices
can be accessed straight away in PennyLane.

You can instantiate these devices in PennyLane as follows:

>>> import pennylane as qml
>>> s3 = ("my-bucket", "my-prefix")
>>> dev_qs1 = qml.device("braket.simulator", s3_destination_folder=s3, wires=2)
>>> dev_rigetti = qml.device("braket.rigetti", s3_destination_folder=s3, shots=1000, wires=3)
>>> dev_ionq= qml.device("braket.ionq", s3_destination_folder=s3, poll_timeout_seconds=86400, shots=1000, wires=3)

These devices can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.


Device options
==============

The Braket devices accept additional arguments beyond the PennyLane default device arguments:

* **s3_destination_folder** (*Tuple[str, str]*) -- A tuple of the S3 bucket and prefix where the results of the run will be stored. This must be provided.

* **poll_timeout_seconds** (*int*) -- Time in seconds to poll for results before timing out. Defaults to 432000 (5 days).

Additionally, ``AWSSimulatorDevice`` accepts

* **backend** (*str*) -- The simulator backend to target; only "QS1" is supported at the moment. Defaults to "QS1".

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
