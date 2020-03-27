.. _usage:

Plugin usage
############

PennyLane-Braket provides three Braket devices for PennyLane:

* :class:`pennylane_braket.AWSSimulatorDevice <~AWSSimulatorDevice>`: provides an PennyLane device for the Braket simulators

* :class:`pennylane_braket.AWSIonQDevice <~AWSIonQDevice>`: provides an PennyLane device for the Braket IonQ QPU

* :class:`pennylane_braket.AWSRigettiDevice <~AWSRigettiDevice>`: provides an PennyLane device for the Braket Rigetti QPU


Using the devices
=================

Once the Braket SDK and the PennyLane-Braket plugin are installed, the devices
can be accessed straight away in PennyLane.

You can instantiate these devices in PennyLane as follows:

>>> import pennylane as qml
>>> dev_qs1 = qml.device('braket.simulator', backend='QS1' wires=2)
>>> dev_qpu = qml.device('braket.ionq', shots=1000, wires=3)

These devices can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.


Device options
==============

The Target Framework simulators accept additional arguments beyond the PennyLane default device arguments.

List available device options here.

``shots=0``
	The number of circuit evaluations/random samples used to estimate expectation values of observables.
	The default value of 0 means that the exact expectation value is returned.



Supported operations
====================

All devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html#qubit-operations>`_, with the exception of the PennyLane ``QubitUnitary`` and ``Rot`` gates and ``QubitStateVector`` state preparation operation.

In addition, the plugin provides the following framework-specific operations for PennyLane. These are all importable from :mod:`pennylane_braket.ops <.ops>`.

These operations include:

.. autosummary::
    pennylane_braket.CPHASE
    pennylane_braket.ISWAP
    pennylane_braket.PSWAP
