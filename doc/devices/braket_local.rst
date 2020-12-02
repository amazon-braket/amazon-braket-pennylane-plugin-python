The local Braket device
=======================

The local device of the PennyLane-Braket plugin runs quantum computations on the local Braket SDK. This
could be either utilizing the processors of your own PC, or from a server that hosts and executes a notebook.

The device is useful for small-scale simulations in which the time of sending a job to a remote service is higher
than simply executing the simulation locally. It can also be used for rapid prototyping before running a computation
on a paid-for remote service.

Usage
~~~~~

After the Braket SDK and the plugin are installed, you immediately have access to the Braket devices in PennyLane.

To instantiate the local Braket simulator, simply use:

>>> import pennylane as qml
>>> device_local = qml.device("braket.local.qubit", wires=2)

You can define and evaluate QNodes with these devices just as you would with any other PennyLane device.

A simple quantum function that returns the expectation value and variance of a measurement and
depends on three classical input parameters would look like:

.. code-block:: python

    @qml.qnode(device_local)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), var(qml.PauliZ(1))

You can then execute the circuit like any other function to get the quantum mechanical expectation value and variance:

>>> circuit(0.2, 0.1, 0.3)
array([0.97517033, 0.04904283])

Device options
~~~~~~~~~~~~~~

When using the remote simulator, you can set ``shots`` to 0 to get exact results instead of results calculated from samples.

The remote device accepts the following additional arguments beyond the PennyLane default device arguments:

* **device_arn** (*Tuple[str, str]*) -- A tuple of the S3 bucket and prefix where the results of the run will be stored. This value must be provided.

* **s3_destination_folder** (*Tuple[str, str]*) -- A tuple of the S3 bucket and prefix where the results of the run will be stored. This value must be provided.

* **poll_timeout_seconds** (*int*) -- Time in seconds to poll for results before timing out. Defaults to 432000 (5 days).

Supported operations
~~~~~~~~~~~~~~~~~~~~

The device support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html#qubit-operations>`_,
with the exception of the PennyLane ``QubitUnitary`` and ``Rot`` gates and ``Hermitian`` observable.

The plugin provides the following framework-specific operations for PennyLane, which can be imported
from :mod:`braket.pennylane_plugin.ops <.ops>`:

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
