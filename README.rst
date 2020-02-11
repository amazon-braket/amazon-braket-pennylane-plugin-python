PennyLane Braket plugin
#######################

Contains the PennyLane-Braket plugin. This plugin allows three AWS devices to work with PennyLane
--- the AWS quantum simulator, as well as Rigetti and IonQ QPUs.

The `Amazon Braket Python SDK  <https://github.com/aws/braket-python-sdk>`_ is an open source
library that provides a framework that you can use to interact with quantum computing hardware
devices through Amazon Braket.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization and automatic differentiation of hybrid quantum-classical computations.


Features
========

* Provides three devices to be used with PennyLane: ``braket.simulator``, ``braket.ionq``,
and ``braket.rigetti``. These provide access to the AWS Quantum Simulator, IonQ QPUs, and
Rigetti QPUs respectively.


* All provided devices support most core qubit PennyLane operations.

* All PennyLane observables are supported with the exception of ``qml.Hermitian``.

* Provides custom PennyLane operations to cover additional Braket operations: ``V``, ``Vi``,
  ``ISWAP``, ``CCNOT``, ``PSWAP``, and many more. Every custom operation supports analytic
  differentiation.

* Combine Braket with PennyLane's automatic differentiation and optimization.


Installation
============

PennyLane-Braket requires both PennyLane and Braket. It can be installed via ``pip``:

.. code-block:: bash

    $ python -m pip install pennylane-braket


Getting started
===============

Once the PennyLane-Braket plugin is installed, the three provided Braket devices can be
accessed straight away in PennyLane.

You can instantiate these devices for PennyLane as follows:

.. code-block:: python

    import pennylane as qml
    dev_qs1 = qml.device('braket.simulator', backend='QS1' wires=2)
    dev_qpu = qml.device('braket.ionq', shots=1000)

These devices can then be used just like other devices for the definition and evaluation of
QNodes within PennyLane. For more details, refer to the PennyLane documentation.


Contributing
============

We welcome contributions - simply fork the PennyLane-Braket repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your
contribution.  All contributers to PennyLane-Braket will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool
projects or applications built on PennyLane and Braket.


Support
=======

- **Source Code:** https://github.com/xanaduai/pennylane-braket
- **Issue Tracker:** https://github.com/xanaduai/pennylane-braket/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.


License
=======

PennyLane-Braket is **free** and **open source**, released under the Apache 2.0 license.
