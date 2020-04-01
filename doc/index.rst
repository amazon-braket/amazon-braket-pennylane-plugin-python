Amazon Braket PennyLane Plugin
##############################

:Release: |release|
:Date: |today|


The Amazon Braket PennyLane plugin allows three AWS quantum devices to work with PennyLane:
the AWS quantum simulator, as well as Rigetti and IonQ Quantum Processing Units (QPUs).

The `Amazon Braket Python SDK  <https://github.com/aws/braket-python-sdk>`_ is an open source
library that provides a framework that you can use to interact with quantum computing hardware
devices through Amazon Braket.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization and automatic differentiation of hybrid quantum-classical computations.


Features
========

* Provides three devices to be used with PennyLane: ``braket.simulator``, ``braket.ionq``,
  and ``braket.rigetti``. These devices provide access to the AWS Quantum Simulator, IonQ QPUs, and
  Rigetti QPUs respectively.

* All provided devices support most core qubit PennyLane operations.

* All PennyLane observables are supported with the exception of ``qml.Hermitian``.

* Provides custom PennyLane operations to cover additional Braket operations: ``ISWAP``, ``PSWAP``, and many more. Every custom operation supports analytic
  differentiation.

* Combine Amazon Braket with PennyLane's automatic differentiation and optimization.

To get started with the plugin, follow the :ref:`installation steps <installation>`, then see the :ref:`usage <usage>` page.

Authors
=======

`Amazon Braket <https://aws.amazon.com/braket/>`_, `Xanadu <https://www.xanadu.ai/>`_

If you are doing research using PennyLane, please cite these papers:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018.
    `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`_

    Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran.
    *Evaluating analytic gradients on quantum hardware.* 2018.
    `Phys. Rev. A 99, 032331 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331>`_


Contents
========

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   usage

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 1
   :caption: Code details

   code/ops
   code/braket_device

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 2
   :caption: Tutorials (external links)

   Demonstrations <https://pennylane.ai/qml/demonstrations.html>