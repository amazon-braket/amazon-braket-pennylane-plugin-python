PennyLane Braket plugin
#######################

Contains the PennyLane-Braket plugin. This plugin allows three AWS quantum devices to work with PennyLane:
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

* Provides custom PennyLane operations to cover additional Braket operations: ``V``, ``Vi``,
  ``ISWAP``, ``CCNOT``, ``PSWAP``, and many more. Every custom operation supports analytic
  differentiation.

* Combine Amazon Braket with PennyLane's automatic differentiation and optimization.


Installation
============

The PennyLane-Braket plugin requires both `PennyLane <https://pennylane.readthedocs.io>`_ and the `Amazon Braket Python SDK  <https://github.com/aws/braket-python-sdk/tree/stable/latest>`_. You should use only the stable/latest branch of the braket-python-sdk repository. Instructions for installing the Amazon Braket SDK are included in the Readme file for the repo.

After you install the Amazon Braket SDK, either clone or download the pennylane-braket repo to your local environment. You must clone or download the repo into a folder in the same virtual environment where you are using the Amazon Braket SDK.

Use the following command to clone the repo.

.. code-block:: bash

    $ git clone git@github.com:XanaduAI/pennylane-braket.git

Note that you must have a valid SSH key created in your local environment that has been added to your GitHub account to clone the repo.

You can also download the repo as a .zip file by using the **Clone or download** button. 

After you add the repo to your local environment, install the plugin with the following ``pip`` command:

.. code-block:: bash

    $ pip install -e pennylane-braket


Getting started
===============

Once the PennyLane-Braket plugin is installed, you can access the three supported Amazon Braket devices in PennyLane.

You can instantiate these devices for PennyLane as follows:

.. code-block:: python

    import pennylane as qml
    dev_qs1 = qml.device('braket.simulator', backend='QS1' wires=2)
    dev_qpu = qml.device('braket.ionq', shots=1000, wires=3)

You can use these devices just like other devices for the definition and evaluation of
QNodes within PennyLane. For more details, refer to the PennyLane documentation.


Contributing
============

We welcome contributions - simply fork the PennyLane-Braket repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your
contribution.  All contributers to PennyLane-Braket will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool
projects or applications built on PennyLane and Braket.


Testing
=======

Before making a pull request, verify the unit tests still pass by running

.. code-block:: bash

    make unit-test

Integration tests that make a full roundtrip through AWS can be run with

.. code-block:: bash

    make integ-test S3_BUCKET=my-s3-bucket S3_PREFIX=my-s3-prefix

replacing my-s3-bucket and my-s3-prefix with the name of your S3 bucket and the S3 key prefix
where you want to save results, respectively.


Support
=======

- **Source Code:** https://github.com/xanaduai/pennylane-braket
- **Issue Tracker:** https://github.com/xanaduai/pennylane-braket/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.


License
=======

PennyLane-Braket is **free** and **open source**, released under the Apache 2.0 license.
