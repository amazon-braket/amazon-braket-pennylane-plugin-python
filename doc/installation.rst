.. _installation:

Setup
#####

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


Software tests
==============

To ensure that Plugin Name is working correctly after installation, the unit test suite can be run by navigating to the source code folder and running
::

	$ make unit-test


Documentation
=============

To build the HTML documentation, go to the top-level directory and run
::

  $ make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.
