.. _installation:

Setup
#####

Installation
============

The Amazon Braket PennyLane plugin requires both `PennyLane <https://pennylane.readthedocs.io>`_ and the `Amazon Braket Python SDK  <https://github.com/aws/braket-python-sdk/tree/stable/latest>`_. You should use only the stable/latest branch of the braket-python-sdk repository. Instructions for installing the Amazon Braket SDK are included in the README file for the repo.

After you install the Amazon Braket SDK, either clone or download the amazon-braket-pennylane-plugin-python repo to your local environment. You must clone or download the repo into a folder in the same virtual environment where you are using the Amazon Braket SDK.

Use the following command to clone the repo.

.. code-block:: bash

    $ git clone https://github.com/aws/amazon-braket-pennylane-plugin-python.git

Note that you must have a valid SSH key created in your local environment that has been added to your GitHub account to clone the repo.

You can also download the repo as a .zip file by using the **Clone or download** button.

After you add the repo to your local environment, install the plugin with the following ``pip`` command:

.. code-block:: bash

    $ pip install -e amazon-braket-pennylane-plugin-python


Software tests
==============

To ensure that the plugin is working correctly after installation, the test suites can be run by navigating to the source code folder and running
::

    $ make unit-test

for the unit tests and
::

    $ make integ-test S3_BUCKET=my-s3-bucket S3_PREFIX=my-s3-prefix

for the integ tests, replacing ``my-s3-bucket`` and ``my-s3-prefix`` with the name of your S3 bucket and the S3 key prefix
where you want to save results, respectively.


Documentation
=============

To build the HTML documentation, go to the top-level directory and run
::

    $ make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.
