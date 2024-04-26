===============================
Calibrated Classification Tools
===============================

Developed by Skylar Stolte of UF's APRIL Lab with Kyle Volle of Torch Technologies, this repository collects tools used for investigating data-driven calibration of ML classifiers.

More documenation to follow

Installation
============

This project uses `Poetry <https://python-poetry.org/docs/>`_ for dependency management and installation, make sure it is installed first. With poetry installed, cd to this directory and run

``$ poetry install``

.. caution::
    Poetry creates a virtual environment in ``/home/<user>/.cache/pypoetry/virtualenvs`` if you wish to use an external virtual environment (venv or conda), activate it before running any Poetry commands

    Poetry creates a poetry.lock file to ensure exact versions of dependencies are installed, if you are unable to resolve dependencies on your machine try deleting this file and rerunning

``$ poetry install``
