Installation
============
The GLAM framework is written in Python and leverages high-performance libraries for spatial indexing (KD-trees) and medical image analysis. It operates as a fully standalone extraction engine, meaning it does not require external radiomics packages to compute conventional texture matrices.

Prerequisites
-------------

Before installing GLAM, ensure you have the following requirements:

* **Python**: Version 3.10 or higher.
* **Pip**: The Python package installer.
* **Virtual Environment**: It is highly recommended to use a virtual environment (e.g., ``venv`` or ``conda``) to avoid dependency conflicts.

Installing from TestPyPI
------------------------

Currently, the GLAM library is hosted on TestPyPI. You can install it using the following command:

.. code-block:: bash

    pip install glam-radiomics

.. note::
   Make sure you have activated your virtual environment before running this command!

Key Dependencies
----------------

When you install GLAM, the following core libraries are automatically integrated:

* **NumPy & SciPy**: Provide the computational backbone for RDF calculations, spatial KD-trees, and Statistical Mechanics descriptors.
* **SimpleITK**: Handles the loading and normalization of 3D medical imaging formats like NIfTI (.nii.gz).
* **Pandas**: Manages the structured output of multiscale Radial Distribution Functions and feature aggregation.
* **Scikit-image & Scikit-learn**: Powers the morphological marching cubes (surface area), K-Means clustering, and advanced geometric descriptors.


Verifying the Installation
--------------------------

To verify that GLAM is correctly installed, you can run a simple version check in your terminal:

.. code-block:: python

    import glam_radiomics
    print(glam_radiomics.__version__)