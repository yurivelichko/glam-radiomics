Installation
============
The GLAM framework is written in Python and leverages high-performance libraries for spatial indexing (KD-trees) and medical image analysis[cite: 1]. It operates as a fully standalone extraction engine, meaning it does not require external radiomics packages to compute conventional texture matrices[cite: 2].

Prerequisites
-------------
Before installing GLAM, ensure you have the following requirements[cite: 3]:

* **Python**: Version 3.10 or higher[cite: 3].
* **Pip**: The Python package installer[cite: 3].
* **Virtual Environment**: It is highly recommended to use a virtual environment (e.g., ``venv`` or ``conda``) to avoid dependency conflicts[cite: 4].

Installing from TestPyPI
------------------------
You can install GLAM-radiomics using the following command:

.. code-block:: bash

    pip install glam-radiomics

.. note::
   This release (v1.2.8) was successfully built and tested using Python 3.12.10 and NumPy 2.3.2. Make sure you have activated your virtual environment before running this command!

.. note::
   GPU Acceleration (Optional but Recommended): > GLAM features automatic GPU-accelerated matrix batching for massive speed improvements. To enable this, you must install CuPy. It is highly recommended to install the pre-compiled binary wheel that matches your system's NVIDIA CUDA version (e.g., pip install cupy-cuda12x) rather than running pip install cupy, which requires a complex C++ build environment and can cause installation failures.

Key Dependencies
----------------
When you install GLAM, the following core libraries are automatically integrated:

* **NumPy & SciPy**: Provide the computational backbone for RDF calculations, spatial KD-trees, and Statistical Mechanics descriptors[cite: 7].
* **SimpleITK**: Handles the loading and normalization of 3D medical imaging formats like NIfTI (.nii.gz)[cite: 8].
* **Pandas**: Manages the structured output of multiscale Radial Distribution Functions and feature aggregation[cite: 9].
* **Scikit-image & Scikit-learn**: Powers the morphological marching cubes (surface area), K-Means clustering, and advanced geometric descriptors[cite: 10].

Verifying the Installation
--------------------------
To verify that GLAM is correctly installed, you can run a simple version check in your Python environment:

.. code-block:: python

    import glam_radiomics
    print(glam_radiomics.__version__)