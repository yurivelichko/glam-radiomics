Installation
============
The GLAM framework is written in Python and leverages high-performance libraries for spatial indexing (KD-trees) and medical image analysis. It operates as a fully standalone extraction engine, meaning it does not require external radiomics packages to compute conventional texture matrices.

Prerequisites
-------------
Before installing GLAM, ensure you have the following requirements:

* **Python**: Version 3.10 or higher.
* **Pip**: The Python package installer.
* **Virtual Environment**: It is highly recommended to use a virtual environment (e.g., ``venv`` or ``conda``) to avoid dependency conflicts.
* **NVIDIA GPU (Optional but Recommended)**: For hardware acceleration, an NVIDIA GPU with updated drivers is required.

Installing from TestPyPI
------------------------
You can install GLAM-radiomics using the following command:

.. code-block:: bash

    pip install glam-radiomics

.. note::
   This release (v1.4.8) was successfully built and tested using Python 3.12.10 and NumPy 2.3.2. Make sure you have activated your virtual environment before running this command!

GPU Acceleration (Optional but Highly Recommended)
--------------------------------------------------
GLAM features automatic GPU-accelerated matrix batching, which provides massive speed improvements for complex 3D texture and spatial feature calculations. To enable this, you must install **CuPy**. 

**CRITICAL:** You must install the pre-compiled CuPy binary wheel that exactly matches your system's NVIDIA CUDA version. Do *not* simply run ``pip install cupy``, as this requires a complex C++ build environment from source and often fails.

1. **Check your CUDA version**: Open your terminal or command prompt and run ``nvidia-smi``. Look for the "CUDA Version" in the top right corner of the output.
2. **Install the matching CuPy version** within your active virtual environment:
   
   * For CUDA 11.x: ``pip install cupy-cuda11x``
   * For CUDA 12.x: ``pip install cupy-cuda12x``

Troubleshooting CUDA/CuPy Mismatches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If your GLAM pipeline abruptly stops or closes silently without a Python error message (a segmentation fault) right before calculating texture matrices (like GLCM), you almost certainly have a CuPy and CUDA version mismatch.

To verify your CuPy installation, open a Python shell in your environment and run:

.. code-block:: python

    import cupy
    cupy.show_config()

Pay close attention to the ``CUDA Build Version`` and ``CUDA Driver Version``. If there is a mismatch, uninstall your current CuPy packages (e.g., ``pip uninstall cupy cupy-cuda12x``) and reinstall the version that strictly matches your workstation's installed toolkit.

Key Dependencies
----------------
When you install GLAM, the following core libraries are automatically integrated:

* **NumPy & SciPy**: Provide the computational backbone for RDF calculations, spatial KD-trees, and Statistical Mechanics descriptors.
* **SimpleITK**: Handles the loading and normalization of 3D medical imaging formats like NIfTI (.nii.gz).
* **Pandas**: Manages the structured output of multiscale Radial Distribution Functions and feature aggregation.
* **Scikit-image & Scikit-learn**: Powers the morphological marching cubes (surface area), K-Means clustering, and advanced geometric descriptors.

Verifying the Installation
--------------------------
To verify that GLAM is correctly installed, you can run a simple version check in your Python environment:

.. code-block:: python

    import glam_radiomics
    print(glam_radiomics.__version__)