Home
====
Theoretical Foundation
----------------------
While modern pathology has advanced toward characterizing tissue architecture through rigorous principles of spatial coupling, conventional radiomics often lacks a parallel framework to capture these fundamental thermodynamic states. 

The foundation of the GLAM framework is the adaptation of rigorous formalism from statistical physics and geometry to quantify the biological organization of a tumor. By treating voxels of differing intensities as interacting particles, we can quantify the effective "interactions" shaping tumor architecture.

GLAM assumes that the spatial heterogeneity observed in an image is the macroscopic result of microscopic biological interactions. By treating voxels of differing intensities as representatives of distinct biological components, GLAM quantifies the effective "attraction" and "repulsion" shaping tumor architecture. This generates a biophysically interpretable signature of the tumor microenvironment that spans **Statistical Mechanics**, **Thermodynamics**, **Liquid Crystal Physics**, **Fractals**, and **Geometric Topology**.

GLAM Radiomics
--------------
**GLAM (Gray-Level Affinity Metrics) Radiomics** is a comprehensive, standalone, physics- and geometry-based radiomics framework for decoding multiscale tissue microstructure and biophysical properties.

Many conventional radiomic classes, such as the GLCM, quantify relationships between pairs of gray levels in a matrix format. To enable direct comparison with these methods, the GLAM framework adopts a similar matrix-based representation. For each GLAM class, the output is a two-dimensional :math:`N \times N` matrix, where :math:`N` is the number of gray levels. The element at matrix position :math:`(\alpha, \beta)` represents the GLAM properties governing the interaction between gray level :math:`\alpha` and gray level :math:`\beta`. By adopting this standard format, GLAM maintains compatibility with traditional workflows while providing a physical interpretation of tissue architecture.

.. figure:: /_static/Radiomics.png
   :width: 600px
   :align: center
   :alt: Conventional Radiomics Matrices

   **Figure:** Conventional and GLAM radiomics matrices derived from the same post-contrast T1-weighted (T1c) MRI scan. 


Key Advantages
--------------
* **Fully Standalone Engine**: Built-in, natively optimized 3D extraction for shape, first-order, and conventional texture matrices with dynamic trimming—no external radiomics dependencies required.
* **Physically Interpretable**: Translates image voxels into interacting particles to measure properties like "Effective Structural Temperature", "Pressure Virial", and "Volumetric Laplacians" (Stress).
* **Soft Matter Descriptors**: Unique capabilities to measure directional tissue organization using the global and local **Nematic Order Parameter (S)** and Orientational Correlation Lengths.
* **3D Feature Mapping**: Integrated sliding-window architecture to generate voxel-wise NIfTI feature maps, allowing for direct visual assessment of the tumor's "affinity landscape."
