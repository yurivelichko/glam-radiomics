GLAM Feature Dictionary
=======================
The GLAM framework provides a fully standalone, standardized feature extraction pipeline that translates complex spatial patterns into quantitative biomarkers. It operates independently of external radiomics packages, offering a highly optimized native 3D extraction engine powered by GPU acceleration (CuPy).

Features are organized into four primary domains: **Standard Radiomics**, **Statistical Mechanics & Thermodynamics**, **Soft Matter Physics**, and **Geometric & Topological Metrics**.

Native Standard Radiomics Classes
---------------------------------
GLAM includes a built-in, natively GPU-optimized engine for calculating standard 3D texture matrices. Unlike conventional implementations, GLAM utilizes **Dynamic Matrix Trimming**, which prevents the calculation of massive, sparse matrices (e.g., in GLRLM and GLSZM) by dynamically truncating empty run-length and zone-size columns, drastically improving computational speed and memory efficiency.

* **Gray Level Co-occurrence Matrix (GLCM)**: Captures localized 3D voxel pairs (offset [0,0,1]).
* **Gray Level Run Length Matrix (GLRLM)**: Quantifies continuous linear runs of identical gray levels.
* **Gray Level Size Zone Matrix (GLSZM)**: Measures the size of contiguous 3D homogenous zones.
* **Gray Level Dependence Matrix (GLDM)**: Captures the number of connected voxels that are dependent on a center voxel.
* **Neighborhood Gray-Tone Difference Matrix (NGTDM)**: Quantifies the difference between a voxel and its neighborhood.
* **Excess and Ratio Matrices**: GLAM automatically bridges conventional radiomics with statistical physics by generating ``_Excess`` (Structured - Random) and ``_Ratio`` (Structured / Random) variants for all standard matrices, quantifying how much the tissue structure deviates from a purely stochastic arrangement.

Statistical Mechanics & Thermodynamic Classes
-------------------------------------------
These features treat the tumor as a many-body physical system, calculating thermodynamic states using the Radial Distribution Function (RDF) and 4x Randomized Baselines.

* **Second Virial Coefficient (B2)**: Quantifies the topological "attraction" (clustering) or "repulsion" between gray levels.
* **Coordination Number (Z)**: Measures the exact number of voxels in the local, first coordination shell.
* **Configurational Disorder Index**: (Formerly Effective Temperature) Quantifies the thermodynamic disorder strictly within the first coordination shell.
* **Structural Pressure Index (SPI)**: Formally analogous to the interaction component of pressure.
* **1-Wasserstein Distance (EMD)**: Measures the 'Biological Work' or 'Assembly Cost' of the tumor's spatial architecture by comparing the structured and random cumulative coordination profiles.

Soft Matter & Geometric Classes
-------------------------------
* **Nematic Order Parameter (S)**: Measures the global and local directional alignment of tissue gradients.
* **Orientational Correlation Length**: Quantifies how far directional alignment persists through the tissue.
* **Topological Betti Numbers**: GPU-accelerated calculation of Connected Components (B0), Tunnels (B1), and Enclosed Voids (B2) using the Euler-Poincaré formula.
* **Fractal Dimension & Lacunarity**: Optimized 3D Box-Counting and GPU Convolutions for multiscale complexity and structural heterogeneity.

Percolation Theory & Network Connectivity
-----------------------------------------
Evaluates the macroscopic connectivity of discrete tissue states to determine if specific microenvironments (e.g., necrosis, hypoxia) form isolated fragments or massive spanning networks.

* **Maximum Cluster Size**: Represents the raw biological burden by measuring the absolute voxel count of the largest contiguous tissue region.
* **Cluster Number Density**: Measures the degree of spatial fragmentation by normalizing the total number of isolated clusters against the ROI volume.
* **Percolation Strength**: A scale-invariant, volume-independent surrogate for the percolation threshold. It measures the probability that any given active site belongs to the primary spanning cluster.

Matrix Reduction Features
-------------------------
Once a multi-dimensional GLAM matrix is generated, the following statistics are extracted to create the final 1D feature vectors for machine learning:

.. list-table:: Feature Category Descriptions
   :widths: 25 50 25
   :header-rows: 1

   * - Feature Category
     - Description
     - Examples
   * - **First-Order Statistics**
     - Global distribution of affinity values in the matrix.
     - Mean, Variance, Skewness, Kurtosis, Energy.
   * - **Second-Order Meta**
     - Structural heterogeneity of the affinity landscape matrix itself.
     - Contrast, Correlation, Joint Entropy.
   * - **Thermodynamic & State**
     - Quantifies the interaction and physical arrangement of tissue clusters.
     - Configurational Disorder Index, Pressure Virial, Coordination Number.
   * - **Profile Shape / Bimodality**
     - Detects structural separation and tissue layering on matrix diagonals.
     - Peak Separation, Bimodality Index, Roughness.
   * - **Topological/Graph**
     - Complexity and stability of the interaction network.
     - Spectral Radius, Eigenvalues, Silhouette Score.
   * - **Symmetry & Diagonal**
     - Reciprocity and "self-affinity" of gray-level interactions.
     - Frobenius Norm, Mean Absolute Asymmetry.
   * - **Percolation / Network**
     - Quantifies the macroscopic connectivity and spatial fragmentation of discrete tissue states.
     - Percolation Strength, Max Cluster Size, Cluster Number Density.

Integration with config.ini
---------------------------
In your ``config.ini`` file, you can specify which of these features to map directly into 3D NIfTI volumes by adding them to the ``MapFeatures`` list (e.g., ``["ConfigurationalDisorderIndex", "PercolationStrength", "CoordNum"]``).