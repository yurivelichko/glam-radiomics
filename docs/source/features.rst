GLAM Feature Dictionary
=======================

The GLAM framework provides a fully standalone, standardized feature extraction pipeline that translates complex spatial patterns into quantitative biomarkers. It operates independently of external radiomics packages, offering a highly optimized native 3D extraction engine.

Features are organized into four primary domains: **Standard Radiomics**, **Statistical Mechanics & Thermodynamics**, **Soft Matter Physics**, and **Geometric & Topological Metrics**.

Native Standard Radiomics Classes
---------------------------------
GLAM includes a built-in, natively optimized engine for calculating standard 3D texture matrices. Unlike conventional implementations, GLAM utilizes **Dynamic Matrix Trimming**, which prevents the calculation of massive, sparse matrices (e.g., in GLRLM and GLSZM) by dynamically truncating empty run-length and zone-size columns, drastically improving computational speed and memory efficiency.

* **Gray Level Co-occurrence Matrix (GLCM)**: Captures localized 3D voxel pairs (offset [0,0,1]).
* **Gray Level Run Length Matrix (GLRLM)**: Quantifies continuous linear runs of identical gray levels.
* **Gray Level Size Zone Matrix (GLSZM)**: Measures the size of contiguous 3D homogenous zones.
* **Gray Level Dependence Matrix (GLDM)**: Captures the number of connected voxels that are dependent on a center voxel.
* **Neighborhood Gray-Tone Difference Matrix (NGTDM)**: Quantifies the difference between a voxel and its average neighborhood intensity to capture coarseness and busyness.

Statistical Mechanics and Thermodynamics Classes
------------------------------------------------
These classes treat voxels as interacting particles within a multi-component system.

* **RDF Shape Statistics**: Transforms key statistical properties of each :math:`g_{\alpha\beta}(r)` curve—including Peak Position, Peak Height, Median, Variance, Skewness, and Kurtosis—into primary GLAM matrices.
* **Second Virial Coefficient (:math:`B_2`)**: Distills distance-dependent RDF information into a single value where negative values indicate net attraction and positive values suggest net repulsion.
* **Potential of Mean Force (PMF)**: Evaluates the energetic stability of texture organization based on the Boltzmann distribution.
* **Isothermal Compressibility**: Quantifies the "sponginess" of the texture. High values imply large-scale density fluctuations (loose clustering), while low values indicate rigid, hyper-uniform distributions.
* **Coordination Number (CN)**: Measures local packing or clustering, representing the average number of :math:`\beta`-voxels surrounding a reference :math:`\alpha`-voxel within the first coordination shell.
* **Correlation Length (:math:`\xi`)**: Characterizes the spatial extent of structural order and the range of voxel interactions.
* **Pressure Virial**: Captures the mechanical response within the texture. Positive values indicate expansion, and negative values indicate compaction.
* **Effective Structural Temperature (:math:`T_{eff}$)**: Characterizes textural disorder and structural "noise" by comparing the observed image structure to a randomized counterpart.

Soft Matter and Liquid Crystal Classes
--------------------------------------
A unique advantage of the GLAM framework is the integration of physical descriptors originally developed for liquid crystals and soft matter. These classes measure the orientational order and mechanical stress of biological tissues.


* **Nematic Order Parameter (:math:`S`)**: Calculates the global alignment of intensity gradients within the tumor. Values approach 1 for highly aligned, fibrous, or directional structures, and 0 for completely isotropic textures.
* **Nematic Order per Gray Level**: Calculates the :math:`S` parameter specifically for populations of individual gray levels to find isolated structural pathways.
* **Local Nematic Alignment**: Measures the average dot-product alignment of local director fields (the primary orientation axis of a local neighborhood) across the entire region of interest.
* **Orientational Correlation Length**: Derived from the :math:`g_2(r)` function, this describes the distance over which directional alignment persists before structural memory is lost.
* **Tissue Stress (Laplacians)**: Calculates the Volumetric and Surface Laplacian means and variances, acting as mathematical analogues to the internal mechanical stress and boundary tension of the tumor.

Geometric and Topological Classes
---------------------------------
These classes quantify the specific spatial relationship between gray levels to model the underlying topology.

* **Fractal Dimension**: Uses an optimized box-counting method to measure the multiscale complexity of both isolated tissue volumes and the interfaces between tissue types.
* **Multifractal Spectrum**: Utilizes the Method of Moments to extract the Generalized Dimensions (:math:`D_q`), Spectral Width, and :math:`\alpha_0`, capturing complex heterogeneity that a single fractal dimension cannot.
* **Lacunarity**: Quantifies the "gappiness" or heterogeneity of void spaces in the spatial arrangement.
* **Topological Invariants**: Includes Betti numbers (:math:`B_0`, :math:`B_1`, :math:`B_2`) and the Euler characteristic to explicitly count disconnected islands, tunnels, and voids.
* **Anisotropy**: Measures structural alignment and preferred orientation using gyration and nematic ordering tensors.

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
   * - **Profile Shape / Bimodality**
     - Detects structural separation and tissue layering on matrix diagonals.
     - Peak Separation, Bimodality Index, Roughness.
   * - **Topological/Graph**
     - Complexity and stability of the interaction network.
     - Spectral Radius, Eigenvalues, Silhouette Score.
   * - **Symmetry & Diagonal**
     - Reciprocity and "self-affinity" of gray-level interactions.
     - Frobenius Norm, Mean Absolute Asymmetry.

Integration with config.ini
---------------------------
In your ``config.ini`` file, you can specify which of these features to map directly into 3D NIfTI volumes by adding them to the ``MapFeatures`` list (e.g., ``["PressureVirial_Symlog", "NematicOrder_S"]``).