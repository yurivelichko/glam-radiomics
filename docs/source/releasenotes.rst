Release Notes
=============

The GLAM framework is written in Python and leverages high-performance libraries for spatial indexing (KD-trees) and medical image analysis. It operates as a fully standalone extraction engine, meaning it does not require external radiomics packages to compute conventional texture matrices.


GLAM 1.5.2 (New Features & Matrices)
---------------------------
* July 2026
* **Assembly Coupling Matrix (Thermodynamic Entanglement):**
  * Added ``calculate_glam_assembly_coupling_matrix`` to compute the mixed partial derivative of the 1-Wasserstein Distance (:math:`\frac{\partial^2 \text{EMD}}{\partial \alpha \partial \beta}`). 
  * This matrix maps the structural alliances (synergistic assembly) and competitions (antagonistic assembly) between different tissue phenotypes.
  * **Technical Note:** Includes a native :math:`\text{SymLog}_{10}` transformation to compress the extreme exponential spikes at phase boundaries while strictly preserving the thermodynamic sign.
* **Phenotypic Distance Matrix (Phase-Space EMD):**
  * Added ``calculate_glam_phenotypic_distance_matrix`` to directly compare the morphological architectures of distinct gray levels.
  * Calculates the 1-Wasserstein Distance between normalized auto-correlation Cumulative Distribution Functions (CDFs).
  * **Technical Note:** Eliminates the randomized baseline to create a true, bounded, non-negative mathematical distance metric in morphological phase-space, making it ideal for downstream K-Means or hierarchical habitat clustering.

GLAM 1.5.0 (Core Engine Update)
---------------------------
* July 2026
* **In-Situ Intersection-Volume Normalization:** Fundamentally overhauled the boundary handling within the Radial Distribution Function (RDF) engine to solve finite-size edge effects.
  * *The Physics:* Previously, the algorithm utilized an ideal spherical volume denominator combined with a global geometric correction curve. For highly irregular, lobulated, or infiltrative tumors (such as glioblastomas and sarcomas), this global approximation under-corrected sharp convexities, mathematically mimicking an artificial "unjammed" or fluid state at the tumor boundary.
  * *The Fix:* Implemented dynamic, voxel-wise 3D convolutions to compute the precise physical intersection volume for every radial shell. This strictly confines the mathematics to the anatomical shape of the lesion, isolating true biological unjamming at the invasive margin from mathematical mask-edge artifacts.
* **Mechanical Phase & Jamming Transitions:** Introduced a new class of descriptors to quantify the physical phase state of the tissue architecture, identifying regions of structural arrest versus active fluidization (unjamming). 
  * *Two New Matrices:* Added extraction for the **Local Packing Fraction** (dimensionless spatial saturation) and the **Structural Frustration Index** (the ratio of structural stress to configurational disorder).
* **Refined Thermodynamic Metrics:** Due to the new boundary precision, the **Structural Pressure Index (SPI)** and **Configurational Disorder Index (CDI)** are now perfectly mathematically stable right up to the outermost voxel of the Volume of Interest (VOI).
* **Deprecated Legacy Correction:** Removed the now-obsolete ``calculate_geometric_factor`` and ``apply_geometric_correction`` modules. The primary ``calculate_rdf_3d`` engine now natively outputs the fully geometry-independent thermodynamic signature. 

GLAM 1.4.9 (Adding Granulometry and Percolation Metrics)
---------------------------
* June 2026
* **Granulometry (Pattern Spectrum) Metrics:** Introduced morphological sieving to evaluate the physical thickness distribution and structural "sponginess" of tissue states. This perfectly complements existing volume-based metrics (like GLSZM) by isolating true physical width.
  * *Five New Matrices:* Added extraction for Granulometry Mean, Variance, Skewness, Kurtosis, and Entropy, capturing the complete statistical distribution of local tissue thickness.
* **Percolation Metrics:** Introduced a new module to quantify the macroscopic connectivity and spatial continuity of discrete tissue states. This addition bridges the gap between local discrete morphology and global tissue architecture (e.g., evaluating whether a specific tissue type forms a massive continuous network or is shattered into isolated fragments).
  * *Three New Matrices:* Added extraction for Maximum Cluster Size (representing raw biological burden), Cluster Number Density (representing spatial fragmentation), and Percolation Strength (a volume-independent, scale-invariant static surrogate for the percolation threshold).
  * *Robust Mathematical Transforms:* Integrated automatic SymLog, Ln, and Log1p transformations for all percolation matrices to safely compress exponential scaling and perfectly handle empty tissue regions (zeros) without throwing mathematical errors.

GLAM 1.4.6 (Feature Update)
---------------------------
* June 2026
* **Native Habitat Radiomics:** Introduced the ability to perform multi-region habitat analysis on the fly. Users can now merge discrete sub-regions (e.g., combining enhancing and non-enhancing cores into a unified "Tumor_Core") directly via the ``LabelsForAnalysis`` dictionary in ``config.ini`` using Boolean union syntax (e.g., ``"1+2"``). This completely eliminates the need to manually preprocess NIfTI masks.
* **Targeted Regional Extraction:** The pipeline now strictly confines matrix calculations and feature extraction to the specifically defined habitats. Unrequested discrete labels are bypassed, drastically reducing computational overhead.
* **GPU/CUDA Stability Enhancements:** Addressed an issue where CuPy/CUDA version mismatches could cause silent C-level segmentation faults prior to GLCM calculation. Updated documentation provides strict version-matching guidelines and diagnostic protocols.

GLAM 1.3.2 (Update)
-------------------
* June 2026
* **Removed Jensen-Shannon (JS) Divergence Matrices:** Deprecated the ``GLAM_JSDivergence_matrix`` and ``GLAM_CumulativeJSDivergence_matrix`` features. A mathematical audit confirmed that due to the thermodynamic law of Global Reciprocity, the L1-normalized radial distribution distributions for cross-pairs (i,j) and (j,i) are mathematically identical. Any non-zero JS divergence previously captured was strictly stochastic sampling noise. True tissue anisotropy is now exclusively measured via the robust Gyration Tensor (``GLAM_Anisotropy_i_j``) and Shape Interface matrices.
* **Stabilized Spatial Correlation Metrics:** Replaced the Positional Correlation Length (``GLAM_corr_length_matrix``) with the Inverse Correlation Length / Decay Rate (``GLAM_InverseCorrelationLength_matrix``). 
  * *The Physics:* Previously, perfectly random or "flat" tissue distributions caused the correlation length (xi) to explode toward infinity, leading to optimizer crashes or massive numerical outliers (NaNs). 
  * *The Fix:* By fitting the decay rate (kappa) directly, the mathematical domain is now safely bounded. A highly structured tissue returns a positive decay rate, while a completely random tissue gracefully evaluates to exactly ``0.0``. This eliminates all curve-fitting crashes and provides a perfectly stable, bounded feature for downstream machine learning.

GLAM 1.2.8 (Major Update)
-------------------------
* May 2026
* **GPU Acceleration:** The entire spatial calculation engine (including GLCM, GLRLM, GLSZM, Topology/Betti numbers, and Lacunarity) has been rewritten in pure CuPy. GLAM now dynamically batches tensor math based on available GPU VRAM, delivering >10x processing speeds.
* **Thermodynamic Edge-Effect Corrections:** Resolved a KD-Tree truncation limitation in the Radial Distribution Function (RDF). Integrals for the Second Virial Coefficient (B2), Coordination Number, and Configurational Disorder are now strictly bounded to the first coordination shell and corrected against a randomized baseline, yielding pure, physically accurate topological signals free of finite-volume boundary artifacts.
* **Terminology Update:** Renamed "Effective Structural Temperature" to "Configurational Disorder Index" to better reflect the biophysical nature of the metric.
* **Configurable Boundaries:** Exposed the ``MaxLocalShellRadius`` parameter in ``config.ini``, allowing users to explicitly define the physical boundary size (in voxels) for the extraction of localized spatial descriptors. Parameter optimization for improved performance and accuracy.

GLAM 1.0.8 (Public Release)
---------------------------
* March 2026 
* First public release