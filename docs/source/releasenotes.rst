Release Notes
=============

The GLAM framework is written in Python and leverages high-performance libraries for spatial indexing (KD-trees) and medical image analysis. It operates as a fully standalone extraction engine, meaning it does not require external radiomics packages to compute conventional texture matrices.

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