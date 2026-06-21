Release Notes
=============

The GLAM framework is written in Python and leverages high-performance libraries for spatial indexing (KD-trees) and medical image analysis. It operates as a fully standalone extraction engine, meaning it does not require external radiomics packages to compute conventional texture matrices.

GLAM 1.0.8 (Public rRelease)
----------------------------
* March 2026 
* First public release

GLAM 1.2.8 (Major Update)
-------------------------
* June 2026
* **GPU Acceleration:** The entire spatial calculation engine (including GLCM, GLRLM, GLSZM, Topology/Betti numbers, and Lacunarity) has been rewritten in pure CuPy. GLAM now dynamically batches tensor math based on available GPU VRAM, delivering >10x processing speeds.
* **Thermodynamic Edge-Effect Corrections:** Resolved a KD-Tree truncation limitation in the Radial Distribution Function (RDF). Integrals for the Second Virial Coefficient (B2), Coordination Number, and Configurational Disorder are now strictly bounded to the first coordination shell and corrected against a randomized baseline, yielding pure, physically accurate topological signals free of finite-volume boundary artifacts.
* **Terminology Update:** Renamed "Effective Structural Temperature" to "Configurational Disorder Index" to better reflect the biophysical nature of the metric.
* **Configurable Boundaries:** Exposed the ``MaxLocalShellRadius`` parameter in ``config.ini``, allowing users to explicitly define the physical scale of the first coordination sphere. This allows GLAM to scale seamlessly from macroscopic MRI down to high-resolution digital pathology.

GLAM 1.3.0 (Update)
-------------------------
* June 2026
* Parameter optimization for improved performance and accuracy

GLAM 1.3.2 (Update)
-------------------------
* June 2026
* **Removed Jensen-Shannon (JS) Divergence Matrices:** Deprecated the ``GLAM_JSDivergence_matrix`` and ``GLAM_CumulativeJSDivergence_matrix`` features. A mathematical audit confirmed that due to the thermodynamic law of Global Reciprocity, the L1-normalized radial distribution distributions for cross-pairs (i,j) and (j,i) are mathematically identical. Any non-zero JS divergence previously captured was strictly stochastic sampling noise. True tissue anisotropy is now exclusively measured via the robust Gyration Tensor (``GLAM_Anisotropy_i_j``) and Shape Interface matrices.
* **Stabilized Spatial Correlation Metrics:** Replaced the Positional Correlation Length (``GLAM_corr_length_matrix``) with the Inverse Correlation Length / Decay Rate (``GLAM_InverseCorrelationLength_matrix``). 
  * *The Physics:* Previously, perfectly random or "flat" tissue distributions caused the correlation length (xi) to explode toward infinity, leading to optimizer crashes or massive numerical outliers (NaNs). 
  * *The Fix:* By fitting the decay rate (kappa) directly, the mathematical domain is now safely bounded. A highly structured tissue returns a positive decay rate, while a completely random tissue gracefully evaluates to exactly ``0.0``. This eliminates all curve-fitting crashes and provides a perfectly stable, bounded feature for downstream machine learning.
