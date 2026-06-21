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
