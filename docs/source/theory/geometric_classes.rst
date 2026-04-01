Geometric Classes
=================

Beyond statistical mechanics, the GLAM framework integrates descriptors from geometry and topology to characterize the spatial arrangement and complexity of biological textures. These metrics are organized into three primary domains:

Structural and Directional Organization
---------------------------------------
These metrics characterize how tissue components are packed and oriented within the tumor microenvironment.

Coordination Number (CN)
~~~~~~~~~~~~~~~~~~~~~~~~
The Coordination Number (CN) measures the local packing or clustering of gray-levels. Adapted from atomic coordination in materials science, it represents the average number of :math:`\beta`-voxels surrounding a reference :math:`\alpha`-voxel within the first coordination shell:

.. math::

   CN_{\alpha,\beta} = 4\pi\rho_\beta \int_{0}^{r_{min}} g_{\alpha\beta}(r) r^2 dr

where :math:`\rho_\beta` is the mean voxel density and :math:`r_{min}` is the first RDF minimum beyond the primary peak. Diagonal terms describe local self-clustering (e.g., tumor cell density), whereas off-diagonal elements quantify the degree of direct interfacing between cancerous and stromal tissue.

.. figure:: /_static/GLAM_CoordNumber.png
   :width: 700px
   :align: center
   :alt: Second Virial Coefficient 

   **Figure:** Coordination Number matrices derived from four co-registered MRI sequences: pre-contrast T1-weighted (T1), post-contrast T1-weighted (T1c), T2-weighted (T2), and Fluid-Attenuated Inversion Recovery (FLAIR). 

* **Interpretation**: This metric asks, *"On average, how many immediate neighbors of type B does a central voxel of type A physically touch?"*
* **Advantage**: It provides a direct, intuitive measure of local packing density and the physical extent of the immediate contact boundary between different tissue types.

Correlation Length (:math:`\xi`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While :math:`B_2` quantifies the magnitude of spatial order, the correlation length :math:`\xi(\alpha, \beta)` characterizes its spatial extent. It describes the range over which voxel interactions persist before structural memory is lost:

.. math::

   h_{\alpha\beta}(r) \propto A \exp \left( -\frac{r}{\xi_{\alpha,\beta}} \right), \quad r > r_{peak}

Longer correlation lengths indicate coherent, organized structures, while shorter ones imply more localized or disordered texture.

.. figure:: /_static/GLAM_CorrelLength.png
   :width: 700px
   :align: center
   :alt: Second Virial Coefficient 

   **Figure:** Correlation Length matrices derived from four co-registered MRI sequences: pre-contrast T1-weighted (T1), post-contrast T1-weighted (T1c), T2-weighted (T2), and Fluid-Attenuated Inversion Recovery (FLAIR). 


* **Interpretation**: This metric asks, *"How far away does a voxel still 'feel' the structural influence of a reference voxel before the tissue arrangement becomes completely random?"*
* **Advantage**: It quantifies the absolute physical size of coherent biological structures (like tumor nests or stromal bands) independently of their intensity values.

Anisotropy Indices
~~~~~~~~~~~~~~~~~~
GLAM captures directional organization (e.g., in aligned stromal bands) using gyration and nematic ordering tensors.

* **Positional Anisotropy**: Uses the Relative Shape Anisotropy index, :math:`A_{\alpha,\beta}`, to quantify geometric elongation derived from the eigenvalues of the local gyration tensor.
* **Orientational Anisotropy**: Calculates the **Nematic Order Parameter (:math:`S`)**, analogous to liquid crystal physics, to quantify the degree of alignment of local intensity gradients. :math:`S=0` represents random orientation, while :math:`S=1` indicates perfect alignment.

.. figure:: /_static/GLAM_Anisotropy.png
   :width: 700px
   :align: center
   :alt: Second Virial Coefficient 

   **Figure:** Anisotropy matrices derived from four co-registered MRI sequences: pre-contrast T1-weighted (T1), post-contrast T1-weighted (T1c), T2-weighted (T2), and Fluid-Attenuated Inversion Recovery (FLAIR). 

* **Interpretation**: This metric asks, *"Are the tissue structures stretched out and aligned in a specific direction, or are they perfectly round and directionless?"*
* **Advantage**: It captures directional gradients and structural elongation, which are critical for identifying invasive fronts, aligned collagen tracks, or collective cell migration patterns.

Complexity and Topology
-----------------------
These measures describe the roughness, multiscale nature, and connectivity of image features.

Fractal and Multifractal Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GLAM utilizes a 3D box-counting algorithm to quantify multiscale self-similarity and complexity.

* **Volume Fractal Dimension (:math:`D_V`)**: Calculated for voxels of a single gray level, indicating its space-filling capacity.
* **Interface Fractal Dimension (:math:`D_I`)**: Measures the roughness and invasiveness of boundaries between two tissue types.
* **Multifractal Spectrum**: Employs Generalized Dimensions (:math:`D_q`) to characterize tissues where scaling properties vary across the region. The spectrum width (:math:`\Delta\alpha`) quantifies the diversity of scaling behaviors, representing the "heterogeneous chaos" of the tissue.

.. figure:: /_static/GLAM_FractalDim.png
   :width: 700px
   :align: center
   :alt: Second Virial Coefficient 

   **Figure:** Fractal Dimension matrices derived from four co-registered MRI sequences: pre-contrast T1-weighted (T1), post-contrast T1-weighted (T1c), T2-weighted (T2), and Fluid-Attenuated Inversion Recovery (FLAIR). 

* **Interpretation**: This metric asks, *"How complex, branching, and space-filling is this tissue structure across different zoom levels?"*
* **Advantage**: It captures the self-similar "roughness" of biological tissues, allowing for robust differentiation between smooth, encapsulated tumors and highly invasive, branching morphologies.

Lacunarity (:math:`\Lambda`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While fractal dimension quantifies space-filling, Lacunarity measures the "gappiness" or heterogeneity of void spaces within the tissue architecture. High Lacunarity indicates large, irregular gaps, while low values suggest a uniform, homogeneous distribution.

Topological Invariants (Betti Numbers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GLAM uses algebraic topology to count discrete features that are invariant under continuous deformation:

* **Betti-0 (:math:`B_0`)**: Counts fragmented "islands" of a specific gray level.
* **Betti-1 (:math:`B_1`)**: Counts tunnels or loops (e.g., vascular networks).
* **Betti-2 (:math:`B_2`)**: Counts enclosed internal cavities or voids.
* **Euler Characteristic (:math:`\chi`)**: A classic measure of topological complexity, where :math:`\chi = B_0 - B_1 + B_2`.

.. figure:: /_static/GLAM_Euler.png
   :width: 700px
   :align: center
   :alt: Second Virial Coefficient 

   **Figure:** Euler Characteristic matrices derived from four co-registered MRI sequences: pre-contrast T1-weighted (T1), post-contrast T1-weighted (T1c), T2-weighted (T2), and Fluid-Attenuated Inversion Recovery (FLAIR). 


* **Interpretation**: This metric asks, *"How many distinct islands, connective tunnels, and hollow voids exist within the tissue, regardless of their exact physical shape?"*
* **Advantage**: Topology is completely invariant to stretching, bending, or scaling. This makes Betti numbers incredibly robust against patient positioning differences, organ deformation, and imaging variations.

Discrete Morphology
-------------------
This category extracts explicit geometric descriptors for distinct tissue clusters identified by gray-level thresholds.

* **Sphericity and Solidity**: Measure the compactness and "ruggedness" of individual gray-level isosurfaces.
* **Interface Area**: Quantifies the total surface of direct contact between two distinct tissue types, representing the extent of physical infiltration.
* **Centroid Distance**: Measures the Euclidean distance between the centers of mass of different tissue components.

.. figure:: /_static/GLAM_CentroidDist.png
   :width: 700px
   :align: center
   :alt: Second Virial Coefficient 

   **Figure:** Centroid Distance matrices derived from four co-registered MRI sequences: pre-contrast T1-weighted (T1), post-contrast T1-weighted (T1c), T2-weighted (T2), and Fluid-Attenuated Inversion Recovery (FLAIR). 

* **Interpretation**: This metric asks, *"What are the tangible physical dimensions, roundness, and contact areas of these specific tissue clumps?"*
* **Advantage**: It provides highly tangible, classic geometric descriptors that correlate directly with standard visual pathological assessments.
