Statistical Mechanics and Thermodynamics
========================================

The foundation of the GLAM framework is the adaptation of rigorous formalism from statistical physics and thermodynamics to quantify the biological organization of a tumor. By treating voxels of differing intensities as interacting particles, we can quantify the effective "interactions" shaping tumor architecture.

Radial Distribution Function (RDF)
----------------------------------
The core descriptor is the pair radial distribution function, :math:`g_{\alpha\beta}(r)`, which measures the spatial correlation between voxels [40]. It quantifies the relative likelihood of finding a voxel with gray level :math:`\beta` at a distance :math:`r` from a reference voxel with gray level :math:`\alpha` [41].

* **Positive Correlation (Attraction)**: :math:`g_{\alpha\beta}(r) > 1` [42].
* **Negative Correlation (Repulsion)**: :math:`g_{\alpha\beta}(r) < 1` [42].
* **Randomness**: :math:`g_{\alpha\beta}(r) = 1` [42].

.. figure:: /_static/GLAM_RDF.png
   :width: 700px
   :align: center
   :alt: Radial Distribution Function 

   **Figure:** RDF as a function of the distance :math:`r` (from 0 to 100 mm). Each colored line represents a specific RDF curve, :math:`g_{\alpha\beta}(r)`, between two different gray levels, :math:`\alpha` and :math:`\beta`. 

* **Interpretation**: This metric asks, *"How much more or less likely am I to find tissue type B at a specific distance from tissue type A, compared to a completely random distribution?"*
* **Advantage**: It provides a fundamental, distance-dependent map of spatial relationships rather than just a single global average, revealing exact distances where tissues cluster or repel.

RDF Shape Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To capture the extensive information embedded in the full RDF curves without incurring the "curse of dimensionality," we transform key statistical properties of each :math:`g_{\alpha\beta}(r)` curve into a set of primary GLAM matrices. For every gray-level pair, we compute the Peak Position, Peak Height, Median, Variance, Skewness, and Kurtosis.

.. figure:: /_static/GLAM_RDFMedian.png
   :width: 700px
   :align: center
   :alt: Second Virial Coefficient 

   **Figure:** RDF Median matrices derived from four co-registered MRI sequences: pre-contrast T1-weighted (T1), post-contrast T1-weighted (T1c), T2-weighted (T2), and Fluid-Attenuated Inversion Recovery (FLAIR). 

Key Thermodynamic Metrics
-------------------------

Second Virial Coefficient (:math:`B_2`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The second virial coefficient distills the complex, distance-dependent information of the entire RDF curve into a single value representing net affinity[cite: 66].

.. math::

   B_{2\alpha, \beta} = -2\pi \int_{0}^{R_{max}} [g_{structured}(r) - g_{randomized}(r)] r^2 dr

* **Negative value**: Indicates net attraction or affinity between gray levels[cite: 66].
* **Positive value**: Suggests net repulsion[cite: 66].

.. figure:: /_static/GLAM_B2.png
   :width: 700px
   :align: center
   :alt: Second Virial Coefficient 

   **Figure:** B2 matrices derived from four co-registered MRI sequences: pre-contrast T1-weighted (T1), post-contrast T1-weighted (T1c), T2-weighted (T2), and Fluid-Attenuated Inversion Recovery (FLAIR). 

* **Interpretation**: This metric asks, *"Overall, do these two tissue types prefer to clump together (attraction) or push each other apart (repulsion) across the entire tumor?"*
* **Advantage**: It condenses a complex multi-distance curve into a single, highly interpretable summary statistic of spatial affinity.

Potential of Mean Force (PMF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To evaluate the energetic stability of texture organization, GLAM computes the PMF based on the Boltzmann distribution[cite: 67, 68].

.. math::

   W_{\alpha\beta}(r) = -k_B T \ln g_{\alpha\beta}(r)

The total PMF energy is obtained by integrating this potential weighted by the RDF[cite: 72]:

.. math::

   U_{PMF\alpha,\beta} = 4\pi \int_{0}^{R_{max}} W_{\alpha\beta}(r) g_{\alpha\beta}(r) r^2 dr

To allow size-independent comparison across ROIs, the UPMF is normalized by volume, producing the Energy Density, which represents the average interaction energy per voxel. 

.. figure:: /_static/GLAM_EnergyDensity.png
   :width: 700px
   :align: center
   :alt: Second Virial Coefficient 

   **Figure:** Energy Density matrices derived from four co-registered MRI sequences: pre-contrast T1-weighted (T1), post-contrast T1-weighted (T1c), T2-weighted (T2), and Fluid-Attenuated Inversion Recovery (FLAIR). 

* **Interpretation**: This metric asks, *"How much 'energy' would it take to maintain this specific spatial arrangement of tissues against natural thermodynamic mixing?"*
* **Advantage**: It translates geometric clustering into a physical energy landscape, allowing the identification of highly stable (deep energy wells) versus unstable, transient tissue architectures.

Isothermal Compressibility (:math:`\kappa_T`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In statistical thermodynamics, the integral of the total correlation function is directly related to the system's density fluctuations and its isothermal compressibility[cite: 628]. 

.. math::

   \kappa_T \propto \int_{0}^{R_{max}} [g_{\alpha\beta}(r) - 1] r^2 dr

In the GLAM framework, this metric quantifies the "sponginess" or susceptibility of the texture to local density variations[cite: 630]. High compressibility implies large-scale density fluctuations and a tendency for the voxel population to cluster loosely, whereas low compressibility indicates a rigid, hyper-uniform distribution typical of highly packed cellular structures[cite: 631].

----

* **Interpretation**: This metric asks, *"How 'spongy' or susceptible to density fluctuations is the tissue architecture?"*
* **Advantage**: It captures large-scale structural heterogeneity. Low compressibility indicates rigid, tightly packed regions (like dense stroma), while high compressibility flags loose, highly variable clustering.

Pressure Virial
~~~~~~~~~~~~~~~
Derived from the virial equation of state, this quantifies the mean internal force between voxel populations, capturing the mechanical response within the texture[cite: 92, 93].

.. math::

   P_{\alpha,\beta} = -\frac{\rho_\alpha \rho_\beta}{6} \int_{0}^{R_{max}} r \frac{dW_{\alpha\beta}(r)}{dr} g_{\alpha\beta}(r) 4\pi r^2 dr

.. figure:: /_static/GLAM_PressVirial.png
   :width: 700px
   :align: center
   :alt: Second Virial Coefficient 

   **Figure:** Pressure Virial matrices derived from four co-registered MRI sequences: pre-contrast T1-weighted (T1), post-contrast T1-weighted (T1c), T2-weighted (T2), and Fluid-Attenuated Inversion Recovery (FLAIR). 

* **Interpretation**: This metric asks, *"What is the net mechanical 'push or pull' (internal stress) exerted between different voxel populations due to their spatial packing?"*
* **Advantage**: It links spatial statistics directly to mechanical properties, offering a non-invasive computational proxy for the internal mechanical stresses within the tumor microenvironment.

Effective Structural Temperature (:math:`T_{eff}`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To complement metrics of energetic stability, the Effective Structural Temperature (:math:`T_{eff}`) characterizes textural disorder. Whereas PMF energy assesses how stable the structure is, :math:`T_{eff}` quantifies how dynamically disordered it is. :math:`T_{eff}` is derived by comparing the measured RDF of the actual image, :math:`g_{structured}(r)`, to that of its randomized counterpart, :math:`g_{randomized}(r)`. 

The Boltzmann relation links the probability of a configuration to its energy and temperature. Because :math:`\ln(g(r))` is proportional to the negative potential divided by the temperature, deviations from randomness reflect an underlying "ordering potential." The observed structure, :math:`\ln(g_{structured}(r))`, represents this potential modulated by thermal-like disorder. The ratio between the structured and randomized forms yields a distance-dependent effective temperature:

.. math::

   T(r) = \frac{\ln(g_{structured}(r))}{\ln(g_{structured}(r)) - \ln(g_{randomized}(r))}

Averaging :math:`T(r)` within the first coordination shell yields a stable, physically interpretable estimate of the local structural temperature :math:`T_{eff}(\alpha,\beta)`. High :math:`T_{eff}(\alpha,\beta)` indicates greater structural "noise," while low :math:`T_{eff}` means rigid or cold ordering. Comparing diagonal versus off-diagonal :math:`T_{eff}(\alpha,\beta)` elements distinguishes self-organized tissue components from dynamically heterogeneous interfaces. 

* **Interpretation**: This metric asks, *"How much structural 'noise' or 'thermal agitation' exists in this tissue compared to a perfectly ordered state?"*
* **Advantage**: It provides a clear indicator of architectural chaos; "hot" tissues are disorganized and random, while "cold" tissues are rigidly structured. This thermodynamic interpretation provides a robust, physically interpretable measure of local structural temperature in medical images.

1-Wasserstein Distance (Assembly Cost)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The 1-Wasserstein Distance, often called the Earth Mover's Distance, measures the "Biological Work" or "Assembly Cost" of the tumor's spatial architecture. It quantifies the total effort required to transform a completely randomized spatial distribution of voxels into the highly ordered, structured state actually observed in the tumor.

First, the Cumulative Coordination Number, :math:`N(R)`, is calculated. This represents the total number of neighbors accumulated up to radius :math:`R`:

.. math::

   N(R) = \int_0^R 4\pi \rho_\beta g(r) r^2 dr

The Wasserstein distance is then defined as the absolute area between the cumulative curves of the structured and randomized states:

.. math::

   W_{\alpha\beta} = \int_{0}^{R_{max}} \left| N_{structured}(R) - N_{random}(R) \right| dR


* **High Value**: Indicates a highly complex, non-random architecture with a high energetic "cost" of assembly, typical of highly organized distinct sub-regions or rigid boundaries.
* **Low Value**: Indicates the tissue architecture is very close to a completely random distribution of cells or voxels.


Jensen-Shannon (JS) Divergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While the standard RDF quantifies the distance-dependent interaction between two specific gray levels, comparing reciprocal RDF curves allows us to evaluate the directional symmetry—or structural anisotropy—of this relationship. In a perfectly isotropic (directionless) texture, the probability of finding gray level :math:`\beta` at distance :math:`r` from gray level :math:`\alpha` should be identical to finding gray level :math:`\alpha` at distance :math:`r` from gray level :math:`\beta`. Therefore, 

.. math::

   g_{\alpha\beta}(r) \approx g_{\beta\alpha}(r). 

JS Divergence quantifies the *disagreement* between these two reciprocal RDF curves. Because standard Kullback-Leibler (KL) Divergence is asymmetric and struggles with zeros, JS Divergence creates a symmetric, bounded comparison by measuring how far both curves deviate from their shared average.

First, the raw RDF curves are L1-normalized into true probability distributions, :math:`P` and :math:`Q`:

.. math::

   P(r) = \frac{g_{\alpha\beta}(r)}{\sum g_{\alpha\beta}(r)}

.. math::

   Q(r) = \frac{g_{\beta\alpha}(r)}{\sum g_{\beta\alpha}(r)}

We define a midpoint distribution :math:`M`:

.. math::

   M(r) = \frac{1}{2} (P(r) + Q(r))

The JS Divergence is then calculated as the average KL Divergence of :math:`P` and :math:`Q` from :math:`M`:

.. math::

   JSD(P \parallel Q) = \frac{1}{2} \sum P(r) \log_2 \left( \frac{P(r)}{M(r)} \right) + \frac{1}{2} \sum Q(r) \log_2 \left( \frac{Q(r)}{M(r)} \right)

* **Perfect Symmetry** (:math:`JSD = 0`): The relationship between tissue A and tissue B is structurally identical in reverse.
* **Structural Anisotropy** (:math:`JSD > 0`): Tissue B tends to cluster around tissue A differently than tissue A clusters around tissue B (e.g., A forms a core while B forms a shell).

----

* **Interpretation**: This metric asks, *"Is the local neighborhood of tissue A around tissue B an exact mirror image of tissue B around tissue A?"*
* **Advantage**: It offers a bounded (0 to 1) and perfectly symmetric measure of local directional bias, avoiding the mathematical instabilities of traditional Kullback-Leibler divergence when encountering empty tissue regions.

Cumulative JS Divergence
~~~~~~~~~~~~~~~~~~~~~~~~
Raw RDF curves (:math:`g(r)`) can be noisy, especially in small tumors or sparse gray levels, causing artificial spikes in standard JS Divergence. 

Cumulative JS Divergence solves this by borrowing intuition from the Wasserstein metric (Earth Mover's Distance). Instead of comparing local densities shell-by-shell, it compares the **Cumulative Coordination Profiles**—the total amount of tissue accumulated up to distance :math:`R`. This acts as a low-pass filter, ignoring high-frequency quantization noise and focusing on global spatial imbalances.

First, we multiply the RDF by the spherical volume element :math:`r^2` and integrate to create a cumulative profile :math:`N(R)`, representing the total interaction mass up to radius :math:`R`:

.. math::

   N_{ij}(R) = \int_0^R g_{ij}(r) r^2 dr

These cumulative profiles are then L1-normalized to create monotonically increasing distributions :math:`P_{cumul}` and :math:`Q_{cumul}`:

.. math::

   P_{cumul}(R) = \frac{N_{ij}(R)}{\max(N_{ij})}

Finally, standard JS Divergence is applied to these smoothed, cumulative curves:

.. math::

   CumulativeJSD = JSD(P_{cumul} \parallel Q_{cumul})

----

* **Interpretation**: This metric asks, *"If we grow a sphere outward, does the total accumulated mass of tissue B around tissue A grow at the same rate as tissue A around tissue B?"*
* **Advantage**: It is highly robust to minor spatial jitter, imaging noise, and voxelation artifacts, providing a much more stable measure of macroscopic architectural bias.
