Usage Guide
===========

This guide demonstrates how to perform a multi-parametric radiomic analysis using the GLAM library. The framework relies on a centralized configuration file to manage the complex physics and radiomics parameters.

Configuration (``config.ini``)
------------------------------
GLAM uses a ``.ini`` file to standardize extraction parameters across different datasets. This ensures reproducibility, which is critical for multi-center studies.

A typical configuration file is structured as follows:

.. code-block:: ini

    # Note: configparser keys are case-insensitive, but values preserve case.

    [System]
    # Number of parallel processes to run. 
    # Use 1 for no parallelism. Use 4, 8, etc. based on your CPU. 
    # Recomendation: NumWorkers ~ 2 * Number of CPU Cores.
    NumWorkers = 8

    [GLAM_Settings]
    MaxRdfRadius = 100
    AnisotropyCutoffRadius = 5
    NumRandomisations = 4
    RdfSamplePoints = 100

    # QuantizationMethod can be "FixedCount" (for MRI) or "FixedWidth" (for CT)
    QuantizationMethod = FixedCount

    # --- FixedCount Parameters ---
    NumGrayLevels = 24

    # --- FixedWidth Parameters ---
    BinWidth = 25
    QuantizationMin = -1000
    QuantizationMax = 1000

    [File_Naming]
    # Segmentation file identifiers. 
    MaskIdentifiers = ["_seg.nii.gz", "-seg.nii.gz", "_mask.nii.gz", "-mask.nii.gz"]

    SequenceIdentifiers = {
        "FLAIR": ["_flair", "-flair"],
        "T2": ["_t2", "-t2"],
        "T1": ["_t1", "-t1"],
        "T1c": ["_t1ce", "-t1ce", "_t1gd", "-t1gd", "_t1c", "-t1c"]
        }

    [Label_Mapping] 
    # JSON keys need to be strings ("1", "99").
    LabelMapping = {
        "1": "Enhancing_Tumor",
        "2": "Non_Enhancing_Tumor_Core",
        "4": "Peritumoral_Edema",
        "99": "Whole_Tumor"
        }

    LabelsForAnalysis = {
        "99": "Whole_Tumor"
        }

    [Algorithm_Parameters]
    SavgolWindow = 7
    SavgolPoly = 3
    PeakProminence = 4

Key settings include:

* **GLAM_Settings**: Defines the discretization and the maximum radius for the Radial Distribution Function (RDF) calculation.
* **Radiomics_Settings**: Ensures parity with conventional libraries by setting matching bin counts.
* **File_Naming**: Uses identifiers to automatically pair MRI sequences (T1, T1c, T2, FLAIR) with their corresponding tumor masks.
* **Label_Mapping**: Maps voxel values to biological compartments, such as the **Whole Tumor** or **Enhancing Tumor**.

Image Quantization Methods
--------------------------
To calculate texture and GLAM matrices, continuous image intensities must first be discretized into discrete bins (gray levels). GLAM supports two primary quantization methods depending on your imaging modality.

**1. Fixed Bin Count (MRI)**
Recommended for MRI, where intensity values are relative and lack an absolute physical meaning. This method rescales the Region of Interest (ROI) intensities directly into a fixed number of bins, :math:`N_g` (``NumGrayLevels``).

.. math::
   X_{d} = \left\lfloor \frac{I - I_{min}}{I_{max} - I_{min}} \times N_g \right\rfloor

Where :math:`I_{min}` and :math:`I_{max}` are the minimum and maximum intensities within the ROI. To ensure the absolute maximum intensity (:math:`I = I_{max}`) does not fall out of bounds, the final discrete value :math:`X_d` is capped at :math:`N_g - 1`. This produces a 0-indexed range from :math:`0` to :math:`N_g - 1`.

**2. Fixed Bin Width (CT)**
Recommended for CT scans, where Hounsfield Units (HU) represent absolute physical densities (e.g., water = 0 HU, bone = ~1000 HU). Using a fixed bin count on CTs would destroy this absolute density mapping. Instead, this method clips the image to a defined range :math:`[Q_{min}, Q_{max}]` and divides the intensities into bins of a fixed physical width, :math:`W` (``BinWidth``).

.. math::
   I_{clipped} = \max(Q_{min}, \min(I, Q_{max}))

.. math::
   X_{d} = \left\lfloor \frac{I_{clipped} - Q_{min}}{W} \right\rfloor

In this mode, GLAM dynamically calculates the total number of gray levels needed to cover the range: :math:`N_g = \lceil (Q_{max} - Q_{min}) / W \rceil`. This ensures the meaning of "Gray Level 5" is identical across every patient in your dataset.

Basic Extraction Script
-----------------------
The following script demonstrates how to initialize the global configuration and process a cohort of scans.

.. note::
   If you are running on a multi-core system, you can increase the ``NumWorkers`` in your config file to enable parallel processing.

.. code-block:: python

    import os
    from glam_radiomics.config import load_config
    from glam_radiomics.run import process_scans

    # 1. Define your project paths
    CONFIG_PATH = "path_to/config.ini"
    INPUT_DIR = "path_to/mri_scans"
    OUTPUT_DIR = "path_to/results"
    PROJECT_NAME = "My_GLAM_Project"

    def main():
        # 2. Load the configuration globally
        load_config(CONFIG_PATH)

        # 3. Ensure the output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # 4. Execute the analysis
        process_scans(
            INPUT_DIR, 
            OUTPUT_DIR, 
            project_name=PROJECT_NAME, 
            config_path=CONFIG_PATH
        )

    if __name__ == '__main__':
        main()

Batch Processing Multiple Scans
-------------------------------
For multi-center studies or large cohorts, you can configure a script to loop through multiple project directories. This approach utilizes parallel processing defined by ``NumWorkers`` in your ``config.ini``.

.. code-block:: python

    import os
    from glam_radiomics.config import load_config
    from glam_radiomics.run import process_scans

    CONFIG_PATH = "path/to/config.ini"

    # Define project batches (Project Name, Input, Output)
    projects_to_run = [
        ("Cohort_A", "C:/data/cohort_a_scans", "C:/results/cohort_a"),
        ("Cohort_B", "C:/data/cohort_b_scans", "C:/results/cohort_b")
    ]

    def main():
        load_config(CONFIG_PATH)

        for proj_name, input_path, output_path in projects_to_run:
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            process_scans(input_path, output_path, 
                          project_name=proj_name, 
                          config_path=CONFIG_PATH)

    if __name__ == '__main__':
        main()

Output Files
------------
After execution, the output directory will contain:

1. **GLAM Matrices**: CSV files containing the :math:`N \times N` interaction data (e.g., ``PressureVirial_Symlog.csv``).
2. **Feature CSVs**: Consolidated tables of physics descriptors (``3d_glam_primary_features.csv``) and topological metrics (``3d_glam_meta_and_radiomic_features.csv``).
3. **Log Files**: Details on the extraction process and data harmonization.