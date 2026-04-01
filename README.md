# GRAY LEVEL AFFINITY METRICS (GLAM) [![Documentation Status](https://readthedocs.org/projects/glam-radiomics/badge/?version=latest)](https://glam-radiomics.readthedocs.io/en/latest/index.html)

This project introduces and implements a novel class of quantitative imaging features, termed Gray Level Affinity Metrics (GLAM), derived from the foundational principles of statistical mechanics. By treating image voxels as interacting particles within a multi-component system, the GLAM framework provides a set of physically interpretable metrics that characterize the spatial organization and texture of 3D medical images.

The GLAM framework is written in Python and leverages high-performance libraries for medical image analysis. It operates as a fully standalone extraction engine, meaning it does not require external radiomics packages to compute conventional texture matrices.

## Installing from GitHub

You can install GLAM-radiomics using the following command:
```Plaintext
pip install glam-radiomics
```

### Key Dependencies
When you install GLAM, the following core libraries are automatically integrated:
* **NumPy & SciPy**: Provide the computational backbone for RDF calculations, spatial KD-trees, and Statistical Mechanics descriptors.
* **SimpleITK**: Handles the loading and normalization of 3D medical imaging formats like NIfTI (`.nii.gz`).
* **Pandas**: Manages the structured output of multiscale Radial Distribution Functions and feature aggregation.
* **Scikit-image & Scikit-learn**: Powers the morphological marching cubes (surface area), K-Means clustering, and advanced geometric descriptors.
* **tqdm**: Used to display progress bars during long-running feature extraction operations.

## Usage Guide

### Configuration (config.ini)
GLAM uses a `.ini` file to standardize extraction parameters across different datasets. This ensures reproducibility, which is critical for multi-center studies.

A typical configuration file is structured as follows:

```ini
# Note: configparser keys are case-insensitive, but values preserve case.

[System]
# Number of parallel processes to run. NumWorkers ~ 2 * Number of CPU Cores.
# Use 1 for no parallelism. Use 4, 8, etc. based on your CPU.
NumWorkers = 16

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
```

### Basic Extraction Script
The following script demonstrates how to initialize the global configuration and process a cohort of scans.

```python
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
```

### Batch Processing Multiple Scans
For multi-center studies or large cohorts, you can configure a script to loop through multiple project directories. This approach utilizes parallel processing defined by NumWorkers in your config.ini.

```python
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
```

### File and Folder Organization
To ensure the pipeline pairs the correct masks with the correct sequences, structure your input directory as follows:

```Plaintext
C:\...\Research\GLAM\scans\   <-- Your 'INPUT_DIR'
├── PT011901\
│   ├── PT011901_t1.nii.gz
│   ├── PT011901_t2.nii.gz
│   └── PT011901_mask.nii.gz
├── PT011902\
│   ├── PT011902_t1.nii.gz
│   ├── PT011902_flair.nii.gz
│   └── PT011902_seg.nii.gz
└── ...
```


