# src/glam_radiomics/config.py
import configparser
import os
import json
import numpy as np

# This is a global, module-level variable that will hold our config
# It starts as None
config = configparser.ConfigParser()
parsed_config = {}

def load_config(config_path):
    """
    Loads and parses the configuration file.
    This function should be called ONCE by the main script.
    """
    global config, parsed_config
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {os.path.abspath(config_path)}")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # --- Pre-parse all values into a simple dict ---
    try:
        parsed = {}
	# System Settings
        # fallback=1 ensures it runs sequentially if the section is missing
        parsed['NumWorkers'] = config.getint('System', 'NumWorkers', fallback=1)

        # GLAM_Settings
        parsed['MaxRdfRadius'] = config.getint('GLAM_Settings', 'MaxRdfRadius')
        parsed['AnisotropyCutoffRadius'] = config.getint('GLAM_Settings', 'AnisotropyCutoffRadius')
        parsed['NumRandomisations'] = config.getint('GLAM_Settings', 'NumRandomisations')
        parsed['RdfSamplePoints'] = config.getint('GLAM_Settings', 'RdfSamplePoints')
        # GLAM_GEOMETRIC_SUPPORT_V1: seed-independent alpha/radius validity from all alpha voxels.
        parsed['RdfMinSupportedReferences'] = config.getint('GLAM_Settings', 'RdfMinSupportedReferences', fallback=50)
        parsed['RdfMinSupportedFraction'] = config.getfloat('GLAM_Settings', 'RdfMinSupportedFraction', fallback=0.01)
        parsed['RdfMinMedianShellVoxels'] = config.getfloat('GLAM_Settings', 'RdfMinMedianShellVoxels', fallback=20.0)
        parsed['RdfMinReferenceShellVoxels'] = config.getint('GLAM_Settings', 'RdfMinReferenceShellVoxels', fallback=1)
        parsed['RdfMinShellVolumeFraction'] = config.getfloat('GLAM_Settings', 'RdfMinShellVolumeFraction', fallback=0.05)

        parsed['QuantizationMethod'] = config.get('GLAM_Settings', 'QuantizationMethod', fallback='FixedCount')
        parsed['NumGrayLevels'] = config.getint('GLAM_Settings', 'NumGrayLevels', fallback=24)
        
        parsed['BinWidth'] = config.getfloat('GLAM_Settings', 'BinWidth', fallback=25.0)
        parsed['QuantizationMin'] = config.getfloat('GLAM_Settings', 'QuantizationMin', fallback=-1000.0)
        parsed['QuantizationMax'] = config.getfloat('GLAM_Settings', 'QuantizationMax', fallback=1000.0)

        # Ensure matrix sizes stay consistent across the dataset by calculating the required levels upfront
        if parsed['QuantizationMethod'].lower() == 'fixedwidth':
            parsed['NumGrayLevels'] = int(np.ceil(
                (parsed['QuantizationMax'] - parsed['QuantizationMin']) / parsed['BinWidth']
            ))

        # File_Naming
        parsed['NormMaskIdentifiers'] = json.loads(config.get('File_Naming', 'NormMaskIdentifiers', fallback='[]'))
        parsed['MaskIdentifiers'] = json.loads(config.get('File_Naming', 'MaskIdentifiers'))
        parsed['SequenceIdentifiers'] = json.loads(config.get('File_Naming', 'SequenceIdentifiers'))

        # Label_Mapping (convert keys to int)
        _label_mapping_str = json.loads(config.get('Label_Mapping', 'LabelMapping'))
        parsed['LabelMapping'] = {int(k): v for k, v in _label_mapping_str.items()}
        
        _labels_for_analysis_str = json.loads(config.get('Label_Mapping', 'LabelsForAnalysis'))
        
        def parse_label_key(k):
            return str(k) if '+' in str(k) else int(k)
            
        parsed['LabelsForAnalysis'] = {parse_label_key(k): v for k, v in _labels_for_analysis_str.items()}

        # Algorithm_Parameters
        parsed['SavgolWindow'] = config.getint('Algorithm_Parameters', 'SavgolWindow')
        parsed['SavgolPoly'] = config.getint('Algorithm_Parameters', 'SavgolPoly')
        parsed['PeakProminence'] = config.getfloat('Algorithm_Parameters', 'PeakProminence')

        # Feature_Mapping (fallback to defaults if section or key is missing)
        parsed['EnableMapping'] = config.getboolean('Feature_Mapping', 'EnableMapping', fallback=False)
        parsed['MapWindowSizeCM'] = config.getfloat('Feature_Mapping', 'MapWindowSizeCM', fallback=2.0)
        parsed['MapMinWindowVoxels'] = config.getint('Feature_Mapping', 'MapMinWindowVoxels', fallback=100)
        parsed['MapFeatures'] = json.loads(config.get('Feature_Mapping', 'MapFeatures', fallback='["CoordNum"]'))

        # In load_config, replace the existing MapMetaMethod line with:
        map_meta_raw = config.get('Feature_Mapping', 'MapMetaMethod', fallback='["Mean"]')
        try:
            parsed['MapMetaMethod'] = json.loads(map_meta_raw)
            if isinstance(parsed['MapMetaMethod'], str):
                parsed['MapMetaMethod'] = [parsed['MapMetaMethod']]
        except json.JSONDecodeError:
            parsed['MapMetaMethod'] = [map_meta_raw.strip('"\'')]

        parsed['MapRDFMaxRadius'] = config.getint('Feature_Mapping', 'MapRDFMaxRadius', fallback=10)
        parsed['MapRDFSamplePoints'] = config.getint('Feature_Mapping', 'MapRDFSamplePoints', fallback=50)

        parsed['MapOverlapPercent'] = config.getfloat('Feature_Mapping', 'MapOverlapPercent', fallback=50.0)
        parsed['MapSaveVisualization'] = config.getboolean('Feature_Mapping', 'MapSaveVisualization', fallback=False)

        from .mapping_mask import parse_mapping_settings
        parsed.update(parse_mapping_settings(config))
        parsed['MapInterpolate'] = config.getboolean('Feature_Mapping', 'MapInterpolate', fallback=False)
        parsed['MapBackend'] = config.get('Feature_Mapping','MapBackend',fallback='CPU').strip()
        if parsed['MapBackend'].lower() not in ('cpu','gpu'): raise ValueError('MapBackend must be CPU or GPU')
        parsed['MapGPUDevice'] = config.getint('Feature_Mapping','MapGPUDevice',fallback=0)
        parsed['MapGPUWindowBatch'] = config.getint('Feature_Mapping','MapGPUWindowBatch',fallback=2)
        parsed['MapGPUStateBatch'] = config.getint('Feature_Mapping','MapGPUStateBatch',fallback=4)
        if parsed['MapGPUDevice'] < 0 or min(parsed['MapGPUWindowBatch'],parsed['MapGPUStateBatch']) < 1: raise ValueError('Invalid GPU mapping settings')
        parsed['MapProfile'] = config.getboolean('Feature_Mapping','MapProfile',fallback=False)
        parsed['MapProfileMaxNodes'] = config.getint('Feature_Mapping','MapProfileMaxNodes',fallback=0)
        parsed['MappingFirst'] = config.getboolean('Feature_Mapping','MappingFirst',fallback=True)
        parsed['MappingOnly'] = config.getboolean('Feature_Mapping','MappingOnly',fallback=False)
        if parsed['MapProfileMaxNodes'] < 0: raise ValueError('MapProfileMaxNodes must be >=0')
        if parsed['MappingOnly'] and not parsed['MappingFirst']: raise ValueError('MappingOnly requires MappingFirst=true')
        parsed_config = parsed
        print(f"Configuration successfully loaded from {config_path}")

    except Exception as e:
        raise ValueError(f"Error parsing config file '{config_path}': {e}")

def get_config(key):
    """
    Gets a pre-parsed configuration value.
    """
    if not parsed_config:
        raise Exception("Configuration is not loaded. Call glam_radiomics.config.load_config(path) first.")
    
    val = parsed_config.get(key)
    if val is None:
        raise KeyError(f"Configuration key '{key}' not found in parsed config.")
    return val
# GLAM_MAPPING_MASK_V3

# GLAM_MAP_INTERPOLATION_V4

# GLAM_BATCHED_GPU_MAPPING_V5

# GLAM_PROFILE_MAPPING_FIRST_V6

FIXED_COUNT_PERCENTILES = (0.25, 99.75)


def get_fixed_count_bounds(values):
    """Centralized FixedCount intensity normalization bounds used identically
    by global analysis (`run.py`) and 3D mapping (`mapping_mask.py`).
    """
    pcts = parsed_config.get('QuantizationPercentiles', FIXED_COUNT_PERCENTILES) if parsed_config else FIXED_COUNT_PERCENTILES
    lo, hi = np.percentile(np.asarray(values, dtype=np.float64), list(pcts))
    return float(lo), float(hi)

