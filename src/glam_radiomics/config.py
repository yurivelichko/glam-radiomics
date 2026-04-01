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
        parsed['MaskIdentifiers'] = json.loads(config.get('File_Naming', 'MaskIdentifiers'))
        parsed['SequenceIdentifiers'] = json.loads(config.get('File_Naming', 'SequenceIdentifiers'))

        # Label_Mapping (convert keys to int)
        _label_mapping_str = json.loads(config.get('Label_Mapping', 'LabelMapping'))
        parsed['LabelMapping'] = {int(k): v for k, v in _label_mapping_str.items()}
        
        _labels_for_analysis_str = json.loads(config.get('Label_Mapping', 'LabelsForAnalysis'))
        parsed['LabelsForAnalysis'] = {int(k): v for k, v in _labels_for_analysis_str.items()}

        # Algorithm_Parameters
        parsed['SavgolWindow'] = config.getint('Algorithm_Parameters', 'SavgolWindow')
        parsed['SavgolPoly'] = config.getint('Algorithm_Parameters', 'SavgolPoly')
        parsed['PeakProminence'] = config.getfloat('Algorithm_Parameters', 'PeakProminence')

        # Feature_Mapping (fallback to defaults if section or key is missing)
        parsed['EnableMapping'] = config.getboolean('Feature_Mapping', 'EnableMapping', fallback=False)
        parsed['MapWindowSizeCM'] = config.getfloat('Feature_Mapping', 'MapWindowSizeCM', fallback=2.0)
        parsed['MapMinWindowVoxels'] = config.getint('Feature_Mapping', 'MapMinWindowVoxels', fallback=100)
        parsed['MapFeatures'] = json.loads(config.get('Feature_Mapping', 'MapFeatures', fallback='["CoordNum"]'))
        parsed['MapMetaMethod'] = config.get('Feature_Mapping', 'MapMetaMethod', fallback='Mean')
        parsed['MapRDFMaxRadius'] = config.getint('Feature_Mapping', 'MapRDFMaxRadius', fallback=10)
        parsed['MapRDFSamplePoints'] = config.getint('Feature_Mapping', 'MapRDFSamplePoints', fallback=50)

        parsed['MapOverlapPercent'] = config.getfloat('Feature_Mapping', 'MapOverlapPercent', fallback=50.0)
        parsed['MapSaveVisualization'] = config.getboolean('Feature_Mapping', 'MapSaveVisualization', fallback=False)

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