# src/glam_radiomics/utils.py
import os
import pandas as pd
import SimpleITK as sitk
from collections import defaultdict
from .config import get_config 
import numpy as np

def find_scan_mask_pairs(directory):
    """
    Finds ONE mask and ALL associated image sequences in a directory.
    Applies the single mask to all found image sequences.
    Uses identifiers from the global config. Search is CASE-INSENSITIVE.
    """

    # --- Get params inside the function ---
    sequence_identifiers = get_config('SequenceIdentifiers')
    mask_identifiers = get_config('MaskIdentifiers')
    # ---

    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]
    
    # --- NEW LOGIC: Find all masks and all images ---
    mask_paths = []
    image_paths = []
    
    # Pre-lower the mask identifiers for efficiency
    mask_id_lower = [m.lower() for m in mask_identifiers]

    for f in files:
        f_lower = f.lower()
        is_mask = False
        for identifier in mask_id_lower:
            # Use endswith for masks like '_seg.nii.gz'
            if f_lower.endswith(identifier): 
                mask_paths.append(os.path.join(directory, f))
                is_mask = True
                break
        if not is_mask:
            image_paths.append(os.path.join(directory, f))
    # ---

    # --- Handle findings ---
    if not mask_paths:
        print(f"  > DEBUG: No mask file found in folder '{directory}'. Skipping.")
        return {}
    
    if not image_paths:
        print(f"  > DEBUG: No image files (non-mask) found in folder '{directory}'. Skipping.")
        return {}
        
    if len(mask_paths) > 1:
        print(f"  > WARNING: Found {len(mask_paths)} masks in '{directory}'. Using the first one: {os.path.basename(mask_paths[0])}")
    
    mask_to_use = mask_paths[0]
    
    # --- Build the image dictionary ---
    images_dict = {}
    
    # Create a list of (seq_name, identifier_str) tuples
    SEQUENCE_ID_CHECKS = []
    for seq_name, identifiers_list in sequence_identifiers.items():
        for identifier in identifiers_list:
            SEQUENCE_ID_CHECKS.append((seq_name, identifier.lower())) # Pre-lower

    for img_path in image_paths:
        img_name_lower = os.path.basename(img_path).lower()
        seq_name_found = 'Unknown' # Default
        
        # Find the *best* (most specific/longest) match
        best_match_len = -1
        for seq_name, identifier in SEQUENCE_ID_CHECKS:
            if identifier in img_name_lower:
                if len(identifier) > best_match_len:
                    seq_name_found = seq_name
                    best_match_len = len(identifier)
                    
        if seq_name_found != 'Unknown':
            if seq_name_found in images_dict:
                 print(f"  > DEBUG: Matched duplicate sequence '{seq_name_found}' from file '{img_name_lower}'. Overwriting.")
            print(f"  > DEBUG: Matched image '{img_name_lower}' as sequence '{seq_name_found}'")
            images_dict[seq_name_found] = img_path
        else:
            print(f"  > DEBUG: Skipping image '{img_name_lower}', could not determine sequence from config identifiers.")

    if not images_dict:
        print(f"  > DEBUG: No images in '{directory}' matched any sequence identifiers. Skipping.")
        return {}

    # --- Create the final 'pairs' object ---
    # The 'prefix' will be the patient folder name
    patient_prefix = os.path.basename(directory)
    
    pairs = defaultdict(lambda: {'mask': None, 'images': {}})
    pairs[patient_prefix]['mask'] = mask_to_use
    pairs[patient_prefix]['images'] = images_dict
    
    return pairs

def generate_binary_mask(multilabel_mask_sitk, label_id):
    """
    Generates a binary SITK mask for a specific label_id.
    Handles the special '99' (Whole_Tumor) case by creating a union.
    """
    # --- Get param inside the function ---
    all_label_definitions = get_config('LabelMapping')
    # ---

    if label_id == 99:
        tumor_component_labels = [k for k in all_label_definitions.keys() if k != 99]
        binary_masks = [(multilabel_mask_sitk == l) for l in tumor_component_labels]
        
        if not binary_masks: # Handle case where no tumor labels are defined
            print("  - WARNING: 'Whole_Tumor' (99) requested, but no tumor component labels (1, 2, 4...) are in config's LabelMapping.")
            # Return an empty mask
            return sitk.Image(multilabel_mask_sitk.GetSize(), sitk.sitkUInt8)

        final_binary_mask = binary_masks[0]
        for i in range(1, len(binary_masks)):
            final_binary_mask = sitk.Or(final_binary_mask, binary_masks[i])
            
        return final_binary_mask
        
    else:
        # Standard case: just select voxels matching the label_id
        return (multilabel_mask_sitk == label_id)    

def reformat_dict_to_matrix(glam_dict, num_levels, key_prefix, diag_prefix=None):
    """Helper to reformat a flat dictionary of GLAM features into a matrix."""
    matrix = np.full((num_levels, num_levels), np.nan)
    
    # 1. Handle off-diagonal / full matrix (ONLY IF PREFIX EXISTS)
    # --- CRITICAL FIX: Skip this loop if key_prefix is None (i.e. for Diagonal-only matrices) ---
    if key_prefix is not None:
        for key, value in glam_dict.items():
            if not key.startswith(key_prefix): continue
            
            # THE FIX: Stop the baseline matrix from sucking in 'L1.0' or 'Decay' keys!
            remainder = key[len(key_prefix):]
            if "L" in remainder or "Decay" in remainder: 
                continue

            try:
                parts = key.split('_')
                
                # Standard case (format: ..._0_1)
                alpha, beta = int(parts[-2]), int(parts[-1])
                matrix[alpha, beta] = value
                    
            except (IndexError, ValueError): continue
    
    # 2. Handle special diagonal prefix (VolumeFD, VolumeLacunarity, VolumeBetti, etc.)
    if diag_prefix:
        for key, value in glam_dict.items():
            if not key.startswith(diag_prefix): continue
            
            # Protect diagonals too
            remainder = key[len(diag_prefix):]
            if "L" in remainder or "Decay" in remainder: 
                continue

            try:
                parts = key.split('_')
                idx = int(parts[-1])
                if 0 <= idx < num_levels:
                    matrix[idx, idx] = value
            except (IndexError, ValueError): continue
            
    return matrix

def save_matrix(matrix, prefix, output_folder, file_suffix, index=None, columns=None):
    """
    Saves a matrix as a CSV.
    TRUNCATION LOGIC: If the matrix has > 100 columns, it saves only the first 100 
    to ensure files are lightweight and readable for visualization.
    """
    if matrix is None: return

    # Create DataFrame (handles both numpy arrays and lists)
    df = pd.DataFrame(matrix, index=index, columns=columns)
    
    # Check dimensions and truncate if necessary
    if df.shape[1] > 100 and "RDF" not in file_suffix:
        # Save only the first 100 columns (e.g. Run Lengths 1-100)
        df = df.iloc[:, :100]
        # We don't change the filename, just the content
    
    output_path = os.path.join(output_folder, f"{prefix}_{file_suffix}.csv")
    
    # Save
    # header=True if columns were provided or if we have a dataframe with headers
    save_header = True if columns is not None else False
    
    df.to_csv(output_path, index=(index is not None), header=save_header)
    print(f"  - Saved {file_suffix} to '{output_path}' (Cols: {df.shape[1]})")

def save_feature_dataframes(primary_features_list, meta_features_list, output_folder):
    """
    Saves the final aggregated primary and meta-feature DataFrames to CSVs.
    The final CSVs are saved to the PARENT directory ("root folder") of the
    output_folder, which contains all the individual matrices.
    """
    
    # os.path.dirname('/path/to/output_dir') will return '/path/to'
    root_folder = os.path.dirname(output_folder)
    # ---

    if primary_features_list:
        df_primary = pd.DataFrame(primary_features_list)
        
        primary_path = os.path.join(root_folder, "3d_glam_primary_features.csv")
        df_primary.to_csv(primary_path, index=False)
        print(f"Saved primary features to '{primary_path}'")
        # ---

    if meta_features_list:
        df_meta = pd.DataFrame(meta_features_list)
        
        final_meta_path = os.path.join(root_folder, "3d_glam_meta_and_radiomic_features.csv")
        df_meta.to_csv(final_meta_path, index=False)
        print(f"Saved meta features to '{final_meta_path}'")
        # ---