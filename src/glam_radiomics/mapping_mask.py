"""Independent mapping ROI for GLAM; global analysis and normalization unchanged."""
from pathlib import Path
import json
import re
import numpy as np


def parse_mapping_settings(config):
    section = 'Feature_Mapping'
    mode = config.get(section, 'MappingMaskMode', fallback='File').strip().lower()
    if mode not in ('file', 'analysis'):
        raise ValueError('MappingMaskMode must be File or Analysis')
    identifiers = json.loads(config.get(section, 'MappingMaskIdentifiers', fallback='["_brain.nii.gz"]'))
    labels = json.loads(config.get(section, 'MappingMaskLabels', fallback='[]'))
    name = config.get(section, 'MappingMaskName', fallback='Brain').strip()
    if not isinstance(identifiers, list) or not all(isinstance(x,str) and x for x in identifiers):
        raise ValueError('MappingMaskIdentifiers must be a JSON list of nonempty filename suffixes')
    if mode == 'file' and not identifiers:
        raise ValueError('File mode requires MappingMaskIdentifiers')
    if not isinstance(labels,list) or not all(type(x) is int and x > 0 for x in labels):
        raise ValueError('MappingMaskLabels must be positive integer labels; [] means all positive voxels')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',name):
        raise ValueError('MappingMaskName must contain only letters, digits, underscores or hyphens')
    return dict(MappingMaskMode=mode, MappingMaskIdentifiers=identifiers,
                MappingMaskLabels=labels, MappingMaskName=name)


def select_mask(array, labels):
    array = np.asarray(array)
    if not np.isfinite(array).all():
        raise ValueError('Mapping mask contains nonfinite values')
    if labels and np.any(array != np.floor(array)):
        raise ValueError('Label-selected mapping mask must contain integer labels')
    result = np.isin(array,labels) if labels else array > 0
    if not result.any():
        raise ValueError('Selected mapping mask is empty')
    return result.astype(np.uint8)


def find_mapping_mask(prefix, paths, identifiers):
    # Exact scan prefix + suffix, never the first arbitrary brain file in a folder.
    folders = {Path(p).resolve().parent for p in paths.get('images',{}).values()}
    if paths.get('mask'):
        folders.add(Path(paths['mask']).resolve().parent)
    expected = {(str(prefix)+suffix).casefold() for suffix in identifiers}
    candidates = {p.resolve() for folder in folders for p in folder.iterdir()
                  if p.is_file() and p.name.casefold() in expected}
    if len(candidates) != 1:
        raise ValueError(f'Expected one mapping mask for {prefix}; found {len(candidates)}. '
                         f'Expected filenames: {sorted(expected)}. No analysis-mask fallback.')
    return next(iter(candidates))


def check_geometry(image, mask, description):
    if image.GetSize() != mask.GetSize():
        raise ValueError(f'{description}: image/mask size mismatch')
    for attr in ('GetSpacing','GetOrigin','GetDirection'):
        if not np.allclose(getattr(image,attr)(),getattr(mask,attr)(),rtol=0,atol=1e-6):
            raise ValueError(f'{description}: {attr} mismatch; register/resample the mask explicitly')


def quantize_mapping(image, mask, method, levels, bounds=None, q_min=None, q_max=None, bin_width=None):
    """Quantize the mapping domain itself, not the analysis crop."""
    image = np.asarray(image,dtype=np.float64)
    roi = np.asarray(mask)>0
    if image.shape != roi.shape or not roi.any():
        raise ValueError('Empty mapping ROI or image/mask shape mismatch')
    values = image[roi]
    if not np.isfinite(values).all():
        raise ValueError('Mapping ROI contains nonfinite intensities')
    if int(levels)!=levels or not 1 <= levels <= 32767:
        raise ValueError('Invalid mapping gray-level count for int16')
    method = method.lower()
    if method == 'fixedcount':
        if bounds is None:
            lo,hi = np.percentile(values,[1,99])
        else:
            lo,hi = bounds
        if not np.isfinite([lo,hi]).all() or hi < lo:
            raise ValueError('Invalid mapping normalization bounds')
        q = np.floor((np.clip(values,lo,hi)-lo)/(hi-lo)*levels) if hi-lo>1e-6 else np.zeros(values.size)
    elif method == 'fixedwidth':
        if not np.isfinite([q_min,q_max,bin_width]).all() or bin_width<=0 or q_max<=q_min:
            raise ValueError('Invalid fixed-width mapping settings')
        if int(np.ceil((q_max-q_min)/bin_width))!=levels:
            raise ValueError('Fixed-width level count mismatch')
        q = np.floor((np.clip(values,q_min,q_max)-q_min)/bin_width)
    else:
        raise ValueError('Unknown mapping quantization method')
    result = np.full(image.shape,-1,dtype=np.int16)
    result[roi] = np.clip(q,0,levels-1).astype(np.int16)
    return result


def generate_scan_maps(prefix, paths, output_dir, config_path, labels_to_process):
    import SimpleITK as sitk
    from .config import get_config
    from .utils import generate_binary_mask
    from . import mapping
    if not get_config('EnableMapping'):
        return
    mode = get_config('MappingMaskMode')
    if mode == 'file':
        path = find_mapping_mask(prefix,paths,get_config('MappingMaskIdentifiers'))
        source = sitk.ReadImage(str(path))
        array = select_mask(sitk.GetArrayFromImage(source),get_config('MappingMaskLabels'))
        roi_image = sitk.GetImageFromArray(array)
        roi_image.CopyInformation(source)
        rois = [(get_config('MappingMaskName'),roi_image)]
        print(f'  - Independent mapping mask: {path.name}; selected voxels: {int(array.sum())}')
    else:
        source = sitk.ReadImage(paths['mask'])
        rois = [(name,generate_binary_mask(source,label)) for label,name in labels_to_process.items()]
        print('  - MappingMaskMode=Analysis: explicitly using analysis label masks')
    method = get_config('QuantizationMethod').lower()
    levels = get_config('NumGrayLevels')
    # Use the same normalization reference policy as analysis, independently of
    # the mapping ROI. A supplied invalid normalization mask fails explicitly.
    norm = None
    if method == 'fixedcount':
        if paths.get('norm_mask'):
            norm = sitk.ReadImage(paths['norm_mask'])
        else:
            norm = generate_binary_mask(sitk.ReadImage(paths['mask']),99)
    for seq,image_path in paths.get('images',{}).items():
        image = sitk.ReadImage(image_path,sitk.sitkFloat32)
        array = sitk.GetArrayFromImage(image)
        bounds = None
        if norm is not None:
            check_geometry(image,norm,'Normalization mask')
            values = array[sitk.GetArrayFromImage(norm)>0]
            if not values.size or not np.isfinite(values).all():
                raise ValueError('Empty/nonfinite normalization reference for mapping')
            bounds = tuple(np.percentile(values,[1,99]))
        for name,roi in rois:
            check_geometry(image,roi,'Mapping mask')
            roi_array = sitk.GetArrayFromImage(roi)
            if not np.any(roi_array>0):
                print(f'  - Skipping empty mapping mask: {name}')
                continue
            quantized = quantize_mapping(array,roi_array,method,levels,bounds,
                get_config('QuantizationMin') if method=='fixedwidth' else None,
                get_config('QuantizationMax') if method=='fixedwidth' else None,
                get_config('BinWidth') if method=='fixedwidth' else None)
            map_prefix = f'{prefix}_{name}_{seq}'
            print(f'  --- Independent mapping: {map_prefix} ---')
            # Preserve parent RNG state so mapping cannot alter later global features.
            state = np.random.get_state()
            try:
                mapping.generate_feature_maps(image,roi,quantized,levels,map_prefix,output_dir,config_path)
            finally:
                np.random.set_state(state)
