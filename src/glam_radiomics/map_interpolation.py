"""Conservative trilinear reconstruction of GLAM's regular sampled grid."""
import itertools
import os
import time
import numpy as np


def interpolate_grid(sampled, mask, strides):
    """Fill only adjacent cells with 8 finite corners and entirely in-mask support.
    Preserve finite sampled values. No extrapolation, smoothing, hole filling,
    triangulation, or bridging missing grid nodes. Returns float32 and cell count.
    """
    sampled = np.asarray(sampled)
    mask = np.asarray(mask) > 0
    if sampled.ndim != 3 or sampled.shape != mask.shape:
        raise ValueError('Interpolation requires matching 3D map/mask arrays')
    if len(strides)!=3 or any(int(s)!=s or s<1 for s in strides):
        raise ValueError('Interpolation strides must be three positive integers')
    strides = tuple(int(s) for s in strides)
    axes = [np.arange(0,d,s) for d,s in zip(mask.shape,strides)]
    out = np.full(sampled.shape,np.nan,dtype=np.float32)
    finite = np.isfinite(sampled) & mask
    out[finite] = sampled[finite]
    if any(len(a)<2 for a in axes):
        return out,0
    coarse = sampled[np.ix_(*axes)]
    corners_valid = np.ones(tuple(len(a)-1 for a in axes),dtype=bool)
    for bits in itertools.product((0,1),repeat=3):
        sl=tuple(slice(b,b+n) for b,n in zip(bits,corners_valid.shape))
        corners_valid &= np.isfinite(coarse[sl])
    # Integer summed-volume table makes full-cell mask checks constant-time.
    integral=np.pad(mask.astype(np.int64),((1,0),)*3)
    for axis in range(3):
        np.cumsum(integral,axis=axis,out=integral)
    def count_box(lo,hi):
        count=0
        for bits in itertools.product((0,1),repeat=3):
            idx=tuple(hi[a]+1 if bits[a] else lo[a] for a in range(3))
            count += (-1)**(3-sum(bits))*int(integral[idx])
        return count
    weights=[]
    for s in strides:
        t=np.arange(s+1,dtype=float)/s
        weights.append((1-t,t))
    cells=0
    for index in np.argwhere(corners_valid):
        lo=tuple(int(axes[a][index[a]]) for a in range(3))
        hi=tuple(int(axes[a][index[a]+1]) for a in range(3))
        if count_box(lo,hi)!=int(np.prod(np.subtract(hi,lo)+1)):
            continue
        block=np.zeros(tuple(s+1 for s in strides),dtype=np.float64)
        for bits in itertools.product((0,1),repeat=3):
            value=coarse[tuple(index[a]+bits[a] for a in range(3))]
            block += value * weights[0][bits[0]][:,None,None] * weights[1][bits[1]][None,:,None] * weights[2][bits[2]][None,None,:]
        sl=tuple(slice(l,h+1) for l,h in zip(lo,hi))
        out[sl]=block.astype(np.float32)
        cells+=1
    # Restore original nodes exactly, including nodes bordering unsupported cells.
    out[finite]=sampled[finite]
    out[~mask]=np.nan
    return out,cells


def save_interpolated_map(sampled, mask, strides, reference, prefix, name,
                          output_dir, save_visualization=False):
    import SimpleITK as sitk
    start=time.perf_counter()
    dense,cells=interpolate_grid(sampled,mask,strides)
    finite=np.isfinite(dense)
    new_count=int(finite.sum())-int((np.isfinite(sampled)&(np.asarray(mask)>0)).sum())
    if new_count<=0:
        print(f'  - WARNING: {name}: no supported interpolation cells; sampled map retained.')
        return
    image=sitk.GetImageFromArray(dense)
    image.CopyInformation(reference)
    stem=f'{prefix}_MAP_{name}_Interpolated'
    sitk.WriteImage(image,os.path.join(output_dir,stem+'.nii.gz'))
    if save_visualization:
        scaled=np.zeros(dense.shape,dtype=np.uint8)
        lo,hi=float(dense[finite].min()),float(dense[finite].max())
        scaled[finite]=(1+254*(dense[finite]-lo)/(hi-lo)).astype(np.uint8) if hi-lo>1e-6 else 128
        viz=sitk.GetImageFromArray(scaled); viz.CopyInformation(reference)
        sitk.WriteImage(viz,os.path.join(output_dir,stem+'_uint8.nii.gz'))
    print(f'  - Saved interpolated map: {stem}.nii.gz; {cells} supported cells, '
          f'{new_count} added voxels; elapsed {time.perf_counter()-start:.2f}s')
