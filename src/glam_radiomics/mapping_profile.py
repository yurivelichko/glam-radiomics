"""Diagnostic stage timings. CUDA stage boundaries synchronize deliberately."""
import contextvars
import contextlib
import functools
import json
from pathlib import Path
import time

_ACTIVE=contextvars.ContextVar('glam_profile',default=None)

@contextlib.contextmanager
def stage(name,cuda=False):
    report=_ACTIVE.get()
    if report is None:
        yield
        return
    cp=None
    if cuda:
        import cupy as cp
        cp.cuda.get_current_stream().synchronize()
        start_event=cp.cuda.Event(); end_event=cp.cuda.Event(); start_event.record()
    start=time.perf_counter()
    try:
        yield
    finally:
        gpu=None
        if cp is not None:
            end_event.record(); end_event.synchronize()
            gpu=float(cp.cuda.get_elapsed_time(start_event,end_event))/1000
        elapsed=time.perf_counter()-start
        row=report['stages'].setdefault(name,dict(calls=0,wall_seconds=0.0,cuda_event_seconds=0.0))
        row['calls']+=1; row['wall_seconds']+=elapsed
        if gpu is not None: row['cuda_event_seconds']+=gpu


def profile_mapping(fn):
    @functools.wraps(fn)
    def wrapper(image_sitk,binary_mask_sitk,quantized_image_array,num_gray_levels,prefix,output_dir,config_path):
        from .config import get_config
        enabled=get_config('MapProfile')
        cap=get_config('MapProfileMaxNodes') if enabled else 0
        actual_prefix=prefix+'_PROFILETEST' if cap>0 else prefix
        if not enabled:
            return fn(image_sitk,binary_mask_sitk,quantized_image_array,num_gray_levels,actual_prefix,output_dir,config_path)
        report=dict(prefix=actual_prefix,backend=get_config('MapBackend'),node_limit=cap,
                    status='running',stages={},notes=[
                        'Stage wall timings include Python work and waiting.',
                        'CUDA event intervals include host-induced idle gaps; not pure kernel time.',
                        'Parent stages overlap children: do not sum all rows.',
                        'Profiling synchronizes CUDA at diagnostic boundaries and changes throughput.',
                        'CPU pool worker-internal stages are not captured by this parent profiler.'])
        token=_ACTIVE.set(report); start=time.perf_counter()
        try:
            result=fn(image_sitk,binary_mask_sitk,quantized_image_array,num_gray_levels,actual_prefix,output_dir,config_path)
            report['status']='completed'; return result
        except BaseException as e:
            report['status']='failed'; report['error']=str(e); raise
        finally:
            report['total_wall_seconds']=time.perf_counter()-start
            _ACTIVE.reset(token)
            folder=Path(output_dir); folder.mkdir(parents=True,exist_ok=True)
            path=folder/(actual_prefix+'_MappingProfile.json')
            path.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('  - Mapping profile:',path)
            for name,row in report['stages'].items():
                print(f"    {name}: {row['wall_seconds']:.3f}s wall / {row['calls']} calls")
    return wrapper


def limit_nodes(coords):
    from .config import get_config
    report=_ACTIVE.get()
    cap=get_config('MapProfileMaxNodes') if get_config('MapProfile') else 0
    if report is not None: report['candidate_nodes']=len(coords)
    if cap>0 and len(coords)>cap:
        import numpy as np
        coords=coords[np.linspace(0,len(coords)-1,cap,dtype=int)]
        print(f'  - PROFILE TEST ONLY: {len(coords)} evenly distributed nodes; interpolation disabled.')
    if report is not None: report['selected_nodes']=len(coords)
    return coords


def allow_interpolation():
    from .config import get_config
    return get_config('MapInterpolate') and not (get_config('MapProfile') and get_config('MapProfileMaxNodes')>0)
