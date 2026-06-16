"""
Custom scaling functions for climate variables.

Scaling strategies:
- tas, ts, hurs: min-max scaling
- pr, sfcWind: skew-aware signed power + one-sided softsign (maps to ~[0,1))
- evspsbl: skew-aware signed power + two-sided softsign (maps to ~(0,1))
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

from weasel.constants import VARIABLES


def signed_power(x: np.ndarray, alpha: float, c: float, inverse: bool = False, eps: float = 1e-12) -> np.ndarray:
    """
    Signed power transformation for handling skewed distributions.
    
    Args:
        x: Input array
        alpha: Power exponent (< 1 compresses large values, > 1 expands)
        c: Scale factor
        inverse: If True, apply inverse transformation
        eps: Small value to ensure positive scale
    
    Returns:
        Transformed array
    """
    c = max(float(c), eps)
    if not inverse:
        return np.sign(x) * np.power(np.abs(x) / c, alpha)
    else:
        return np.sign(x) * np.power(np.abs(x), 1.0 / alpha) * c


def softsign01(z: np.ndarray, inverse: bool = False, eps: float = 1e-8) -> np.ndarray:
    """
    Two-sided softsign: maps (-inf, inf) to (0, 1).
    Suitable for variables with both positive and negative values (e.g., evspsbl).
    
    Args:
        z: Input array
        inverse: If True, apply inverse transformation
        eps: Small value to avoid division by zero
    
    Returns:
        Transformed array in (0, 1) for forward, (-inf, inf) for inverse
    """
    if not inverse:
        u = z / (1.0 + np.abs(z))
        return 0.5 * (u + 1.0)
    else:
        y = np.clip(z, eps, 1.0 - eps)
        u = 2.0 * y - 1.0
        return u / np.maximum(1.0 - np.abs(u), eps)


def softsign01_nonneg(z: np.ndarray, inverse: bool = False, eps: float = 1e-8) -> np.ndarray:
    """
    One-sided softsign: maps [0, inf) to [0, 1).
    Suitable for non-negative variables (e.g., pr, sfcWind).
    
    Args:
        z: Input array
        inverse: If True, apply inverse transformation
        eps: Small value to avoid division by zero
    
    Returns:
        Transformed array in [0, 1) for forward, [0, inf) for inverse
    """
    if not inverse:
        return z / (1.0 + z)
    else:
        y = np.minimum(z, 1.0 - eps)
        return y / np.maximum(1.0 - y, eps)


def alpha_from_skew(skew: float) -> float:
    """
    Determine power exponent based on skewness.
    Higher skew -> lower alpha (more compression).
    
    Args:
        skew: Skewness value
    
    Returns:
        Alpha exponent for signed_power
    """
    if skew > 5.0:
        return 0.3
    if skew > 1.0:
        return 0.5
    if skew < -0.5:
        return 1.5
    return 1.0


def scale_c_from_minmax(vmin: float, vmax: float, eps: float = 1e-12) -> float:
    """
    Compute scale factor c from min/max values.
    For non-negative variables, uses vmax.
    For signed variables, uses max(|min|, |max|).
    
    Args:
        vmin: Minimum value
        vmax: Maximum value
        eps: Small value to ensure positive result
    
    Returns:
        Scale factor c
    """
    if vmin >= 0:
        return max(vmax, eps)
    return max(abs(vmin), abs(vmax), eps)


def scale_variable(
    data: np.ndarray,
    variable: str,
    stats: Dict,
    inverse: bool = False,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Scale a single variable array using appropriate transformation.
    
    Args:
        data: Input array (any shape)
        variable: Variable name
        stats: Dict with '_min', '_max', '_skew', '_q95' keys
        inverse: If True, apply inverse scaling
        eps: Small value for numerical stability
    
    Returns:
        Scaled array (same shape as input)
    """
    vmin = float(stats['_min'])
    vmax = float(stats['_max'])
    q95 = float(stats.get('_q95', vmax))
    skew = float(stats.get('_skew', 0.0))
    
    # Min-max scaling for temperature and humidity
    if variable in ('tas', 'ts'):
        den = max(vmax - vmin, eps)
        if not inverse:
            return ((data - vmin) / den).astype(np.float32)
        else:
            return (data * den + vmin).astype(np.float32)
    
    if variable == 'hurs':
        # Min-max scaling for humidity (can exceed 100%)
        den = max(vmax - vmin, eps)
        if not inverse:
            return ((data - vmin) / den).astype(np.float32)
        else:
            return (data * den + vmin).astype(np.float32)
    
    # Skew-aware + one-sided softsign for non-negative skewed variables
    if variable in ('pr', 'sfcWind'):
        alpha = alpha_from_skew(skew)
        cscale = scale_c_from_minmax(vmin, q95, eps=eps)
        if not inverse:
            z = signed_power(data, alpha=alpha, c=cscale, inverse=False)
            return softsign01_nonneg(z, inverse=False).astype(np.float32)
        else:
            z = softsign01_nonneg(data, inverse=True, eps=eps)
            return signed_power(z, alpha=alpha, c=cscale, inverse=True).astype(np.float32)
    
    # Skew-aware + two-sided softsign for signed variables
    if variable == 'evspsbl':
        alpha = alpha_from_skew(skew)
        cscale = scale_c_from_minmax(vmin, q95, eps=eps)
        if not inverse:
            z = signed_power(data, alpha=alpha, c=cscale, inverse=False)
            return softsign01(z, inverse=False).astype(np.float32)
        else:
            z = softsign01(data, inverse=True, eps=eps)
            return signed_power(z, alpha=alpha, c=cscale, inverse=True).astype(np.float32)
    
    # Fallback: simple min-max scaling
    den = max(vmax - vmin, eps)
    if not inverse:
        return ((data - vmin) / den).astype(np.float32)
    else:
        return (data * den + vmin).astype(np.float32)


def scale_channels(
    arr: np.ndarray,
    global_stats: Dict,
    inverse: bool = False,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Scale all channels in a (C, H, W) array.
    Channel order must match VARIABLES constant.
    
    Args:
        arr: Input array with shape (C, H, W) where C == len(VARIABLES)
        global_stats: Dict mapping variable name to stats dict
        inverse: If True, apply inverse scaling
        eps: Small value for numerical stability
    
    Returns:
        Scaled array with same shape
    """
    assert arr.ndim == 3 and arr.shape[0] == len(VARIABLES), \
        f"Expected (C,H,W) with C == {len(VARIABLES)}, got shape {arr.shape}"
    
    out = np.empty_like(arr, dtype=np.float32)
    
    for c, var in enumerate(VARIABLES):
        out[c] = scale_variable(
            arr[c],
            variable=var,
            stats=global_stats[var],
            inverse=inverse,
            eps=eps
        )
    
    return out


def _parse_dict_field(value):
    """Parse a dict field that may be stored as string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import ast
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
    return {}


def process_and_scale_row(args: Tuple) -> Tuple[str, Dict, np.ndarray]:
    """
    Process a single row: load data from NC files, extract date slice, scale.
    
    Args:
        args: Tuple of (row_dict, global_stats, date_str)
    
    Returns:
        Tuple of (date_str, meta_dict, scaled_data)
    """
    row_dict, global_stats, date_str = args
    
    variable_paths = _parse_dict_field(row_dict['variable_paths'])
    
    # Initialize output array
    sample_path = list(variable_paths.values())[0]
    with xr.open_dataset(sample_path, decode_times=True) as ds:
        sample_var = list(ds.data_vars)[0]
        if 'time' in ds.dims:
            times = pd.to_datetime(ds['time'].values)
            target_date = pd.Timestamp(date_str)
            date_mask = times.date == target_date.date()
            shape = ds[sample_var].isel(time=date_mask).shape
            n_times = sum(date_mask)
        else:
            shape = ds[sample_var].shape
            n_times = 1
        
        lat_dim = ds.dims.get('lat', ds.dims.get('latitude', shape[-2]))
        lon_dim = ds.dims.get('lon', ds.dims.get('longitude', shape[-1]))
    
    # Load all variables for this date
    data = np.zeros((len(VARIABLES), n_times, lat_dim, lon_dim), dtype=np.float32)
    
    for c, var in enumerate(VARIABLES):
        file_path = variable_paths.get(var)
        if not file_path:
            continue
        
        try:
            with xr.open_dataset(file_path, decode_times=True) as ds:
                if var not in ds.data_vars:
                    continue
                
                if 'time' in ds.dims:
                    times = pd.to_datetime(ds['time'].values)
                    target_date = pd.Timestamp(date_str)
                    date_mask = times.date == target_date.date()
                    var_data = ds[var].isel(time=date_mask).values
                else:
                    var_data = ds[var].values[np.newaxis, ...]
                
                # Flip vertically (latitude axis) - NC files have south-to-north order
                var_data = np.flip(var_data, axis=-2)
                
                data[c] = var_data
        except Exception:
            pass
    
    # Scale each time step
    scaled_data = np.zeros_like(data)
    for t in range(n_times):
        scaled_data[:, t, :, :] = scale_channels(data[:, t, :, :], global_stats)
    
    # Prepare metadata
    meta = {
        'date': date_str,
        'source': row_dict.get('source'),
        'source_id': row_dict.get('source_id'),
        'driving_source_id': row_dict.get('driving_source_id'),
        'driving_experiment_id': row_dict.get('driving_experiment_id'),
    }
    
    return (date_str, meta, scaled_data)


META_KEYS = [
    'month_sin',
    'month_cos',
    'day_sin',
    'day_cos',
    'dow_sin',
    'dow_cos'
]


def _extract_meta_array(meta_dict: Dict) -> np.ndarray:
    """
    Extract meta features in fixed order as float32 array.
    
    Order: month_sin, month_cos, day_sin, day_cos, dow_sin, dow_cos (6 cyclic features)
    """
    return np.array([float(meta_dict.get(k, 0.0)) for k in META_KEYS], dtype=np.float32)


def _trim_to_divisible_by_8(data: np.ndarray) -> np.ndarray:
    """
    Trim spatial dimensions to be divisible by 8 from bottom and right.
    
    Args:
        data: Array with shape (..., H, W)
    
    Returns:
        Trimmed array with H and W divisible by 8
    """
    H, W = data.shape[-2], data.shape[-1]
    new_H = H - (H % 8)
    new_W = W - (W % 8)
    return data[..., :new_H, :new_W]


def _process_single_sample(args: Tuple) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Process a single sample: load data, scale, extract meta.
    
    Args:
        args: Tuple of (idx, row_dict, global_stats)
    
    Returns:
        Tuple of (idx, scaled_data, meta_array) or (idx, None, None) on failure
    """
    idx, row_dict, global_stats = args
    
    try:
        variable_paths = _parse_dict_field(row_dict['variable_paths'])
        meta_dict = _parse_dict_field(row_dict['meta'])
        date_str = str(row_dict['date'])
        
        if not variable_paths:
            return (idx, None, None)
        
        # Get first file to determine dimensions
        sample_path = list(variable_paths.values())[0]
        with xr.open_dataset(sample_path, decode_times=True) as ds:
            target_date = pd.Timestamp(date_str)
            
            if 'time' in ds.dims:
                times = pd.to_datetime(ds['time'].values)
                date_mask = times.date == target_date.date()
                n_times = int(sum(date_mask))
                if n_times == 0:
                    return (idx, None, None)
            else:
                n_times = 1
            
            lat_dim = ds.sizes.get('lat', ds.sizes.get('latitude', 1018))
            lon_dim = ds.sizes.get('lon', ds.sizes.get('longitude', 1298))
        
        # Load all variables for this date - shape (C, H, W)
        # We take the mean across time steps if multiple exist
        data = np.zeros((len(VARIABLES), lat_dim, lon_dim), dtype=np.float32)
        
        for c, var in enumerate(VARIABLES):
            file_path = variable_paths.get(var)
            if not file_path:
                continue
            
            with xr.open_dataset(file_path, decode_times=True) as ds:
                if var not in ds.data_vars:
                    continue
                
                if 'time' in ds.dims:
                    times = pd.to_datetime(ds['time'].values)
                    date_mask = times.date == target_date.date()
                    var_data = ds[var].isel(time=date_mask).values
                    # Average across time steps for daily mean
                    var_data = np.nanmean(var_data, axis=0)
                else:
                    var_data = ds[var].values
                
                # Flip vertically (latitude axis) - NC files have south-to-north order
                var_data = np.flip(var_data, axis=-2)
                
                # Handle NaNs
                var_data = np.nan_to_num(var_data, nan=0.0).astype(np.float32)
                data[c] = var_data
        
        # Scale all channels
        scaled_data = scale_channels(data, global_stats)  # (C, H, W)
        
        # Trim spatial dims to be divisible by 8
        scaled_data = _trim_to_divisible_by_8(scaled_data)
        
        # Extract meta as float32 array
        meta_array = _extract_meta_array(meta_dict)
        
        return (idx, scaled_data, meta_array)
        
    except Exception as e:
        return (idx, None, None)


def create_scaled_datasets(
    df: pd.DataFrame,
    global_stats: Dict,
    output_dir: str = '/g/data/x77/ha2606/barra',
    max_workers: int = None,
    samples_per_shard: int = 500
) -> Dict[str, Path]:
    """
    Create scaled train/test/val datasets from DataFrame using sharded storage.
    
    For large datasets (500GB+), data is saved in shards to avoid memory issues.
    Each shard contains a fixed number of samples with corresponding metadata.
    A manifest file tracks all shards and maintains sample ordering.
    
    Output structure:
        output_dir/
        ├── scaling_config.json
        ├── train/
        │   ├── manifest.json      # Tracks shards and sample indices
        │   ├── shard_0000_data.npy  # (samples_per_shard, 6, H, W)
        │   ├── shard_0000_meta.npy  # (samples_per_shard, 9)
        │   ├── shard_0001_data.npy
        │   └── ...
        ├── test/
        └── val/
    
    Args:
        df: DataFrame with 'split', 'variable_paths', 'date', 'meta' columns
        global_stats: Dict with global min/max/skew per variable
        output_dir: Directory to save output files
        max_workers: Number of parallel workers
        samples_per_shard: Number of samples per shard file (default 500)
    
    Returns:
        Dict mapping split name to output directory paths
    """
    import json
    
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 64)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save global stats and scaling config
    config = {
        'variables': list(VARIABLES),
        'global_stats': global_stats,
        'meta_order': META_KEYS,
        'samples_per_shard': samples_per_shard,
        'scaling_info': {
            'tas': 'min-max',
            'ts': 'min-max', 
            'hurs': 'min-max',
            'pr': 'signed_power + softsign01_nonneg',
            'sfcWind': 'signed_power + softsign01_nonneg',
            'evspsbl': 'signed_power + softsign01 (two-sided)'
        }
    }
    with open(output_path / 'scaling_config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    output_dirs = {}
    
    for split in ['train', 'test', 'val']:
        split_df = df[df['split'] == split].reset_index(drop=True)
        
        if len(split_df) == 0:
            print(f"No data for split: {split}")
            continue
        
        n_samples = len(split_df)
        n_shards = (n_samples + samples_per_shard - 1) // samples_per_shard
        
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split: {n_samples} samples in {n_shards} shards")
        print(f"{'='*60}")
        
        # Process in shard batches to control memory
        manifest = {
            'split': split,
            'total_samples': 0,
            'n_shards': 0,
            'samples_per_shard': samples_per_shard,
            'shards': []
        }
        
        global_sample_idx = 0  # Track global sample index across shards
        
        for shard_idx in range(n_shards):
            start_idx = shard_idx * samples_per_shard
            end_idx = min(start_idx + samples_per_shard, n_samples)
            shard_df = split_df.iloc[start_idx:end_idx]
            
            print(f"\nShard {shard_idx + 1}/{n_shards}: samples {start_idx}-{end_idx-1}")
            
            # Prepare tasks for this shard
            tasks = []
            for local_idx, (_, row) in enumerate(shard_df.iterrows()):
                row_dict = row.to_dict()
                tasks.append((local_idx, row_dict, global_stats))
            
            # Process shard in parallel
            results = {}
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_process_single_sample, task): task[0] for task in tasks}
                
                with tqdm(total=len(tasks), desc=f"Shard {shard_idx}", unit="sample") as pbar:
                    for future in as_completed(futures):
                        try:
                            idx, scaled_data, meta_array = future.result()
                            if scaled_data is not None:
                                results[idx] = (scaled_data, meta_array)
                        except Exception as e:
                            pass
                        pbar.update(1)
            
            if not results:
                print(f"  No successful samples in shard {shard_idx}")
                continue
            
            # Sort by local index to maintain sequence
            sorted_indices = sorted(results.keys())
            n_valid = len(sorted_indices)
            
            # Get dimensions from first sample
            first_data, _ = results[sorted_indices[0]]
            _, H, W = first_data.shape
            
            # Allocate shard arrays
            shard_data = np.zeros((n_valid, len(VARIABLES), H, W), dtype=np.float32)
            shard_meta = np.zeros((n_valid, len(META_KEYS)), dtype=np.float32)
            
            # Fill arrays in sequence order
            for new_idx, orig_idx in enumerate(sorted_indices):
                scaled_data, meta = results[orig_idx]
                shard_data[new_idx] = scaled_data
                shard_meta[new_idx] = meta
            
            # Save shard files
            shard_data_file = split_dir / f'shard_{shard_idx:04d}_data.npy'
            shard_meta_file = split_dir / f'shard_{shard_idx:04d}_meta.npy'
            
            np.save(shard_data_file, shard_data)
            np.save(shard_meta_file, shard_meta)
            
            # Update manifest
            shard_info = {
                'shard_idx': shard_idx,
                'data_file': shard_data_file.name,
                'meta_file': shard_meta_file.name,
                'n_samples': n_valid,
                'global_start_idx': global_sample_idx,
                'global_end_idx': global_sample_idx + n_valid,
                'shape': list(shard_data.shape)
            }
            manifest['shards'].append(shard_info)
            manifest['total_samples'] += n_valid
            manifest['n_shards'] += 1
            
            global_sample_idx += n_valid
            
            print(f"  Saved shard {shard_idx}: {n_valid} samples, shape {shard_data.shape}")
            
            # Free memory
            del shard_data, shard_meta, results
        
        # Save manifest
        manifest_file = split_dir / 'manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        output_dirs[split] = split_dir
        print(f"\n{split.upper()} complete: {manifest['total_samples']} samples in {manifest['n_shards']} shards")
    
    print(f"\n{'='*60}")
    print(f"All data saved to {output_path}")
    print(f"{'='*60}")
    
    return output_dirs


def load_shard(split_dir: Union[str, Path], shard_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single shard from a split directory.
    
    Args:
        split_dir: Path to split directory (e.g., output_dir/train)
        shard_idx: Index of shard to load
    
    Returns:
        Tuple of (data_array, meta_array)
    """
    split_path = Path(split_dir)
    data = np.load(split_path / f'shard_{shard_idx:04d}_data.npy')
    meta = np.load(split_path / f'shard_{shard_idx:04d}_meta.npy')
    return data, meta


def load_manifest(split_dir: Union[str, Path]) -> Dict:
    """
    Load manifest for a split directory.
    
    Args:
        split_dir: Path to split directory
    
    Returns:
        Manifest dict with shard info
    """
    import json
    manifest_path = Path(split_dir) / 'manifest.json'
    with open(manifest_path, 'r') as f:
        return json.load(f)


def get_sample_by_index(split_dir: Union[str, Path], global_idx: int, manifest: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a single sample by its global index.
    
    Args:
        split_dir: Path to split directory
        global_idx: Global sample index
        manifest: Optional pre-loaded manifest
    
    Returns:
        Tuple of (data, meta) for single sample
    """
    split_path = Path(split_dir)
    
    if manifest is None:
        manifest = load_manifest(split_path)
    
    # Find which shard contains this index
    for shard_info in manifest['shards']:
        if shard_info['global_start_idx'] <= global_idx < shard_info['global_end_idx']:
            local_idx = global_idx - shard_info['global_start_idx']
            data, meta = load_shard(split_path, shard_info['shard_idx'])
            return data[local_idx], meta[local_idx]
    
    raise IndexError(f"Global index {global_idx} not found in manifest")


class ShardedDataLoader:
    """
    Memory-efficient data loader for sharded datasets.
    Loads shards on-demand and provides iteration/indexing.
    """
    
    def __init__(self, split_dir: Union[str, Path]):
        self.split_dir = Path(split_dir)
        self.manifest = load_manifest(self.split_dir)
        self.total_samples = self.manifest['total_samples']
        self.n_shards = self.manifest['n_shards']
        self._current_shard_idx = -1
        self._current_data = None
        self._current_meta = None
    
    def __len__(self):
        return self.total_samples
    
    def _load_shard(self, shard_idx: int):
        """Load a shard into memory."""
        if shard_idx != self._current_shard_idx:
            self._current_data, self._current_meta = load_shard(self.split_dir, shard_idx)
            self._current_shard_idx = shard_idx
    
    def _find_shard_for_idx(self, global_idx: int) -> Tuple[int, int]:
        """Find shard index and local index for a global index."""
        for shard_info in self.manifest['shards']:
            if shard_info['global_start_idx'] <= global_idx < shard_info['global_end_idx']:
                local_idx = global_idx - shard_info['global_start_idx']
                return shard_info['shard_idx'], local_idx
        raise IndexError(f"Index {global_idx} out of range")
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get sample by index."""
        if idx < 0:
            idx = self.total_samples + idx
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples})")
        
        shard_idx, local_idx = self._find_shard_for_idx(idx)
        self._load_shard(shard_idx)
        return self._current_data[local_idx], self._current_meta[local_idx]
    
    def iter_shards(self):
        """Iterate over shards, yielding (shard_data, shard_meta) tuples."""
        for shard_info in self.manifest['shards']:
            yield load_shard(self.split_dir, shard_info['shard_idx'])
    
    def iter_samples(self):
        """Iterate over all samples in order."""
        for data, meta in self.iter_shards():
            for i in range(len(data)):
                yield data[i], meta[i]


def load_scaling_config(output_dir: Union[str, Path]) -> Dict:
    """
    Load scaling configuration from output directory.
    
    Args:
        output_dir: Path to scaled data directory
    
    Returns:
        Dict with variables, global_stats, meta_order, and scaling_info
    """
    import json
    config_path = Path(output_dir) / 'scaling_config.json'
    with open(config_path, 'r') as f:
        return json.load(f)


def inverse_scale_sample(
    scaled_data: np.ndarray,
    global_stats: Dict
) -> np.ndarray:
    """
    Apply inverse scaling to recover original values.
    
    Args:
        scaled_data: Scaled array with shape (C, H, W) or (N, C, H, W)
        global_stats: Dict with global stats per variable
    
    Returns:
        Unscaled array in original units
    """
    if scaled_data.ndim == 3:
        # (C, H, W)
        return scale_channels(scaled_data, global_stats, inverse=True)
    elif scaled_data.ndim == 4:
        # (N, C, H, W)
        out = np.zeros_like(scaled_data)
        for n in range(scaled_data.shape[0]):
            out[n] = scale_channels(scaled_data[n], global_stats, inverse=True)
        return out
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {scaled_data.shape}")
