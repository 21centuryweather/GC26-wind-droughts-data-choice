"""Extract statistics from NetCDF files for each variable and date."""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import xarray as xr
from scipy import stats as scipy_stats
from tqdm import tqdm

from weasel.catalogue_downloader import RESOURCE_DIR, is_file_stale
from weasel.logger import get_logger

logger = get_logger()

DEFAULT_WORKERS = min(os.cpu_count() or 4, 20)

QUANTILES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.25, 0.50, 0.75, 0.95, 0.96, 0.97, 0.98, 0.99]

STATS_CATALOGUE_FILE = "catalogue_with_stats.csv.gz"


def compute_stats_for_array(data: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for a flattened array efficiently.
    
    Args:
        data: 1D numpy array of values (NaN values will be ignored)
    
    Returns:
        Dict with all statistics
    """
    # Remove NaN values
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return None
    
    # Compute basic stats
    stats_dict = {
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'median': float(np.median(data)),
        'skewness': float(scipy_stats.skew(data)),
    }
    
    # Compute quantiles
    quantile_values = np.quantile(data, QUANTILES)
    for q, val in zip(QUANTILES, quantile_values):
        stats_dict[f'q{q:.2f}'] = float(val)
    
    return stats_dict


def extract_stats_from_netcdf(args: Tuple[str, str, str]) -> Dict[str, Any]:
    """
    Extract statistics for a specific variable and date from a NetCDF file.
    Memory-efficient: processes one time slice at a time.
    
    Args:
        args: Tuple of (file_path, variable_id, date_str)
    
    Returns:
        Dict with path, variable_id, date, and stats dictionary
    """
    file_path, variable_id, date_str = args
    
    try:
        with xr.open_dataset(file_path, decode_times=True) as ds:
            # Find the variable
            if variable_id not in ds.data_vars:
                return {
                    'path': file_path,
                    'variable_id': variable_id,
                    'date': date_str,
                    'stats': None
                }
            
            # Select the specific date
            target_date = pd.Timestamp(date_str)
            
            if 'time' in ds.dims:
                # Find matching time index
                times = pd.to_datetime(ds['time'].values)
                date_mask = times.date == target_date.date()
                
                if not any(date_mask):
                    return {
                        'path': file_path,
                        'variable_id': variable_id,
                        'date': date_str,
                        'stats': None
                    }
                
                # Select data for this date and flatten
                data_slice = ds[variable_id].isel(time=date_mask).values
            else:
                data_slice = ds[variable_id].values
            
            # Flatten and compute stats
            flat_data = data_slice.flatten()
            stats = compute_stats_for_array(flat_data)
            
            return {
                'path': file_path,
                'variable_id': variable_id,
                'date': date_str,
                'stats': stats
            }
            
    except Exception as e:
        return {
            'path': file_path,
            'variable_id': variable_id,
            'date': date_str,
            'stats': None,
            'error': str(e)
        }


def extract_stats_concurrent(
    catalogue_path: Optional[Path] = None,
    max_workers: int = DEFAULT_WORKERS,
    force_refresh: bool = False,
) -> Path:
    """
    Extract statistics from all NetCDF files in the catalogue.
    Adds a 'stats' column with computed statistics for each row.
    
    Args:
        catalogue_path: Path to catalogue with dates. If None, uses default.
        max_workers: Number of concurrent workers for parallel processing.
        force_refresh: If True, reprocess even if cache exists.
    
    Returns:
        Path to the catalogue with stats.
    """
    output_path = RESOURCE_DIR / STATS_CATALOGUE_FILE
    
    # Check cache freshness
    if not force_refresh and not is_file_stale(output_path):
        logger.info(f"Catalogue with stats exists and is fresh: {output_path}")
        return output_path
    
    if catalogue_path is None:
        catalogue_path = RESOURCE_DIR / "catalogue_with_dates.csv.gz"
    
    logger.info(f"Loading catalogue from {catalogue_path}")
    catalogue = pd.read_csv(catalogue_path, compression='gzip')
    
    # Prepare tasks: (path, variable_id, date) for each row
    tasks = [
        (row['path'], row['variable_id'], row['date'])
        for _, row in catalogue.iterrows()
        if pd.notna(row['date'])
    ]
    
    total_tasks = len(tasks)
    logger.info(f"Extracting stats from {total_tasks} file-date combinations using {max_workers} workers...")
    
    # Results storage: keyed by (path, variable_id, date)
    results = {}
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(extract_stats_from_netcdf, task): task
            for task in tasks
        }
        
        # Process results as they complete with progress bar
        with tqdm(total=total_tasks, desc="Extracting stats", unit="row") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                key = (result['path'], result['variable_id'], result['date'])
                results[key] = result.get('stats')
                pbar.update(1)
    
    # Add stats to catalogue
    def get_stats(row):
        if pd.isna(row['date']):
            return None
        key = (row['path'], row['variable_id'], row['date'])
        return results.get(key)
    
    catalogue['stats'] = catalogue.apply(get_stats, axis=1)
    
    # Count stats
    successful = catalogue['stats'].notna().sum()
    logger.info(f"Stats extracted: {successful}/{len(catalogue)} rows")
    
    # Save result
    catalogue.to_csv(output_path, index=False, compression='gzip')
    logger.success(f"Saved catalogue with stats to {output_path}")
    
    return output_path
