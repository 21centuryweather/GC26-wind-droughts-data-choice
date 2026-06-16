"""Extract dates from NetCDF files in the filtered catalogue."""

import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, Any, List
import xarray as xr
from tqdm import tqdm

from weasel.catalogue_downloader import RESOURCE_DIR, BARRA_CATALOGUE_FILE
from weasel.logger import get_logger

logger = get_logger()

# Default to number of CPUs available
DEFAULT_WORKERS = min(os.cpu_count() or 4, 20)


def extract_metadata_from_netcdf(file_path: str) -> Dict[str, Any]:
    """
    Extract all timestamps and data dimensions from a single NetCDF file.
    Memory-efficient: only reads metadata, not data values.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        Dict with file_path, list of dates, and dimensions string.
    """
    try:
        with xr.open_dataset(file_path, decode_times=True, chunks={}) as ds:
            # Extract all timestamps
            dates = []
            if 'time' in ds.dims:
                time_values = ds['time'].values
                dates = [pd.Timestamp(t).strftime('%Y-%m-%d') for t in time_values]

            # Extract spatial dimensions only (exclude time since each row is a single date)
            data_vars = [v for v in ds.data_vars if v not in ds.coords]
            if data_vars:
                var_name = data_vars[0]
                dims = {k: v for k, v in ds[var_name].sizes.items() if k != 'time'}
                dimensions = str(dims)
            else:
                dimensions = None

            return {'path': file_path, 'dates': dates, 'dimensions': dimensions}
    except Exception as e:
        return {'path': file_path, 'dates': [], 'dimensions': None}


CATALOGUE_WITH_DATES_FILE = "catalogue_with_dates.csv.gz"


def extract_dates_concurrent(
    catalogue_path: Optional[Path] = None,
    max_workers: int = DEFAULT_WORKERS,
    force_refresh: bool = False,
) -> Path:
    """
    Extract dates and dimensions from all NetCDF files in the catalogue.
    Creates one row per timestamp with a single 'date' column.
    Caches result and reuses if less than 30 days old.

    Args:
        catalogue_path: Path to filtered catalogue. If None, uses default.
        max_workers: Number of concurrent workers for parallel processing.
        force_refresh: If True, reprocess even if cache exists.

    Returns:
        Path to the expanded catalogue file.
    """
    from weasel.catalogue_downloader import is_file_stale

    output_path = RESOURCE_DIR / CATALOGUE_WITH_DATES_FILE

    # Check cache freshness
    if not force_refresh and not is_file_stale(output_path):
        logger.info(f"Catalogue with dates exists and is fresh: {output_path}")
        return output_path

    if catalogue_path is None:
        catalogue_path = RESOURCE_DIR / BARRA_CATALOGUE_FILE

    logger.info(f"Loading catalogue from {catalogue_path}")
    catalogue = pd.read_csv(catalogue_path, compression='gzip')
    
    # Remove old date columns if present
    cols_to_drop = ['start_date', 'end_date']
    catalogue = catalogue.drop(columns=[c for c in cols_to_drop if c in catalogue.columns])
    
    file_paths = catalogue['path'].tolist()
    total_files = len(file_paths)
    logger.info(f"Extracting metadata from {total_files} NetCDF files using {max_workers} workers...")

    # Results storage
    results = {}

    # Use ProcessPoolExecutor for true parallelism across CPU cores
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(extract_metadata_from_netcdf, path): path 
            for path in file_paths
        }

        # Process results as they complete with progress bar
        with tqdm(total=total_files, desc="Extracting metadata", unit="file") as pbar:
            for future in as_completed(future_to_path):
                result = future.result()
                results[result['path']] = result
                pbar.update(1)

    # Expand catalogue: one row per timestamp
    expanded_rows = []
    for _, row in catalogue.iterrows():
        path = row['path']
        metadata = results.get(path, {'dates': [], 'dimensions': None})
        dates = metadata.get('dates', [])
        dimensions = metadata.get('dimensions')
        
        if dates:
            for date in dates:
                new_row = row.to_dict()
                new_row['date'] = date
                new_row['dimensions'] = dimensions
                expanded_rows.append(new_row)
        else:
            # Keep row even if no dates found
            new_row = row.to_dict()
            new_row['date'] = None
            new_row['dimensions'] = dimensions
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)

    # Count stats
    successful = expanded_df['date'].notna().sum()
    logger.info(f"Expanded catalogue: {len(catalogue)} files -> {len(expanded_df)} rows ({successful} with dates)")

    # Save result
    expanded_df.to_csv(output_path, index=False, compression='gzip')
    logger.success(f"Saved expanded catalogue to {output_path}")

    return output_path


def get_catalogue_with_dates(force_refresh: bool = False) -> Path:
    """
    Get the catalogue with extracted dates. Uses cached version if available.

    Args:
        force_refresh: If True, re-extract dates even if cache exists.

    Returns:
        Path to the catalogue with dates.
    """
    return extract_dates_concurrent(force_refresh=force_refresh)
