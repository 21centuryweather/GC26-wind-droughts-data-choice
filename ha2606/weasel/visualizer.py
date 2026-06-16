"""Visualization utilities for frequency distribution plots."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from tqdm import tqdm
import xarray as xr
import json
import os

from weasel.constants import VARIABLES


def extract_data_from_nc(args: Tuple[str, str, str]) -> Tuple[str, str, np.ndarray]:
    """
    Extract flattened data values from a NetCDF file for a specific variable and date.
    
    Args:
        args: Tuple of (file_path, variable_id, date_str)
    
    Returns:
        Tuple of (variable_id, date_str, flattened_values)
    """
    file_path, variable_id, date_str = args
    
    try:
        with xr.open_dataset(file_path, decode_times=True) as ds:
            if variable_id not in ds.data_vars:
                return (variable_id, date_str, np.array([]))
            
            target_date = pd.Timestamp(date_str)
            
            if 'time' in ds.dims:
                times = pd.to_datetime(ds['time'].values)
                date_mask = times.date == target_date.date()
                
                if not any(date_mask):
                    return (variable_id, date_str, np.array([]))
                
                data_slice = ds[variable_id].isel(time=date_mask).values
            else:
                data_slice = ds[variable_id].values
            
            # Flatten and remove NaNs
            flat = data_slice.flatten()
            flat = flat[~np.isnan(flat)]
            
            return (variable_id, date_str, flat)
            
    except Exception as e:
        return (variable_id, date_str, np.array([]))


def compute_histogram_from_nc(args: Tuple[str, str, str, np.ndarray]) -> Tuple[str, np.ndarray, int]:
    """
    Extract data from NetCDF and compute histogram counts for pre-defined bins.
    Memory efficient - only returns counts, not raw data.
    
    Args:
        args: Tuple of (file_path, variable_id, date_str, bin_edges)
    
    Returns:
        Tuple of (group_key, histogram_counts, total_points)
    """
    file_path, variable_id, date_str, bin_edges = args
    
    try:
        with xr.open_dataset(file_path, decode_times=True) as ds:
            if variable_id not in ds.data_vars:
                return (None, np.zeros(len(bin_edges) - 1), 0)
            
            target_date = pd.Timestamp(date_str)
            
            if 'time' in ds.dims:
                times = pd.to_datetime(ds['time'].values)
                date_mask = times.date == target_date.date()
                
                if not any(date_mask):
                    return (None, np.zeros(len(bin_edges) - 1), 0)
                
                data_slice = ds[variable_id].isel(time=date_mask).values
            else:
                data_slice = ds[variable_id].values
            
            # Flatten and remove NaNs
            flat = data_slice.flatten()
            flat = flat[~np.isnan(flat)]
            
            if len(flat) == 0:
                return (None, np.zeros(len(bin_edges) - 1), 0)
            
            # Compute histogram counts for this file
            counts, _ = np.histogram(flat, bins=bin_edges)
            
            return (variable_id, counts, len(flat))
            
    except Exception as e:
        return (None, np.zeros(len(bin_edges) - 1) if bin_edges is not None else np.array([]), 0)


def compute_streaming_histogram(
    df: pd.DataFrame,
    variable: str,
    global_stats: Dict,
    nbins: int = 100,
    max_workers: int = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """
    Compute histogram by streaming through files - memory efficient.
    Accumulates counts per bin without storing raw data.
    
    Args:
        df: DataFrame with 'variable_paths' and 'date' columns
        variable: Variable name to extract
        global_stats: Dict with global min/max for bin edges
        nbins: Number of histogram bins
        max_workers: Number of parallel workers
    
    Returns:
        Dict mapping group label to (bin_edges, counts, total_points)
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 64)
    
    # Get global min/max for this variable to define bin edges
    var_stats = global_stats.get(variable, {})
    var_min = var_stats.get('_min')
    var_max = var_stats.get('_max')
    
    if var_min is None or var_max is None:
        raise ValueError(f"Global stats missing for {variable}")
    
    # Add small padding to capture edge values
    padding = (var_max - var_min) * 0.01
    bin_edges = np.linspace(var_min - padding, var_max + padding, nbins + 1)
    
    group_cols = ['source_id', 'driving_source_id', 'driving_experiment_id']
    
    # Prepare tasks with bin edges
    tasks = []
    task_to_group = {}
    
    for keys, group in df.groupby(group_cols):
        for _, row in group.iterrows():
            var_paths = row.get('variable_paths')
            date_str = row.get('date')
            
            if not var_paths or not isinstance(var_paths, dict):
                continue
            
            file_path = var_paths.get(variable)
            if file_path and date_str:
                task = (file_path, variable, date_str, bin_edges)
                tasks.append(task)
                task_to_group[id(task)] = keys
                # Store task id mapping since tuples with arrays aren't hashable
                task_to_group[(file_path, date_str)] = keys
    
    if not tasks:
        return {}
    
    # Initialize accumulators per group
    group_keys_set = set(df.groupby(group_cols).groups.keys())
    accumulators = {
        keys: {'counts': np.zeros(nbins, dtype=np.int64), 'total': 0}
        for keys in group_keys_set
    }
    
    # Process in parallel - each worker returns histogram counts
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_histogram_from_nc, task): (task[0], task[2])
            for task in tasks
        }
        
        with tqdm(total=len(tasks), desc=f"Processing {variable}", unit="file") as pbar:
            for future in as_completed(futures):
                task_key = futures[future]
                group_keys = task_to_group.get(task_key)
                
                try:
                    _, counts, n_points = future.result()
                    if group_keys and n_points > 0:
                        accumulators[group_keys]['counts'] += counts.astype(np.int64)
                        accumulators[group_keys]['total'] += n_points
                except Exception:
                    pass
                
                pbar.update(1)
    
    # Format results
    final_results = {}
    for keys, acc in accumulators.items():
        if acc['total'] > 0:
            label = ' | '.join(str(k) for k in keys)
            final_results[label] = (bin_edges, acc['counts'], acc['total'])
    
    return final_results


def extract_all_values_parallel(
    df: pd.DataFrame,
    variable: str,
    max_workers: int = None,
    sample_fraction: float = 0.1
) -> Dict[str, np.ndarray]:
    """
    Extract all data values for a variable from NetCDF files using multiprocessing.
    Groups by source_id/driving_source_id/driving_experiment_id.
    WARNING: Memory intensive - use compute_streaming_histogram for large datasets.
    
    Args:
        df: DataFrame with 'variable_paths' and 'date' columns
        variable: Variable name to extract
        max_workers: Number of parallel workers
        sample_fraction: Fraction of spatial points to sample (for memory efficiency)
    
    Returns:
        Dict mapping group label to sampled values array
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 32)
    
    group_cols = ['source_id', 'driving_source_id', 'driving_experiment_id']
    results = {label: [] for label in df.groupby(group_cols).groups.keys()}
    
    # Prepare tasks
    tasks = []
    task_to_group = {}
    
    for keys, group in df.groupby(group_cols):
        for _, row in group.iterrows():
            var_paths = row.get('variable_paths')
            date_str = row.get('date')
            
            if not var_paths or not isinstance(var_paths, dict):
                continue
            
            file_path = var_paths.get(variable)
            if file_path and date_str:
                task = (file_path, variable, date_str)
                tasks.append(task)
                task_to_group[task] = keys
    
    if not tasks:
        return {}
    
    # Process in parallel
    all_values = {keys: [] for keys in results.keys()}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_data_from_nc, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc=f"Extracting {variable}", unit="file") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                group_keys = task_to_group[task]
                
                try:
                    _, _, values = future.result()
                    if len(values) > 0:
                        # Sample for memory efficiency
                        if sample_fraction < 1.0 and len(values) > 1000:
                            n_samples = max(int(len(values) * sample_fraction), 1000)
                            idx = np.random.choice(len(values), n_samples, replace=False)
                            values = values[idx]
                        all_values[group_keys].append(values)
                except Exception:
                    pass
                
                pbar.update(1)
    
    # Combine and format results
    final_results = {}
    for keys, value_list in all_values.items():
        if value_list:
            combined = np.concatenate(value_list)
            label = ' | '.join(str(k) for k in keys)
            final_results[label] = combined
    
    return final_results


def extract_stat_values(
    df: pd.DataFrame, 
    variable: str, 
    stat_key: str = 'mean',
    group_cols: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Extract stat values for a variable grouped by categorical columns.
    Uses pre-computed stats from the stats column (fast method).
    
    Args:
        df: DataFrame with 'stats' column
        variable: Variable name (e.g., 'pr', 'tas')
        stat_key: Statistic to extract (e.g., 'mean', 'min', 'max')
        group_cols: Columns to group by
    
    Returns:
        Dict mapping group label to array of values
    """
    if group_cols is None:
        group_cols = ['source_id', 'driving_source_id', 'driving_experiment_id']
    
    results = {}
    
    for keys, group in df.groupby(group_cols):
        label = ' | '.join(str(k) for k in keys)
        values = []
        
        for _, row in group.iterrows():
            stats = row.get('stats')
            if stats and isinstance(stats, dict):
                var_stats = stats.get(variable)
                if var_stats and isinstance(var_stats, dict):
                    val = var_stats.get(stat_key)
                    if val is not None and np.isfinite(val):
                        values.append(val)
        
        if values:
            results[label] = np.array(values)
    
    return results


def create_distribution_plot(
    data_dict: Dict[str, np.ndarray],
    variable: str,
    stat_key: str = 'mean',
    nbins: int = 50,
    title_suffix: str = ''
) -> go.Figure:
    """
    Create a frequency distribution plot comparing multiple groups.
    
    Args:
        data_dict: Dict mapping group label to array of values
        variable: Variable name for title
        stat_key: Statistic name for axis label
        nbins: Number of histogram bins
        title_suffix: Additional title text
    
    Returns:
        Plotly Figure object
    """
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
    ]
    
    fig = go.Figure()
    
    for i, (label, values) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Histogram(
            x=values,
            name=label,
            nbinsx=nbins,
            opacity=0.7,
            marker_color=color,
            histnorm='probability density'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'{variable.upper()} - {stat_key.capitalize()} Distribution{title_suffix}',
            font=dict(size=20, family='Arial Black')
        ),
        xaxis_title=dict(
            text=f'{stat_key.capitalize()} Value',
            font=dict(size=14, family='Arial')
        ),
        yaxis_title=dict(
            text='Density',
            font=dict(size=14, family='Arial')
        ),
        barmode='overlay',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=11)
        ),
        width=900,
        height=600,
        margin=dict(l=80, r=40, t=100, b=80)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='Gray'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )
    
    return fig


def create_all_distribution_plots(
    df: pd.DataFrame,
    stat_key: str = 'mean',
    nbins: int = 50,
    save_dir: Optional[str] = None,
    max_workers: Optional[int] = None
) -> Dict[str, go.Figure]:
    """
    Create frequency distribution plots for all variables using pre-computed stats.
    Fast method - uses stats column.
    
    Args:
        df: DataFrame with 'stats' column
        stat_key: Statistic to plot (e.g., 'mean', 'min', 'max')
        nbins: Number of histogram bins
        save_dir: Directory to save figures (optional)
        max_workers: Number of parallel workers
    
    Returns:
        Dict mapping variable name to Figure
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(VARIABLES))
    
    figures = {}
    
    def process_variable(variable: str) -> Tuple[str, go.Figure]:
        data_dict = extract_stat_values(df, variable, stat_key)
        fig = create_distribution_plot(data_dict, variable, stat_key, nbins)
        return variable, fig
    
    with tqdm(total=len(VARIABLES), desc="Creating plots") as pbar:
        for var in VARIABLES:
            var_name, fig = process_variable(var)
            figures[var_name] = fig
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.write_html(f"{save_dir}/{var_name}_{stat_key}_distribution.html")
                fig.write_image(f"{save_dir}/{var_name}_{stat_key}_distribution.png", scale=2)
            
            pbar.update(1)
    
    return figures


def create_histogram_plot_from_counts(
    hist_data: Dict[str, Tuple[np.ndarray, np.ndarray, int]],
    variable: str,
    title_suffix: str = ''
) -> go.Figure:
    """
    Create a distribution plot from pre-computed histogram counts.
    
    Args:
        hist_data: Dict mapping label to (bin_edges, counts, total_points)
        variable: Variable name for title
        title_suffix: Additional title text
    
    Returns:
        Plotly Figure object
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = go.Figure()
    
    for i, (label, (bin_edges, counts, total)) in enumerate(hist_data.items()):
        color = colors[i % len(colors)]
        
        # Compute bin centers and normalize to density
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        density = counts / (total * bin_width) if total > 0 else counts
        
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=density,
            name=f"{label} (n={total:,})",
            opacity=0.7,
            marker_color=color,
            width=bin_width * 0.9
        ))
    
    fig.update_layout(
        title=dict(
            text=f'{variable.upper()} - Full Distribution{title_suffix}',
            font=dict(size=20, family='Arial Black')
        ),
        xaxis_title=dict(
            text='Value',
            font=dict(size=14, family='Arial')
        ),
        yaxis_title=dict(
            text='Density',
            font=dict(size=14, family='Arial')
        ),
        barmode='overlay',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=11)
        ),
        width=900,
        height=600,
        margin=dict(l=80, r=40, t=100, b=80)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig


def create_streaming_distribution_plots(
    df: pd.DataFrame,
    global_stats: Dict,
    nbins: int = 100,
    save_dir: Optional[str] = None,
    max_workers: Optional[int] = None
) -> Dict[str, go.Figure]:
    """
    Create frequency distribution plots using streaming histogram computation.
    Memory efficient - processes all data without loading into memory.
    
    Args:
        df: DataFrame with 'variable_paths' and 'date' columns
        global_stats: Dict with global min/max per variable (from extract_global_stats)
        nbins: Number of histogram bins
        save_dir: Directory to save figures (optional)
        max_workers: Number of parallel workers
    
    Returns:
        Dict mapping variable name to Figure
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 64)
    
    figures = {}
    
    for var in VARIABLES:
        print(f"\n{'='*60}")
        print(f"Processing {var.upper()}")
        print(f"{'='*60}")
        
        # Compute streaming histogram
        hist_data = compute_streaming_histogram(
            df, var,
            global_stats=global_stats,
            nbins=nbins,
            max_workers=max_workers
        )
        
        if not hist_data:
            print(f"No data found for {var}")
            continue
        
        # Print stats
        for label, (_, counts, total) in hist_data.items():
            print(f"  {label}: {total:,} total points")
        
        # Create plot from histogram counts
        fig = create_histogram_plot_from_counts(hist_data, var)
        figures[var] = fig
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.write_html(f"{save_dir}/{var}_distribution.html")
            fig.write_image(f"{save_dir}/{var}_distribution.png", scale=2)
            print(f"  Saved to {save_dir}/{var}_distribution.png")
    
    return figures


def create_combined_streaming_figure(
    df: pd.DataFrame,
    global_stats: Dict,
    nbins: int = 100,
    save_path: Optional[str] = None,
    max_workers: Optional[int] = None
) -> go.Figure:
    """
    Create a single figure with subplots for all 6 variables using streaming histograms.
    Publication-ready format, memory efficient.
    
    Args:
        df: DataFrame with 'variable_paths' and 'date' columns
        global_stats: Dict with global min/max per variable
        nbins: Number of histogram bins
        save_path: Path to save figure (optional)
        max_workers: Number of parallel workers
    
    Returns:
        Combined Plotly Figure
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 64)
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[v.upper() for v in VARIABLES],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    legend_added = set()
    
    for idx, variable in enumerate(VARIABLES):
        print(f"\nProcessing {variable}...")
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        hist_data = compute_streaming_histogram(
            df, variable,
            global_stats=global_stats,
            nbins=nbins,
            max_workers=max_workers
        )
        
        for i, (label, (bin_edges, counts, total)) in enumerate(hist_data.items()):
            color = colors[i % len(colors)]
            show_legend = label not in legend_added
            
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            density = counts / (total * bin_width) if total > 0 else counts
            
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=density,
                    name=label,
                    opacity=0.7,
                    marker_color=color,
                    width=bin_width * 0.9,
                    showlegend=show_legend,
                    legendgroup=label
                ),
                row=row, col=col
            )
            
            legend_added.add(label)
    
    fig.update_layout(
        title=dict(
            text='Variable Distribution Comparison (Full Data)',
            font=dict(size=24, family='Arial Black'),
            x=0.5
        ),
        barmode='overlay',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        width=1400,
        height=900,
        margin=dict(l=60, r=40, t=100, b=120)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    if save_path:
        fig.write_html(save_path.replace('.png', '.html'))
        fig.write_image(save_path, scale=2)
        print(f"\nSaved to {save_path}")
    
    return fig


def create_raw_distribution_plots(
    df: pd.DataFrame,
    nbins: int = 100,
    save_dir: Optional[str] = None,
    max_workers: Optional[int] = None,
    sample_fraction: float = 0.05
) -> Dict[str, go.Figure]:
    """
    Create frequency distribution plots from actual NetCDF data.
    Uses multiprocessing to extract values from files.
    WARNING: Memory intensive - use create_streaming_distribution_plots instead.
    
    Args:
        df: DataFrame with 'variable_paths' and 'date' columns
        nbins: Number of histogram bins
        save_dir: Directory to save figures (optional)
        max_workers: Number of parallel workers for NC extraction
        sample_fraction: Fraction of spatial points to sample per file
    
    Returns:
        Dict mapping variable name to Figure
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 64)
    
    figures = {}
    
    for var in VARIABLES:
        print(f"\n{'='*60}")
        print(f"Processing {var.upper()}")
        print(f"{'='*60}")
        
        # Extract actual values from NetCDF files
        data_dict = extract_all_values_parallel(
            df, var, 
            max_workers=max_workers,
            sample_fraction=sample_fraction
        )
        
        if not data_dict:
            print(f"No data found for {var}")
            continue
        
        # Print stats
        for label, values in data_dict.items():
            print(f"  {label}: {len(values):,} values")
        
        # Create plot
        fig = create_distribution_plot(
            data_dict, var, 
            stat_key='raw values',
            nbins=nbins
        )
        figures[var] = fig
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.write_html(f"{save_dir}/{var}_raw_distribution.html")
            fig.write_image(f"{save_dir}/{var}_raw_distribution.png", scale=2)
            print(f"  Saved to {save_dir}/{var}_raw_distribution.png")
    
    return figures


def create_combined_raw_distribution_figure(
    df: pd.DataFrame,
    nbins: int = 100,
    save_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    sample_fraction: float = 0.05
) -> go.Figure:
    """
    Create a single figure with subplots for all 6 variables using actual NetCDF data.
    Publication-ready format with multiprocessing extraction.
    
    Args:
        df: DataFrame with 'variable_paths' and 'date' columns
        nbins: Number of histogram bins
        save_path: Path to save figure (optional)
        max_workers: Number of parallel workers
        sample_fraction: Fraction of spatial points to sample
    
    Returns:
        Combined Plotly Figure
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 64)
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[v.upper() for v in VARIABLES],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    legend_added = set()
    
    for idx, variable in enumerate(VARIABLES):
        print(f"\nExtracting {variable}...")
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        data_dict = extract_all_values_parallel(
            df, variable,
            max_workers=max_workers,
            sample_fraction=sample_fraction
        )
        
        for i, (label, values) in enumerate(data_dict.items()):
            color = colors[i % len(colors)]
            show_legend = label not in legend_added
            
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=label,
                    nbinsx=nbins,
                    opacity=0.7,
                    marker_color=color,
                    histnorm='probability density',
                    showlegend=show_legend,
                    legendgroup=label
                ),
                row=row, col=col
            )
            
            legend_added.add(label)
    
    fig.update_layout(
        title=dict(
            text='Variable Distribution Comparison (Raw Data)',
            font=dict(size=24, family='Arial Black'),
            x=0.5
        ),
        barmode='overlay',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        width=1400,
        height=900,
        margin=dict(l=60, r=40, t=100, b=120)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    if save_path:
        fig.write_html(save_path.replace('.png', '.html'))
        fig.write_image(save_path, scale=2)
        print(f"\nSaved to {save_path}")
    
    return fig


def create_combined_distribution_figure(
    df: pd.DataFrame,
    stat_key: str = 'mean',
    nbins: int = 50,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create a single figure with subplots for all 6 variables.
    Publication-ready format.
    
    Args:
        df: DataFrame with 'stats' column
        stat_key: Statistic to plot
        nbins: Number of histogram bins
        save_path: Path to save figure (optional)
    
    Returns:
        Combined Plotly Figure
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[v.upper() for v in VARIABLES],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    legend_added = set()
    
    for idx, variable in enumerate(VARIABLES):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        data_dict = extract_stat_values(df, variable, stat_key)
        
        for i, (label, values) in enumerate(data_dict.items()):
            color = colors[i % len(colors)]
            show_legend = label not in legend_added
            
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=label,
                    nbinsx=nbins,
                    opacity=0.7,
                    marker_color=color,
                    histnorm='probability density',
                    showlegend=show_legend,
                    legendgroup=label
                ),
                row=row, col=col
            )
            
            legend_added.add(label)
    
    fig.update_layout(
        title=dict(
            text=f'Variable Distribution Comparison ({stat_key.capitalize()})',
            font=dict(size=24, family='Arial Black'),
            x=0.5
        ),
        barmode='overlay',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        width=1400,
        height=900,
        margin=dict(l=60, r=40, t=100, b=120)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    if save_path:
        fig.write_html(save_path.replace('.png', '.html'))
        fig.write_image(save_path, scale=2)
    
    return fig


def _process_shard_minmax(args: Tuple[str, int]) -> Tuple[List[float], List[float]]:
    """Process a single shard to get min/max per channel."""
    shard_path, shard_idx = args
    data = np.load(shard_path, mmap_mode='r')
    
    mins = [np.inf] * 6
    maxs = [-np.inf] * 6
    
    for i in range(data.shape[0]):
        sample = np.array(data[i])
        for c in range(6):
            channel = sample[c].ravel()
            valid = channel[~np.isnan(channel)]
            if len(valid) > 0:
                mins[c] = min(mins[c], float(valid.min()))
                maxs[c] = max(maxs[c], float(valid.max()))
        del sample
    del data
    return mins, maxs


def _process_shard_histogram(args: Tuple[str, str, int, List[np.ndarray], int]) -> Dict:
    """Process a single shard to compute histogram counts."""
    shard_path, meta_path, shard_idx, bin_edges_list, nbins = args
    
    data = np.load(shard_path, mmap_mode='r')
    meta = np.load(meta_path, mmap_mode='r')
    
    group_results = {}
    
    for i in range(data.shape[0]):
        group_key = (int(meta[i, 0]), int(meta[i, 1]), int(meta[i, 2]))
        
        if group_key not in group_results:
            group_results[group_key] = {
                c: {'counts': np.zeros(nbins, dtype=np.int64), 'total': 0}
                for c in range(6)
            }
        
        sample = np.array(data[i])
        
        for c in range(6):
            channel = sample[c].ravel()
            valid = channel[~np.isnan(channel)]
            if len(valid) > 0:
                counts, _ = np.histogram(valid, bins=bin_edges_list[c])
                group_results[group_key][c]['counts'] += counts
                group_results[group_key][c]['total'] += len(valid)
        del sample
    
    del data, meta
    return group_results


def compute_scaled_histogram_from_shards(
    split_dir: Union[str, Path],
    nbins: int = 100,
    max_workers: int = None
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, int]]]:
    """
    Streaming histogram from scaled sharded data with multiprocessing.
    Two-pass approach: (1) get min/max edges, (2) accumulate histogram counts.
    Groups by source/driving/experiment. Minimal memory usage.
    
    Args:
        split_dir: Path to split directory (e.g., output_dir/train)
        nbins: Number of histogram bins
        max_workers: Number of parallel workers (default: CPU count)
    
    Returns:
        Dict mapping variable name to Dict[group_label -> (bin_edges, counts, total)]
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 32)
    
    split_path = Path(split_dir)
    manifest_path = split_path / 'manifest.json'
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    n_shards = manifest['n_shards']
    
    # ========== PASS 1: Parallel min/max scan ==========
    print(f"Pass 1: Parallel min/max ({n_shards} shards, {max_workers} workers)...")
    channel_mins = [np.inf] * 6
    channel_maxs = [-np.inf] * 6
    
    tasks = [
        (str(split_path / f'shard_{i:04d}_data.npy'), i)
        for i in range(n_shards)
    ]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_shard_minmax, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=n_shards, desc="Min/Max scan"):
            mins, maxs = future.result()
            for c in range(6):
                channel_mins[c] = min(channel_mins[c], mins[c])
                channel_maxs[c] = max(channel_maxs[c], maxs[c])
    
    # Create bin edges
    print("Creating bin edges...")
    bin_edges_per_channel = []
    for c in range(6):
        padding = (channel_maxs[c] - channel_mins[c]) * 0.01
        edges = np.linspace(channel_mins[c] - padding, channel_maxs[c] + padding, nbins + 1)
        bin_edges_per_channel.append(edges)
        print(f"  {VARIABLES[c]}: [{channel_mins[c]:.4f}, {channel_maxs[c]:.4f}]")
    
    # ========== PASS 2: Parallel histogram accumulation ==========
    print(f"Pass 2: Parallel histogram counts ({n_shards} shards, {max_workers} workers)...")
    
    tasks = [
        (
            str(split_path / f'shard_{i:04d}_data.npy'),
            str(split_path / f'shard_{i:04d}_meta.npy'),
            i,
            bin_edges_per_channel,
            nbins
        )
        for i in range(n_shards)
    ]
    
    group_results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_shard_histogram, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=n_shards, desc="Histogram"):
            shard_results = future.result()
            
            # Merge shard results into global results
            for group_key, channels in shard_results.items():
                if group_key not in group_results:
                    group_results[group_key] = {
                        c: {'counts': np.zeros(nbins, dtype=np.int64), 'total': 0}
                        for c in range(6)
                    }
                
                for c in range(6):
                    group_results[group_key][c]['counts'] += channels[c]['counts']
                    group_results[group_key][c]['total'] += channels[c]['total']
    
    # Format results: variable -> {group_label -> (edges, counts, total)}
    results = {var: {} for var in VARIABLES}
    
    for group_key, channels in group_results.items():
        label = f"src:{group_key[0]} | drv:{group_key[1]} | exp:{group_key[2]}"
        
        for c, var in enumerate(VARIABLES):
            results[var][label] = (
                bin_edges_per_channel[c],
                channels[c]['counts'],
                channels[c]['total']
            )
    
    print(f"Found {len(group_results)} experiment groups")
    return results


def create_scaled_distribution_plot(
    hist_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, int]]],
    variable: str,
    title_suffix: str = ' (Scaled)'
) -> go.Figure:
    """
    Create a distribution plot for a single scaled variable with group comparisons.
    
    Args:
        hist_data: Dict with variable -> {group_label -> (bin_edges, counts, total)}
        variable: Variable name to plot
        title_suffix: Additional title text
    
    Returns:
        Plotly Figure object
    """
    if variable not in hist_data:
        raise ValueError(f"Variable {variable} not found in histogram data")
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig = go.Figure()
    
    for i, (label, (bin_edges, counts, total)) in enumerate(hist_data[variable].items()):
        color = colors[i % len(colors)]
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        density = counts / (total * bin_width) if total > 0 else counts
        
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=density,
            name=f"{label} (n={total:,})",
            opacity=0.7,
            marker_color=color,
            width=bin_width * 0.9
        ))
    
    fig.update_layout(
        title=dict(
            text=f'{variable.upper()} - Distribution{title_suffix}',
            font=dict(size=20, family='Arial Black')
        ),
        xaxis_title=dict(
            text='Scaled Value',
            font=dict(size=14, family='Arial')
        ),
        yaxis_title=dict(
            text='Density',
            font=dict(size=14, family='Arial')
        ),
        barmode='overlay',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        width=900,
        height=600,
        margin=dict(l=80, r=40, t=120, b=80)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig


def create_scaled_distribution_plots(
    split_dir: Union[str, Path],
    nbins: int = 100,
    save_dir: Optional[str] = None
) -> Dict[str, go.Figure]:
    """
    Create frequency distribution plots for all 6 variables from scaled sharded data.
    
    Args:
        split_dir: Path to split directory with shards
        nbins: Number of histogram bins
        save_dir: Directory to save figures (optional)
    
    Returns:
        Dict mapping variable name to Figure
    """
    # Compute histograms from shards
    hist_data = compute_scaled_histogram_from_shards(split_dir, nbins)
    
    figures = {}
    
    for var in VARIABLES:
        print(f"Creating plot for {var}...")
        fig = create_scaled_distribution_plot(hist_data, var)
        figures[var] = fig
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.write_html(f"{save_dir}/{var}_scaled_distribution.html")
            fig.write_image(f"{save_dir}/{var}_scaled_distribution.png", scale=2)
            print(f"  Saved to {save_dir}/{var}_scaled_distribution.png")
    
    return figures


def create_combined_scaled_distribution_figure(
    split_dir: Union[str, Path],
    nbins: int = 100,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create a single figure with subplots for all 6 scaled variables.
    Shows all experiment groups in each subplot. Publication-ready format.
    
    Args:
        split_dir: Path to split directory with shards
        nbins: Number of histogram bins
        save_path: Path to save figure (optional)
    
    Returns:
        Combined Plotly Figure
    """
    # Compute histograms from shards (grouped by experiment)
    hist_data = compute_scaled_histogram_from_shards(split_dir, nbins)
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[v.upper() for v in VARIABLES],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    legend_added = set()
    
    for idx, variable in enumerate(VARIABLES):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        for i, (label, (bin_edges, counts, total)) in enumerate(hist_data[variable].items()):
            color = colors[i % len(colors)]
            show_legend = label not in legend_added
            
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            density = counts / (total * bin_width) if total > 0 else counts
            
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=density,
                    name=label,
                    opacity=0.7,
                    marker_color=color,
                    width=bin_width * 0.9,
                    showlegend=show_legend,
                    legendgroup=label
                ),
                row=row, col=col
            )
            legend_added.add(label)
    
    fig.update_layout(
        title=dict(
            text='Scaled Variable Distribution (All 6 Channels)',
            font=dict(size=24, family='Arial Black'),
            x=0.5
        ),
        barmode='overlay',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.12,
            xanchor='center',
            x=0.5,
            font=dict(size=9)
        ),
        width=1400,
        height=900,
        margin=dict(l=60, r=40, t=100, b=140)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path.replace('.png', '.html'))
        fig.write_image(save_path, scale=2)
        print(f"\nSaved to {save_path} and .html")
    
    return fig


def create_scaled_distribution_all_splits(
    output_dir: Union[str, Path],
    nbins: int = 100,
    save_dir: Optional[str] = None,
    max_workers: int = None
) -> Dict[str, go.Figure]:
    """
    Create frequency distribution plots for all 6 variables with experiment group comparisons.
    Uses streaming histogram computation with multiprocessing (memory-efficient).
    
    Args:
        output_dir: Base output directory containing train/test/val subdirs
        nbins: Number of histogram bins
        save_dir: Directory to save figures (optional)
        max_workers: Number of parallel workers
    
    Returns:
        Dict mapping variable name to Figure
    """
    output_path = Path(output_dir)
    
    # Use train split for distribution analysis (largest dataset)
    train_dir = output_path / 'train'
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")
    
    print("Computing histograms from train split (grouped by experiment)...")
    hist_data = compute_scaled_histogram_from_shards(train_dir, nbins, max_workers=max_workers)
    
    # Create individual plots per variable
    figures = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for var in VARIABLES:
        print(f"\nCreating plot for {var}...")
        fig = go.Figure()
        
        for i, (label, (bin_edges, counts, total)) in enumerate(hist_data[var].items()):
            color = colors[i % len(colors)]
            
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            density = counts / (total * bin_width) if total > 0 else counts
            
            fig.add_trace(go.Bar(
                x=bin_centers,
                y=density,
                name=f"{label} (n={total:,})",
                opacity=0.7,
                marker_color=color,
                width=bin_width * 0.9
            ))
        
        fig.update_layout(
            title=dict(
                text=f'{var.upper()} - Scaled Distribution by Experiment',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title=dict(
                text='Scaled Value',
                font=dict(size=14, family='Arial')
            ),
            yaxis_title=dict(
                text='Density',
                font=dict(size=14, family='Arial')
            ),
            barmode='overlay',
            template='plotly_white',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                font=dict(size=10)
            ),
            width=900,
            height=600,
            margin=dict(l=80, r=40, t=120, b=80)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        figures[var] = fig
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.write_html(f"{save_dir}/{var}_scaled_distribution.html")
            fig.write_image(f"{save_dir}/{var}_scaled_distribution.png", scale=2)
            print(f"  Saved to {save_dir}/{var}_scaled_distribution.html/png")
    
    return figures


def create_combined_scaled_all_splits_figure(
    output_dir: Union[str, Path],
    nbins: int = 100,
    save_path: Optional[str] = None,
    max_workers: int = None
) -> go.Figure:
    """
    Create a single figure with subplots for all 6 scaled variables.
    Shows experiment group comparisons. Publication-ready format.
    
    Args:
        output_dir: Base output directory containing train/test/val subdirs
        nbins: Number of histogram bins
        save_path: Path to save figure (optional)
        max_workers: Number of parallel workers
    
    Returns:
        Combined Plotly Figure
    """
    output_path = Path(output_dir)
    
    # Use train split for distribution analysis
    train_dir = output_path / 'train'
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")
    
    print("Computing histograms from train split (grouped by experiment)...")
    hist_data = compute_scaled_histogram_from_shards(train_dir, nbins, max_workers=max_workers)
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[v.upper() for v in VARIABLES],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    legend_added = set()
    
    for idx, variable in enumerate(VARIABLES):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        for i, (label, (bin_edges, counts, total)) in enumerate(hist_data[variable].items()):
            color = colors[i % len(colors)]
            show_legend = label not in legend_added
            
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            density = counts / (total * bin_width) if total > 0 else counts
            
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=density,
                    name=label,
                    opacity=0.7,
                    marker_color=color,
                    width=bin_width * 0.9,
                    showlegend=show_legend,
                    legendgroup=label
                ),
                row=row, col=col
            )
            
            legend_added.add(label)
    
    fig.update_layout(
        title=dict(
            text='Scaled Variable Distribution by Experiment',
            font=dict(size=24, family='Arial Black'),
            x=0.5
        ),
        barmode='overlay',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.12,
            xanchor='center',
            x=0.5,
            font=dict(size=12)
        ),
        width=1400,
        height=900,
        margin=dict(l=60, r=40, t=100, b=120)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path.replace('.png', '.html'))
        fig.write_image(save_path, scale=2)
        print(f"\nSaved to {save_path} and .html")
    
    return fig


def compare_raw_vs_scaled_distributions(
    df: pd.DataFrame,
    split_dir: Union[str, Path],
    global_stats: Dict,
    nbins: int = 100,
    save_path: Optional[str] = None,
    max_workers: Optional[int] = None
) -> go.Figure:
    """
    Create a side-by-side comparison of raw vs scaled distributions.
    
    Args:
        df: DataFrame with 'variable_paths' and 'date' columns (for raw data)
        split_dir: Path to split directory with scaled shards
        global_stats: Dict with global min/max per variable
        nbins: Number of histogram bins
        save_path: Path to save figure (optional)
        max_workers: Number of parallel workers for raw data extraction
    
    Returns:
        Combined comparison Figure
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 64)
    
    # Compute scaled histograms
    print("Computing scaled data histograms...")
    scaled_hist = compute_scaled_histogram_from_shards(split_dir, nbins)
    
    # Create figure with 2 columns per variable (raw | scaled)
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[
            f'{VARIABLES[0].upper()} Raw', f'{VARIABLES[0].upper()} Scaled',
            f'{VARIABLES[1].upper()} Raw', f'{VARIABLES[1].upper()} Scaled',
            f'{VARIABLES[2].upper()} Raw', f'{VARIABLES[2].upper()} Scaled',
            f'{VARIABLES[3].upper()} Raw', f'{VARIABLES[3].upper()} Scaled',
            f'{VARIABLES[4].upper()} Raw', f'{VARIABLES[4].upper()} Scaled',
            f'{VARIABLES[5].upper()} Raw', f'{VARIABLES[5].upper()} Scaled',
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.10
    )
    
    colors_raw = '#1f77b4'
    colors_scaled = '#2ca02c'
    
    for idx, variable in enumerate(VARIABLES):
        row = idx // 2 + 1
        col_raw = (idx % 2) * 2 + 1
        col_scaled = col_raw + 1
        
        # Scaled histogram
        bin_edges, counts, total = scaled_hist[variable]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        density = counts / (total * bin_width) if total > 0 else counts
        
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=density,
                name=f"{variable} Scaled",
                opacity=0.8,
                marker_color=colors_scaled,
                width=bin_width * 0.9,
                showlegend=(idx == 0)
            ),
            row=row, col=col_scaled
        )
        
        # Compute raw histogram using streaming method
        print(f"Computing raw histogram for {variable}...")
        raw_hist = compute_streaming_histogram(
            df, variable,
            global_stats=global_stats,
            nbins=nbins,
            max_workers=max_workers
        )
        
        # Combine all groups for overall raw distribution
        if raw_hist:
            all_counts = np.zeros(nbins, dtype=np.int64)
            all_total = 0
            raw_edges = None
            for label, (edges, cnts, tot) in raw_hist.items():
                all_counts += cnts.astype(np.int64)
                all_total += tot
                raw_edges = edges
            
            if raw_edges is not None and all_total > 0:
                raw_centers = (raw_edges[:-1] + raw_edges[1:]) / 2
                raw_width = raw_edges[1] - raw_edges[0]
                raw_density = all_counts / (all_total * raw_width)
                
                fig.add_trace(
                    go.Bar(
                        x=raw_centers,
                        y=raw_density,
                        name=f"{variable} Raw",
                        opacity=0.8,
                        marker_color=colors_raw,
                        width=raw_width * 0.9,
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col_raw
                )
    
    fig.update_layout(
        title=dict(
            text='Raw vs Scaled Distribution Comparison',
            font=dict(size=24, family='Arial Black'),
            x=0.5
        ),
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.10,
            xanchor='center',
            x=0.5,
            font=dict(size=11)
        ),
        width=1600,
        height=1000,
        margin=dict(l=50, r=40, t=100, b=100)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    if save_path:
        fig.write_html(save_path.replace('.png', '.html'))
        fig.write_image(save_path, scale=2)
        print(f"\nSaved to {save_path}")
    
    return fig


def plot_sample_from_splits(
    output_dir: Union[str, Path],
    sample_idx: int = 0,
    save_path: Optional[str] = None
) -> 'plt.Figure':
    """
    Plot a sample from each split (train/val/test) showing all 6 channels.
    Creates a 3x6 grid: rows=splits, cols=channels.
    
    Args:
        output_dir: Base output directory containing train/test/val subdirs
        sample_idx: Index of sample to plot from first shard (default 0)
        save_path: Path to save figure (optional)
    
    Returns:
        Matplotlib Figure with sample visualizations
    """
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    splits = ['train', 'val', 'test']
    
    # Create figure with matplotlib for better image handling
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    
    print("Loading samples from each split...")
    
    for row_idx, split in enumerate(splits):
        split_dir = output_path / split
        if not split_dir.exists():
            print(f"  {split}: directory not found, skipping")
            for col_idx in range(6):
                axes[row_idx, col_idx].text(0.5, 0.5, 'N/A', ha='center', va='center')
                axes[row_idx, col_idx].axis('off')
            continue
        
        # Load first shard
        shard_path = split_dir / 'shard_0000_data.npy'
        meta_path = split_dir / 'shard_0000_meta.npy'
        
        if not shard_path.exists():
            print(f"  {split}: shard not found, skipping")
            continue
        
        data = np.load(shard_path, mmap_mode='r')
        meta = np.load(meta_path, mmap_mode='r')
        
        # Get sample
        idx = min(sample_idx, data.shape[0] - 1)
        sample = np.array(data[idx])  # (6, H, W)
        sample_meta = meta[idx]
        
        print(f"  {split}: sample shape = {sample.shape}, meta = {sample_meta[:3]}")
        
        for col_idx, var in enumerate(VARIABLES):
            channel = sample[col_idx]
            
            # Plot with colorbar
            im = axes[row_idx, col_idx].imshow(channel, cmap='viridis', aspect='auto')
            
            # Title for top row only
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(var.upper(), fontsize=12, fontweight='bold')
            
            # Y-axis label for first column
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(f'{split.upper()}\n({sample.shape[1]}x{sample.shape[2]})', fontsize=10)
            
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            
            # Add colorbar
            plt.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046, pad=0.04)
        
        del data, meta
    
    plt.suptitle(f'Sample {sample_idx} from Each Split (All 6 Channels)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    
    return fig


def plot_static_variables(
    static_paths: List[str],
    save_path: Optional[str] = None
) -> 'plt.Figure':
    """
    Plot static variables from BARRA catalogue.
    
    Args:
        static_paths: List of paths to static variable NetCDF files
        save_path: Path to save figure (optional)
    
    Returns:
        Matplotlib Figure with static variable visualizations
    """
    import matplotlib.pyplot as plt
    
    n_vars = len(static_paths)
    if n_vars == 0:
        print("No static variable paths provided")
        return None
    
    # Calculate grid size
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_vars == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    print(f"Loading {n_vars} static variables...")
    
    for idx, path in enumerate(static_paths):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        try:
            with xr.open_dataset(path) as ds:
                # Get the first data variable
                var_names = list(ds.data_vars)
                if not var_names:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.axis('off')
                    continue
                
                var_name = var_names[0]
                data = ds[var_name].values
                
                # Handle 3D data (take first slice)
                if data.ndim > 2:
                    data = data[0] if data.shape[0] < data.shape[-1] else data
                    while data.ndim > 2:
                        data = data[0]
                
                print(f"  {var_name}: shape = {data.shape}, range = [{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]")
                
                im = ax.imshow(data, cmap='terrain', aspect='auto')
                ax.set_title(f'{var_name}\n{Path(path).name}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}', ha='center', va='center', fontsize=8)
            ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_vars, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('BARRA Static Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    
    return fig


def print_split_shapes(output_dir: Union[str, Path]) -> Dict[str, Dict]:
    """
    Print shapes and info for all splits in the output directory.
    
    Args:
        output_dir: Base output directory containing train/test/val subdirs
    
    Returns:
        Dict with shape info per split
    """
    output_path = Path(output_dir)
    splits = ['train', 'val', 'test']
    
    info = {}
    
    print("=" * 60)
    print("DATASET SHAPES AND INFO")
    print("=" * 60)
    
    for split in splits:
        split_dir = output_path / split
        manifest_path = split_dir / 'manifest.json'
        
        if not manifest_path.exists():
            print(f"\n{split.upper()}: Not found")
            continue
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Load first shard to get shape
        shard_path = split_dir / 'shard_0000_data.npy'
        if shard_path.exists():
            data = np.load(shard_path, mmap_mode='r')
            sample_shape = data.shape[1:]  # (C, H, W)
            del data
        else:
            sample_shape = "N/A"
        
        info[split] = {
            'total_samples': manifest['total_samples'],
            'n_shards': manifest['n_shards'],
            'samples_per_shard': manifest['samples_per_shard'],
            'sample_shape': sample_shape
        }
        
        print(f"\n{split.upper()}:")
        print(f"  Total samples:     {manifest['total_samples']:,}")
        print(f"  Number of shards:  {manifest['n_shards']}")
        print(f"  Samples per shard: {manifest['samples_per_shard']}")
        print(f"  Sample shape:      {sample_shape} (C, H, W)")
        print(f"  Variables:         {VARIABLES}")
    
    print("\n" + "=" * 60)
    
    return info


def create_static_maps(
    static_paths: List[str],
    output_dir: Union[str, Path],
    filename: str = 'static_maps.npy'
) -> np.ndarray:
    """
    Load static variables (topography, land sea mask), apply min-max scaling,
    and save as (2, H, W) array.
    
    Args:
        static_paths: List of paths to static variable NetCDF files
        output_dir: Directory to save the static maps
        filename: Name of the output file
    
    Returns:
        numpy array of shape (2, H, W) with scaled static maps
        index 0 = topography (orog), index 1 = land sea mask (sftlf)
    """
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find topography (orog) and land sea mask (sftlf)
    topo_data = None
    lsm_data = None
    
    print("Loading static variables...")
    
    for path in static_paths:
        try:
            with xr.open_dataset(path) as ds:
                var_names = list(ds.data_vars)
                if not var_names:
                    continue
                
                var_name = var_names[0]
                data = ds[var_name].values
                
                # Handle 3D data
                while data.ndim > 2:
                    data = data[0]
                
                # Flip vertically (latitude axis) - NC files have south-to-north order
                data = np.flip(data, axis=-2)
                
                if 'orog' in var_name.lower() or 'orog' in path.lower():
                    topo_data = data.astype(np.float32)
                    print(f"  Found topography (orog): shape={data.shape}, range=[{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
                elif 'sftlf' in var_name.lower() or 'sftlf' in path.lower():
                    lsm_data = data.astype(np.float32)
                    print(f"  Found land sea mask (sftlf): shape={data.shape}, range=[{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
                    
        except Exception as e:
            print(f"  Error loading {path}: {e}")
    
    if topo_data is None:
        raise ValueError("Topography (orog) not found in static paths")
    if lsm_data is None:
        raise ValueError("Land sea mask (sftlf) not found in static paths")
    
    # Apply min-max scaling to each
    def minmax_scale(data: np.ndarray) -> np.ndarray:
        """Apply min-max scaling to [0, 1] range."""
        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            return data
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        if max_val - min_val > 0:
            scaled = (data - min_val) / (max_val - min_val)
        else:
            scaled = np.zeros_like(data)
        return scaled.astype(np.float32)
    
    print("\nApplying min-max scaling...")
    topo_scaled = minmax_scale(topo_data)
    lsm_scaled = minmax_scale(lsm_data)
    
    print(f"  Topography scaled: range=[{np.nanmin(topo_scaled):.4f}, {np.nanmax(topo_scaled):.4f}]")
    print(f"  Land sea mask scaled: range=[{np.nanmin(lsm_scaled):.4f}, {np.nanmax(lsm_scaled):.4f}]")
    
    # Create (2, H, W) array: index 0 = topography, index 1 = land sea mask
    static_maps = np.stack([topo_scaled, lsm_scaled], axis=0)
    
    # Trim spatial dims to be divisible by 8 (from bottom and right)
    H, W = static_maps.shape[-2], static_maps.shape[-1]
    new_H = H - (H % 8)
    new_W = W - (W % 8)
    static_maps = static_maps[..., :new_H, :new_W]
    print(f"\nStatic maps shape (trimmed to div by 8): {static_maps.shape}")
    
    # Save to output directory
    save_path = output_path / filename
    np.save(save_path, static_maps)
    print(f"Saved static maps to: {save_path}")
    
    return static_maps


def plot_static_maps(
    static_maps: np.ndarray,
    save_path: Optional[str] = None
) -> 'plt.Figure':
    """
    Plot scaled static maps (topography and land sea mask).
    
    Args:
        static_maps: Array of shape (2, H, W) with index 0=topography, 1=land sea mask
        save_path: Path to save figure (optional)
    
    Returns:
        Matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    titles = ['Topography (orog)', 'Land Sea Mask (sftlf)']
    cmaps = ['terrain', 'Blues']
    
    for idx, (ax, title, cmap) in enumerate(zip(axes, titles, cmaps)):
        data = static_maps[idx]
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f'{title}\n(scaled: [{vmin:.4f}, {vmax:.4f}])', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Scaled Value', fontsize=10)
    
    plt.suptitle(f'Scaled Static Maps - Shape: {static_maps.shape}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig


def plot_sample_channels(
    output_dir: Union[str, Path],
    split: str = 'train',
    shard_idx: int = 0,
    sample_idx: int = 0,
    save_path: Optional[str] = None
) -> 'plt.Figure':
    """
    Plot a single sample from a shard showing all 6 channels with matplotlib.
    Each channel shows colorbar with actual min/max values.
    
    Args:
        output_dir: Base output directory containing train/test/val subdirs
        split: Which split to use ('train', 'val', 'test')
        shard_idx: Index of shard to load
        sample_idx: Index of sample within shard
        save_path: Path to save figure (optional)
    
    Returns:
        Matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    split_dir = output_path / split
    
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    # Load shard
    shard_path = split_dir / f'shard_{shard_idx:04d}_data.npy'
    meta_path = split_dir / f'shard_{shard_idx:04d}_meta.npy'
    
    if not shard_path.exists():
        raise FileNotFoundError(f"Shard not found: {shard_path}")
    
    print(f"Loading {split} split, shard {shard_idx}, sample {sample_idx}...")
    
    data = np.load(shard_path, mmap_mode='r')
    meta = np.load(meta_path, mmap_mode='r')
    
    # Get sample
    idx = min(sample_idx, data.shape[0] - 1)
    sample = np.array(data[idx])  # (6, H, W)
    sample_meta = meta[idx]
    
    print(f"Sample shape: {sample.shape}")
    print(f"Meta: src={int(sample_meta[0])}, drv={int(sample_meta[1])}, exp={int(sample_meta[2])}")
    
    # Create 2x3 subplot for 6 channels
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for ch_idx, (ax, var) in enumerate(zip(axes, VARIABLES)):
        channel = sample[ch_idx]
        
        # Get min/max for this channel
        valid = channel[~np.isnan(channel)]
        if len(valid) > 0:
            vmin, vmax = float(valid.min()), float(valid.max())
        else:
            vmin, vmax = 0, 1
        
        print(f"  {var}: shape={channel.shape}, range=[{vmin:.4f}, {vmax:.4f}]")
        
        # Plot with colorbar showing actual min/max
        im = ax.imshow(channel, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f'{var.upper()}\n[{vmin:.4f}, {vmax:.4f}]', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Scaled Value', fontsize=9)
    
    plt.suptitle(
        f'{split.upper()} Split - Shard {shard_idx} - Sample {idx}\n'
        f'Shape: {sample.shape} (C, H, W)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    del data, meta
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    
    return fig
