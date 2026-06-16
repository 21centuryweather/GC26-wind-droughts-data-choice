"""Group catalogue data by date with aggregated variables and paths."""

import ast
import numpy as np
import pandas as pd
from typing import Dict, Any

from weasel.constants import VARIABLES


def extract_time_features(date_str: str) -> Dict[str, float]:
    """
    Extract cyclical time features from a date string.
    
    Args:
        date_str: Date string in YYYY-MM-DD format.
    
    Returns:
        Dict with month_sin, month_cos, day_sin, day_cos, dow_sin, dow_cos.
    """
    ts = pd.Timestamp(date_str)
    month = int(ts.month)
    day = int(ts.day)
    day_of_week = int(ts.dayofweek)

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    day_sin = np.sin(2 * np.pi * day / 31)
    day_cos = np.cos(2 * np.pi * day / 31)

    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    return {
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
    }


def group_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group catalogue data by source, driving_source_id, driving_experiment_id, source_id, and date.
    Aggregates variable_id and path into a dictionary {var: path, ...} ordered by VARIABLES constant.
    Updates dimensions to include var count: {var: N, lat: X, lon: Y}.
    Adds meta column with time features and label encoded categorical columns.

    Args:
        df: DataFrame with columns: source, driving_source_id, driving_experiment_id,
            source_id, variable_id, path, date, dimensions

    Returns:
        Grouped DataFrame with variable_paths, dimensions, meta, and label encoded columns.
    """
    group_cols = ['source', 'driving_source_id', 'driving_experiment_id', 'source_id', 'date']
    
    # Create label encoders for categorical columns
    source_labels = {v: i for i, v in enumerate(sorted(df['source'].unique()))}
    driving_source_labels = {v: i for i, v in enumerate(sorted(df['driving_source_id'].unique()))}
    experiment_labels = {v: i for i, v in enumerate(sorted(df['driving_experiment_id'].unique()))}
    
    def aggregate_row(group: pd.DataFrame) -> pd.Series:
        # Create variable -> path mapping ordered by VARIABLES constant
        var_path_map = dict(zip(group['variable_id'], group['path']))
        variable_paths = {v: var_path_map[v] for v in VARIABLES if v in var_path_map}
        
        # Create variable -> stats mapping ordered by VARIABLES constant
        has_stats = 'stats' in group.columns
        if has_stats:
            def parse_stats(s):
                if pd.isna(s) or s is None:
                    return None
                if isinstance(s, dict):
                    return s
                if isinstance(s, str):
                    try:
                        return ast.literal_eval(s)
                    except (ValueError, SyntaxError):
                        return None
                return None
            
            var_stats_map = {
                var_id: parse_stats(stats_val)
                for var_id, stats_val in zip(group['variable_id'], group['stats'])
            }
            variable_stats = {v: var_stats_map.get(v) for v in VARIABLES if v in var_path_map}
        else:
            variable_stats = None
        
        # Get dimensions from first row and add var count
        dims_str = group['dimensions'].iloc[0]
        if dims_str and isinstance(dims_str, str):
            try:
                dims = ast.literal_eval(dims_str)
            except (ValueError, SyntaxError):
                dims = {}
        else:
            dims = {}
        
        # Add actual var count at the beginning
        dims_with_var = {'var': len(variable_paths), **dims}
        
        # Extract time features and add label encoded values to meta
        date_val = group['date'].iloc[0]
        meta = extract_time_features(date_val)
        
        # Add label encoded values to meta
        source = group['source'].iloc[0]
        driving_source = group['driving_source_id'].iloc[0]
        experiment = group['driving_experiment_id'].iloc[0]
        
        meta['source_encoded'] = source_labels[source]
        meta['driving_source_encoded'] = driving_source_labels[driving_source]
        meta['experiment_encoded'] = experiment_labels[experiment]
        
        result = {
            'variable_paths': variable_paths,
            'dimensions': dims_with_var,
            'meta': meta,
        }
        
        if has_stats:
            result['stats'] = variable_stats
        
        return pd.Series(result)
    
    # Group and aggregate
    rows = []
    for keys, group in df.groupby(group_cols):
        agg = aggregate_row(group)
        row = dict(zip(group_cols, keys))
        row.update(agg.to_dict())
        rows.append(row)
    
    result = pd.DataFrame(rows)
    return result


def extract_global_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Extract global statistics per variable for data scaling.
    
    Computes:
        - _min: min of min across all rows
        - _max: max of max across all rows
        - _skew: max of skewness across all rows
        - _q98: min of q0.98 across all rows
        - _q95: min of q0.95 across all rows
    
    Args:
        df: DataFrame with 'stats' column containing {variable: {stat: value}}
    
    Returns:
        Dict[variable, Dict[stat_name, value]] for scaling
    """
    from weasel.constants import VARIABLES
    
    # Initialize collectors per variable
    collectors = {var: {'min': [], 'max': [], 'skewness': [], 'q0.98': [], 'q0.95': []} 
                  for var in VARIABLES}
    
    # Collect stats from all rows
    for _, row in df.iterrows():
        stats = row.get('stats')
        if not stats or not isinstance(stats, dict):
            continue
        
        for var in VARIABLES:
            var_stats = stats.get(var)
            if not var_stats or not isinstance(var_stats, dict):
                continue
            
            if 'min' in var_stats and var_stats['min'] is not None:
                collectors[var]['min'].append(var_stats['min'])
            if 'max' in var_stats and var_stats['max'] is not None:
                collectors[var]['max'].append(var_stats['max'])
            if 'skewness' in var_stats and var_stats['skewness'] is not None:
                collectors[var]['skewness'].append(var_stats['skewness'])
            if 'q0.98' in var_stats and var_stats['q0.98'] is not None:
                collectors[var]['q0.98'].append(var_stats['q0.98'])
            if 'q0.95' in var_stats and var_stats['q0.95'] is not None:
                collectors[var]['q0.95'].append(var_stats['q0.95'])
    
    # Compute global stats
    global_stats = {}
    for var in VARIABLES:
        c = collectors[var]
        global_stats[var] = {
            '_min': min(c['min']) if c['min'] else None,
            '_max': max(c['max']) if c['max'] else None,
            '_skew': max(c['skewness']) if c['skewness'] else None,
            '_q98': min(c['q0.98']) if c['q0.98'] else None,
            '_q95': min(c['q0.95']) if c['q0.95'] else None,
        }
    
    return global_stats


def apply_dataset_split(df: pd.DataFrame, val_ratio: float = 0.4, seed: int = 42) -> pd.DataFrame:
    """
    Apply train/test/validation split labels based on year-day pattern.
    
    Training: odd years + even days, OR even years + odd days (~50%)
    Testing: remaining samples, split further into test and validation
    Validation: 40% of the non-training samples (default), randomly selected
    
    Final split: ~50% train, ~30% test, ~20% validation
    
    Args:
        df: DataFrame with 'date' column
        val_ratio: Ratio of non-training samples to use as validation (default 0.4)
        seed: Random seed for reproducibility (default 42)
    
    Returns:
        DataFrame with 'split' column containing labels ('train', 'test', 'val')
    """
    df = df.copy()
    
    # Shuffle BEFORE splitting for randomness
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Extract year and day from date
    dates = pd.to_datetime(df['date'])
    years = dates.dt.year
    days = dates.dt.day
    
    # Training: (odd year AND even day) OR (even year AND odd day)
    is_odd_year = years % 2 == 1
    is_even_day = days % 2 == 0
    is_even_year = years % 2 == 0
    is_odd_day = days % 2 == 1
    
    is_train = (is_odd_year & is_even_day) | (is_even_year & is_odd_day)
    
    # Initialize split column
    df['split'] = 'test'
    df.loc[is_train, 'split'] = 'train'
    
    # From non-training, randomly select val_ratio as validation with fixed seed
    non_train_idx = df[~is_train].index.tolist()
    n_val = int(len(non_train_idx) * val_ratio)
    
    # Random selection with fixed seed for reproducibility
    rng = np.random.RandomState(seed)
    val_idx = rng.choice(non_train_idx, size=n_val, replace=False)
    
    df.loc[val_idx, 'split'] = 'val'
    
    # Shuffle with fixed seed and reset index
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df
