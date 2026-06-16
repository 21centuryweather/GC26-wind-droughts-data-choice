"""NCI Gadi catalogue downloader and data processing for BARRA-C2."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import paramiko
from tqdm import tqdm

from weasel.constants import BARRA_PATH, NCI_PASSWORD, NCI_USERNAME, VARIABLES
from weasel.logger import get_logger

logger = get_logger()

# Resource folder for storing downloaded files
RESOURCE_DIR = Path(__file__).parent / "resources"

# Maximum age for cached files (in days)
MAX_FILE_AGE_DAYS = 30


def is_on_nci() -> bool:
    """Check if running on NCI Gadi by checking if /g/data exists."""
    return Path("/g/data").exists()


def is_file_stale(file_path: Path, max_age_days: int = MAX_FILE_AGE_DAYS) -> bool:
    """
    Check if a file is older than the specified maximum age.

    Args:
        file_path: Path to the file to check.
        max_age_days: Maximum age in days before file is considered stale.

    Returns:
        True if file doesn't exist or is older than max_age_days, False otherwise.
    """
    if not file_path.exists():
        return True

    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
    age = datetime.now() - file_mtime

    if age > timedelta(days=max_age_days):
        logger.info(f"File {file_path.name} is {age.days} days old (max: {max_age_days} days)")
        return True

    logger.debug(f"File {file_path.name} is {age.days} days old, still fresh")
    return False


class TqdmCallback:
    """Callback class for tracking SFTP download progress with tqdm."""

    def __init__(self, tqdm_instance: tqdm):
        self.tqdm = tqdm_instance
        self.last_transferred = 0

    def __call__(self, transferred: int, total: int):
        """Update tqdm progress bar with bytes transferred."""
        delta = transferred - self.last_transferred
        self.tqdm.update(delta)
        self.last_transferred = transferred


def download_file_via_ssh(
    remote_path: str,
    local_path: Path,
    username: str,
    password: str,
    hostname: str = "gadi.nci.org.au",
) -> Path:
    """
    Download a single file from remote server via SSH/SFTP with progress bar.

    Args:
        remote_path: Full path to the file on the remote server.
        local_path: Local path where the file should be saved.
        username: SSH username.
        password: SSH password.
        hostname: SSH hostname (default: gadi.nci.org.au).

    Returns:
        Path to the downloaded file.
    """
    # Create parent directories if they don't exist
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Establish SSH connection
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        logger.info(f"Connecting to {hostname}...")
        ssh.connect(hostname, username=username, password=password)
        logger.debug(f"SSH connection established to {hostname}")

        # Open SFTP session
        sftp = ssh.open_sftp()

        # Get remote file size for progress bar
        file_stat = sftp.stat(remote_path)
        file_size = file_stat.st_size
        filename = os.path.basename(remote_path)

        logger.info(f"Downloading {filename} ({file_size / (1024 * 1024):.2f} MB)...")

        # Download with tqdm progress bar
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename,
            ncols=80,
        ) as pbar:
            callback = TqdmCallback(pbar)
            sftp.get(remote_path, str(local_path), callback=callback)

        sftp.close()
        logger.success(f"Downloaded: {local_path}")
        return local_path

    except paramiko.AuthenticationException:
        logger.error(f"Authentication failed for user {username} on {hostname}")
        raise
    except paramiko.SSHException as e:
        logger.error(f"SSH connection error: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Remote file not found: {remote_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        raise
    finally:
        ssh.close()
        logger.debug("SSH connection closed")


def download_file(
    remote_path: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    resource_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """
    Download catalog file from NCI Gadi for offline usage.
    If running on NCI, directly returns path to the source file.

    Args:
        username: NCI username. If None, uses NCI_UNAME from constants/environment.
        password: NCI password. If None, uses NCI_PASSWORD from constants/environment.
        resource_dir: Directory to store downloaded files. Defaults to weasel/resources.
        force_download: If True, re-download even if files exist locally.

    Returns:
        Path to local catalogue file.

    Raises:
        ValueError: If username or password is not provided and not in environment (when not on NCI).
        ConnectionError: If SSH connection fails.
    """
    # If running on NCI, directly return the source file path
    if is_on_nci():
        logger.info("Running on NCI Gadi - using direct file access")
        return Path(remote_path)

    # Resolve credentials (only needed when not on NCI)
    uname = username or NCI_USERNAME
    passwd = password or NCI_PASSWORD

    if not uname or not passwd:
        logger.error("NCI credentials not provided")
        raise ValueError(
            "NCI credentials not provided. Either pass username/password arguments "
            "or set NCI_USERNAME and NCI_PASSWORD environment variables."
        )

    logger.info(f"Starting '{remote_path}' download for user: {uname}")

    # Set up resource directory
    res_dir = resource_dir or RESOURCE_DIR
    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Define local file path
    _local = res_dir / Path(remote_path).name

    # Download file
    if force_download or is_file_stale(_local):
        logger.info(f"Downloading catalog {remote_path}...")
        file_local = download_file_via_ssh(
            remote_path=remote_path,
            local_path=_local,
            username=uname,
            password=passwd,
        )
        return file_local


    logger.info(f"file exists and is fresh: {_local}")
    return _local


def parse_version_date(version_str: str) -> datetime:
    """
    Convert version string (e.g., 'v20241201') to datetime.

    Args:
        version_str: Version string in format 'vYYYYMMDD'.

    Returns:
        Parsed datetime object.
    """
    if pd.isna(version_str):
        return pd.NaT
    # Remove 'v' prefix and parse date
    date_str = str(version_str).lstrip('v')
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        return pd.NaT


def parse_float_date(float_date: float) -> datetime:
    """
    Convert float date (e.g., 202505.0) to datetime.

    Args:
        float_date: Date as float in format YYYYMM.0 (year and month only).

    Returns:
        Parsed datetime object (first day of the month).
    """
    if pd.isna(float_date):
        return pd.NaT
    try:
        date_str = str(int(float_date))
        # Handle YYYYMM format (6 digits)
        if len(date_str) == 6:
            return datetime.strptime(date_str, '%Y%m')
        # Handle YYYYMMDD format (8 digits) as fallback
        elif len(date_str) == 8:
            return datetime.strptime(date_str, '%Y%m%d')
        else:
            return pd.NaT
    except (ValueError, TypeError):
        return pd.NaT


def select_latest_version(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select rows with latest version for each variable + start_time combination.

    Args:
        df: DataFrame with 'version', 'variable_id', 'start_time' columns.

    Returns:
        DataFrame with only latest version per variable + start_time.
    """
    df = df.copy()

    # Parse version to datetime for comparison
    df['version_date'] = df['version'].apply(parse_version_date)

    # Group by variable and start_time, keep latest version
    df = df.sort_values('version_date', ascending=False)
    df = df.drop_duplicates(subset=['variable_id', 'start_time'], keep='first')

    # Sort by variable and start_time for clean output
    df = df.sort_values(['variable_id', 'start_time'])
    df = df.reset_index(drop=True)

    return df


def load_barra_catalogue(path: Path) -> pd.DataFrame:
    """
    Load and filter BARRA catalogue data.

    Args:
        path: Path to the BARRA catalogue CSV.gz file.

    Returns:
        Filtered DataFrame with BARRA data.
    """
    logger.info(f"Loading BARRA catalogue from {path}")
    data = pd.read_csv(path, compression='gzip')

    # Apply BARRA-specific filters
    mask = (
        (data.domain_id == 'AUST-04') &
        (data.file_type == 'f') &
        (data.activity_id == 'reanalysis') &
        (data.project_id == 'output') &
        (data.RCM_institution_id == 'BOM') &
        (data.driving_source_id == 'ERA5') &
        (data.driving_experiment_id == 'historical') &
        (data.driving_variant_label == 'hres') &
        (data.source_id == 'BARRA-C2') &
        (data.version_realisation == 'v1') &
        (data.freq == 'day') &
        (data.variable_id.isin(VARIABLES))
    )
    data = data.loc[mask].copy()
    data['source'] = 'BARRA'

    # Select and reorder columns
    data = data[[
        'source',
        'driving_source_id',
        'driving_experiment_id',
        'source_id',
        'variable_id',
        'start_time',
        'end_time',
        'version',
        'path'
    ]].copy()

    logger.info(f"BARRA catalogue filtered: {len(data)} records (before version filter)")

    # Select latest version per variable + start_time
    data = select_latest_version(data)
    logger.info(f"BARRA catalogue: {len(data)} records (after selecting latest versions)")

    return data


def load_static_vars(path: Path) -> list[str]:
    """
    Load static variables (fx frequency) from BARRA catalogue.

    Args:
        path: Path to the BARRA catalogue CSV.gz file.

    Returns:
        List of paths to static variable files.
    """
    logger.info(f"Loading static variables from {path}")
    data = pd.read_csv(path, compression='gzip')

    # Filter for static variables
    mask = (
        (data['domain_id'] == 'AUST-04') &
        (data['file_type'] == 'f') &
        (data['freq'] == 'fx')
    )
    data = data.loc[mask].copy()

    paths = data.path.to_list()
    logger.info(f"Found {len(paths)} static variable files")

    return paths


def process_combined_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process combined catalogue data: convert dates to proper format and clean up columns.
    Note: Latest version selection is already done in individual load functions.

    Args:
        df: DataFrame with combined catalogue data.

    Returns:
        Processed DataFrame with proper date columns and cleaned up.
    """
    logger.info("Processing combined catalogue data...")

    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Convert start_time and end_time to year-month period strings (e.g., "2024-01")
    df['start_date'] = df['start_time'].apply(
        lambda x: parse_float_date(x).strftime('%Y-%m') if pd.notna(parse_float_date(x)) else None
    )
    df['end_date'] = df['end_time'].apply(
        lambda x: parse_float_date(x).strftime('%Y-%m') if pd.notna(parse_float_date(x)) else None
    )

    # Sort by source, variable, experiment, and start_date for cleaner output
    df = df.sort_values(['source', 'driving_experiment_id', 'variable_id', 'start_date'])
    df = df.reset_index(drop=True)

    # Remove unnecessary columns
    columns_to_drop = ['version', 'version_date', 'start_time', 'end_time']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Reorder columns for cleaner output
    column_order = [
        'source',
        'driving_source_id',
        'driving_experiment_id',
        'source_id',
        'variable_id',
        'start_date',
        'end_date',
        'path'
    ]
    df = df[[col for col in column_order if col in df.columns]]

    logger.info(f"Processed catalogue: {len(df)} records")
    return df


BARRA_CATALOGUE_FILE = "filtered_barra.csv.gz"


def load_combined_catalogue(
    barra_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Path:
    """
    Load, filter and process BARRA-C2 catalogue.
    Caches the result to disk and reuses if less than 30 days old.

    Args:
        barra_path: Path to BARRA catalogue. If None, downloads first.
        force_refresh: If True, ignore cache and reprocess.

    Returns:
        Path to the processed BARRA catalogue CSV file.
    """
    # Cached catalogue path
    cached_path = RESOURCE_DIR / BARRA_CATALOGUE_FILE

    # Check if cached file exists and is fresh
    if not force_refresh and not is_file_stale(cached_path):
        logger.info(f"BARRA catalogue exists and is fresh: {cached_path}")
        return cached_path

    logger.info("Processing BARRA catalogue (cache missing or stale)...")

    # Download if path not provided
    if barra_path is None:
        barra_path = download_file(BARRA_PATH)

    # Load, filter, and select latest versions
    barra_df = load_barra_catalogue(barra_path)

    # Process dates
    processed = process_combined_catalogue(barra_df)

    # Save to cache with gzip compression
    processed.to_csv(cached_path, index=False, compression='gzip')
    logger.success(f"BARRA catalogue saved: {cached_path} ({len(processed)} records)")

    return cached_path
