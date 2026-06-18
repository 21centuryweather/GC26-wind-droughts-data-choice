"""Calculate wind drought probabilities from BARRA-C2"""

import time

import dask
import intake
import numpy as np
import xarray as xr
from dask_setup import setup_dask_client  # pylint: disable=import-error

CUT_IN_SPEED = 3.5
RATED_SPEED = 13.0
CUT_OUT_SPEED = 25.0
COARSEN_FACTOR = 6
WINDOW_DAYS = 3
CAPACITY_FACTOR_THRESHOLDS = [0.0, 0.1, 0.2]
OUT_DIR = "/scratch/nf33/ts1181"


def coarsen(da, factor):
    """Coarsen an array by averaging over blocks."""
    return da.coarsen({"lon": factor, "lat": factor}, boundary="trim").mean()


def capacity_factor(wspd):
    """Calculate the van der Wiel wind turbine capacity factor."""
    return xr.where(
        wspd < CUT_IN_SPEED,
        0.0,
        xr.where(
            wspd < RATED_SPEED,
            (wspd**3 - CUT_IN_SPEED**3) / (RATED_SPEED**3 - CUT_IN_SPEED**3),
            xr.where(wspd < CUT_OUT_SPEED, 1.0, 0.0),
        ),
    )


def is_drought(wspd, thresh, window):
    """Identify wind droughts using a capacity factor threshold and time window."""
    lt_thresh = capacity_factor(wspd) <= thresh
    all_window_lt_thresh = (
        lt_thresh.rolling({"time": window}).sum("time") == window
    ).shift({"time": -(window - 1)}, fill_value=False)
    any_all_window_lt_thresh = (
        all_window_lt_thresh.rolling({"time": window}, min_periods=1).sum("time") > 0
    )
    return any_all_window_lt_thresh


def probability_drought(wspd, thresh, window):
    """Calculate climatological wind drought probability."""
    return is_drought(wspd, thresh, window).mean("time")


def get_data():
    """Retrieve wind speed data from BARRA-C2."""
    catalog = intake.open_esm_datastore("/g/data/ob53/catalog/v2/esm/catalog.json")
    ua100m = catalog.search(
        source_id="BARRA-C2",
        variable_id="ua100m",
        freq="day",
        version="latest",
    ).to_dask()["ua100m"]
    va100m = catalog.search(
        source_id="BARRA-C2",
        variable_id="va100m",
        freq="day",
        version="latest",
    ).to_dask()["va100m"]
    return np.sqrt(ua100m**2 + va100m**2)


def main():
    """Main script function."""
    client, cluster, _ = setup_dask_client(
        mode="local", workload_type="io", reserve_mem_gb=2, dashboard=False
    )

    timing = time.time()
    wspd = get_data()
    print(f"got data in {(time.time() - timing)/60:.1f} min")

    thresh = xr.DataArray(CAPACITY_FACTOR_THRESHOLDS, dims="thresh")
    thresh = thresh.assign_coords({"thresh": thresh})
    coarsen_first = probability_drought(
        coarsen(wspd, factor=COARSEN_FACTOR), thresh=thresh, window=WINDOW_DAYS
    )
    coarsen_last = coarsen(
        probability_drought(wspd, thresh=thresh, window=WINDOW_DAYS),
        factor=COARSEN_FACTOR,
    )

    timing = time.time()
    coarsen_first, coarsen_last = dask.compute(coarsen_first, coarsen_last)
    print(f"finished computing in {(time.time() - timing)/60:.1f} min")

    timing = time.time()
    coarsen_first.to_netcdf(OUT_DIR + "/coarsen_first.nc")
    coarsen_last.to_netcdf(OUT_DIR + "/coarsen_last.nc")
    print(f"finished writing in {(time.time() - timing)/60:.1f} min")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
