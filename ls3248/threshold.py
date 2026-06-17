import xarray as xr
import numpy as np
from glob import glob

# =====================
# Settings
# =====================
threshold = 20  # percentile, e.g. 10 / 20 / 25

# Australia region
lon_min, lon_max = 110, 160
lat_max, lat_min = -10, -45   # ERA5 latitude is descending

clim_start = "1991-01-01"
clim_end   = "2020-12-31"

u_path = "/g/data/nf33/WindDroughts_Group3/ERA5/100u/*.nc"
v_path = "/g/data/nf33/WindDroughts_Group3/ERA5/100v/*.nc"

print("Parameters setted")

# =====================
# Read data
# =====================
u_files = sorted(glob(u_path))
v_files = sorted(glob(v_path))

ds_u = xr.open_mfdataset(
    u_files,
    combine="by_coords",
    chunks={"time": 90, "latitude": 50, "longitude": 50}
)

ds_v = xr.open_mfdataset(
    v_files,
    combine="by_coords",
    chunks={"time": 90, "latitude": 50, "longitude": 50}
)

print("Data loaded")

# =====================
# Daily wind speed
# =====================
ws = np.hypot(ds_u["u100"], ds_v["v100"]).astype("float32").rename("ws")
print("Wind speed calculated")

# =====================
# Subset region first
# =====================
ws = ws.sel(
    longitude=slice(lon_min, lon_max),
    latitude=slice(lat_max, lat_min)
)
print("Region subsetted")

# =====================
# 1991–2020 climatology
# =====================
ws_clim = ws.sel(time=slice(clim_start, clim_end))
print("Climatology selected")


# =====================
# Fixed percentile threshold
# =====================
p_threshold = ws_clim.quantile(threshold / 100.0, dim="time")
p_threshold = p_threshold.astype("float32").rename(f"ws_p{threshold}")

# remove the quantile coordinate if present
if "quantile" in p_threshold.coords:
    p_threshold = p_threshold.drop_vars("quantile")
print("Threshold calculated")

p_threshold.attrs["long_name"] = (
    f"{threshold}th percentile of daily 100m wind speed from ERA5 (1991-2020)"
)
p_threshold.attrs["units"] = "m s-1"

# =====================
# Save threshold field
# =====================
outfile = f"ERA5_WS_P{threshold}_1991_2020_AUS.nc"
p_threshold.to_netcdf(outfile)

print(f"Saved: {outfile}")