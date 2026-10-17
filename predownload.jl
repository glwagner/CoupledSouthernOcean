# Fetch every ERA5 and GLORYS file a hindcast window needs before the GPU run, so the run never
# waits on the CDS or Copernicus Marine queues. Uses the same regions (hence the same cache
# filenames) as hindcast.jl / ocean.jl / nested_atmosphere_model:
#
#   SO_START=2018-01-29T00:00:00 SO_HOURS=672 julia --project predownload.jl
#
# SO_ERA5=false skips the ERA5 pressure levels, SO_GLORYS=false skips GLORYS, SO_SNAPSHOTS=false skips
# the single-level snapshots; SO_ERA5_DIR (case.jl) puts the ERA5 files in a private directory.

using NumericalEarth
using NumericalEarth.DataWrangling: Metadata, Metadatum, MetadataSet, BoundingBox, default_horizontal_padding
using NumericalEarth.DataWrangling.GLORYS: GLORYSStatic
using CopernicusClimateDataStore
using CopernicusMarine
using Oceananigans
using Dates

include("case.jl")
const download = NumericalEarth.DataWrangling.Downloads.download

pressure_levels = ERA5HourlyPressureLevels()
single_levels = ERA5HourlySingleLevel()

grid = LatitudeLongitudeGrid(CPU(); longitude, latitude, size = (Nx, Ny, 1), z = (0, 1))
parent_region = BoundingBox(grid; padding = default_horizontal_padding(pressure_levels))

# The pressure-level bundle PrescribedAtmosphere downloads up front, without building it: building
# also fetches a single-level surface geopotential at the window's first date, a separate CDS
# request that can queue for an hour. Only the run's start date needs the single-level snapshots
# (surface geopotential, MSLP, skin temperature), so a window fetches them only when
# SO_SNAPSHOTS=true (the default; predownload_windows.batch sets it for the first window only).
if get(ENV, "SO_ERA5", "true") == "true"
    @info "ERA5 parent pressure levels, $start_date → $stop_date over $(parent_region)"
    era5_dates = NumericalEarth.DataWrangling.expand_dates(pressure_levels, :temperature, (start_date, stop_date))
    download(MetadataSet(:temperature, :eastward_velocity, :northward_velocity, :specific_humidity,
                         :specific_cloud_liquid_water_content, :specific_rain_water_content,
                         :specific_cloud_ice_water_content, :specific_snow_water_content, :geopotential;
                         dataset = pressure_levels, dates = era5_dates, region = parent_region, dir = era5_dir))
end

if get(ENV, "SO_SNAPSHOTS", "true") == "true"
    @info "ERA5 single-level snapshots at $start_date"
    for (name, region) in ((:geopotential, parent_region), (:mean_sea_level_pressure, BoundingBox(grid)),
                           (:skin_temperature, parent_region))
        download(Metadatum(name; dataset = single_levels, date = start_date, region, dir = era5_dir))
    end
end

glorys_region = BoundingBox(longitude = (longitude[1] - 0.5, longitude[2] + 0.5),
                            latitude  = (latitude[1]  - 0.5, latitude[2]  + 0.5))
# ocean.jl reads one day past the stop date; SO_GLORYS_EXTRA_DAYS=0 stops at the stop date, so a
# window can be fetched in pieces without two jobs writing the same daily file.
glorys_dates = start_date:Day(1):(stop_date + Day(parse(Int, get(ENV, "SO_GLORYS_EXTRA_DAYS", "1"))))

if get(ENV, "SO_GLORYS", "true") == "true"
    @info "GLORYS daily u, v, T, S, zos for $(length(glorys_dates)) days over $(glorys_region)"
    download(MetadataSet(:u_velocity, :v_velocity, :temperature, :salinity, :free_surface;
                         dataset = GLORYSDaily(), dates = glorys_dates, region = glorys_region, dir = glorys_dir))
    download(Metadata(:depth; dataset = GLORYSStatic(), dir = glorys_dir, region = glorys_region))
end

@info "done: $(length(readdir(era5_dir))) ERA5 files, $(length(readdir(glorys_dir))) GLORYS files"
