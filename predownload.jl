# Fetch every ERA5 and GLORYS file a hindcast window needs before the GPU run, so the run never
# waits on the CDS or Copernicus Marine queues. Uses the same regions (hence the same cache
# filenames) as hindcast.jl / ocean.jl / nested_atmosphere_model:
#
#   SO_START=2018-01-29T00:00:00 SO_HOURS=672 julia --project predownload.jl

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

@info "ERA5 parent (pressure levels + surface geopotential), $start_date → $stop_date over $(parent_region)"
PrescribedAtmosphere(parent_region, (start_date, stop_date), pressure_levels;
                     architecture = CPU(), dir = era5_dir, time_indices_in_memory = 3)

@info "ERA5 single-level snapshots at $start_date"
for (name, region) in ((:mean_sea_level_pressure, BoundingBox(grid)), (:skin_temperature, parent_region))
    download(Metadatum(name; dataset = single_levels, date = start_date, region, dir = era5_dir))
end

glorys_region = BoundingBox(longitude = (longitude[1] - 0.5, longitude[2] + 0.5),
                            latitude  = (latitude[1]  - 0.5, latitude[2]  + 0.5))
glorys_dates = start_date:Day(1):(stop_date + Day(1))

@info "GLORYS daily u, v, T, S, zos for $(length(glorys_dates)) days over $(glorys_region)"
download(MetadataSet(:u_velocity, :v_velocity, :temperature, :salinity, :free_surface;
                     dataset = GLORYSDaily(), dates = glorys_dates, region = glorys_region, dir = glorys_dir))
download(Metadata(:depth; dataset = GLORYSStatic(), dir = glorys_dir, region = glorys_region))

@info "done: $(length(readdir(era5_dir))) ERA5 files, $(length(readdir(glorys_dir))) GLORYS files"
