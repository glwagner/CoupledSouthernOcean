# Fetch every ERA5 and GLORYS file a hindcast window needs before the GPU run, so the run never
# waits on the CDS or Copernicus Marine queues. Uses the same regions (hence the same cache
# filenames) as hindcast.jl / ocean.jl / nested_atmosphere_model:
#
#   SO_START=2018-01-29T00:00:00 SO_HOURS=672 julia --project predownload.jl
#
# SO_ERA5=false skips the ERA5 pressure levels, SO_GLORYS=false skips GLORYS, SO_SNAPSHOTS=false skips
# the single-level snapshots; SO_ERA5_DIR (case.jl) puts the ERA5 files in a private directory.
# SO_ARCO_PLAN=<file> lists the missing ERA5 files for tools/arco/arco.jl instead of asking the CDS.

using NumericalEarth
using NumericalEarth.DataWrangling: Metadata, Metadatum, MetadataSet, BoundingBox, default_horizontal_padding
using NumericalEarth.DataWrangling.GLORYS: GLORYSStatic
using CopernicusClimateDataStore
using CopernicusMarine
using Oceananigans
using Dates

include("case.jl")

# The CDS-MARS queue can hold a request for hours (7566 queued, 460 running on 2026-10-17), but
# CopernicusClimateDataStore gives up after 1 h, and the request then starts again at the back of
# the queue on retry. `hourly` does not pass `max_wait` through, so the plain three-argument
# `retrieve` it calls waits SO_CDS_MAX_WAIT_HOURS instead.
const cds_max_wait = parse(Float64, get(ENV, "SO_CDS_MAX_WAIT_HOURS", "20")) * 3600
CopernicusClimateDataStore.retrieve(dataset::String, params::Dict, output_path::String) =
    CopernicusClimateDataStore.retrieve(dataset, params, output_path; max_wait = cds_max_wait)
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
hydrometeor_names = [NumericalEarth.DataWrangling.ERA5.ERA5_hydrometeor_names[q] for q in parent_hydrometeors]
pressure_level_names = (:temperature, :eastward_velocity, :northward_velocity, :specific_humidity,
                        hydrometeor_names..., :geopotential)
era5_dates = NumericalEarth.DataWrangling.expand_dates(pressure_levels, :temperature, (start_date, stop_date))
snapshots = ((:geopotential, parent_region), (:mean_sea_level_pressure, BoundingBox(grid)),
             (:skin_temperature, parent_region))

# SO_ARCO_PLAN=<file>: instead of asking the CDS, append every missing ERA5 file to <file> for
# tools/arco/arco.jl, which cuts it from Google's ARCO-ERA5 copy (no queue). One row per file:
# ARCO variable, NetCDF variable, level kind, date, padded west/east/south/north, path.
arco_plan = get(ENV, "SO_ARCO_PLAN", nothing)
cds_extension = Base.get_extension(NumericalEarth, :NumericalEarthCopernicusClimateDataStoreExt)
arco_names = Dict(:geopotential => "geopotential_at_surface")   # single-level names that differ

function plan_or_download(metadata::Metadata, kind)
    isnothing(arco_plan) && return download(metadata)
    dataset_names = kind == "pl" ? NumericalEarth.DataWrangling.ERA5.ERA5PL_dataset_variable_names :
                                   NumericalEarth.DataWrangling.ERA5.ERA5_dataset_variable_names
    nc_names = NumericalEarth.DataWrangling.ERA5.nc_varnames(metadata.dataset)
    open(arco_plan, "a") do io
        for metadatum in metadata
            path = NumericalEarth.DataWrangling.metadata_path(metadatum)
            isfile(path) && continue
            area = cds_extension.padded_era5_region(metadatum.region, metadatum.dataset, [metadatum.name])
            arco_name = kind == "sl" ? get(arco_names, metadatum.name, dataset_names[metadatum.name]) :
                                       dataset_names[metadatum.name]
            println(io, join((arco_name, nc_names[metadatum.name], kind, metadatum.dates,
                              area.longitude..., area.latitude..., path), '\t'))
        end
    end
end

if get(ENV, "SO_ERA5", "true") == "true"
    @info "ERA5 parent pressure levels $(pressure_level_names), $start_date → $stop_date over $(parent_region)"
    for name in pressure_level_names
        plan_or_download(Metadata(name; dataset = pressure_levels, dates = era5_dates, region = parent_region, dir = era5_dir), "pl")
    end
end

if get(ENV, "SO_SNAPSHOTS", "true") == "true"
    @info "ERA5 single-level snapshots at $start_date"
    for (name, region) in snapshots
        plan_or_download(Metadata(name; dataset = single_levels, dates = [start_date], region, dir = era5_dir), "sl")
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
    # SO_GLORYS_STATIC=false leaves the static depth to one of several parallel jobs
    get(ENV, "SO_GLORYS_STATIC", "true") == "true" &&
        download(Metadata(:depth; dataset = GLORYSStatic(), dir = glorys_dir, region = glorys_region))
end

@info "done: $(length(readdir(era5_dir))) ERA5 files, $(length(readdir(glorys_dir))) GLORYS files"
