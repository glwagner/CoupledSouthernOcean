# Cut ERA5 files from Google's ARCO-ERA5 copy (0.25°, 37 pressure levels, hourly; no CDS queue)
# into the per-variable, per-hour NetCDF files NumericalEarth's ERA5 readers expect, in the layout
# the CDS delivers: longitude × latitude (north to south) × pressure_level (1000 → 1 hPa) × valid_time.
#
#   julia --project=tools/arco tools/arco/arco.jl <plan.tsv> [worker] [nworkers]
#
# The plan comes from `SO_ARCO_PLAN=<plan.tsv> julia --project predownload.jl`: one row per missing
# file (ARCO variable, NetCDF variable, "pl" or "sl", date, padded west east south north, path).
# Rows are grouped by (variable, hour) so each global chunk is read once; worker k of n takes every
# n-th group.

using Zarr
using NCDatasets
using Dates
using Printf

const store = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
const Δ = 0.25

plan_path = ARGS[1]
worker = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 1
nworkers = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 1

struct Row
    arco_name :: String
    nc_name   :: String
    kind      :: String
    date      :: DateTime
    west :: Float64
    east :: Float64
    south :: Float64
    north :: Float64
    path :: String
end

function read_plan(path)
    rows = Row[]
    for line in eachline(path)
        isempty(line) && continue
        a, n, k, d, w, e, s, no, p = split(line, '\t')
        push!(rows, Row(a, n, k, DateTime(d), parse.(Float64, (w, e, s, no))..., p))
    end
    return unique(r -> r.path, rows)
end

# Grid points inside the padded box, as the CDS returns them: longitudes in the box's own range
# (e.g. 149 … 191 across the dateline), latitudes from north to south.
box_longitudes(r) = collect(ceil(r.west / Δ) * Δ : Δ : floor(r.east / Δ) * Δ)
box_latitudes(r)  = collect(floor(r.north / Δ) * Δ : -Δ : ceil(r.south / Δ) * Δ)
longitude_index(λ) = round(Int, mod(λ, 360) / Δ) + 1   # ARCO: 0 … 359.75
latitude_index(φ)  = round(Int, (90 - φ) / Δ) + 1       # ARCO: 90 … -90

function write_file(r, field, levels)
    λ, φ = box_longitudes(r), box_latitudes(r)
    i, j = longitude_index.(λ), latitude_index.(φ)
    tmp = r.path * ".arco-tmp"
    isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do ds
        ds.attrib["Conventions"] = "CF-1.7"
        ds.attrib["institution"] = "European Centre for Medium-Range Weather Forecasts"
        ds.attrib["history"] = "cut from ARCO-ERA5 ($store) by southern-ocean/tools/arco/arco.jl"
        defDim(ds, "valid_time", 1)
        defDim(ds, "latitude", length(φ))
        defDim(ds, "longitude", length(λ))
        defVar(ds, "number", Int64(0), (); attrib = ["long_name" => "ensemble member numerical id", "units" => "1",
                                                      "standard_name" => "realization"])
        defVar(ds, "valid_time", [r.date], ("valid_time",); attrib = ["long_name" => "time", "standard_name" => "time",
                                                                      "units" => "seconds since 1970-01-01",
                                                                      "calendar" => "proleptic_gregorian"])
        dims = ("longitude", "latitude")
        if r.kind == "pl"
            # ARCO stores 1 → 1000 hPa; the CDS files run 1000 → 1.
            order = sortperm(levels; rev = true)
            defDim(ds, "pressure_level", length(levels))
            defVar(ds, "pressure_level", Float64.(levels[order]), ("pressure_level",);
                   fillvalue = NaN, attrib = ["long_name" => "pressure", "units" => "hPa", "positive" => "down",
                                              "stored_direction" => "decreasing", "standard_name" => "air_pressure"])
            data = reshape(field[i, j, order], length(λ), length(φ), length(levels), 1)
            dims = (dims..., "pressure_level")
        else
            data = reshape(field[i, j], length(λ), length(φ), 1)
        end
        defVar(ds, "latitude", φ, ("latitude",); fillvalue = NaN,
               attrib = ["units" => "degrees_north", "standard_name" => "latitude", "long_name" => "latitude",
                         "stored_direction" => "decreasing"])
        defVar(ds, "longitude", λ, ("longitude",); fillvalue = NaN,
               attrib = ["units" => "degrees_east", "standard_name" => "longitude", "long_name" => "longitude"])
        defVar(ds, "expver", ["0001"], ("valid_time",))
        defVar(ds, r.nc_name, Float32.(data), (dims..., "valid_time"); fillvalue = NaN32)
    end
    mv(tmp, r.path; force = true)
    return nothing
end

rows = filter(r -> !isfile(r.path), read_plan(plan_path))
grouped = Dict{Tuple{String, DateTime}, Vector{Row}}()
foreach(r -> push!(get!(grouped, (r.arco_name, r.date), Row[]), r), rows)
groups = collect(values(grouped))
sort!(groups, by = g -> (first(g).date, first(g).arco_name))
mine = groups[worker:nworkers:end]
@info "worker $worker/$nworkers: $(length(mine)) of $(length(groups)) (variable, hour) groups, $(sum(length, mine; init = 0)) files"

g = zopen(store; consolidated = true)
levels = g["level"][:]
hours_since_1900(t) = (t - DateTime(1900)).value ÷ 3_600_000
times = g["time"]

t₀ = time()
for (n, group) in enumerate(mine)
    r = first(group)
    index = hours_since_1900(r.date) + 1
    times[index] == hours_since_1900(r.date) || error("ARCO time index mismatch at $(r.date)")
    array = g[r.arco_name]
    field = ndims(array) == 4 ? array[:, :, :, index] : array[:, :, index]
    any(isnan, field) && @warn "$(r.arco_name) at $(r.date) has NaNs"
    foreach(s -> write_file(s, field, levels), group)
    if n % 20 == 0 || n == length(mine)
        @info @sprintf("worker %d: %d / %d groups, %.1f s per group", worker, n, length(mine), (time() - t₀) / n)
        flush(stderr)
    end
end
