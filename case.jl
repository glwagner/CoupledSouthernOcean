# Case configuration shared by the drivers and analysis scripts.
#
# February 2018 around New Zealand: ex-tropical cyclone Fehi made landfall on the West Coast of
# the South Island on 1 February and ex-tropical cyclone Gita crossed central New Zealand on
# 20 February, both during the 2017/18 Tasman Sea marine heatwave. Summer stratification is
# shallow, so storm-driven mixing should leave a clear imprint on the sea surface temperature.
#
# GLORYS daily reanalysis (GLORYSDaily) ends on 2021-06-30, which rules out later storms
# (e.g. Gabrielle, February 2023) until the interim product is wired up.

using Dates: DateTime, Day, Hour

start_date = DateTime(get(ENV, "SO_START", "2018-02-19T00:00:00"))
run_hours  = parse(Float64, get(ENV, "SO_HOURS", "72"))
stop_date  = start_date + Hour(ceil(Int, run_hours))

# ≈ CONUS-sized box: the Tasman Sea, New Zealand and its subantarctic plateau, and the
# Southern Ocean storm track to 57°S. It crosses the dateline; longitudes stay in (150, 190).
longitude = Tuple(parse.(Float64, split(get(ENV, "SO_LONGITUDE", "150,190"), ',')))
latitude  = Tuple(parse.(Float64, split(get(ENV, "SO_LATITUDE",  "-57,-27"), ',')))

cells_per_degree = parse(Float64, get(ENV, "SO_CELLS_PER_DEGREE", "9"))   # 9 ⇒ 1/9° ≈ 12 km; 4.5 ⇒ ≈ 25 km
Nx = round(Int, (longitude[2] - longitude[1]) * cells_per_degree)
Ny = round(Int, (latitude[2]  - latitude[1])  * cells_per_degree)

datadir = get(ENV, "SO_DATA", joinpath(@__DIR__, "data"))
era5_dir   = get(ENV, "SO_ERA5_DIR", joinpath(datadir, "era5"))   # a private directory lets ERA5 jobs run concurrently
glorys_dir = joinpath(datadir, "glorys")
mkpath(era5_dir); mkpath(glorys_dir)

# Hydrometeors the ERA5 parent carries (SO_PARENT_HYDROMETEORS, comma-separated from qᶜˡ, qʳ, qᶜⁱ, qˢ).
# Rain and snow water content are left out by default: they only add to the boundary total water,
# and they are the two pressure-level variables Google's ARCO-ERA5 copy lacks.
parent_hydrometeors = Tuple(Symbol.(split(get(ENV, "SO_PARENT_HYDROMETEORS", "qᶜˡ,qᶜⁱ"), ',')))
