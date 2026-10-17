# Ocean-only diagnosis of the regional ocean in ocean.jl: the same grid, GLORYS initial state and
# open boundaries, no atmosphere (zero surface fluxes), so it can step at minutes per step.
#
#   SO_OCEAN_FT=Float32|Float64   SO_OCEAN_DT=60 (s)   SO_OCEAN_HOURS=24
#   SO_OCEAN_BOUNDARIES=open|closed (closed: drop every lateral open-boundary condition)

using NumericalEarth
using NumericalEarth.DataWrangling: Metadata, BoundingBox
using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: Clamp
using Oceananigans.Architectures: on_architecture
using CopernicusMarine
using CUDA
using JLD2
using Printf
using Statistics: mean
import Dates
using Dates: DateTime, Day, Hour, Millisecond, value

include("case.jl")

arch = GPU()
Oceananigans.defaults.FloatType = eval(Symbol(get(ENV, "SO_OCEAN_FT", "Float32")))
Δt = parse(Float64, get(ENV, "SO_OCEAN_DT", "60"))
hours_to_run = parse(Float64, get(ENV, "SO_OCEAN_HOURS", "24"))
boundaries = get(ENV, "SO_OCEAN_BOUNDARIES", "open")
name = get(ENV, "SO_NAME", "ocean_only")
rundir = joinpath(@__DIR__, "runs", name)
mkpath(rundir)

glorys_region = BoundingBox(longitude = (longitude[1] - 0.5, longitude[2] + 0.5),
                            latitude  = (latitude[1]  - 0.5, latitude[2]  + 0.5))
glorys_dates = start_date:Day(1):(stop_date + Day(1))

@info "ocean only" name Oceananigans.defaults.FloatType Δt hours_to_run boundaries

close_boundaries = boundaries == "closed"   # read by ocean.jl

include("ocean.jl")

ocean.Δt = Δt
ocean.stop_time = hours_to_run * hours

function progress(sim)
    m = sim.model
    umax, I = findmax(abs.(interior(m.velocities.u)))
    @info @sprintf("iter %5d  t = %s  max|u| = %.2f at %s  max|U| = %.1f  max|η| = %.2f  T ∈ [%.2f, %.2f]",
                   iteration(sim), prettytime(sim), umax, Tuple(I),
                   maximum(abs, m.free_surface.barotropic_velocities.U),
                   maximum(abs, m.free_surface.displacement), minimum(m.tracers.T), maximum(m.tracers.T))
    flush(stderr)
    return nothing
end

add_callback!(ocean, progress, TimeInterval(30minutes))
add_ocean_output!(ocean, ocean, rundir, 1hour)
run!(ocean)
