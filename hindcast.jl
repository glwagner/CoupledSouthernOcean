# # Regional coupled hindcast around New Zealand
#
# A 12 km limited-area Breeze atmosphere, nested in ERA5 through open lateral boundaries and
# Davies relaxation, over a surface split by the ocean fraction þ (NumericalEarth's
# `SurfacePartition`) into a `SlabLand` and an ocean:
#
#   SO_OCEAN=prescribed   a `PrescribedOcean` carrying GLORYS daily sea surface temperature
#   SO_OCEAN=prognostic   a 12 km hydrostatic regional ocean with GLORYS open boundaries (ocean.jl)
#
# Radiation is all-sky RRTMGP on the child, solved hourly. Configuration comes from case.jl and
# SO_* environment variables; see slurm/hindcast.batch.

using NumericalEarth
using NumericalEarth.DataWrangling: Metadata, Metadatum, BoundingBox
using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: Clamp
using Oceananigans.Architectures: on_architecture
using Breeze
using CopernicusClimateDataStore   # ERA5 + Copernicus albedo downloads
using CopernicusMarine             # GLORYS downloads
using CloudMicrophysics            # the nest's default one-moment mixed-phase microphysics
using RRTMGP                       # Breeze's radiative-transfer extension
using CUDA
using JLD2
using Printf
using Statistics: mean
import Dates   # not `using`: Dates' hour/day/second clash with Oceananigans.Units
using Dates: DateTime, Day, Hour, Second, Millisecond, value

include("case.jl")

arch = GPU()
Oceananigans.defaults.FloatType = Float32

ocean_kind = get(ENV, "SO_OCEAN", "prescribed")
ocean_kind in ("prescribed", "prognostic") || error("SO_OCEAN must be prescribed or prognostic")
closure_kind = get(ENV, "SO_CLOSURE", "tke")
name = get(ENV, "SO_NAME", "nz_$(ocean_kind)")
rundir = joinpath(@__DIR__, "runs", name)
mkpath(rundir)

@info "case" name ocean_kind closure_kind start_date stop_date longitude latitude Nx Ny

# ## Atmosphere grid
#
# 50 levels to 19.5 km with 60 m spacing at the surface (as in the AtmosphericRivers.jl and
# NumericalEarth downscaling examples), terrain-following over ETOPO2022 orography.

z = ReferenceToStretchedDiscretization(extent = 19525.0,
                                       bias = :left,
                                       bias_edge = 0,
                                       constant_spacing = 60,
                                       constant_spacing_extent = 60,
                                       maximum_spacing = 490,
                                       stretching = LinearStretching(0.15))
Nz = length(z)

grid = LatitudeLongitudeGrid(arch; longitude, latitude,
                             z = TerrainFollowingVerticalDiscretization(z),
                             size = (Nx, Ny, Nz),
                             halo = (5, 5, 5),
                             topology = (Bounded, Bounded, Bounded))

# ## The nest
#
# Without a boundary-layer closure the surface fluxes pile into the lowest 60 m cell (the ENA
# hindcast's drizzling fog), so the child carries Breeze's `TKEBasedTurbulenceClosure`,
# vertically implicit. Its TKE density ρe has no ERA5 counterpart: it is advected with plain
# WENO(5) and has neither lateral values nor a Davies target.

dataset = ERA5HourlyPressureLevels()
dates = (start_date, stop_date)

breeze_extension = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
microphysics = breeze_extension.default_nested_microphysics()
scalar_advection = breeze_extension.default_nested_scalar_advection(microphysics)

closure_kw = if closure_kind == "tke"
    closure = TKEBasedTurbulenceClosure(Float32)
    scalar_advection = merge(scalar_advection, (; ρe = WENO(order = 5)))
    (; closure)
elseif closure_kind == "none"
    NamedTuple()
else
    error("SO_CLOSURE must be tke or none")
end

# SO_DIVERGENCE_DAMPING=α: acoustic divergence damping (Klemp, Skamarock & Ha 2018) with coefficient α
# (MPAS's 0.1). NumericalEarth's default child has none, and whole columns over steep terrain went NaN
# (CFL 0.5 at 8 h, CFL 0.4 at 3 days). The direct form avoids the thermal proxy's cold-start force, the
# reason NumericalEarth turned damping off. The lid sponge and base pressure are NumericalEarth's defaults.
divergence_damping = parse(Float64, get(ENV, "SO_DIVERGENCE_DAMPING", "0"))
dynamics_kw = if divergence_damping > 0
    base_pressure = breeze_extension.mean_sea_level_pressure(dataset, grid, start_date, era5_dir)
    sponge = UpperSponge(; damping_rate = 1/5, depth = breeze_extension.default_lid_depth(grid))
    damping = DirectDivergenceDamping(coefficient = divergence_damping)
    (; base_pressure, dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(; sponge, damping); base_pressure))
else
    NamedTuple()
end

nest = nested_atmosphere_model(grid, dataset;
                               dates,
                               dir = era5_dir,
                               terrain = ETOPO2022(),
                               relaxation_rate = 1/300,
                               relaxation_width = 5,
                               parent_time_indices_in_memory = 4,
                               parent_hydrometeors,   # case.jl: cloud liquid and ice by default
                               microphysics,
                               momentum_advection = WENO(order = 5),
                               scalar_advection,
                               # Float64 time on the Float32 grid (NE's default clock is Float32,
                               # 0.25 s resolution at 4 weeks); also matches the ocean's clock
                               clock = Clock(grid),
                               closure_kw...,
                               dynamics_kw...)

child = nest.child
# The parent's download box (the child box padded as in nested_atmosphere_model), so the land's
# skin-temperature snapshot is the file predownload.jl fetches.
era5_region = BoundingBox(grid; padding = NumericalEarth.DataWrangling.default_horizontal_padding(dataset))

# ## Surface: land, ocean, and the partition between them

surface_grid = LatitudeLongitudeGrid(arch; longitude, latitude,
                                     size = (Nx, Ny),
                                     halo = (5, 5),
                                     topology = (Bounded, Bounded, Flat))

land = SlabLand(surface_grid)
set!(land.temperature, Metadatum(:skin_temperature; dataset = ERA5HourlySingleLevel(),
                                 date = start_date, region = era5_region, dir = era5_dir))
set!(land; M = 20)

# GLORYS covers a box padded beyond the model domain so that its cell centres bracket every
# model face (needed by the open-boundary interpolation in ocean.jl, harmless here).
glorys_region = BoundingBox(longitude = (longitude[1] - 0.5, longitude[2] + 0.5),
                            latitude  = (latitude[1]  - 0.5, latitude[2]  + 0.5))

# GLORYS daily means are stamped at 00 UTC but represent the whole day; NumericalEarth places
# each at noon, measured from the first date. That first date must be the model's t = 0, so the
# series starts on `start_date` (00 UTC) and is held at its first value (Clamp) until noon.
Dates.hour(start_date) == 0 || error("SO_START must be at 00 UTC so GLORYS days align with t = 0")
glorys_dates = start_date:Day(1):(stop_date + Day(1))
glorys_times = [value(Millisecond(d + Hour(12) - start_date)) / 1000 for d in glorys_dates]

if ocean_kind == "prescribed"
    # Sea surface temperature from the top GLORYS level, interpolated in time between daily means.
    top_layer = LatitudeLongitudeGrid(arch; longitude, latitude, size = (Nx, Ny, 1),
                                      halo = (5, 5, 5), z = (-1, 0))
    T_top = CenterField(top_layer)
    sst = FieldTimeSeries{Center, Center, Nothing}(surface_grid, glorys_times; time_indexing = Clamp())
    for (n, date) in enumerate(glorys_dates)
        set!(T_top, Metadatum(:temperature; dataset = GLORYSDaily(), date,
                              region = glorys_region, dir = glorys_dir))
        interior(sst[n]) .= interior(T_top) .+ 273.15f0
    end
    ocean = PrescribedOcean(surface_grid, glorys_times; sea_surface_temperature = sst)
else
    include("ocean.jl")   # defines `ocean` and `ocean_wet` from the regional ocean grid
end

# Ocean fraction þ from ETOPO2022 (the dataset of the child's terrain). Land-locked basins below
# sea level (the Fiordland lakes' beds reach −250 m) read as ocean in the raw fraction, so þ is
# zeroed wherever the ocean bathymetry — connected basin only — is dry.
ocean_fraction = regrid_ocean_fraction(surface_grid; dataset = ETOPO2022())
if ocean_kind == "prescribed"
    ocean_bathymetry_grid = LatitudeLongitudeGrid(arch; longitude, latitude, size = (Nx, Ny, 1),
                                                  halo = (5, 5, 5), z = (-1, 0))
    bottom = regrid_bathymetry(ocean_bathymetry_grid; dataset = ETOPO2022(),
                               minimum_depth = 10, major_basins = 1)
    ocean_wet = Array(interior(bottom))[:, :, 1] .< 0
end
interior(ocean_fraction) .*= on_architecture(arch, Float32.(reshape(ocean_wet, Nx, Ny, 1)))
þ_host = Array(interior(ocean_fraction))
@info @sprintf("ocean fraction: mean %.3f, %d wet columns of %d, %d fractional",
               mean(þ_host), count(ocean_wet), Nx * Ny, count(x -> 0 < x < 1, þ_host))

# ## Radiation

radiation = RadiativeTransferModel(grid, AllSkyOptics(), child.thermodynamic_constants;
                                   solar_position = ApparentSolarPosition(epoch = start_date),
                                   surface_albedo = CopernicusAlbedo(),
                                   schedule = TimeInterval(1hour))

# ## Coupled model

Δt = parse(Float64, get(ENV, "SO_DT", "10"))
atmosphere = Simulation(nest; Δt)   # the coupled model manages Δt; this sets only the initial value
model = EarthSystemModel(; atmosphere, land, ocean, radiation, ocean_fraction)

simulation = Simulation(model; Δt, stop_time = run_hours * hours)
conjure_time_step_wizard!(simulation, IterationInterval(3);
                          cfl = parse(Float64, get(ENV, "SO_CFL", "0.5")),
                          max_Δt = parse(Float64, get(ENV, "SO_MAX_DT", string(Δt))))

haskey(ENV, "SO_STOP_ITERATION") && (simulation.stop_iteration = parse(Int, ENV["SO_STOP_ITERATION"]))

# ## Output
#
# Hourly 2-D maps for the animation: near-surface wind and θᵛ, surface pressure, precipitation,
# column water vapour, the skin temperature the atmosphere sees, and the surface heat fluxes.

u, v, w = child.velocities
ρ = child.dynamics.total_density
qᵛ = specific_humidity(child)
p = Breeze.AtmosphereModels.dynamics_pressure(child.dynamics)
k_aloft = searchsortedfirst(Array(znodes(grid, Center())), 1500)

ao = model.interfaces.atmosphere_ocean_interface.fluxes
al = model.interfaces.atmosphere_land_interface.fluxes

surface_fields = (u = Field(u; indices = (:, :, 1)),
                  v = Field(v; indices = (:, :, 1)),
                  θᵛ = Field(VirtualPotentialTemperature(child); indices = (:, :, 1)),
                  p = Field(p; indices = (:, :, 1)),
                  w_aloft = Field(w; indices = (:, :, k_aloft)),
                  precipitation = Field(bottom_precipitation_flux(child)),
                  pw = Field(Integral(ρ * qᵛ, dims = 3)),
                  Tₛ = surface_temperature(model.interfaces),
                  Qᵒᶜ_sensible = ao.sensible_heat,
                  Qᵒᶜ_latent = ao.latent_heat,
                  τˣ = ao.x_momentum,
                  τʸ = ao.y_momentum)

output_interval = parse(Float64, get(ENV, "SO_OUTPUT_MINUTES", "60")) * minutes
simulation.output_writers[:surface] = JLD2Writer(model, surface_fields;
                                                 schedule = TimeInterval(output_interval),
                                                 filename = joinpath(rundir, "surface.jld2"),
                                                 overwrite_files = true)

# SO_OCEAN_OUTPUT=false skips the ocean files (each carries the serialized GLORYS boundary conditions,
# GBs) for diagnostic replays. Their hourly schedule coincides with the surface file's, so the time
# steps, which align to output times, are unchanged within the first day.
ocean_kind == "prognostic" && get(ENV, "SO_OCEAN_OUTPUT", "true") == "true" &&
    add_ocean_output!(simulation, ocean, rundir, output_interval)

jldsave(joinpath(rundir, "static.jld2");
        ocean_fraction = Array(interior(ocean_fraction))[:, :, 1],
        λ =Array(λnodes(surface_grid, Center())), φ = Array(φnodes(surface_grid, Center())),
        start_date)

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    wall_clock[] = time_ns()
    child = sim.model.atmosphere.model.child
    u, v, w = child.velocities
    qᵛ = specific_humidity(child)
    Tₛ = surface_temperature(sim.model.interfaces)
    msg = @sprintf("iter %6d  t = %s (%s)  Δt = %5.2f s  wall %6.1f s  max|u,v,w| = (%5.1f, %5.1f, %5.2f)  qᵛ ∈ [%.2e, %.2e]  Tₛ ∈ [%.1f, %.1f]",
                   iteration(sim), prettytime(sim), start_date + Millisecond(round(Int, 1000 * sim.model.clock.time)),
                   sim.Δt, elapsed, maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   minimum(qᵛ), maximum(qᵛ), minimum(Tₛ), maximum(Tₛ))
    if ocean_kind == "prognostic"
        oc = sim.model.ocean.model
        # Where the fastest current is (i, j, k), to catch boundary or shallow-water trouble early
        umax, I = findmax(abs.(interior(oc.velocities.u)))
        U = oc.free_surface.barotropic_velocities.U
        msg *= @sprintf("  ocean max|u| = %.2f at %s  max|U| = %.1f  max|η| = %.2f  T ∈ [%.2f, %.2f]",
                        umax, Tuple(I), maximum(abs, U), maximum(abs, oc.free_surface.displacement),
                        minimum(oc.tracers.T), maximum(oc.tracers.T))
    end
    @info msg
    flush(stderr)   # a log file on the shared filesystem is otherwise written only when Julia yields
    return nothing
end

add_callback!(simulation, progress, IterationInterval(parse(Int, get(ENV, "SO_PROGRESS", "180"))))

# SO_NAN_CHECK_AFTER_HOURS: from that time on, check every step for a NaN or Inf in the atmosphere prognostics,
# land, ocean and interface fluxes (cheap: device reductions and copies); on one, report where, save the
# last clean state and the broken one to nan_dump.jld2, and stop the run.
function first_nan(name, field)
    A = Array(interior(field))
    n = count(isnan, A)
    n == 0 && return nothing
    return @sprintf("%s: %d NaN, first at %s", name, n, Tuple(findfirst(isnan, A)))
end

function nan_check_candidates(sim)
    child = sim.model.atmosphere.model.child
    candidates = Any[("atmosphere $name", f) for (name, f) in pairs(Oceananigans.fields(child)) if f isa Field]
    append!(candidates, [("land T", land.temperature), ("land water", land.water_storage), ("surface Tₛ", surface_temperature(sim.model.interfaces)),
                         ("ao sensible", ao.sensible_heat), ("ao latent", ao.latent_heat), ("ao τˣ", ao.x_momentum), ("ao τʸ", ao.y_momentum)])
    append!(candidates, [("al $name", getproperty(al, name)) for name in propertynames(al) if getproperty(al, name) isa Field])
    if ocean_kind == "prognostic"
        oc = sim.model.ocean.model
        append!(candidates, [("ocean T", oc.tracers.T), ("ocean S", oc.tracers.S), ("ocean u", oc.velocities.u),
                             ("ocean η", oc.free_surface.displacement)])
    end
    return candidates
end

# The cheap part, every step: one reduction per field on the device, and a device copy of each field
# while all are finite, so the last clean state survives the step that breaks.
function nan_check(sim)
    candidates = nan_check_candidates(sim)
    if isnothing(previous_state[])
        previous_state[] = [copy(parent(f)) for (_, f) in candidates]
    end
    if all(((_, f),) -> isfinite(sum(f)), candidates)
        foreach(((saved, (_, f)),) -> copyto!(saved, parent(f)), zip(previous_state[], candidates))
        return nothing
    end
    return nan_report(sim, candidates)
end

function nan_report(sim, candidates)
    child = sim.model.atmosphere.model.child
    reports = filter(!isnothing, [first_nan(name, f) for (name, f) in candidates])
    w = Array(interior(child.velocities.w))
    ρᵈ = Array(interior(child.dynamics.dry_density))
    wmax, I = findmax(abs.(replace(w, NaN32 => 0)))
    ρmin, J = findmin(replace(ρᵈ, NaN32 => Inf32))
    msg = @sprintf("nan check: iter %d  t = %s  Δt = %.2f  max|w| = %.2f at %s  min ρᵈ = %.4f at %s  land T ∈ [%.1f, %.1f]",
                   iteration(sim), prettytime(sim), sim.Δt, wmax, Tuple(I), ρmin, Tuple(J), extrema(Array(interior(land.temperature)))...)
    if haskey(Oceananigans.fields(child), :ρe)
        ρe = replace(Array(interior(Oceananigans.fields(child).ρe)), NaN32 => 0)
        (emin, K), (emax, L) = findmin(ρe), findmax(ρe)
        msg *= @sprintf("  ρe ∈ [%.3e at %s, %.3e at %s]", emin, Tuple(K), emax, Tuple(L))
    end
    @info msg
    foreach(r -> @error(r), reports)
    @error "NaN or Inf at iteration $(iteration(sim)), t = $(prettytime(sim))"
    # The last clean state and the first broken one (parent arrays, halos included), for working out
    # where the NaN started
    names = [name for (name, _) in candidates]
    jldsave(joinpath(rundir, "nan_dump.jld2"); names,
            previous = [Array(a) for a in previous_state[]], current = [Array(parent(f)) for (_, f) in candidates],
            iteration = iteration(sim), Δt = sim.Δt)
    sim.running = false
    flush(stderr)
    return nothing
end

const previous_state = Ref{Any}(nothing)

# SO_SYNC_EVERY_STEP=true: wait for the GPU after every step. coupled_28d (Jan 29) hit a NaN Δt at
# 7.96 h twice without it, and ran clean past that point when a per-step check synchronized the GPU.
get(ENV, "SO_SYNC_EVERY_STEP", "false") == "true" &&
    add_callback!(simulation, sim -> CUDA.synchronize(), IterationInterval(1))

if haskey(ENV, "SO_NAN_CHECK_AFTER_HOURS")
    nan_check_start = parse(Float64, ENV["SO_NAN_CHECK_AFTER_HOURS"]) * hours
    add_callback!(simulation, sim -> sim.model.clock.time ≥ nan_check_start && nan_check(sim), IterationInterval(1))
end

haskey(ENV, "SO_DEBUG_ITERATION") && include("debug_step.jl")

run!(simulation)
@info "done" name
