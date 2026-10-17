# # Regional ocean with GLORYS open boundaries (included by hindcast.jl when SO_OCEAN=prognostic)
#
# A hydrostatic Oceananigans ocean on the atmosphere's horizontal grid (Δ = 1/9° ≈ 12 km), driven
# at its open boundaries by GLORYS daily means following enri66/NumericalEarthSandbox
# scripts/02_mab_glorys_obc.jl ("native" Flather exterior):
#
#   u, v normal       NormalFlowBoundaryCondition(Interpolated(GLORYS); PerturbationAdvection)
#   u, v tangential   ValueBoundaryCondition(Interpolated(GLORYS); NormalRadiation)
#   T, S              ValueBoundaryCondition(Interpolated(GLORYS); NormalRadiation)
#   U, V barotropic   Flather (GravityWaveRadiation), fed the GLORYS depth-mean velocity over
#                     GLORYS's own levels times the model's wet depth, plus GLORYS zos
#   η                 Chapman (SurfaceWaveRadiation)
#
# Differences from the sandbox: everything that the boundary conditions read lives on the GPU —
# the Flather exterior is a parameterized discrete function over device arrays rather than a
# closure over host vectors — and the GLORYS source grid is the model grid padded by `n_pad` cells
# (the sandbox insets the model inside a fixed data box instead). Expects from hindcast.jl: `arch`,
# `longitude`, `latitude`, `Nx`, `Ny`, `cells_per_degree`, `start_date`, `glorys_dates`,
# `glorys_region`, `glorys_dir`. Defines `ocean`, `ocean_wet` and `add_ocean_output!`.

using NumericalEarth.NestedModels: Interpolated
using NumericalEarth.DataWrangling: MetadataSet, z_interfaces
using NumericalEarth.Diagnostics: MixedLayerDepthField
using Oceananigans.BoundaryConditions: PerturbationAdvection, NormalRadiation,
                                       GravityWaveRadiationBoundaryCondition,
                                       SurfaceWaveRadiationBoundaryCondition
using Oceananigans.Grids: znodes

include("glorys_bathymetry.jl")

# ## Vertical grid: 4 m in the top 32 m (storm mixing lives there), stretching by 12% per level
# to a 400 m cap, to 6.27 km (deeper trenches are truncated).

Δz = fill(4.0, 8)
while sum(Δz) < 6000
    push!(Δz, min(1.12 * Δz[end], 400.0))
end
ocean_z = vcat(-reverse(cumsum(Δz)), 0.0)
Nzᵒ = length(Δz)

host_ocean_grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, Nzᵒ), longitude, latitude,
                                        z = ocean_z, halo = (7, 7, 7))

bottom = Array(interior(regrid_bathymetry(host_ocean_grid; dataset = ETOPO2022(), height_above_water = 1,
                                          minimum_depth = 10, major_basins = 1,
                                          interpolation_passes = 10)))[:, :, 1]
match_boundary_bathymetry!(bottom, host_ocean_grid, glorys_dir; region = glorys_region, n_match = 4)
ocean_wet = bottom .< 0

underlying_ocean_grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nzᵒ), longitude, latitude,
                                              z = ocean_z, halo = (7, 7, 7))
bottom_height = Field{Center, Center, Nothing}(underlying_ocean_grid)
set!(bottom_height, reshape(bottom, Nx, Ny, 1))
ocean_grid = ImmersedBoundaryGrid(underlying_ocean_grid, GridFittedBottom(bottom_height))
@info @sprintf("ocean grid: %d×%d×%d, %d wet columns, Δz from %.1f to %.0f m", Nx, Ny, Nzᵒ,
               count(ocean_wet), first(Δz), last(Δz))

# ## GLORYS on a source grid that brackets the model's boundary faces
#
# `Interpolated` evaluates the parent at the child's boundary face, half a cell outside the
# outermost centre, so the source grid is the model grid padded by `n_pad` cells on every side.

n_pad = 2
Δ = 1 / cells_per_degree
source_longitude = (longitude[1] - n_pad * Δ, longitude[2] + n_pad * Δ)
source_latitude  = (latitude[1]  - n_pad * Δ, latitude[2]  + n_pad * Δ)
Nx_source, Ny_source = Nx + 2n_pad, Ny + 2n_pad

source_grid = LatitudeLongitudeGrid(arch; size = (Nx_source, Ny_source, Nzᵒ),
                                    longitude = source_longitude, latitude = source_latitude,
                                    z = ocean_z, halo = (7, 7, 7))

glorys = GLORYSDaily()
glorys_metadata(name) = Metadata(name; dataset = glorys, dates = glorys_dates, dir = glorys_dir,
                                 region = glorys_region)

# Every day resident: the series are a few GB on the device, and nothing has to reload a window
# inside a boundary-condition kernel.
Nt = length(glorys_dates)
glorys_series(name, grid) = FieldTimeSeries(glorys_metadata(name), grid; time_indexing = Clamp(),
                                            time_indices_in_memory = Nt)

@info "GLORYS boundary series ($Nt days)…"
u_glorys = glorys_series(:u_velocity, source_grid)
v_glorys = glorys_series(:v_velocity, source_grid)
T_glorys = glorys_series(:temperature, source_grid)
S_glorys = glorys_series(:salinity, source_grid)
glorys_times = u_glorys.times

# ## The Flather exterior: GLORYS depth-mean velocity × model wet depth, and GLORYS zos
#
# The depth mean is taken over GLORYS's own levels down to its own floor (`deptho`), so neither
# a model-z resampling nor the inpainted values below GLORYS's real bottom enter it; only the
# wet depth H is the model's, since that is the H the model's barotropic mode carries.

host_source_native_z = LatitudeLongitudeGrid(CPU(); size = (Nx_source, Ny_source, length(z_interfaces(glorys_metadata(:u_velocity))) - 1),
                                             longitude = source_longitude, latitude = source_latitude,
                                             z = z_interfaces(glorys_metadata(:u_velocity)), halo = (7, 7, 7))
host_source_grid = LatitudeLongitudeGrid(CPU(); size = (Nx_source, Ny_source, 1),
                                         longitude = source_longitude, latitude = source_latitude,
                                         z = (-1, 0), halo = (7, 7, 7))

u_native = FieldTimeSeries(glorys_metadata(:u_velocity), host_source_native_z; time_indexing = Clamp(), time_indices_in_memory = Nt)
v_native = FieldTimeSeries(glorys_metadata(:v_velocity), host_source_native_z; time_indexing = Clamp(), time_indices_in_memory = Nt)
η_native = FieldTimeSeries(glorys_metadata(:free_surface), host_source_grid; time_indexing = Clamp(), time_indices_in_memory = Nt)
glorys_floor = -glorys_deptho_on_grid(host_source_grid, glorys_dir; region = glorys_region)   # NaN where GLORYS is land

z_native_centers = collect(znodes(host_source_native_z, Center()))
Δz_native = diff(collect(znodes(host_source_native_z, Face())))
Δz_model = diff(ocean_z)
z_model_centers = (ocean_z[1:end-1] .+ ocean_z[2:end]) ./ 2

# Model wet depth: the levels the GridFittedBottom leaves wet (cell centre above the floor).
wet_depth(b) = sum(Δz_model[k] for k in 1:Nzᵒ if z_model_centers[k] > b; init = 0.0)

# Depth mean of a GLORYS column over its own valid levels (0 where GLORYS has no ocean).
function glorys_depth_mean(column, floor)
    transport = depth = 0.0
    for k in eachindex(column)
        if z_native_centers[k] > floor && isfinite(column[k])
            transport += Δz_native[k] * column[k]
            depth += Δz_native[k]
        end
    end
    return depth > 0 ? transport / depth : 0.0
end

# One boundary line of the source grid, indexed along the boundary (source indices, including
# the padding) and mapped to the model's floor at the clamped model index.
model_index(s, N) = clamp(s - n_pad, 1, N)

function flather_exterior_tables(side)
    Nb = side in (:west, :east) ? Ny_source : Nx_source
    U = zeros(Float32, Nb, Nt)
    η = zeros(Float32, Nb, Nt)
    velocity = side in (:west, :east) ? u_native : v_native

    for n in 1:Nt
        V = Array(interior(velocity[n]))
        H = Array(interior(η_native[n]))[:, :, 1]
        for s in 1:Nb
            # Source cell on the boundary line, one cell inside the padding (as the sandbox does).
            i, j = side === :west  ? (n_pad + 1, s) :
                   side === :east  ? (Nx_source - n_pad, s) :
                   side === :south ? (s, n_pad + 1) : (s, Ny_source - n_pad)
            im, jm = side in (:west, :east) ? (i - n_pad, model_index(s, Ny)) : (model_index(s, Nx), j - n_pad)
            im, jm = clamp(im, 1, Nx), clamp(jm, 1, Ny)
            ū = glorys_depth_mean(view(V, i, j, :), glorys_floor[i, j])
            U[s, n] = ū * wet_depth(bottom[im, jm])
            η[s, n] = isfinite(H[i, j]) ? H[i, j] : 0
        end
    end

    return U, η
end

# Remove the initial domain-mean GLORYS sea level from the exterior, as from the initial η below,
# so Flather does not drive a uniform setup toward GLORYS's arbitrary datum.
flather_tables = Dict(side => flather_exterior_tables(side) for side in (:west, :east, :south, :north))
# GLORYS zos comes back as exactly 0 over land, so the datum averages GLORYS ocean cells only.
η_reference = let η₀ = Array(interior(η_native[1]))[:, :, 1]
    Float32(mean(η₀[isfinite.(glorys_floor) .& isfinite.(η₀)]))
end
for side in keys(flather_tables)
    flather_tables[side][2] .-= η_reference
end

@inline function flather_exterior(ξ, k, grid, clock, model_fields, p)
    s = (clock.time - p.t₁) / p.Δτ
    n = clamp(unsafe_trunc(Int, floor(s)) + 1, 1, p.Nt - 1)
    w = clamp(s - (n - 1), zero(s), one(s))
    m = clamp(ξ + p.n_pad, 1, size(p.U, 1))
    @inbounds U = (1 - w) * p.U[m, n] + w * p.U[m, n + 1]
    @inbounds η = (1 - w) * p.η[m, n] + w * p.η[m, n + 1]
    FT = eltype(grid)
    return (convert(FT, U), convert(FT, η))
end

function flather_parameters(side)
    U, η = flather_tables[side]
    return (U = on_architecture(arch, U), η = on_architecture(arch, η), t₁ = first(glorys_times),
            Δτ = glorys_times[2] - glorys_times[1], Nt, n_pad)
end

flather(side) = GravityWaveRadiationBoundaryCondition(flather_exterior; discrete_form = true,
                                                      parameters = flather_parameters(side))

# ## Boundary conditions

τ_in = parse(Float64, get(ENV, "SO_OCEAN_TAU_IN_DAYS", "1")) * days
normal_scheme  = PerturbationAdvection(inflow_timescale = τ_in, outflow_timescale = Inf)
radiated       = NormalRadiation(inflow_timescale = τ_in, outflow_timescale = Inf)

tangential(fts) = ValueBoundaryCondition(Interpolated(fts); scheme = radiated)
tracer(fts) = FieldBoundaryConditions(west  = ValueBoundaryCondition(Interpolated(fts); scheme = radiated),
                                      east  = ValueBoundaryCondition(Interpolated(fts); scheme = radiated),
                                      south = ValueBoundaryCondition(Interpolated(fts); scheme = radiated),
                                      north = ValueBoundaryCondition(Interpolated(fts); scheme = radiated))

u_bcs = FieldBoundaryConditions(west  = NormalFlowBoundaryCondition(Interpolated(u_glorys); scheme = normal_scheme),
                                east  = NormalFlowBoundaryCondition(Interpolated(u_glorys); scheme = normal_scheme),
                                south = tangential(u_glorys),
                                north = tangential(u_glorys))

v_bcs = FieldBoundaryConditions(south = NormalFlowBoundaryCondition(Interpolated(v_glorys); scheme = normal_scheme),
                                north = NormalFlowBoundaryCondition(Interpolated(v_glorys); scheme = normal_scheme),
                                west  = tangential(v_glorys),
                                east  = tangential(v_glorys))

U_bcs = FieldBoundaryConditions(ocean_grid, (Face(), Center(), nothing); west = flather(:west), east = flather(:east))
V_bcs = FieldBoundaryConditions(ocean_grid, (Center(), Face(), nothing); south = flather(:south), north = flather(:north))
η_bcs = FieldBoundaryConditions(ocean_grid, (Center(), Center(), Face());
                                west  = SurfaceWaveRadiationBoundaryCondition(),
                                east  = SurfaceWaveRadiationBoundaryCondition(),
                                south = SurfaceWaveRadiationBoundaryCondition(),
                                north = SurfaceWaveRadiationBoundaryCondition())

ocean_boundary_conditions = (u = u_bcs, v = v_bcs, T = tracer(T_glorys), S = tracer(S_glorys),
                             U = U_bcs, V = V_bcs, η = η_bcs)

# ## The ocean model, initialized from GLORYS at the start date

ocean = ocean_simulation(ocean_grid; boundary_conditions = ocean_boundary_conditions, warn = false)

set!(ocean.model, MetadataSet(:temperature, :salinity, :u_velocity, :v_velocity;
                              dataset = glorys, date = start_date, dir = glorys_dir, region = glorys_region))
let η = ocean.model.free_surface.displacement
    set!((; free_surface = η), MetadataSet(:free_surface; dataset = glorys, date = start_date,
                                           dir = glorys_dir, region = glorys_region))
    interior(η) .-= η_reference
end

# ## Ocean output: hourly surface maps and mixed-layer depth, daily 3-D state

function add_ocean_output!(simulation, ocean, rundir, interval)
    model = ocean.model
    grid = model.grid
    Nz = size(grid, 3)
    u, v, w = model.velocities
    T, S = model.tracers.T, model.tracers.S
    mld = MixedLayerDepthField(model.buoyancy.formulation, grid, model.tracers)

    surface = (T = Field(T; indices = (:, :, Nz)), S = Field(S; indices = (:, :, Nz)),
               u = Field(u; indices = (:, :, Nz)), v = Field(v; indices = (:, :, Nz)),
               η = model.free_surface.displacement, mld = mld)

    simulation.output_writers[:ocean_surface] = JLD2Writer(model, surface; schedule = TimeInterval(interval),
                                                           filename = joinpath(rundir, "ocean_surface.jld2"),
                                                           overwrite_existing = true)

    simulation.output_writers[:ocean_daily] = JLD2Writer(model, (; u, v, T, S); schedule = TimeInterval(1day),
                                                         filename = joinpath(rundir, "ocean_daily.jld2"),
                                                         overwrite_existing = true)
    return nothing
end
