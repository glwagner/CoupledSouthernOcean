using NumericalEarth, Oceananigans, CUDA
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.Atmospheres: PrescribedAtmosphere

variant = ARGS[1]   # "default" ocean_simulation or "stripped" (no advection / closure / coriolis)
kw = (longitude = (0, 4), latitude = (0, 4))
ocean_grid = LatitudeLongitudeGrid(GPU(); size = (4, 4, 4), kw..., z = (-100, 0))
surface_grid = LatitudeLongitudeGrid(GPU(); size = (4, 4, 1), kw..., z = (-1, 0))
ocean = variant == "default" ? ocean_simulation(ocean_grid; warn = false) :
        ocean_simulation(ocean_grid; momentum_advection = nothing, tracer_advection = nothing,
                         closure = nothing, coriolis = nothing, warn = false)
set!(ocean.model; T = 15, S = 35)
land = SlabLand(surface_grid); set!(land; T = 290)
atmosphere = PrescribedAtmosphere(surface_grid; surface_layer_height = 10, boundary_layer_height = 512)
fill!(parent(atmosphere.temperature), 285.0); fill!(parent(atmosphere.specific_humidity), 0.005)
fill!(parent(atmosphere.velocities.u), 8.0); fill!(parent(atmosphere.pressure), 101325.0)
model = EarthSystemModel(; atmosphere, land, ocean, ocean_fraction = 0.5)
time_step!(model, 60)
@show variant extrema(Array(interior(model.interfaces.net_fluxes.ocean.T)))
