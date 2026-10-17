using NumericalEarth, Oceananigans, CUDA
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.EarthSystemModels.InterfaceComputations: computed_fluxes
import NumericalEarth.Oceans as O
import NumericalEarth.EarthSystemModels as E

# Print the argument types `update_net_ocean_fluxes!` would pass to `_assemble_net_ocean_fluxes!`.
function O.update_net_ocean_fluxes!(cm, ocean_model, grid::Oceananigans.Grids.AbstractGrid)
    land_exchanger = cm.interfaces.exchanger.land
    args = (net = cm.interfaces.net_fluxes.ocean,
            atmos_ocean = O.atmos_ocean_flux(cm),
            sea_ice_ocean = computed_fluxes(cm.interfaces.sea_ice_ocean_interface),
            sst = E.ocean_surface_temperature(ocean_model),
            ℵ = E.sea_ice_concentration(cm.sea_ice),
            rain = O.rainfall_flux(cm), snow = O.snowfall_flux(cm),
            intercepted = E.intercepted_snowfall(cm.sea_ice),
            runoff = O.land_freshwater_flux(land_exchanger, :runoff_freshwater_flux),
            iceberg = O.land_freshwater_flux(land_exchanger, :iceberg_freshwater_flux),
            props = cm.interfaces.ocean_properties)
    for (k, v) in pairs(args)
        println(k, " :: ", typeof(v))
    end
    flush(stdout)
    return nothing
end

kw = (longitude = (0, 4), latitude = (0, 4))
ocean_grid = LatitudeLongitudeGrid(GPU(); size = (4, 4, 4), kw..., z = (-100, 0))
surface_grid = LatitudeLongitudeGrid(GPU(); size = (4, 4, 1), kw..., z = (-1, 0))
ocean = ocean_simulation(ocean_grid; momentum_advection = nothing, tracer_advection = nothing,
                         closure = nothing, coriolis = nothing, warn = false)
land = SlabLand(surface_grid)
atmosphere = PrescribedAtmosphere(surface_grid; surface_layer_height = 10, boundary_layer_height = 512)
for (with_land, l) in ((false, nothing), (true, land))
    println("==== land = ", with_land)
    model = with_land ? EarthSystemModel(; atmosphere, land = l, ocean, ocean_fraction = 0.5) :
                        EarthSystemModel(; atmosphere, ocean)
end
