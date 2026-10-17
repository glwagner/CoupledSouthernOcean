# Bisect the NVPTX crash in `_assemble_net_ocean_fluxes!`: build the coupled model with the
# assembly stubbed out (stashing its arguments), then compile a copy of the kernel with one
# piece removed, selected by ARGS[1]. Each variant runs in its own process (the crash is a segfault).
using NumericalEarth, Oceananigans, CUDA
using NumericalEarth.Oceans: @kernel, @index
using Oceananigans.Grids: inactive_node, Center
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Utils: launch!
using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.EarthSystemModels.InterfaceComputations: computed_fluxes
import NumericalEarth.Oceans as O
import NumericalEarth.EarthSystemModels as E

const variant = Symbol(ARGS[1])
const stash = Ref{Any}(nothing)

function O.update_net_ocean_fluxes!(cm, ocean_model, grid::Oceananigans.Grids.AbstractGrid)
    land_exchanger = cm.interfaces.exchanger.land
    stash[] = (cm.interfaces.net_fluxes.ocean, grid, cm.clock, O.atmos_ocean_flux(cm),
               computed_fluxes(cm.interfaces.sea_ice_ocean_interface),
               E.ocean_surface_temperature(ocean_model), E.sea_ice_concentration(cm.sea_ice),
               O.rainfall_flux(cm), O.snowfall_flux(cm), E.intercepted_snowfall(cm.sea_ice),
               O.land_freshwater_flux(land_exchanger, :runoff_freshwater_flux),
               O.land_freshwater_flux(land_exchanger, :iceberg_freshwater_flux),
               cm.interfaces.ocean_properties)
    return nothing
end

@inline τᶜᶜᶜ(i, j, k, grid, ρᵒᶜ⁻¹, ℵ, ρτᶜᶜᶜ) = @inbounds ρᵒᶜ⁻¹ * (1 - ℵ[i, j, k]) * ρτᶜᶜᶜ[i, j, k]

@kernel function _bisect!(net_ocean_fluxes, grid, clock, atmos_ocean_fluxes, sea_ice_ocean_fluxes,
                          ocean_surface_temperature, sea_ice_concentration, rainfall_flux, snowfall_flux,
                          intercepted_snowfall_flux, runoff_freshwater_flux, iceberg_freshwater_flux,
                          ocean_properties, ::Val{v}) where v
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    ρτˣᵃᵒ = atmos_ocean_fluxes.x_momentum
    ρτʸᵃᵒ = atmos_ocean_fluxes.y_momentum
    ρτˣⁱᵒ = sea_ice_ocean_fluxes.x_momentum
    ρτʸⁱᵒ = sea_ice_ocean_fluxes.y_momentum
    ρλˣⁱᵒ = sea_ice_ocean_fluxes.x_momentum_coefficient
    ρλʸⁱᵒ = sea_ice_ocean_fluxes.y_momentum_coefficient

    @inbounds begin
        ℵᵢ = sea_ice_concentration[i, j, 1]
        Tᵒᶜ = v === :nosst ? 0.0 : ocean_surface_temperature[i, j, 1]
        Jʳⁿ = v === :norain ? 0.0 : rainfall_flux[i, j, 1]
        Jˢⁿ = v === :norain ? 0.0 : snowfall_flux[i, j, 1]
        Pˢⁿ = intercepted_snowfall_flux[i, j, 1]
        Jˡⁿ = runoff_freshwater_flux[i, j, 1]
        Jⁱᵇ = iceberg_freshwater_flux[i, j, 1]
        𝒬ᵀ = atmos_ocean_fluxes.sensible_heat[i, j, 1]
        𝒬ᵛ = atmos_ocean_fluxes.latent_heat[i, j, 1]
        Jᵛ = atmos_ocean_fluxes.water_vapor[i, j, 1]
    end

    ℒᶠ = ocean_properties.latent_heat_of_fusion
    ΣQao = (𝒬ᵀ + 𝒬ᵛ) * (1 - ℵᵢ) + ℒᶠ * (Jˢⁿ - Pˢⁿ + Jⁱᵇ)
    ρᵒᶜ⁻¹ = 1 / ocean_properties.reference_density
    ΣFao  = - (Jʳⁿ + Jˡⁿ + Jⁱᵇ + Jˢⁿ - Pˢⁿ) * ρᵒᶜ⁻¹ + (1 - ℵᵢ) * Jᵛ * ρᵒᶜ⁻¹
    Jʷao  = - ΣFao

    τˣ = net_ocean_fluxes.u
    τʸ = net_ocean_fluxes.v
    λˣ = net_ocean_fluxes.u_coefficient
    λʸ = net_ocean_fluxes.v_coefficient
    Jᵀ = net_ocean_fluxes.T
    Jˢ = net_ocean_fluxes.S
    Jʷ = net_ocean_fluxes.η
    Jᴴ = net_ocean_fluxes.freshwater_heat_content
    ℵ  = sea_ice_concentration
    cᵒᶜ⁻¹ = 1 / ocean_properties.heat_capacity
    inactive = v === :noinactive ? false :
               v === :ijonly ? (i < 1) | (i > size(grid, 1)) | (j < 1) | (j > size(grid, 2)) :
               inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    @inbounds begin
        𝒬ⁱⁿ = sea_ice_ocean_fluxes.interface_heat[i, j, 1]
        Jˢio = sea_ice_ocean_fluxes.salt[i, j, 1]
        Jʷio = sea_ice_ocean_fluxes.freshwater[i, j, 1]
        Jᴴio = sea_ice_ocean_fluxes.freshwater_heat_content[i, j, 1]
        Jᵀao = ΣQao * ρᵒᶜ⁻¹ * cᵒᶜ⁻¹
        Jᵀio =  𝒬ⁱⁿ * ρᵒᶜ⁻¹ * cᵒᶜ⁻¹
        if v === :nointerp
            τˣᵃᵒ = ρτˣᵃᵒ[i, j, 1] * ρᵒᶜ⁻¹
            τʸᵃᵒ = ρτʸᵃᵒ[i, j, 1] * ρᵒᶜ⁻¹
        else
            τˣᵃᵒ = ℑxᶠᵃᵃ(i, j, 1, grid, τᶜᶜᶜ, ρᵒᶜ⁻¹, ℵ, ρτˣᵃᵒ)
            τʸᵃᵒ = ℑyᵃᶠᵃ(i, j, 1, grid, τᶜᶜᶜ, ρᵒᶜ⁻¹, ℵ, ρτʸᵃᵒ)
        end
        τˣⁱᵒ = ρτˣⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑxᶠᵃᵃ(i, j, 1, grid, ℵ)
        τʸⁱᵒ = ρτʸⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑyᵃᶠᵃ(i, j, 1, grid, ℵ)

        if v === :branch
            if inactive
                τˣ[i, j, 1] = zero(grid); τʸ[i, j, 1] = zero(grid)
                Jᵀ[i, j, 1] = zero(grid); Jˢ[i, j, 1] = zero(grid)
                Jʷ[i, j, 1] = zero(grid); Jᴴ[i, j, 1] = zero(grid)
            else
                τˣ[i, j, 1] = τˣᵃᵒ + τˣⁱᵒ; τʸ[i, j, 1] = τʸᵃᵒ + τʸⁱᵒ
                Jᵀ[i, j, 1] = Jᵀao + Jᵀio; Jˢ[i, j, 1] = Jˢio
                Jʷ[i, j, 1] = Jʷao + Jʷio; Jᴴ[i, j, 1] = Tᵒᶜ * Jʷao + Jᴴio
            end
        else
        τˣ[i, j, 1] = ifelse(inactive, zero(grid), τˣᵃᵒ + τˣⁱᵒ)
        τʸ[i, j, 1] = ifelse(inactive, zero(grid), τʸᵃᵒ + τʸⁱᵒ)
        if v !== :noimplicit && !isnothing(λˣ)
            λˣ[i, j, 1] = ifelse(inactive, zero(grid), ρλˣⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑxᶠᵃᵃ(i, j, 1, grid, ℵ))
            λʸ[i, j, 1] = ifelse(inactive, zero(grid), ρλʸⁱᵒ[i, j, 1] * ρᵒᶜ⁻¹ * ℑyᵃᶠᵃ(i, j, 1, grid, ℵ))
        end
        Jᵀ[i, j, 1] = ifelse(inactive, zero(grid), Jᵀao + Jᵀio)
        Jˢ[i, j, 1] = ifelse(inactive, zero(grid), Jˢio)
        Jʷ[i, j, 1] = ifelse(inactive, zero(grid), Jʷao + Jʷio)
        if v !== :noheatcontent
            Jᴴ[i, j, 1] = ifelse(inactive, zero(grid), Tᵒᶜ * Jʷao + Jᴴio)
        end
        end
    end
end

kw = (longitude = (0, 4), latitude = (0, 4))
ocean_grid = LatitudeLongitudeGrid(GPU(); size = (8, 8, 4), kw..., z = (-100, 0), halo = (6, 6, 3))
surface_grid = LatitudeLongitudeGrid(GPU(); size = (8, 8, 1), kw..., z = (-1, 0))
ocean = ocean_simulation(ocean_grid; warn = false)
land = SlabLand(surface_grid)
atmosphere = PrescribedAtmosphere(surface_grid; surface_layer_height = 10, boundary_layer_height = 512)
model = EarthSystemModel(; atmosphere, land, ocean, ocean_fraction = 0.5)

args = stash[]
println("variant $variant: sea-ice fluxes ", typeof(args[5]).name.name, ", λ ", typeof(args[1].u_coefficient))
flush(stdout)
launch!(GPU(), args[2], :xy, _bisect!, args..., Val(variant))
CUDA.synchronize()
println("variant $variant: compiled and ran"); flush(stdout)
