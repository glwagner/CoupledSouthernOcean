# The saturation adjustment and θˡⁱ → T inversion of the cell where the coupled run first goes NaN
# (diag_step, iteration 6396, stage 1, (20, 262, 14)), evaluated on the CPU and on the GPU from the
# same Float32 inputs (analysis/debug_cell.jl reads them from debug_columns.jld2).
#   julia --project tests/gpu_cell.jl runs/diag_step/debug_columns.jld2 "stage 1 implicit_substep!" 14
using JLD2
using Printf
using Oceananigans
using Breeze
using CloudMicrophysics
using NumericalEarth
using CUDA

include(joinpath(@__DIR__, "..", "breeze_fixes.jl"))

Oceananigans.defaults.FloatType = Float32
const BT = Breeze.Thermodynamics
const BM = Breeze.Microphysics
const AM = Breeze.AtmosphereModels

file, label, k = ARGS[1], ARGS[2], parse(Int, ARGS[3])
labels, columns, constants = jldopen(f -> (f["labels"], f["columns"], f["constants"]), file)
c = columns[findfirst(==(label), labels)]
column(names...) = c[names[findfirst(n -> haskey(c, n), names)]]

ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
microphysics = ext.default_nested_microphysics()
newton = Breeze.Solvers.NewtonSolver(Float32)
pˢᵗ = 1f5

# The two kernels of update_state! for one cell: saturation adjustment from the previous condensate
# (_compute_auxiliary_thermodynamic_variables!), then T and p from the adjusted condensate
# (_compute_temperature_and_pressure!).
@inline function diagnose(ρᵈ, ρθ, ρqᵉ, ρqʳ, ρqˢⁿ, qᶜˡ, qᶜⁱ, microphysics, constants, solver, pˢᵗ)
    ρ = ρᵈ + ρqᵉ + ρqʳ + ρqˢⁿ
    θ = ρθ / ρᵈ
    qᵉ = ρqᵉ / ρ
    qʳ, qˢⁿ = ρqʳ / ρ, ρqˢⁿ / ρ
    q₀ = BT.MoistureMassFractions(qᵉ - qᶜˡ - qᶜⁱ, qᶜˡ + qʳ, qᶜⁱ + qˢⁿ)
    𝒰₀ = BT.LiquidIceDensityState(θ, q₀, pˢᵗ, ρ, solver)
    𝒰₁ = AM.maybe_adjust_thermodynamic_state(𝒰₀, microphysics, qᵉ, constants, (; ρqʳ, ρqˢⁿ), ρ)
    T₁ = BT.temperature(𝒰₁, constants)
    q₁ = 𝒰₁.moisture_mass_fractions
    𝒰₂ = BT.LiquidIceDensityState(θ, q₁, pˢᵗ, ρ, solver)
    T₂ = BT.temperature(𝒰₂, constants)
    return (T₁, T₂, q₁.vapor, q₁.liquid, q₁.ice)
end

# A function barrier, so the GPU closure captures concretely typed arguments rather than globals
gpu_diagnose(args, microphysics, constants, solver, pˢᵗ) =
    Array(map(a -> diagnose(a..., microphysics, constants, solver, pˢᵗ), CuArray([args])))[1]

for kk in (k,)
    args = (column("fields.ρᵈ", "dynamics.dry_density")[kk], column("fields.ρθ")[kk], column("fields.ρqᵉ")[kk],
            column("fields.ρqʳ")[kk], column("fields.ρqˢⁿ")[kk], column("fields.qᶜˡ")[kk], column("fields.qᶜⁱ")[kk])
    cpu = diagnose(args..., microphysics, constants, newton, pˢᵗ)
    gpu = gpu_diagnose(args, microphysics, constants, newton, pˢᵗ)
    @printf("k = %2d  CPU T = %.4f %.4f q = %.4e %.4e %.4e\n", kk, cpu...)
    @printf("        GPU T = %.4f %.4f q = %.4e %.4e %.4e\n", gpu...)
end

# The adjustment step by step (adjust_thermodynamic_state and secant_solve, same expressions), written
# to `out`: T₁, qᵛ⁺₁, T₂, r₁, r₂, then (T, r, slope) for 10 secant iterations.
@inline function trace!(out, ρᵈ, ρθ, ρqᵉ, ρqʳ, ρqˢⁿ, qᶜˡ, qᶜⁱ, microphysics, constants, solver, pˢᵗ)
    ρ = ρᵈ + ρqᵉ + ρqʳ + ρqˢⁿ
    θ = ρθ / ρᵈ
    qᵉ = ρqᵉ / ρ
    qʳ, qˢⁿ = ρqʳ / ρ, ρqˢⁿ / ρ
    q₀ = BT.MoistureMassFractions(qᵉ - qᶜˡ - qᶜⁱ, qᶜˡ + qʳ, qᶜⁱ + qˢⁿ)
    𝒰₀ = BT.LiquidIceDensityState(θ, q₀, pˢᵗ, ρ, solver)
    𝒰₁ = BT.with_moisture(𝒰₀, BT.MoistureMassFractions(qᵉ, qʳ, qˢⁿ))
    adjustment = microphysics.cloud_formation
    equilibrium = adjustment.equilibrium
    qᵗ = BT.total_specific_moisture(𝒰₁)
    𝒰ᵃ = BT.with_moisture(𝒰₁, BT.MoistureMassFractions(qᵗ - qʳ - qˢⁿ, qʳ, qˢⁿ))
    T₁ = BT.temperature(𝒰ᵃ, constants)
    qᵛ⁺₁ = BT.saturation_specific_humidity(T₁, ρ, constants, equilibrium)
    𝒰s = BM.adjust_state(𝒰₁, T₁, constants, equilibrium, (qʳ, qˢⁿ))
    q̃ = 𝒰s.moisture_mass_fractions
    cᵖᵐ = BT.mixture_heat_capacity(q̃, constants)
    ΔT = (constants.liquid.reference_latent_heat * (q̃.liquid - qʳ) + constants.ice.reference_latent_heat * (q̃.ice - qˢⁿ)) / cᵖᵐ
    T₂ = T₁ + max(0.01f0, ΔT / 2)
    residual(T) = BM.saturation_adjustment_residual(T, 𝒰₁, constants, equilibrium, (qʳ, qˢⁿ))
    x₁, x₂ = T₁, T₂
    r₁, r₂ = residual(x₁), residual(x₂)
    @inbounds begin
        out[1], out[2], out[3], out[4], out[5] = T₁, qᵛ⁺₁, T₂, r₁, r₂
        for n in 1:10
            s = (x₂ - x₁) / (r₂ - r₁)
            s = ifelse(isfinite(s), s, zero(s))
            x₁, r₁ = x₂, r₂
            x₂ -= r₂ * s
            r₂ = residual(x₂)
            out[3n + 3], out[3n + 4], out[3n + 5] = x₂, r₂, s
        end
    end
    return nothing
end

function trace_kernel!(out, args, microphysics, constants, solver, pˢᵗ)
    trace!(out, args..., microphysics, constants, solver, pˢᵗ)
    return nothing
end

function gpu_trace(args, microphysics, constants, solver, pˢᵗ)
    out = CUDA.zeros(Float32, 35)
    @cuda threads=1 trace_kernel!(out, args, microphysics, constants, solver, pˢᵗ)
    return Array(out)
end

function cpu_trace(args, microphysics, constants, solver, pˢᵗ)
    out = zeros(Float32, 35)
    trace!(out, args..., microphysics, constants, solver, pˢᵗ)
    return out
end

args = (column("fields.ρᵈ", "dynamics.dry_density")[k], column("fields.ρθ")[k], column("fields.ρqᵉ")[k],
        column("fields.ρqʳ")[k], column("fields.ρqˢⁿ")[k], column("fields.qᶜˡ")[k], column("fields.qᶜⁱ")[k])
for (where, t) in (("CPU", cpu_trace(args, microphysics, constants, newton, pˢᵗ)),
                   ("GPU", gpu_trace(args, microphysics, constants, newton, pˢᵗ)))
    @printf("%s  T₁ = %.6f  qᵛ⁺₁ = %.6e  T₂ = %.6f  r₁ = % .6e  r₂ = % .6e\n", where, t[1:5]...)
    for n in 1:10
        @printf("     iter %2d  T = %.6f  r = % .6e  slope = % .6e\n", n, t[5 + 3n - 2], t[5 + 3n - 1], t[5 + 3n])
    end
end
