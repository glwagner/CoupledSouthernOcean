# Replay, on the CPU, the thermodynamic diagnosis of one cell from the column values debug_step.jl
# saves (debug_columns.jld2): total density, θ, equilibrium moisture, precipitation and the previous
# cloud condensate, then the saturation adjustment and θˡⁱ → T inversion, printing each iterate.
#   julia --project analysis/debug_cell.jl runs/<name>/debug_columns.jld2 <label> <k>
using JLD2
using Printf
using Oceananigans
using Breeze
using CloudMicrophysics
using NumericalEarth

Oceananigans.defaults.FloatType = Float32
const BT = Breeze.Thermodynamics
const BM = Breeze.Microphysics

file, label, k = ARGS[1], ARGS[2], parse(Int, ARGS[3])
labels, columns, constants = jldopen(f -> (f["labels"], f["columns"], f["constants"]), file)
n = findfirst(==(label), labels)
isnothing(n) && error("no checkpoint \"$label\"; have $(labels)")
c = columns[n]

println("checkpoint \"$label\", k = $k; fields at k-2:k+2:")
for name in sort(collect(keys(c)))
    v = c[name]
    length(v) ≥ k + 2 || continue
    @printf("  %-60s %s\n", name, join([@sprintf("% .7e", v[kk]) for kk in k-2:k+2], " "))
end

get(names...) = c[names[findfirst(n -> haskey(c, n), names)]][k]
ρᵈ  = get("fields.ρᵈ", "dynamics.dry_density")
ρθ  = get("fields.ρθ")
ρqᵉ = get("fields.ρqᵉ")
ρqʳ = get("fields.ρqʳ")
ρqˢⁿ = get("fields.ρqˢⁿ")
qᶜˡ = get("fields.qᶜˡ")
qᶜⁱ = get("fields.qᶜⁱ")
ρ = ρᵈ + ρqᵉ + ρqʳ + ρqˢⁿ
θ = ρθ / ρᵈ
qᵉ = ρqᵉ / ρ
qʳ, qˢⁿ = ρqʳ / ρ, ρqˢⁿ / ρ
@info "inputs" ρᵈ ρ θ qᵉ qʳ qˢⁿ qᶜˡ qᶜⁱ ρ_total_saved = get("dynamics.total_density", "fields.ρ")

pˢᵗ = 1f5
newton = Breeze.Solvers.NewtonSolver(Float32)
ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
microphysics = ext.default_nested_microphysics()
adjustment = microphysics.cloud_formation
@info "schemes" newton adjustment.solver adjustment.equilibrium

# The kernel: grid moisture fractions from the previous condensate, then the adjustment
q₀ = BT.MoistureMassFractions(qᵉ - qᶜˡ - qᶜⁱ, qᶜˡ + qʳ, qᶜⁱ + qˢⁿ)
𝒰₀ = BT.LiquidIceDensityState(θ, q₀, pˢᵗ, ρ, newton)
@info "state before adjustment" q₀ T = BT.temperature(𝒰₀, constants)

q₁ = BT.MoistureMassFractions(qᵉ, qʳ, qˢⁿ)
𝒰₁ = BT.with_moisture(𝒰₀, q₁)

# adjust_thermodynamic_state, step by step
qᵗ = BT.total_specific_moisture(𝒰₁)
qᵉ_adj = qᵗ - qʳ - qˢⁿ
𝒰ᵃ = BT.with_moisture(𝒰₁, BT.MoistureMassFractions(qᵉ_adj, qʳ, qˢⁿ))
T₁ = BT.temperature(𝒰ᵃ, constants)
qᵛ⁺₁ = BT.saturation_specific_humidity(T₁, ρ, constants, adjustment.equilibrium)
@info "first guess" qᵗ qᵉ_adj T₁ qᵛ⁺₁ saturated = qᵉ_adj > qᵛ⁺₁
if qᵉ_adj > qᵛ⁺₁
    𝒰s = BM.adjust_state(𝒰₁, T₁, constants, adjustment.equilibrium, (qʳ, qˢⁿ))
    q̃ = 𝒰s.moisture_mass_fractions
    cᵖᵐ = BT.mixture_heat_capacity(q̃, constants)
    ΔT = (constants.liquid.reference_latent_heat * (q̃.liquid - qʳ) + constants.ice.reference_latent_heat * (q̃.ice - qˢⁿ)) / cᵖᵐ
    T₂ = T₁ + max(0.01f0, ΔT / 2)
    residual(T) = BM.saturation_adjustment_residual(T, 𝒰₁, constants, adjustment.equilibrium, (qʳ, qˢⁿ))
    @info "secant start" q̃ ΔT T₂ r₁ = residual(T₁) r₂ = residual(T₂)
    let x₁ = T₁, x₂ = T₂
    r₁, r₂ = residual(x₁), residual(x₂)
    for iter in 1:20
        abs(r₂) > 1f-4 || break
        s = (x₂ - x₁) / (r₂ - r₁)
        isfinite(s) || (println("  iter $iter: non-finite slope"); break)
        x₁, r₁ = x₂, r₂
        x₂ -= r₂ * s
        r₂ = residual(x₂)
        @printf("  iter %2d: T = %.6f  r = % .6e\n", iter, x₂, r₂)
    end end
end
𝒰★ = BM.adjust_thermodynamic_state(𝒰₁, adjustment, constants, (qʳ, qˢⁿ))
@info "adjusted (Breeze)" 𝒰★.moisture_mass_fractions T = BT.temperature(𝒰★, constants)
