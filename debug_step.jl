# SO_DEBUG_ITERATION=n: run the child atmosphere's step from iteration n to n + 1 through a copy of
# Breeze's AcousticRungeKutta3 `time_step!` that checks every field it can reach for NaN/Inf after each
# phase, so the phase that first breaks a field is named. Included by hindcast.jl; the copy performs
# the same operations in the same order, so the trajectory is unchanged.

const TS = Breeze.TimeSteppers
const debug_step = Ref(false)
const debug_column = Ref((0, 0))   # interior (i, j) to print a profile of, if nonzero

# Every Field or device array reachable from `x` within a few levels of struct/NamedTuple nesting.
function reachable_arrays!(out, prefix, x, depth = 0)
    depth > 3 && return out
    if x isa Field
        push!(out, (prefix, x))
    elseif x isa CUDA.CuArray{<:AbstractFloat}
        push!(out, (prefix, x))
    elseif x isa AbstractArray
        return out
    elseif x isa Union{NamedTuple, Tuple}
        for (k, v) in pairs(x)
            reachable_arrays!(out, "$prefix.$k", v, depth + 1)
        end
    elseif !isbits(x) && isstructtype(typeof(x)) && !(x isa Module) && !(x isa Function) && fieldcount(typeof(x)) > 0
        for name in fieldnames(typeof(x))
            isdefined(x, name) || continue
            reachable_arrays!(out, "$prefix.$name", getfield(x, name), depth + 1)
        end
    end
    return out
end

function debug_arrays(model)
    out = Any[]
    reachable_arrays!(out, "fields", Oceananigans.fields(model), 0)
    reachable_arrays!(out, "Gⁿ", model.timestepper.Gⁿ, 0)
    reachable_arrays!(out, "U⁰", model.timestepper.U⁰, 0)
    reachable_arrays!(out, "substepper", model.timestepper.substepper, 0)
    reachable_arrays!(out, "dynamics", model.dynamics, 0)
    reachable_arrays!(out, "closure_fields", model.closure_fields, 0)
    # Distinct arrays only (several names alias one field)
    seen = IdDict()
    return [(name, a) for (name, a) in out if !haskey(seen, a) && (seen[a] = true)]
end

finite(a::Field) = isfinite(sum(interior(a)))
finite(a) = isfinite(sum(a))

const already_bad = Set{String}()
const debug_columns = Any[]

function debug_check(model, label)
    debug_step[] || return nothing
    bad = String[]
    for (name, a) in debug_arrays(model)
        finite(a) && continue
        name in already_bad && continue
        push!(already_bad, name)
        A = Array(a isa Field ? interior(a) : a)
        n = count(!isfinite, A)
        first_bad = findfirst(!isfinite, A)
        push!(bad, @sprintf("%s (%d non-finite, first at %s, size %s)", name, n, Tuple(first_bad), size(A)))
    end
    i, j = debug_column[]
    profile = ""
    if i > 0
        # Every 3-D field's column at (i, j), saved after the step to debug_columns.jld2
        columns = Dict{String, Vector}()
        for (name, a) in debug_arrays(model)
            a isa Field && ndims(interior(a)) == 3 && size(interior(a), 3) > 1 || continue
            columns[name] = Array(interior(a, i:i, j:j, :))[1, 1, :]
        end
        push!(debug_columns, (label, columns))
        ρw = Array(interior(model.momentum.ρw))[i, j, :]
        ρθ = Array(interior(thermodynamic_density(model.formulation)))[i, j, :]
        profile = @sprintf("  column (%d, %d): max|ρw| = %.3g, ρθ ∈ [%.4g, %.4g]", i, j,
                           maximum(abs, ρw), extrema(ρθ)...)
    end
    @info "debug step: $label" * profile
    foreach(b -> @warn("  newly non-finite after $label: $b"), bad)
    flush(stderr)
    return nothing
end

thermodynamic_density(formulation) = Breeze.AtmosphereModels.thermodynamic_density(formulation)

function debug_rk3_substep!(model, Δt, β, stage)
    ts = model.timestepper
    substepper = ts.substepper
    CE = Breeze.CompressibleEquations
    TS.prepare_acoustic_cache!(substepper, model);             debug_check(model, "stage $stage prepare_acoustic_cache!")
    TS.cache_advecting_state!(model);                          debug_check(model, "stage $stage cache_advecting_state!")
    TS.compute_flux_bc_tendencies!(model);                     debug_check(model, "stage $stage compute_flux_bc_tendencies!")
    TS.compute_closure_tendencies!(model);                     debug_check(model, "stage $stage compute_closure_tendencies!")
    TS.add_implicit_advection_tendency!(model);                debug_check(model, "stage $stage add_implicit_advection_tendency!")
    θ_advection = TS.field_advection_scheme(model.advection, TS.thermodynamic_density_name(model.formulation))
    TS.acoustic_rk3_substep_loop!(model, substepper, Δt, β, ts.U⁰, θ_advection)
    debug_check(model, "stage $stage acoustic_rk3_substep_loop!")
    TS.scalar_rk3_substep!(model, β * Δt);                     debug_check(model, "stage $stage scalar_rk3_substep!")
    TS.implicit_substep!(model, β * Δt);                       debug_check(model, "stage $stage implicit_substep!")
    return nothing
end

function Oceananigans.TimeSteppers.time_step!(model::TS.CompressibleAcousticModel, Δt; callbacks = [])
    TS.maybe_prepare_first_time_step!(model, Δt, callbacks)
    ts = model.timestepper
    β₁, β₂, β₃ = ts.β₁, ts.β₂, ts.β₃
    tⁿ⁺¹ = model.clock.time + Δt

    debug_check(model, "step start")
    TS.store_initial_state!(model)
    TS.freeze_linearization_state!(ts.substepper, model);      debug_check(model, "freeze_linearization_state!")

    if debug_step[]
        debug_rk3_substep!(model, Δt, β₁, 1)
    else
        TS.acoustic_rk3_substep!(model, Δt, β₁)
    end
    TS.tick_stage!(model.clock, β₁ * Δt)
    TS.update_state!(model, callbacks; compute_tendencies = true); debug_check(model, "stage 1 update_state!")
    TS.cache_transport_velocity!(model)

    if debug_step[]
        debug_rk3_substep!(model, Δt, β₂, 2)
    else
        TS.acoustic_rk3_substep!(model, Δt, β₂)
    end
    TS.tick_stage!(model.clock, (β₂ - β₁) * Δt)
    TS.update_state!(model, callbacks; compute_tendencies = true); debug_check(model, "stage 2 update_state!")
    TS.cache_transport_velocity!(model)

    if debug_step[]
        debug_rk3_substep!(model, Δt, β₃, 3)
    else
        TS.acoustic_rk3_substep!(model, Δt, β₃)
    end
    corrected_Δt = TS.time_difference_seconds(tⁿ⁺¹, model.clock.time)
    TS.tick_stage!(model.clock, corrected_Δt, Δt)

    TS.step_closure_prognostics!(model.closure_fields, model.closure, model, Δt); debug_check(model, "step_closure_prognostics!")
    TS.update_state!(model, callbacks; compute_tendencies = true);              debug_check(model, "stage 3 update_state!")
    TS.cache_transport_velocity!(model)
    TS.microphysics_model_update!(model.microphysics, model);                   debug_check(model, "microphysics_model_update!")
    TS.step_lagrangian_particles!(model, Δt)
    return nothing
end

debug_iteration = parse(Int, ENV["SO_DEBUG_ITERATION"])
haskey(ENV, "SO_DEBUG_COLUMN") && (debug_column[] = Tuple(parse.(Int, split(ENV["SO_DEBUG_COLUMN"], ','))))
function arm_debug_step(sim)
    if iteration(sim) == debug_iteration + 1 && !isempty(debug_columns)
        jldsave(joinpath(rundir, "debug_columns.jld2"); labels = first.(debug_columns), columns = last.(debug_columns),
                column = debug_column[], constants = sim.model.atmosphere.model.child.thermodynamic_constants)
    end
    debug_step[] = iteration(sim) == debug_iteration
    debug_step[] && @info "debug step: checking every phase of the step from iteration $debug_iteration"
    iteration(sim) > debug_iteration && (sim.running = false)
    return nothing
end

add_callback!(simulation, arm_debug_step, IterationInterval(1))
