# Fixes to Breeze, overriding its methods until they are fixed upstream.

# Saturation adjustment's secant solve (Breeze.Solvers.secant_solve) stops only when the residual
# falls below abstol = 1e-4 K. In Float32 that is the rounding noise of the residual, a difference
# of potential temperatures near 300 K (one ulp = 3e-5 K). The coupled runs went NaN where an
# iteration converged to the root with a residual of 1.04e-4: the next step moved T by one ulp, the
# residual did not change, the slope Δx/Δr was −4e6, and T jumped to −155 K (tests/gpu_cell.jl,
# the cell at (20, 262, 14) of diag_step). As in NumericalEarth/Breeze.jl#1108, the solve stops
# once |r| ≤ 30 eps(scale), the residual's rounding noise, whatever the solver's tolerances.
# (coupled_28d_v2 and big12_28d_v2 ran with an earlier form that refused steps growing past twice
# the previous one.)
@inline function Breeze.Solvers.secant_solve(residual, solver::Breeze.Solvers.SecantSolver, x₁, x₂, scale)
    r₁ = residual(x₁)
    r₂ = residual(x₂)
    tolerance = max(solver.abstol, solver.reltol * abs(scale), 30 * eps(abs(scale)))
    iter = 0
    while abs(r₂) > tolerance && iter < solver.maxiter
        ΔxΔr = (x₂ - x₁) / (r₂ - r₁)
        valid_step = isfinite(ΔxΔr)
        ΔxΔr = ifelse(valid_step, ΔxΔr, zero(ΔxΔr))

        x₁, r₁ = x₂, r₂
        x₂ -= r₂ * ΔxΔr
        r₂ = residual(x₂)

        # Ends the loop at the next header check when the slope was degenerate
        r₂ = ifelse(valid_step, r₂, zero(r₂))
        iter += 1
    end
    return x₂
end
