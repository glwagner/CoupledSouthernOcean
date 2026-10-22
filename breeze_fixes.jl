# Fixes to Breeze, overriding its methods until they are fixed upstream.

# Saturation adjustment's secant solve (Breeze.Solvers.secant_solve) stops only when the residual
# falls below abstol = 1e-4 K. In Float32 that is the rounding noise of the residual, a difference
# of potential temperatures near 300 K (one ulp = 3e-5 K). The coupled runs went NaN where an
# iteration converged to the root with a residual of 1.04e-4: the next step moved T by one ulp, the
# residual did not change, the slope Δx/Δr was −4e6, and T jumped to −155 K (tests/gpu_cell.jl,
# the cell at (20, 262, 14) of diag_step). Breeze guards only a zero denominator. Here a step is
# also refused when it is more than twice the previous one: near the root secant steps shrink, so a
# growing step is noise.
@inline function Breeze.Solvers.secant_solve(residual, solver::Breeze.Solvers.SecantSolver, x₁, x₂, scale)
    r₁ = residual(x₁)
    r₂ = residual(x₂)
    iter = 0
    while abs(r₂) > max(solver.abstol, solver.reltol * abs(scale)) && iter < solver.maxiter
        Δx = - r₂ * (x₂ - x₁) / (r₂ - r₁)
        valid_step = isfinite(Δx) & (abs(Δx) ≤ 2 * abs(x₂ - x₁))
        Δx = ifelse(valid_step, Δx, zero(Δx))

        x₁, r₁ = x₂, r₂
        x₂ += Δx
        r₂ = residual(x₂)

        # Ends the loop at the next header check when the step was refused
        r₂ = ifelse(valid_step, r₂, zero(r₂))
        iter += 1
    end
    return x₂
end
