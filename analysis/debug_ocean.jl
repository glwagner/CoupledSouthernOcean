# Maps from ocean_debug.jld2 (SO_OCEAN_DEBUG_MINUTES): η, barotropic U and V, and bottom-level
# u, w and T change, at the first and last saved times.
#   julia --project=analysis analysis/debug_ocean.jl runs/<name>

using CairoMakie
using JLD2
using Printf

rundir = ARGS[1]
static = load(joinpath(rundir, "static.jld2"))
λ, φ = static["λ"], static["φ"]
Nx, Ny = length(λ), length(φ)

interior_range(n, N) = ((n - N) ÷ 2 + 1):(n - (n - N) ÷ 2)

function centered(file, name, iteration)
    A = file["timeseries/$name/$iteration"]
    A = A[interior_range(size(A, 1), Nx), interior_range(size(A, 2), Ny), 1]
    size(A, 1) == Nx + 1 && (A = (A[1:end-1, :] .+ A[2:end, :]) ./ 2)
    size(A, 2) == Ny + 1 && (A = (A[:, 1:end-1] .+ A[:, 2:end]) ./ 2)
    return Float64.(A)
end

file = jldopen(joinpath(rundir, "ocean_debug.jld2"))
iterations = sort(parse.(Int, filter(!=("serialized"), keys(file["timeseries/t"]))))
first_i, last_i = first(iterations), last(iterations)
t = file["timeseries/t/$last_i"]

panels = [("η (m)", centered(file, "η", last_i), 2.0),
          ("U (m² s⁻¹)", centered(file, "U", last_i), 1000.0),
          ("V (m² s⁻¹)", centered(file, "V", last_i), 1000.0),
          ("bottom u (m s⁻¹)", centered(file, "u_bottom", last_i), 1.0),
          ("w above bottom level (m s⁻¹)", centered(file, "w_bottom", last_i), 1e-2),
          ("bottom T change (°C)", centered(file, "T_bottom", last_i) .- centered(file, "T_bottom", first_i), 0.5)]

fig = Figure(size = (1800, 1000), fontsize = 14)
Label(fig[0, 1:6], @sprintf("%s at t = %.1f min (iteration %d)", rundir, t / 60, last_i); fontsize = 20)
for (n, (label, A, scale)) in enumerate(panels)
    row, col = fldmod1(n, 3)
    ax = Axis(fig[row, 2col - 1]; title = label, aspect = DataAspect())
    finite = filter(isfinite, A)
    hm = heatmap!(ax, λ, φ, A; colormap = :balance, colorrange = (-scale, scale))
    Colorbar(fig[row, 2col], hm)
    m, I = findmax(abs.(replace(A, NaN => 0)))
    @info @sprintf("%-28s max|·| = %.3g at λ = %.2f, φ = %.2f (i, j = %d, %d)", label, m, λ[I[1]], φ[I[2]], I[1], I[2])
end
save(joinpath(rundir, "ocean_debug.png"), fig)
@info "wrote $(joinpath(rundir, "ocean_debug.png"))"
