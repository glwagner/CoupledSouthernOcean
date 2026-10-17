# Time series at the surface point where the ocean ends coldest (or at a given i, j):
# ocean surface T, S, u, v, MLD, η, the air–sea fluxes and the wind, plus the column's wet levels.
#   julia --project=analysis analysis/point_series.jl runs/<name> [i j]

using JLD2
using Printf

rundir = ARGS[1]
static = load(joinpath(rundir, "static.jld2"))
λ, φ, ℵ = static["λ"], static["φ"], static["ocean_fraction"]
Nx, Ny = length(λ), length(φ)
Nz = 53  # ocean levels (ocean.jl)

interior_range(n, N) = ((n - N) ÷ 2 + 1):(n - (n - N) ÷ 2)

function centered(file, name, iteration)
    A = file["timeseries/$name/$iteration"]
    A = A[interior_range(size(A, 1), Nx), interior_range(size(A, 2), Ny), size(A, 3) > 1 ? interior_range(size(A, 3), Nz) : (:)]
    size(A, 1) == Nx + 1 && (A = (A[1:end-1, :, :] .+ A[2:end, :, :]) ./ 2)
    size(A, 2) == Ny + 1 && (A = (A[:, 1:end-1, :] .+ A[:, 2:end, :]) ./ 2)
    return Float64.(A)
end

iterations(file) = sort(parse.(Int, filter(!=("serialized"), keys(file["timeseries/t"]))))

ocean = jldopen(joinpath(rundir, "ocean_surface.jld2"))
atmos = jldopen(joinpath(rundir, "surface.jld2"))
daily = jldopen(joinpath(rundir, "ocean_daily.jld2"))
oi, ai = iterations(ocean), iterations(atmos)

if length(ARGS) ≥ 3
    i, j = parse(Int, ARGS[2]), parse(Int, ARGS[3])
else
    T_last = centered(ocean, "T", last(oi))[:, :, 1]
    _, I = findmin(replace(T_last, NaN => Inf))
    i, j = Tuple(I)
end

T₀ = centered(daily, "T", first(iterations(daily)))[i, j, :]
wet = findall(!iszero, T₀)
@info @sprintf("point i, j = %d, %d at λ = %.3f, φ = %.3f; ocean fraction %.3f; %d wet levels of %d",
               i, j, λ[i], φ[j], ℵ[i, j], length(wet), length(T₀))
@info "ocean fraction around it (rows j+1 … j-1):"
for jj in j+1:-1:j-1
    println("    ", join([@sprintf("%5.2f", ℵ[ii, jj]) for ii in max(i-2, 1):min(i+2, Nx)], " "))
end

println(" hours      T      S      u      v    mld     η |  Tₛ(atm)   Q_sens   Q_lat    τˣ     τʸ   u_air  v_air")
for (n, it) in enumerate(oi)
    t = ocean["timeseries/t/$it"]
    o = [centered(ocean, name, it)[i, j, 1] for name in ("T", "S", "u", "v", "mld", "η")]
    m = argmin([abs(atmos["timeseries/t/$a"] - t) for a in ai])
    a = [centered(atmos, name, ai[m])[i, j, 1] for name in ("Tₛ", "Qᵒᶜ_sensible", "Qᵒᶜ_latent", "τˣ", "τʸ", "u", "v")]
    @printf("%6.1f %6.2f %6.2f %6.2f %6.2f %6.1f %5.2f | %8.2f %8.1f %7.1f %6.2f %6.2f %6.1f %6.1f\n",
            t / 3600, o..., a...)
end
