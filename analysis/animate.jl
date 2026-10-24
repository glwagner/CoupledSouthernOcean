# Animate a hindcast run directory: julia --project=analysis analysis/animate.jl runs/<name> [stride]
#
# Atmosphere panels from surface.jld2: near-surface wind speed with surface-pressure contours over
# the ocean, precipitation rate, surface skin temperature, and surface turbulent heat loss over the
# ocean. When the run has a prognostic ocean (ocean_surface.jld2), ocean panels are added: SST change
# since the start and surface current speed (six panels in all). Coastlines are the þ = 1/2 contour.

using CairoMakie
using JLD2
using Dates
using Printf
using Statistics

rundir = ARGS[1]
stride = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 1

static = load(joinpath(rundir, "static.jld2"))
λ, φ = static["λ"], static["φ"]
þ = static["ocean_fraction"]
start_date = static["start_date"]
Nx, Ny = length(λ), length(φ)
ocean_mask = ifelse.(þ .> 0.5, 1.0, NaN)

# Interior points of a saved dimension of length n: the writer keeps halos, so a centre-located
# dimension has N + 2H points and a face-located one N + 1 + 2H (the parity tells them apart).
function interior_range(n, N)
    H = (n - N) ÷ 2
    return H+1:n-H
end

# Interior of a saved 2-D slice, averaged to cell centres (face-located fields carry N + 1 points).
function centered(file, name, iteration)
    A = dropdims(file["timeseries/$name/$iteration"]; dims = 3)
    A = A[interior_range(size(A, 1), Nx), interior_range(size(A, 2), Ny)]
    size(A, 1) == Nx + 1 && (A = (A[1:end-1, :] .+ A[2:end, :]) ./ 2)
    size(A, 2) == Ny + 1 && (A = (A[:, 1:end-1] .+ A[:, 2:end]) ./ 2)
    size(A) == (Nx, Ny) || error("$name has interior size $(size(A)), expected ($Nx, $Ny)")
    return Float64.(A)
end

iterations(file) = sort(parse.(Int, filter(k -> k != "serialized", keys(file["timeseries/t"]))))

atmosphere = jldopen(joinpath(rundir, "surface.jld2"))
ocean_path = joinpath(rundir, "ocean_surface.jld2")
ocean = isfile(ocean_path) ? jldopen(ocean_path) : nothing

atmosphere_iterations = iterations(atmosphere)[1:stride:end]
times = [atmosphere["timeseries/t/$i"] for i in atmosphere_iterations]

# Match each atmosphere frame to the ocean output written at the same time.
if !isnothing(ocean)
    ocean_iterations = iterations(ocean)
    ocean_times = [ocean["timeseries/t/$i"] for i in ocean_iterations]
    ocean_frame(t) = ocean_iterations[argmin(abs.(ocean_times .- t))]
    T₀ = centered(ocean, "T", first(ocean_iterations))
end

n = Observable(1)
frame_iteration = @lift atmosphere_iterations[$n]
frame_time = @lift times[$n]

wind = @lift sqrt.(centered(atmosphere, "u", $frame_iteration) .^ 2 .+ centered(atmosphere, "v", $frame_iteration) .^ 2)
pressure = @lift centered(atmosphere, "p", $frame_iteration) .* ocean_mask ./ 100
precipitation = @lift max.(centered(atmosphere, "precipitation", $frame_iteration) .* 3600, 0)
skin = @lift centered(atmosphere, "Tₛ", $frame_iteration) .- 273.15
heat_loss = @lift (centered(atmosphere, "Qᵒᶜ_sensible", $frame_iteration) .+
                   centered(atmosphere, "Qᵒᶜ_latent", $frame_iteration)) .* ocean_mask

has_ocean = !isnothing(ocean)
fig = Figure(size = has_ocean ? (1800, 820) : (1400, 1050), fontsize = 15)
title = @lift @sprintf("New Zealand hindcast — %s UTC",
                       Dates.format(start_date + Millisecond(round(Int, 1000 * $frame_time)), "yyyy-mm-dd HH:MM"))
Label(fig[0, 1:(has_ocean ? 6 : 4)], title; fontsize = 22, tellwidth = false)

function map_axis(position, label)
    ax = Axis(fig[position...]; title = label, aspect = DataAspect(), xlabel = "longitude (°E)", ylabel = "latitude (°)")
    return ax
end

coastline!(ax) = contour!(ax, λ, φ, þ; levels = [0.5], color = :black, linewidth = 0.8)

ax = map_axis((1, 1), "Lowest-level wind speed (m s⁻¹), surface pressure over ocean (hPa)")
hm = heatmap!(ax, λ, φ, wind; colormap = :viridis, colorrange = (0, 30))
contour!(ax, λ, φ, pressure; levels = 940:4:1040, color = :white, linewidth = 0.6)
coastline!(ax); Colorbar(fig[1, 2], hm)

ax = map_axis((1, 3), "Precipitation (mm h⁻¹)")
hm = heatmap!(ax, λ, φ, precipitation; colormap = Reverse(:davos), colorrange = (0, 10), lowclip = :white)
coastline!(ax); Colorbar(fig[1, 4], hm)

ax = map_axis((2, 1), "Surface skin temperature (°C)")
hm = heatmap!(ax, λ, φ, skin; colormap = :thermal, colorrange = (0, 28))
coastline!(ax); Colorbar(fig[2, 2], hm)

ax = map_axis((2, 3), "Ocean turbulent heat loss, sensible + latent (W m⁻²)")
hm = heatmap!(ax, λ, φ, heat_loss; colormap = :balance, colorrange = (-400, 400))
coastline!(ax); Colorbar(fig[2, 4], hm)

if has_ocean
    oi = @lift ocean_frame($frame_time)
    ΔT = @lift (centered(ocean, "T", $oi) .- T₀) .* ocean_mask
    speed = @lift sqrt.(centered(ocean, "u", $oi) .^ 2 .+ centered(ocean, "v", $oi) .^ 2) .* ocean_mask

    ax = map_axis((1, 5), "SST change since start (°C)")
    hm = heatmap!(ax, λ, φ, ΔT; colormap = :balance, colorrange = (-3, 3), nan_color = :gray80)
    coastline!(ax); Colorbar(fig[1, 6], hm)

    ax = map_axis((2, 5), "Surface current speed (m s⁻¹)")
    hm = heatmap!(ax, λ, φ, speed; colormap = :speed, colorrange = (0, 1), nan_color = :gray80)
    coastline!(ax); Colorbar(fig[2, 6], hm)
end

output = joinpath(rundir, "animation.mp4")
record(fig, output, 1:length(atmosphere_iterations); framerate = 12) do i
    n[] = i
end
@info "wrote $output ($(length(atmosphere_iterations)) frames)"

# A still of the last frame for quick inspection
save(joinpath(rundir, "last_frame.png"), fig)
