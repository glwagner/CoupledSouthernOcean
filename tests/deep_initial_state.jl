# Initial GLORYS T and S on the model's deepest levels (wet cells only), to check the inpainting.
using NumericalEarth, Oceananigans, CopernicusMarine, CUDA, Printf, JLD2
using NumericalEarth.DataWrangling: BoundingBox, Metadata
using Oceananigans.OutputReaders: Clamp
using Oceananigans.Architectures: on_architecture
using Oceananigans.Units
using Statistics: mean
import Dates
using Dates: Day
include(joinpath(@__DIR__, "..", "case.jl"))
arch = GPU()
Oceananigans.defaults.FloatType = Float32
glorys_region = BoundingBox(longitude = (longitude[1] - 0.5, longitude[2] + 0.5),
                            latitude  = (latitude[1]  - 0.5, latitude[2]  + 0.5))
glorys_dates = start_date:Day(1):(stop_date + Day(1))
include(joinpath(@__DIR__, "..", "ocean.jl"))
T, S = Array(interior(ocean.model.tracers.T)), Array(interior(ocean.model.tracers.S))
zc = Array(znodes(ocean.model.grid, Center()))
for k in 1:8
    wet = [bottom[i, j] < zc[k] for i in 1:Nx, j in 1:Ny]
    n = count(wet)
    n == 0 && continue
    Tk, Sk = T[:, :, k][wet], S[:, :, k][wet]
    @info @sprintf("k = %d (z = %.0f m): %6d wet  T ∈ [%.3f, %.3f] mean %.3f   S ∈ [%.3f, %.3f] mean %.3f",
                   k, zc[k], n, extrema(Tk)..., mean(Tk), extrema(Sk)..., mean(Sk))
end
