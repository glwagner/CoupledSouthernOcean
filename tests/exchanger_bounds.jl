# The StateExchanger's initial fill on the GPU (NumericalEarth test_nested_simulation.jl:573, which
# hits an illegal memory access). Run with --check-bounds=yes so the out-of-range index is reported.
#   julia --project=../NumericalEarth-so/test --check-bounds=yes tests/exchanger_bounds.jl
using NumericalEarth
using Oceananigans
using Breeze
using CUDA

ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
arch = GPU()

parent_grid = RectilinearGrid(arch; size = (8, 8, 4), x = (-1.5, 1.5), y = (-1.5, 1.5),
                              z = (0, 1), topology = (Bounded, Bounded, Bounded))
times  = [0.0, 1.0, 2.0, 3.0, 4.0]
parent = PrescribedAtmosphere(parent_grid, times)
set!(parent.temperature,       (x, y, z, t) -> 280 + t)
set!(parent.specific_humidity, (x, y, z, t) -> 0.005)
set!(parent.velocities.u,      (x, y, z, t) -> 1.0)
set!(parent.velocities.v,      (x, y, z, t) -> 0.0)
set!(parent.pressure,          (x, y, z, t) -> 9.0e4)

for (name, fts) in pairs((T = parent.temperature, q = parent.specific_humidity, p = parent.pressure,
                          u = parent.velocities.u, v = parent.velocities.v))
    @info name typeof(fts.backend) size(parent(fts)) location(fts)
end

constants = ThermodynamicConstants()
exchanger = ext.state_exchanger(parent, 1.0e5, constants; condensates = (qᶜˡ = nothing, qᶜⁱ = nothing))
CUDA.synchronize()
@info "initial fill ok" extrema(Array(interior(exchanger.prognostic.ρθ[1])))

exchange = NumericalEarth.NestedModels.exchange_state!
exchange(exchanger, 2.5)
CUDA.synchronize()
@info "cycled" exchanger.prognostic.ρᵈ.backend extrema(Array(interior(exchanger.prognostic.ρθ[3])))
