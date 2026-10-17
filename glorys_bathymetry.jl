# Match the model bathymetry to GLORYS near the open boundaries.
#
# Adapted from enri66/NumericalEarthSandbox scripts/glorys_bathymetry.jl (d275bb1). The
# barotropic transport through a boundary is U = ∫u dz over the whole column, so a depth mismatch
# between the model and GLORYS at a boundary cell is a transport error: on a slope a 10% depth
# error drives a spurious boundary jet. Changes from the sandbox: longitudes are unwrapped onto the
# grid's convention so a GLORYS file crossing the dateline (stored in [-180, 180]) interpolates
# correctly onto a grid labelled in (150, 190), and the bottom height is edited on the host.

using NumericalEarth
using NumericalEarth.DataWrangling: Metadata, BoundingBox
using NumericalEarth.DataWrangling.GLORYS: GLORYSStatic
using NCDatasets
using Oceananigans
using Oceananigans.Grids: λnodes, φnodes
using Printf

"""
    glorys_deptho_on_grid(grid, data_dir; region) -> Matrix

GLORYS static sea-floor depth (positive, metres) bilinearly interpolated to the `Center, Center`
points of `grid`. `NaN` where GLORYS has no ocean.
"""
function glorys_deptho_on_grid(grid, data_dir; region)
    metadata = Metadata(:depth; dataset = GLORYSStatic(), dir = data_dir, region)
    NumericalEarth.DataWrangling.Downloads.download(metadata)
    path = NumericalEarth.DataWrangling.metadata_path(metadata)
    path = path isa AbstractVector ? first(path) : path

    ds = NCDataset(path)
    depth = Array{Float64}(replace(ds["deptho"][:, :], missing => NaN))
    λg = Array{Float64}(ds["longitude"][:])
    φg = Array{Float64}(ds["latitude"][:])
    close(ds)

    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    λm = collect(λnodes(underlying, Center()))
    φm = collect(φnodes(underlying, Center()))

    # Unwrap the file's longitudes into a window starting just west of the grid, then sort.
    λ₀ = minimum(λm) - 180
    λg = mod.(λg .- λ₀, 360) .+ λ₀
    order = sortperm(λg)
    λg, depth = λg[order], depth[order, :]

    function bilerp(x, y)
        i = clamp(searchsortedlast(λg, x), 1, length(λg) - 1)
        j = clamp(searchsortedlast(φg, y), 1, length(φg) - 1)
        tx = (x - λg[i]) / (λg[i+1] - λg[i])
        ty = (y - φg[j]) / (φg[j+1] - φg[j])
        corners = ((depth[i, j], (1-tx) * (1-ty)), (depth[i+1, j], tx * (1-ty)),
                   (depth[i, j+1], (1-tx) * ty),   (depth[i+1, j+1], tx * ty))
        v = sum(c[1] * c[2] for c in corners)
        if !isfinite(v)   # a land corner: fall back to the heaviest finite corner
            finite = filter(c -> isfinite(c[1]), corners)
            v = isempty(finite) ? NaN : argmax(c -> c[2], finite)[1]
        end
        return v
    end

    return [bilerp(x, y) for x in λm, y in φm]
end

"""
    match_boundary_bathymetry!(bottom, grid, data_dir; region, n_match = 4, minimum_depth = 10)

Blend the host array `bottom` (bottom height, negative in the ocean) toward GLORYS `deptho` within
`n_match` cells of every lateral boundary, with a raised-cosine weight that is 1 on the boundary.
Cells that are land in either the model or GLORYS are left alone. Returns the GLORYS depth.
"""
function match_boundary_bathymetry!(bottom, grid, data_dir; region, n_match = 4, minimum_depth = 10)
    Hg = glorys_deptho_on_grid(grid, data_dir; region)
    Nx, Ny = size(bottom)

    weight(d) = d >= n_match ? 0.0 : 0.5 * (1 + cos(π * d / n_match))
    edge_distance(i, j) = min(i - 1, Nx - i, j - 1, Ny - j)

    modified = 0
    for j in 1:Ny, i in 1:Nx
        isfinite(Hg[i, j]) || continue
        α = weight(edge_distance(i, j))
        α == 0 && continue
        Hm = -bottom[i, j]
        Hm > 0 || continue
        bottom[i, j] = -max(minimum_depth, (1 - α) * Hm + α * Hg[i, j])
        modified += 1
    end

    @printf("bathymetry matched to GLORYS in %d cells (%d-cell band)\n", modified, n_match)
    return Hg
end
