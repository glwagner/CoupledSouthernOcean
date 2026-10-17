# Is the all-NaN deepest GLORYS level (5728 m here) filled by NumericalEarth's inpainting?
# Inpaint one salinity snapshot on CPU and on GPU, with the cache off, and report the bottom levels.
using NumericalEarth, Oceananigans, CopernicusMarine, CUDA
using NumericalEarth.DataWrangling: Metadatum, BoundingBox
include(joinpath(@__DIR__, "..", "case.jl"))

region = BoundingBox(longitude = (longitude[1] - 0.5, longitude[2] + 0.5),
                     latitude  = (latitude[1]  - 0.5, latitude[2]  + 0.5))
md = Metadatum(:salinity; dataset = GLORYSDaily(), date = start_date, region, dir = glorys_dir)

for arch in (CPU(), GPU())
    S = Field(md, arch; cache_inpainted_data = false)
    s = Array(interior(S))
    for k in 1:3
        level = s[:, :, k]
        println(arch, " level $k: extrema ", extrema(level), "  zeros ", count(==(0), level), " / ", length(level))
    end
    flush(stdout)
end
