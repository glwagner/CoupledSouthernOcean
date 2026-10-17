# Does NumericalEarth's data path place data correctly on a region that crosses the dateline?
#
# For GLORYS temperature, GLORYS free surface and ERA5 SST, one snapshot is set on a model grid
# labelled 174…186°E (crossing 180°) and, independently, on two grids that do not cross it:
# 174…179°E, and 179…174°W in the dataset's own [-180, 180] convention. Both share the crossing
# grid's 1/9° lattice, so their cells coincide with crossing-grid cells and the values must agree.
# Finite but misplaced data (a shifted, mirrored or wrapped longitude axis) fails the comparison;
# missing data fails the finiteness checks. The seam itself is checked for continuity, and the
# downloaded files for longitude coverage.

using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum, BoundingBox
using CopernicusMarine
using CopernicusClimateDataStore
using Oceananigans
using NCDatasets
using Dates
using Printf
using Statistics: median
using Test

const datadir = joinpath(@__DIR__, "..", "data", "dateline_test")
mkpath(datadir)

const date = DateTime(2018, 2, 19)
const Δ = 1/9
const latitude = (-42, -36)
const Ny = round(Int, (latitude[2] - latitude[1]) / Δ)

grid_for(longitude) = LatitudeLongitudeGrid(CPU(); size = (round(Int, (longitude[2] - longitude[1]) / Δ), Ny, 1),
                                            longitude, latitude, z = (-1, 0))

crossing = grid_for((174, 186))
west     = grid_for((174, 179))
east     = grid_for((-179, -174))
i_west = 1:size(west, 1)                                       # 174…179°E
i_east = (size(crossing, 1) - size(east, 1) + 1):size(crossing, 1)   # 181…186°E ≡ 179…174°W

λc = collect(λnodes(crossing, Center()))
@assert all(λc[i_west] .≈ collect(λnodes(west, Center())))
@assert all(λc[i_east] .- 360 .≈ collect(λnodes(east, Center())))

function load(name, dataset, grid, field_type)
    region = BoundingBox(grid)
    field = field_type(grid)
    set!(field, Metadatum(name; dataset, date, region, dir = datadir))
    return Array(interior(field))[:, :, 1]
end

function check(label, name, dataset; field_type = CenterField, finite_fraction = 1)
    @testset "$label" begin
        Ac = load(name, dataset, crossing, field_type)
        Aw = load(name, dataset, west, field_type)
        Ae = load(name, dataset, east, field_type)

        fc = count(isfinite, Ac) / length(Ac)
        @printf("%-26s crossing grid: %.1f%% finite, range [%.3f, %.3f]\n", label, 100fc,
                minimum(filter(isfinite, Ac)), maximum(filter(isfinite, Ac)))
        @test fc ≥ finite_fraction

        # Agreement with the independent, non-crossing loads (NaN where both are NaN counts as agreement).
        for (side, A, i) in (("west", Aw, i_west), ("east", Ae, i_east))
            C = Ac[i, :]
            same_mask = isfinite.(C) .== isfinite.(A)
            both = isfinite.(C) .& isfinite.(A)
            d = abs.(C[both] .- A[both])
            close = count(d .≤ 1e-3 * max(1, maximum(abs, A[both]))) / length(d)
            @printf("   vs %s: %d compared, median |Δ| = %.2e, max |Δ| = %.2e, %.1f%% within tolerance, NaN mask agrees at %.1f%%\n",
                    side, length(d), median(d), maximum(d), 100close, 100count(same_mask) / length(same_mask))
            @test length(d) > 0.5 * length(C)
            @test close ≥ 0.98          # inpainted land near box edges may legitimately differ
            @test count(same_mask) / length(same_mask) ≥ 0.98
        end

        # Continuity across the seam: the jump between the last cell west of 180° and the first
        # east of it should look like any other neighbouring difference along the row.
        i⁻ = findlast(<(180), λc)
        jumps = [abs(Ac[i⁻ + 1, j] - Ac[i⁻, j]) for j in 1:Ny if isfinite(Ac[i⁻, j]) && isfinite(Ac[i⁻ + 1, j])]
        typical = [abs(Ac[i + 1, j] - Ac[i, j]) for i in 1:size(Ac, 1) - 1, j in 1:Ny
                   if i != i⁻ && isfinite(Ac[i, j]) && isfinite(Ac[i + 1, j])]
        q99 = sort(typical)[ceil(Int, 0.99 * length(typical))]
        @printf("   seam jump: median %.2e, max %.2e; 99th percentile of other neighbour jumps %.2e\n",
                median(jumps), maximum(jumps), q99)
        @test !isempty(jumps)
        @test maximum(jumps) ≤ 2 * q99
    end
end

@testset "dateline crossing" begin
    check("GLORYS temperature (°C)", :temperature, GLORYSDaily())
    check("GLORYS free surface (m)", :free_surface, GLORYSDaily(); field_type = g -> Field{Center, Center, Nothing}(g))
    check("ERA5 SST (K)", :sea_surface_temperature, ERA5HourlySingleLevel();
          field_type = g -> Field{Center, Center, Nothing}(g), finite_fraction = 0.8)

    # The child terrain, the ocean fraction and the ocean bathymetry all regrid ETOPO2022.
    @testset "ETOPO2022 bathymetry and ocean fraction" begin
        regrid(grid) = Array(interior(regrid_bathymetry(grid; dataset = ETOPO2022(), minimum_depth = 0,
                                                       major_basins = Inf, interpolation_passes = 1)))[:, :, 1]
        Bc, Bw, Be = regrid(crossing), regrid(west), regrid(east)
        for (side, B, i) in (("west", Bw, i_west), ("east", Be, i_east))
            d = abs.(Bc[i, :] .- B)
            @printf("ETOPO bottom height vs %s: median |Δ| = %.2f m, %.1f%% within 1 m\n",
                    side, median(d), 100count(d .≤ 1) / length(d))
            @test count(d .≤ 1) / length(d) ≥ 0.98
        end
        # Ground truth independent of any data path: the central North Island between Rotorua and
        # Taupo (176.5°E, 38.5°S) is land; the Kermadec Trench region (179°W, 36.5°S) is deep ocean.
        φc = collect(φnodes(crossing, Center()))
        i_nz = findfirst(≥(176.5), λc); j_nz = findfirst(≥(-38.5), φc)
        i_tr = findfirst(≥(181.0), λc); j_tr = findfirst(≥(-36.5), φc)
        @printf("bottom height at (176.5°E, 38.5°S) = %.0f m, at (179°W, 36.5°S) = %.0f m\n", Bc[i_nz, j_nz], Bc[i_tr, j_tr])
        @test Bc[i_nz, j_nz] ≥ 0
        @test Bc[i_tr, j_tr] < -2000

        surface = LatitudeLongitudeGrid(CPU(); size = (size(crossing, 1), Ny), longitude = (174, 186), latitude,
                                        topology = (Bounded, Bounded, Flat))
        þ = Array(interior(regrid_ocean_fraction(surface; dataset = ETOPO2022())))[:, :, 1]
        @test 0 ≤ minimum(þ) && maximum(þ) ≤ 1
        @test þ[i_nz, j_nz] < 0.5 && þ[i_tr, j_tr] > 0.99
    end

    @testset "downloaded files cover the crossing box" begin
        files = filter(f -> endswith(f, ".nc"), readdir(datadir; join = true))
        covers(span, lo, hi) = span[1] ≤ lo && span[2] ≥ hi
        full_cover = 0
        for f in files
            NCDataset(f) do ds
                lon = haskey(ds, "longitude") ? Float64.(ds["longitude"][:]) : Float64.(ds["lon"][:])
                unwrapped = mod.(lon .- 170, 360) .+ 170   # into [170, 530)
                span = extrema(unwrapped)
                @printf("%s: %d longitudes, first %.3f last %.3f, monotone %s, unwrapped span [%.3f, %.3f]\n",
                        basename(f), length(lon), lon[1], lon[end], issorted(lon), span...)
                # Every file holds one of the three boxes; none may come back short of its box.
                @test covers(span, 174, 179) || covers(span, 181, 186)
                full_cover += covers(span, 174, 186)
            end
        end
        @test length(files) ≥ 3
        @test full_cover ≥ 2   # at least the two GLORYS variables were fetched across the seam
    end
end
