# Southern Ocean / New Zealand coupled hindcast — plan and status

Greg's ask (2026-10-17): (1) a ~CONUS-sized regional 12 km Breeze atmosphere, ERA5-forced, over a
GLORYS-SST PrescribedOcean, a few days; (2) replace PrescribedOcean with a prognostic 12 km regional
ocean using the regional (GLORYS open-boundary) machinery from enri66/NumericalEarthSandbox, fixing
bugs along the way; (3) run 14–28 days with a strong storm, NZ terrain via SurfacePartition (PR #664);
animate; (4) maybe higher resolution later.

## Decisions (made without asking, per Greg's workflow preference)

- **Case: Feb 2018, New Zealand.** Ex-TC Fehi (landfall ~1 Feb) and ex-TC Gita (~20 Feb) during the
  2017/18 Tasman Sea marine heatwave: strong storms, terrain, and a shallow summer mixed layer, so
  storm mixing should show in SST. No sea ice in summer at these latitudes.
- **Why 2018:** NumericalEarth's `GLORYSDaily` (`cmems_mod_glo_phy_my_0.083deg_P1D-m`) ends 2021-06-30;
  later dates (e.g. Cyclone Gabrielle 2023) need the GLORYS interim product (`myint`), not wired up.
- **Domain:** 150°E–190°E (170°W) × 57°S–27°S, Δ = 1/9° (≈12 km) → 360×270 columns, ≈ CONUS area.
  Crosses the dateline: ERA5 (CDS area) and GLORYS (CopernicusMarine subset) handling of
  longitudes > 180 still TO VERIFY (ext/NumericalEarthCopernicusClimateDataStoreExt.jl ~l.600–650,
  ext/NumericalEarthCopernicusMarineExt.jl ~l.100–160).
- **Windows:** stage 1 test 19–22 Feb 2018 (Gita); long coupled run 29 Jan → 26 Feb 2018 (28 d, both storms).

## Software

- Integration branch `glw/southern-ocean` in worktree `~/NumericalEarth-so` (local only, not pushed):
  NE main (c38e1e1f9) + #753 `glw/bottom-precipitation-flux` + #664 `glw/surface-partition`
  (3 conflicts resolved: ρe→ρE seam rename, `surface_radiation.surface_emissivity`, exports).
- Env `~/southern-ocean/Project.toml`: Breeze main (0.12.1), Oceananigans 0.114.0, NE path source.
  Isolated depot `~/.julia-so` (slurm/env.sh); never touches ~/.julia.
- Instantiate job 2962 (gpua100largex4) → logs/so-instantiate-2962.log.

## Known gaps to fix for stage 2 (prognostic ocean)

1. `ext/NumericalEarthBreezeExt/breeze_air_land_radiation.jl`: `apply_breeze_air_sea_radiative_fluxes!`
   has no method for a prognostic ocean (MethodError). Implement: Jᵀ += [εσTₛ⁴ + ε ℐˡʷꜜ +
   (1−α) ℐˢʷꜜ, with penetrating SW via `shortwave_radiative_forcing`] / (ρ₀ cₚ), weighted (1−ℵ).
   Breeze stores downwelling fluxes negative.
2. EarthSystemModel steps every component with one Δt (atmosphere ~10–20 s). First try the ocean at
   the atmospheric Δt; if too costly, add ocean subcycling with averaged fluxes.
3. Ocean fraction þ must be 0 where the ocean grid is immersed (land), else the atmosphere sees
   garbage ocean temperature in coastal cells.
4. Ocean OBCs: port the triad from sandbox `scripts/02_mab_glorys_obc.jl` (PerturbationAdvection
   normal u/v, NormalRadiation tangential + T/S, Flather `GravityWaveRadiationBoundaryCondition` for
   U/V fed GLORYS depth-mean transport, `SurfaceWaveRadiationBoundaryCondition` on η). All available
   on Oceananigans 0.114 main. Helper `glorys_bathymetry.jl` (boundary bathymetry matching).
   Sandbox lessons: GLORYS source grid must bracket the model grid (inset model by n_pad cells);
   clamp GLORYS time indexing at t=0; bounded WENO for T/S near the bottom.

## Templates

- Atmosphere nest + partition: `~/NumericalEarth-so/examples/breeze_downscaling_era5.jl` (from #664:
  SlabLand + PrescribedOcean FTS + `regrid_ocean_fraction` + RRTMGP + `EarthSystemModel(; ...)`).
- `~/AtmosphericRivers.jl/downscale.jl` (12 km nest, Float32, WENO5, relaxation 1/300 s, width 5).

## Status 2026-10-17 05:40 UTC

- PR #664 merged with main, load failure fixed (atmosphere_feels_surface trait), #753's PrescribedOcean
  BGC one-liner cherry-picked; pushed as 4d049a2, description updated. Tests 139 pass / 2 pre-existing
  broken on CPU (2974) and GPU (2973).
- Integration branch also carries the Breeze-RTM → prognostic-ocean radiation kernel + test:
  140 / 2 on CPU (2969) and GPU (2968).
- Dateline test (tests/test_dateline.jl) 44/44 (job 2971): GLORYS, ERA5 and ETOPO on 174–186°E match
  independent non-crossing loads; files cover the box. GLORYS zos is exactly 0 on land (η datum
  now averages GLORYS ocean cells only).
- Stage 1 72 h run: job 2972 (runs/atm_gita_72h). Predownload piece A (Jan 29–Feb 17): job 2975.
  Piece B (Feb 18–Feb 27, GLORYS +1 day) to submit after 2972 finishes its downloads.
- ERA5: one CDS request per variable per window (≤6 threads), ~5–10 min per round.

## Status 08:30 UTC

- 2972 (stage 1) died at setup: `using Dates` made `hour` ambiguous with Oceananigans.Units (now
  `import Dates`); GPU scalar `mean` in a log line. 2976 then failed at `JLD2Writer`: the keyword is
  `overwrite_files` on Oceananigans 0.114. Rerun: 2981. Setup takes ~25 min (albedo, nest, RRTMGP).
- Julia output to the NFS log appears only when Julia yields or exits; progress lines now `flush(stderr)`.
- Coupled smoke (prognostic ocean, 90 iterations): 2978 failed because the ocean grid (6.27 km) was
  deeper than GLORYS's levels (5.95 km); grid now 54 levels to 5.87 km. Rerun: 2982.
- Predownload A (2975) collided with 2972 on era5cli's shared temp file `_tmp_era5cli_201802_<var>…`:
  never run two ERA5 jobs over the same month at once. B (2977) runs now; A requeued after B (2980).
  ERA5 cache is one file per hour, so pieces combine.

## Status 10:20 UTC

- Stage 1 (2981) stepping, Δt ≈ 4–5 s (vertical CFL from grid-scale convection), ~1.3 s wall per
  simulated minute → 72 h in ~2.4 h. Animation pipeline verified on a partial copy (Gita split low
  over NZ, Southern Alps wind jet, cold-air-outbreak heat loss over the Tasman).
- Stage 2 fixes: (1) Float64 clock for the nest via `clock = Clock(grid)` (NE's Breeze default is
  Float32 — mismatched the ocean and would lose 0.25 s per step at 4 weeks); (2) NE: ocean flux
  assembly read `runoff_freshwater_flux` from SlabLand's exchanger (FieldError) → ZeroField for lands
  without runoff; (3) NE: `_assemble_net_ocean_fluxes!` segfaulted the external NVPTX back-end
  (NVPTX_LLVM_Backend_jll 23.1.1+3 and 23.1.2+1, `tryLDG`) — bisected (tests/gpu_flux_kernel_bisect.jl)
  to the eight `ifelse`-guarded stores; rewritten as one branch. NE 3f7cab57e (integration) /
  c2abec997 (PR #664 branch, tests pending).
- Coupled smoke 3023: 90 iterations OK, 0.14 s/iteration (atmosphere alone 0.094). 72 h coupled: 3025.

## Next steps

1. Check 2962 succeeded. 2. Verify dateline handling (small ERA5 + GLORYS download test).
3. Write `atmosphere.jl` (stage 1) + batch; 3-day run. 4. Stage 2 coupling fixes + `coupled.jl`.
