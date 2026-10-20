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

## Status 12:20 UTC

- Stage 1 done (2981): runs/atm_gita_72h, animation.mp4 (73 frames). Stable, Δt 3.7–5 s.
- Coupled run blew up within 1 h at the deepest ocean level. Diagnosis (ocean_only.jl,
  SO_OCEAN_DEBUG_MINUTES, tests/deep_initial_state.jl): wet cells at the 5669 m level started with
  S = 4.6, T = 0.07 — GLORYS T/S below ~5.5 km came back as zeros here, not continued downward.
  Independent of precision (Float32/64) and of the open boundaries (closed also blew up).
  Fix: ocean grid to 5.47 km (53 levels). **NE bug to report/fix**: the `inpaint_mask!` path
  continues values downward, so the zeros come from somewhere else (a missing deepest level in
  the subset? `_fill_nans!` after an all-NaN level?) — reproducer tests/deep_initial_state.jl
  with SO_OCEAN_GRID_DEPTH=5900.
- Coupled 72 h (3039) stable so far; 0.14 s/iteration. 28 d estimate ≈ 22–26 h on one A100.
- ERA5 predownload: whole-month CDS requests time out after 1 h; windowed job 3041 (48 h pieces,
  serial) running for Jan 29 → Feb 26. Missing so far: Feb 6–17, Feb 23–26.

## Next steps

1. Check 2962 succeeded. 2. Verify dateline handling (small ERA5 + GLORYS download test).
3. Write `atmosphere.jl` (stage 1) + batch; 3-day run. 4. Stage 2 coupling fixes + `coupled.jl`.

## Status 14:55 UTC

- Coupled 72 h (3039) crashed at 2.6 d (DomainError, sqrt of a negative) after surface T went to
  −16 °C, S to 9 psu in single cells on the open boundaries: west edge (150.06°E, 40.5°S, i = 1,
  j ≈ 148–154) from hour 42, east edge (i = 360, j ≈ 62) from hour ~34. Surface fluxes there were
  small, so the cause is the T/S open boundary (NormalRadiation, from the sandbox's script 02), not
  the coupling. The sandbox has since moved to `TracerReservoir` (Oceananigans #5964) for T/S.
  ocean.jl now takes SO_OCEAN_TRACER_OBC=reservoir (default) | radiation; ocean-only 72 h runs of
  both (3069 radiation, 3070 reservoir) to confirm. analysis/point_series.jl: time series at the
  coldest surface point (or any i, j).
- NE inpainting fix: the new kernel read `size(field, 3)` on the device — fails for 2-D fields
  (Field adapted with grid = nothing). Nz now passed from the host; test extended with a 2-D
  field. CPU 3068 / GPU 3067.
- ERA5: each 48 h window re-requested three single-level snapshots (surface geopotential, MSLP,
  skin T), each queueing ~1 h at the CDS. predownload.jl now downloads the pressure-level bundle
  directly and the snapshots only for the run start. NE clears every `_tmp_era5cli_<yyyymm>*` in its
  directory before a request (that was the earlier "collision"), so concurrent jobs use private
  SO_ERA5_DIRs merged into data/era5 after each window: 3072 (Feb 6–12), 3073 (Feb 12–18),
  3074 (Feb 22–26); GLORYS Jan 29 → Feb 27 + Jan 29 snapshots: 3071. Download jobs use 2 CPUs
  (6 CPUs × 4 jobs blocked GPU jobs on the 24-CPU nodes).

## Status 16:30 UTC — root cause of the boundary blow-up

- Ocean-only 72 h runs (Δt 60 s and 5 s; NormalRadiation and TracerReservoir) all stay clean, so
  neither the boundary scheme nor the small Δt is the cause.
- In the coupled run the west-face normal velocity never changes from its t = 0 GLORYS value (every
  j, 62 h), and the S halo is the Dirichlet reflection about the t = 0 value (c₀ + c₁ = 2 × 35.655).
  Cause (NE, from #753's commit 02505c5b9 "Fill the exchanged state's halos centrally"): with the
  exchange grid equal to the ocean grid (the default), the ocean exchanger *is* the ocean's u, v, S,
  and `update_state!(::EarthSystemModel)` refills their halos every step with no clock or fields.
  Open boundaries then fall back to their exterior value at t = 0 — a frozen boundary transport
  against a wind-driven interior, a divergent edge cell, and S, T drifting without bound.
  Fix (integration branch, to go into #753): `fill_state_halo_regions!(exchanger, component)`; the
  Oceananigans-ocean method skips fields that belong to the ocean model. Regression test in
  test_ocean_only_model.jl (time-dependent west S boundary value). GPU 3080 / CPU 3081; the same
  test on the unfixed code: 3083.
- CDS: the CDS-MARS queue is congested (7566 queued, 460 running). CopernicusClimateDataStore
  waits 1 h, then the retry re-queues at the back: no request could ever finish. 57 abandoned
  requests of ours were still queued (deleted). predownload.jl now waits SO_CDS_MAX_WAIT_HOURS
  (20 h) per request; windows of 6 days (fewer requests): 3085 (Feb 6–12), 3086 (Feb 12–18),
  3087 (Feb 22–26); GLORYS + Jan 29 snapshots 3084.

## Status 19:10 UTC — coupled 72 h works

- 3091 (runs/coupled_gita_72h_fix, NE fc750222d + TracerReservoir T/S): 72 h clean, 2.65 h wall.
  Ocean T ∈ [−0.37, 28.5] throughout; edge cells T ∈ [1.8, 27.6], S ∈ [33.7, 36.0]; the west-face
  normal velocity now evolves (0.25 m/s in the first hour, as in the ocean-only runs).
- 3 days of Gita: mean ΔSST −0.36 °C over the ocean (1–99 %: −2.2 … +1.1); ΔSSS ±0.8. Largest
  cooling 5.2 °C off Cape Farewell (172.6°E, 40.5°S) under a NW gale (τ ≈ 0.5 N/m², latent
  200–270 W/m²) with S rising 34.7 → 35.2: upwelling/advection (Kahurangi upwelling), MLD ~10 m.
  Surface currents up to 1.7–2.8 m/s in the 4 m top cell near Cook Strait / Taranaki — high,
  to keep an eye on. Median MLD 21 m. Animation: 3098.
- NE fix also on local branch glw/exchanger-halo-fix-753 (off #753, CPU tests pass); not pushed.

## Status 2026-10-20 02:20 UTC

- Greg (2026-10-18): two more runs on a doubled domain, northwest corner fixed, expanding south and
  east: 150°E–130°W × 27°S–67°S (latitude capped at 67°S; doubling to 87°S would hit the Antarctic ice
  sheet and the pole), at 12 km (720×360) and 25 km (360×180, SO_CELLS_PER_DEGREE=4.5). All runs
  already Float32 (Float64 clock).
- ERA5 now comes from Google's ARCO-ERA5 (tools/arco, slurm/arco.batch): 7032 files for both domains
  in 15 min, identical to CDS files to GRIB packing precision. The parent carries cloud liquid + ice
  only (ARCO has no rain/snow water content; NE 89a28bfee adds `hydrometeors`).
- GLORYS: Copernicus Marine timed out with 16 concurrent jobs; 2 per domain worked.
- 2026-10-20: both the in-session launchers (thought dead) and the Slurm launchers fired, so every run
  was submitted twice into the same run directory. Duplicates segfaulted in setup (3669, 3671) and
  two of the originals hit a NaN Δt (3668 at 8 h, 3674 at once); the same configuration as 3674 runs
  fine (3676), so the failures were not set by the data. Survivors: big12 3670, big25 3676.
  coupled_28d resubmitted alone as 3687. Throughput: big12 0.35 s/step (12× real time, ~56 h);
  big25 0.10 s/step (49×, ~14 h).
- Output files carry the serialized boundary conditions (all GLORYS days): ~5.7 GB each for big12.

## Status 2026-10-20 04:10 UTC — coupled_28d NaN

- coupled_28d failed twice at the same step (iteration 6300–6480, Jan 29 ~08:00, NaN Δt in Breeze's
  acoustic substep count), alone the second time (3687): deterministic, not the duplicate. Its ERA5
  inputs for Jan 29 are clean (no NaN, right times/shapes).
- With SO_NAN_CHECK_AFTER_HOURS=7.9 (every step copies all fields to the host, i.e. synchronizes)
  the same run went to 12 h clean (3693). Suspect a GPU memory bug sensitive to timing / memory
  contents — the StateExchanger's moving window is the prime candidate (NE's own GPU test of it,
  test_nested_simulation.jl:573, hits an illegal memory access, also on the unmodified branch).
- Relaunched as 3701 with SO_SYNC_EVERY_STEP=true (CUDA.synchronize() each step). To do: find the
  out-of-bounds access with the failing test under CUDA_LAUNCH_BLOCKING / compute-sanitizer.

## Status 2026-10-20 07:20 UTC — NaN diagnosis and CFL 0.4

- The 28-day coupled_28d NaN is deterministic (also with per-step GPU sync, 3701). Per-step check
  (3708): clean through iteration 6396, then at 6397 (Jan 29 08:04) 113 whole columns NaN at once
  (5650 = 113 × 50 levels), first at i = 20, j = 254–257 (≈152.2°E, 28.7°S, the ranges behind the
  Gold Coast). The strongest updraft (20 m/s over the North Island) is elsewhere. A column-wise
  failure over steep terrain, not radiation (the hourly RRTMGP update is 62 steps earlier) and not
  the inputs. big25 (CFL 0.5) died the same way at 7.3 days (Feb 5 ~08 UTC).
- CFL 0.4 (Δt ≈ 3.5 s) takes coupled_28d past the failure (3709). Restarted big25 (3712) and big12
  (3713, big12_28d_cfl04) at CFL 0.4; cancelled the CFL 0.5 big12 (3670, at 2.7 days) for disk:
  /home has 65 GB free and a 28-day big12 run writes ~33 GB.
- To do: the column instability itself (terrain-following acoustic substeps over steep terrain?);
  NE GPU illegal access in compute_child_prognostics! (test_nested_simulation.jl:573, blocking
  launches point to breeze_state_exchanger.jl:162); output files carry serialized boundary conditions.

## Status 2026-10-20 15:30 UTC — divergence damping

- coupled_28d at CFL 0.4 (3709) went NaN at 3.08 days (Feb 1 02 UTC). A replay with the per-step
  check from 73.8 h (3716) ran clean to 80 h: the failure is not reproducible over days. Together with
  the 7.96 h failure (3 of 4 runs, one clean), this reads as a marginal instability tipped by GPU
  round-off, not a memory bug. NE's GPU test failure (test_nested_simulation.jl:573) is an NVPTX
  compiler segfault on the Float64 test grid (tests/exchanger_bounds.jl), unrelated.
- NE's nested child has no acoustic divergence damping (NoDivergenceDamping; the thermal-proxy damper
  pushes on a cold start). SO_DIVERGENCE_DAMPING=α now builds the child with Breeze's
  DirectDivergenceDamping (Klemp et al. 2018, no cold-start force) and NE's default sponge and base
  pressure. α = 0.1 at CFL 0.5 (3722) passed the 7.96 h failure.
- Running: coupled_28d (3721, CFL 0.4, undamped), coupled_28d_damp (3723, CFL 0.5, α = 0.1),
  big25_28d (3712), big12_28d_cfl04 (3713). If big12 fails, restart it damped.
