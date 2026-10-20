# CoupledSouthernOcean

Regional coupled atmosphere–ocean hindcasts around New Zealand and the Southern Ocean, February 2018
(ex-tropical cyclones Fehi and Gita, during the 2017/18 Tasman Sea marine heatwave):

- a [Breeze](https://github.com/NumericalEarth/Breeze.jl) compressible atmosphere nested in ERA5,
- a hydrostatic [Oceananigans](https://github.com/CliMA/Oceananigans.jl) ocean with GLORYS open boundaries
  (radiation / Flather / tracer reservoirs),
- coupled by [NumericalEarth](https://github.com/NumericalEarth/NumericalEarth.jl) over a fractional
  land–ocean surface (`SurfacePartition`, with a `SlabLand` for New Zealand).

The default domain is 150°E–170°W × 57°S–27°S at 1/9° (≈ 12 km); the case is configured by
environment variables in `case.jl` (`SO_START`, `SO_HOURS`, `SO_LONGITUDE`, `SO_LATITUDE`,
`SO_CELLS_PER_DEGREE`, …). `PLAN.md` is the running log of decisions, diagnoses and status.

## Layout

| path | what |
|---|---|
| `case.jl` | dates, domain, resolution, data directories |
| `hindcast.jl` | the coupled model: nested Breeze atmosphere, land, ocean (`SO_OCEAN=prescribed` or `prognostic`), output |
| `ocean.jl` | the regional ocean: grid, bathymetry, GLORYS open boundaries and initial state |
| `ocean_only.jl` | the ocean alone (zero surface fluxes), for diagnosing it |
| `predownload.jl` | fetches the ERA5 and GLORYS inputs a window needs; `SO_ARCO_PLAN` lists missing ERA5 files instead |
| `tools/arco/arco.jl` | cuts those ERA5 files from Google's ARCO-ERA5 (no CDS queue) |
| `analysis/` | animation and diagnostics of the output |
| `slurm/` | batch scripts (run, download, input check, launcher, NumericalEarth tests) |
| `tests/` | probes written while debugging (GPU kernels, dateline handling, deep GLORYS levels) |

## Running

```sh
sbatch slurm/arco.batch data/arco_plan.tsv                     # ERA5 (after SO_ARCO_PLAN=… predownload.jl)
SO_ERA5=false SO_SNAPSHOTS=false julia --project predownload.jl # GLORYS
sbatch --job-name=so-run --export=ALL,SO_NAME=coupled_28d,SO_OCEAN=prognostic,SO_HOURS=672,SO_START=2018-01-29T00:00:00 slurm/hindcast.batch
sbatch slurm/animate.batch runs/coupled_28d
```

`slurm/env.sh` uses an isolated Julia depot (`~/.julia-so`).

## Dependencies

`Project.toml` takes NumericalEarth from a local checkout (`../NumericalEarth-so`) of the integration
branch `glw/southern-ocean`: NumericalEarth main plus PRs #664 (`SurfacePartition`) and #753, and fixes
made for this project (prognostic-ocean radiation and flux assembly, GLORYS inpainting of empty deep
levels, the coupler's halo fill of the ocean's own fields, a choice of ERA5 hydrometeors). That branch
is not yet on GitHub.
