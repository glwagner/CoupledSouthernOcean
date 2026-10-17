# Common environment for all southern-ocean jobs (isolated depot; never writes ~/.julia).
export JULIA_DEPOT_PATH="$HOME/.julia-so:/home/greg_aeolus_earth/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/local/share/julia:/home/greg_aeolus_earth/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/share/julia"
export JULIA_NUM_PRECOMPILE_TASKS=${SLURM_CPUS_PER_TASK:-6}
export HDF5_USE_FILE_LOCKING=FALSE
unset LD_LIBRARY_PATH
JULIA="/home/greg_aeolus_earth/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia --startup-file=no"
cd $HOME/southern-ocean
