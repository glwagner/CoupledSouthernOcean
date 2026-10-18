#!/bin/bash
# Count the inputs a hindcast window needs and print what is missing; exit 0 when complete.
# The domain comes from SO_LONGITUDE / SO_LATITUDE (case.jl defaults); the ERA5 parent and GLORYS
# boxes are the domain padded by 0.5°, the MSLP box is the domain itself.
#   SO_LONGITUDE=150,230 SO_LATITUDE=-67,-27 slurm/check_inputs.sh 2018-01-29T00 2018-02-26T00
start=$1; stop=$2
cd $HOME/southern-ocean/data
IFS=, read west east <<< "${SO_LONGITUDE:-150,190}"
IFS=, read south north <<< "${SO_LATITUDE:--57,-27}"
f1() { printf "%.1f" "$1"; }
padded=$(f1 $(echo "$west - 0.5" | bc))_$(f1 $(echo "$east + 0.5" | bc))_$(f1 $(echo "$south - 0.5" | bc))_$(f1 $(echo "$north + 0.5" | bc))
domain=$(f1 $west)_$(f1 $east)_$(f1 $south)_$(f1 $north)
missing=0
# cloud liquid and ice only (case.jl's default parent_hydrometeors)
variables="geopotential temperature u_component_of_wind v_component_of_wind specific_humidity
           specific_cloud_liquid_water_content specific_cloud_ice_water_content"

hours=()
t=$(date -u -d "${start/T/ }:00" +%s); t_end=$(date -u -d "${stop/T/ }:00" +%s)
while (( t <= t_end )); do hours+=("$(date -u -d @$t +%Y-%m-%dT%H)"); t=$((t + 3600)); done

for v in $variables; do
    n=0
    for h in "${hours[@]}"; do
        [[ -f era5/${v}_ERA5HourlyPressureLevels_${h}_${padded}.nc ]] || n=$((n + 1))
    done
    (( n > 0 )) && echo "ERA5 $v: $n of ${#hours[@]} hours missing"
    missing=$((missing + n))
done

day0=${start%T*}
for snapshot in geopotential_ERA5HourlySingleLevel_${day0}T00_${padded} \
                mean_sea_level_pressure_ERA5HourlySingleLevel_${day0}T00_${domain} \
                skin_temperature_ERA5HourlySingleLevel_${day0}T00_${padded}; do
    [[ -f era5/$snapshot.nc ]] || { echo "missing snapshot $snapshot"; missing=$((missing + 1)); }
done

# GLORYS: u, v, T, S, zos for every day through one day past the stop date
d=$day0; last=$(date -u -d "${stop%T*} + 1 day" +%F)
while [[ ! "$d" > "$last" ]]; do
    n=$(ls glorys 2>/dev/null | grep -v inpainted | grep "GLORYSDaily_${d}T00-00-00" | grep -c "_${padded}.nc")
    (( n < 5 )) && { echo "GLORYS $d: $n of 5 variables"; missing=$((missing + 1)); }
    d=$(date -u -d "$d + 1 day" +%F)
done
[[ -f glorys/deptho_GLORYSStatic_nothing_nothing_${padded}.nc ]] || { echo "missing GLORYS deptho"; missing=$((missing + 1)); }

(( missing == 0 )) && echo "all inputs present for ${domain} ${start} → ${stop}"
exit $(( missing > 0 ))
