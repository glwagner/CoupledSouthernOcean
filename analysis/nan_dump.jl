# Read the nan_dump.jld2 a run with SO_NAN_CHECK_AFTER_HOURS writes when a field goes NaN or Inf:
# where the broken step's NaNs are, and the last clean state around them and at its extremes.
#   julia --project analysis/nan_dump.jl runs/<name>/nan_dump.jld2
using JLD2
using Printf

file = ARGS[1]
names, previous, current = jldopen(f -> (f["names"], f["previous"], f["current"]), file)
@info "dump" file iteration = jldopen(f -> f["iteration"], file) Δt = jldopen(f -> f["Δt"], file)

finite_extrema(A) = (B = filter(isfinite, A); isempty(B) ? (NaN, NaN) : extrema(B))

println("\n== broken step: NaN/Inf counts and their index box (parent indices, halos included)")
for (name, A) in zip(names, current)
    bad = findall(!isfinite, A)
    isempty(bad) && continue
    lo = Tuple(minimum(bad)); hi = Tuple(maximum(bad))
    @printf("%-28s %8d bad  box %s – %s\n", name, length(bad), lo, hi)
end

# The column where the atmosphere first broke, and the clean state there
ρ_index = findfirst(n -> occursin("ρ", n) && startswith(n, "atmosphere"), names)
bad = findall(!isfinite, current[ρ_index])
I = isempty(bad) ? nothing : first(bad)

println("\n== last clean state: extrema of every field")
for (name, A) in zip(names, previous)
    lo, hi = finite_extrema(A)
    @printf("%-28s [% .4e, % .4e]  size %s\n", name, lo, hi, size(A))
end

if !isnothing(I)
    i, j = I[1], I[2]
    println("\n== last clean column at (i, j) = ($i, $j) (parent indices)")
    for (name, A) in zip(names, previous)
        ndims(A) == 3 || continue
        startswith(name, "atmosphere") || continue
        size(A, 3) > 3 || continue
        column = A[i, j, :]
        @printf("%-24s %s\n", name, join([@sprintf("%.3g", c) for c in column[1:min(end, 20)]], " "))
    end
    println("\n== surface around (i, j), last clean state (±2 cells)")
    for (name, A) in zip(names, previous)
        startswith(name, "atmosphere") && continue
        startswith(name, "ocean") && continue
        B = ndims(A) == 3 ? A[:, :, 1] : A
        (i + 2 ≤ size(B, 1) && j + 2 ≤ size(B, 2)) || continue
        @printf("%-24s %s\n", name, join([@sprintf("%.3g", b) for b in B[i-2:i+2, j]], " "))
    end
end
