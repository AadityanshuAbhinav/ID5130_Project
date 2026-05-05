#!/usr/bin/env bash
# run_scaling_omp.sh  —  OMP strong + weak scaling studies
# Usage: bash run_scaling_omp.sh
#
# Outputs: results/scaling_omp.csv
# CSV columns: study,solver,paradigm,p,lc,elements,setup_time,solver_time,total_time,extra

set -e

GMSH_INC=/home/aadityanshu/.local/include
GMSH_LIB=/home/aadityanshu/.local/lib
CXX="g++ -O2 -fopenmp -I${GMSH_INC}"
LIBS="-L${GMSH_LIB} -lgmsh -Wl,-rpath,${GMSH_LIB}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------ #
# Compile                                                             #
# ------------------------------------------------------------------ #
echo "=== Compiling OMP executables ==="
$CXX main_omp.cpp      -o main_omp      $LIBS
$CXX unsteady_omp.cpp  -o unsteady_omp  $LIBS
echo "  done."

# ------------------------------------------------------------------ #
# Output CSV                                                          #
# ------------------------------------------------------------------ #
mkdir -p results
CSV="results/scaling_omp.csv"
echo "study,solver,paradigm,p,lc,elements,setup_time,solver_time,total_time,extra" > "$CSV"

# Helper: parse PERF line and append one row to CSV
# $1 = study (strong|weak)   output already in $TMPOUT
append_perf() {
    local study="$1"
    grep "^PERF" "$TMPOUT" | while read -r line; do
        # PERF <solver> <paradigm> <p> <lc> <elements> <setup> <solve> <total> <extra>
        echo "$line" | awk -v study="$study" \
            '{print study","$2","$3","$4","$5","$6","$7","$8","$9","$10}' >> "$CSV"
    done
}

TMPOUT=$(mktemp /tmp/omp_run.XXXXXX)

# ------------------------------------------------------------------ #
# Strong scaling — steady                                             #
# ------------------------------------------------------------------ #
LC_STRONG=0.07
THREADS_LIST="1 2 4"
OMEGA=1.5

echo ""
echo "=== Strong scaling — steady (lc=${LC_STRONG}, omega=${OMEGA}) ==="
for T in $THREADS_LIST; do
    export OMP_NUM_THREADS=$T
    echo "  threads=$T"
    ./main_omp "$LC_STRONG" "$OMEGA" 2>&1 | tee "$TMPOUT"
    append_perf strong
done

# ------------------------------------------------------------------ #
# Strong scaling — unsteady                                           #
# ------------------------------------------------------------------ #
NSTEPS=1000

echo ""
echo "=== Strong scaling — unsteady (lc=${LC_STRONG}, nsteps=${NSTEPS}) ==="
for T in $THREADS_LIST; do
    export OMP_NUM_THREADS=$T
    echo "  threads=$T"
    ./unsteady_omp "$LC_STRONG" "$NSTEPS" 2>&1 | tee "$TMPOUT"
    append_perf strong
done

# ------------------------------------------------------------------ #
# Weak scaling — steady                                               #
# Each thread gets roughly the same number of elements:
#   lc(p) = lc_base / sqrt(p)   →  elements ≈ p × elements(1 thread)
# ------------------------------------------------------------------ #
LC_BASE=0.15     # lc for 1 thread

echo ""
echo "=== Weak scaling — steady ==="
for T in $THREADS_LIST; do
    export OMP_NUM_THREADS=$T
    # bc -l for floating-point arithmetic
    LC=$(echo "scale=6; $LC_BASE / sqrt($T)" | bc -l)
    echo "  threads=$T  lc=$LC"
    ./main_omp "$LC" "$OMEGA" 2>&1 | tee "$TMPOUT"
    grep "^PERF" "$TMPOUT" | awk -v study=weak \
        '{print study","$2","$3","$4","$5","$6","$7","$8","$9","$10}' >> "$CSV"
done

# ------------------------------------------------------------------ #
# Weak scaling — unsteady                                             #
# ------------------------------------------------------------------ #
echo ""
echo "=== Weak scaling — unsteady (nsteps=${NSTEPS}) ==="
for T in $THREADS_LIST; do
    export OMP_NUM_THREADS=$T
    LC=$(echo "scale=6; $LC_BASE / sqrt($T)" | bc -l)
    echo "  threads=$T  lc=$LC"
    ./unsteady_omp "$LC" "$NSTEPS" 2>&1 | tee "$TMPOUT"
    grep "^PERF" "$TMPOUT" | awk -v study=weak \
        '{print study","$2","$3","$4","$5","$6","$7","$8","$9","$10}' >> "$CSV"
done

rm -f "$TMPOUT"

echo ""
echo "=== Done. Results in $CSV ==="
