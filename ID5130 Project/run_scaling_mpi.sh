#!/usr/bin/env bash
# run_scaling_mpi.sh  —  MPI strong + weak scaling studies
# Usage: bash run_scaling_mpi.sh
#
# Outputs: results/scaling_mpi.csv
# CSV columns: study,solver,paradigm,p,lc,elements,setup_time,solver_time,total_time,extra

set -e

GMSH_INC=/home/aadityanshu/.local/include
GMSH_LIB=/home/aadityanshu/.local/lib
MPICXX="mpicxx -O2 -I${GMSH_INC}"
LIBS="-L${GMSH_LIB} -lgmsh -Wl,-rpath,${GMSH_LIB}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Compile MPI executables
echo "=== Compiling MPI executables ==="
$MPICXX main_mpi.cpp      -o main_mpi      $LIBS
$MPICXX unsteady_mpi.cpp  -o unsteady_mpi  $LIBS
echo "  done."

# Output CSV header
mkdir -p results
CSV="results/scaling_mpi.csv"
echo "study,solver,paradigm,p,lc,elements,setup_time,solver_time,total_time,extra" > "$CSV"

TMPOUT=$(mktemp /tmp/mpi_run.XXXXXX)

# Strong scaling — steady
LC_STRONG=0.07
RANKS_LIST="1 2 4"
OMEGA=1.5

echo ""
echo "=== Strong scaling — steady (lc=${LC_STRONG}, omega=${OMEGA}) ==="
for P in $RANKS_LIST; do
    echo "  ranks=$P"
    mpirun -n "$P" ./main_mpi "$LC_STRONG" "$OMEGA" 2>&1 | tee "$TMPOUT"
    grep "^PERF" "$TMPOUT" | awk -v study=strong \
        '{print study","$2","$3","$4","$5","$6","$7","$8","$9","$10}' >> "$CSV"
done

# Strong scaling — unsteady
NSTEPS=1000

echo ""
echo "=== Strong scaling — unsteady (lc=${LC_STRONG}, nsteps=${NSTEPS}) ==="
for P in $RANKS_LIST; do
    echo "  ranks=$P"
    mpirun -n "$P" ./unsteady_mpi "$LC_STRONG" "$NSTEPS" 2>&1 | tee "$TMPOUT"
    grep "^PERF" "$TMPOUT" | awk -v study=strong \
        '{print study","$2","$3","$4","$5","$6","$7","$8","$9","$10}' >> "$CSV"
done

# Weak scaling — steady
LC_BASE=0.15

echo ""
echo "=== Weak scaling — steady ==="
for P in $RANKS_LIST; do
    LC=$(echo "scale=6; $LC_BASE / sqrt($P)" | bc -l)
    echo "  ranks=$P  lc=$LC"
    mpirun -n "$P" ./main_mpi "$LC" "$OMEGA" 2>&1 | tee "$TMPOUT"
    grep "^PERF" "$TMPOUT" | awk -v study=weak \
        '{print study","$2","$3","$4","$5","$6","$7","$8","$9","$10}' >> "$CSV"
done

# Weak scaling — unsteady
echo ""
echo "=== Weak scaling — unsteady (nsteps=${NSTEPS}) ==="
for P in $RANKS_LIST; do
    LC=$(echo "scale=6; $LC_BASE / sqrt($P)" | bc -l)
    echo "  ranks=$P  lc=$LC"
    mpirun -n "$P" ./unsteady_mpi "$LC" "$NSTEPS" 2>&1 | tee "$TMPOUT"
    grep "^PERF" "$TMPOUT" | awk -v study=weak \
        '{print study","$2","$3","$4","$5","$6","$7","$8","$9","$10}' >> "$CSV"
done

rm -f "$TMPOUT"

echo ""
echo "=== Done. Results in $CSV ==="
