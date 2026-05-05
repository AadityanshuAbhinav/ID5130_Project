# ID5130 Project — FVM Heat Diffusion: Build, Run, and Evaluate

Parallelization of steady and unsteady 2D diffusion on an unstructured triangular mesh.
Solvers are written in C++ and parallelized with **OpenMP** and **MPI**.

---

## 1. Prerequisites

Tested on **Ubuntu 22.04 LTS**.

| Package | Purpose |
|---------|---------|
| `g++` (≥ 11) | C++ compiler with OpenMP support (`-fopenmp`) |
| `openmpi-bin`, `libopenmpi-dev` | `mpirun` and `mpicxx` |
| `bc` | Floating-point arithmetic in the shell scripts |
| **Gmsh SDK** (≥ 4.11) | Mesh generation C++ API (`gmsh.h`, `libgmsh.so`) |
| `python3`, `pip3` | Plotting (`numpy`, `matplotlib`) |

---

## 2. Install System Packages

```bash
sudo apt update
sudo apt install -y g++ openmpi-bin libopenmpi-dev bc python3-pip
```

---

## 3. Install the Gmsh C++ SDK

The C++ API requires the **SDK** package, which is **different** from the
regular Gmsh desktop application.

- Regular binary (`gmsh-X.Y.Z-Linux64.tgz`) → contains only `bin/` and `share/` — **not what you want**
- SDK (`gmsh-X.Y.Z-Linux64-sdk.tgz`) → contains `include/`, `lib/`, `share/` — **this is the one**

### Step 1 — Download the SDK

Visit [https://gmsh.info/#Download](https://gmsh.info/#Download) and scroll down
to the **"Software Development Kit (SDK)"** section (below the stable release
binaries). Download the **Linux 64-bit SDK**.

Or directly from the terminal:

```bash
cd ~
wget https://gmsh.info/bin/Linux/gmsh-4.13.1-Linux64-sdk.tgz
tar -xzf gmsh-4.13.1-Linux64-sdk.tgz
```

> If the version above is outdated, check the download page for the current
> SDK filename and substitute accordingly.

### Step 2 — Confirm you have the right package

```bash
ls ~/gmsh-4.13.1-Linux64-sdk/
# Must show: include/  lib/  share/
# If you only see bin/ and share/, you have the binary — re-download the SDK.
```

### Step 3 — Install headers and library to `~/.local`

```bash
SDK=~/gmsh-4.13.1-Linux64-sdk     # adjust to match your extracted folder name

mkdir -p ~/.local/include ~/.local/lib

cp "$SDK/include/gmsh.h"       ~/.local/include/
cp "$SDK/lib/libgmsh.so"       ~/.local/lib/
cp "$SDK/lib/libgmsh.so".*     ~/.local/lib/   2>/dev/null || true
```

### Step 4 — Verify

```bash
ls ~/.local/include/gmsh.h
ls ~/.local/lib/libgmsh.so
```

Both files must exist before proceeding.

---

## 4. Install Python Dependencies

```bash
pip3 install numpy matplotlib
```

---

## 5. Update Paths in the Shell Scripts

The scripts hardcode the user's home directory. On a new machine, open both
shell scripts and replace the `GMSH_INC` / `GMSH_LIB` / `MPICXX` / `CXX` lines
with your actual home directory.

```bash
# in run_scaling_omp.sh and run_scaling_mpi.sh, change:
GMSH_INC=/home/aadityanshu/.local/include
GMSH_LIB=/home/aadityanshu/.local/lib

# to (replace <your-username>):
GMSH_INC=/home/<your-username>/.local/include
GMSH_LIB=/home/<your-username>/.local/lib
```

Use `sed` to do both files at once:

```bash
OLD="aadityanshu"
NEW="<your-username>"

sed -i "s|/home/$OLD/|/home/$NEW/|g" run_scaling_omp.sh run_scaling_mpi.sh
```

---

## 6. Project File Overview

```
ID5130 Project/
├── main_omp.cpp        Steady FVM solver — OpenMP (graph-colored GS-SOR)
├── main_mpi.cpp        Steady FVM solver — MPI (block GS, ω=1.0)
├── unsteady_omp.cpp    Unsteady FVM solver — OpenMP (explicit Euler)
├── unsteady_mpi.cpp    Unsteady FVM solver — MPI (explicit Euler)
├── main.cpp            Serial reference steady solver
├── unsteady.cpp        Serial reference unsteady solver
├── run_scaling_omp.sh  Compile + strong/weak scaling study (OMP)
├── run_scaling_mpi.sh  Compile + strong/weak scaling study (MPI)
├── plot_all.py         Generate all performance and solution plots
├── Unsteady_plot.py    Standalone unsteady snapshot plotter
├── results/            CSV output from scaling scripts
└── plots/              PNG output from plot_all.py
```

### CLI Arguments

| Executable | Arg 1 | Arg 2 | Defaults |
|------------|-------|-------|---------|
| `main_omp` | `lc` (mesh size) | `omega` (relaxation) | `0.15`, `1.5` |
| `main_mpi` | `lc` | `omega` | `0.15`, `1.0` |
| `unsteady_omp` | `lc` | `nsteps` | `0.15`, `10000` |
| `unsteady_mpi` | `lc` | `nsteps` | `0.15`, `10000` |

---

## 7. Manual Compilation

Set these variables once in your shell (adjust your username):

```bash
GMSH_INC=~/.local/include
GMSH_LIB=~/.local/lib
```

### OpenMP executables

```bash
g++ -O2 -fopenmp -I$GMSH_INC main_omp.cpp      -o main_omp      -L$GMSH_LIB -lgmsh -Wl,-rpath,$GMSH_LIB
g++ -O2 -fopenmp -I$GMSH_INC unsteady_omp.cpp  -o unsteady_omp  -L$GMSH_LIB -lgmsh -Wl,-rpath,$GMSH_LIB
```

### MPI executables

```bash
mpicxx -O2 -I$GMSH_INC main_mpi.cpp      -o main_mpi      -L$GMSH_LIB -lgmsh -Wl,-rpath,$GMSH_LIB
mpicxx -O2 -I$GMSH_INC unsteady_mpi.cpp  -o unsteady_mpi  -L$GMSH_LIB -lgmsh -Wl,-rpath,$GMSH_LIB
```

> **Note:** Libraries (`-L`, `-lgmsh`) must come **after** the source file.
> Putting them before the source file causes linker errors.

---

## 8. Run a Single Solver

### Steady — OpenMP

```bash
# 1 thread, default mesh
./main_omp

# 2 threads, lc=0.07, omega=1.5
OMP_NUM_THREADS=2 ./main_omp 0.07 1.5
```

Expected stdout:

```
Triangles: 544
omega = 1.5  colors = 9
Converged in 552 iterations
Max difference: ...
Max error: 2.33679
PERF steady omp 2 0.07 544 <setup_s> <solver_s> <total_s> 552
```

Output file: `steady_result.txt` (columns: `x  y  T_fvm  T_exact`)

### Steady — MPI

```bash
# 2 ranks, lc=0.07, omega=1.0
mpirun -n 2 ./main_mpi 0.07 1.0
```

Output file: `steady_result_mpi.txt`

### Unsteady — OpenMP

```bash
# 2 threads, lc=0.07, 1000 steps
OMP_NUM_THREADS=2 ./unsteady_omp 0.07 1000
```

Output file: `Usteady.txt` (columns: `t  x  y  T`, one snapshot per time step)

### Unsteady — MPI

```bash
mpirun -n 2 ./unsteady_mpi 0.07 1000
```

---

## 9. Run the Full Scaling Studies

These scripts compile, then run strong and weak scaling sweeps, and write CSV results.

```bash
bash run_scaling_omp.sh   # → results/scaling_omp.csv
bash run_scaling_mpi.sh   # → results/scaling_mpi.csv
```

### What the scripts test

| Study | lc | Parallelism levels |
|-------|----|--------------------|
| Strong scaling (steady + unsteady) | 0.07 (fixed) | p = 1, 2, 4 |
| Weak scaling (steady + unsteady) | 0.15 / √p | p = 1, 2, 4 |

> The scripts use `RANKS_LIST="1 2 4"` / `THREADS_LIST="1 2 4"`.
> If your machine has fewer than 4 physical cores, the p=4 results will be
> oversubscribed (contention — not a parallelization failure).
> To test only valid core counts, edit the list to `"1 2"`.

### CSV columns

```
study, solver, paradigm, p, lc, elements, setup_time, solver_time, total_time, extra
```

`extra` = iteration count (steady) or number of time steps (unsteady).

---

## 10. Generate All Plots

```bash
python3 plot_all.py --all
```

Individual subsets:

```bash
python3 plot_all.py --steady    # solution + error contour
python3 plot_all.py --scaling   # speedup, efficiency, weak scaling, time breakdown
python3 plot_all.py --unsteady  # unsteady snapshots
```

Output: `plots/*.png`

### Unsteady snapshots only

```bash
# Requires Usteady.txt to exist (run unsteady_omp or unsteady_mpi first)
python3 Unsteady_plot.py
```

> `Unsteady_plot.py` calls `plt.show()` — run it in a desktop session or
> replace `plt.show()` with `plt.savefig(f"snapshot_{k:04d}.png")` for
> headless/SSH use.

---

## 11. Evaluating Performance

### Accuracy (steady solver)

```
Max error: X.XX       ← max|T_fvm - T_analytical| over all elements
```

Reference value at lc=0.07 (544 elements): **2.34 °C**.
Lower lc → finer mesh → smaller error.

### Strong scaling

From `results/scaling_omp.csv` or `scaling_mpi.csv`, compute:

```
Speedup   S(p) = solver_time(p=1) / solver_time(p)
Efficiency E(p) = S(p) / p
```

Expected at p=2: S ≈ 1.94, E ≈ 0.97 (OMP); S ≈ 1.46, E ≈ 0.73 (MPI).

### Weak scaling

```
Weak efficiency  Ew(p) = solver_time(p=1) / solver_time(p)
```

Expected to degrade for this problem because the matrix is stored densely
(O(N²) memory and work per iteration). This is a known algorithmic
limitation, not a bug.

### Convergence (unsteady)

Correctness of the parallel unsteady solvers is checked by:

1. Running with `OMP_NUM_THREADS=1` (or `mpirun -n 1`) and comparing
   `Usteady.txt` output to the multi-thread/rank run — they should match
   to floating-point precision.
2. Running enough steps that the final snapshot matches `steady_result.txt`
   (the unsteady solution must converge to the steady state as t → ∞).

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `error while loading shared libraries: libgmsh.so` | Runtime linker can't find `libgmsh.so` | Recompile with `-Wl,-rpath,<path>`, or run `export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH` |
| `fatal error: gmsh.h: No such file or directory` | Wrong `GMSH_INC` path | Verify `~/.local/include/gmsh.h` exists; update path |
| `mpirun` not found | OpenMPI not installed | `sudo apt install openmpi-bin libopenmpi-dev` |
| T values = `-nan`, solver diverges (MPI) | `omega > 1` with multiple ranks | Use `omega=1.0` for MPI (`./main_mpi <lc> 1.0`) |
| `bc` not found | Missing utility | `sudo apt install bc` |
| Scaling script fails to compile | Libraries before source in linker flags | Use the manual compilation commands in Section 7 |
