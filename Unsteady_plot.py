import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# ---------------- PARAMETERS ----------------
filename = "Usteady.txt"

T_left = 50.0
T_right = 200.0
T_top = 100.0
T_bottom = 300.0

# ---------------- LOAD DATA ----------------
data = np.loadtxt(filename)

time = data[:, 0]
x = data[:, 1]
y = data[:, 2]
T = data[:, 3]

unique_times = np.unique(time)
n_nodes = len(x) // len(unique_times)

# ---------------- ADD BOUNDARY POINTS ----------------
def add_boundary_points(x, y, T):
    xb, yb, Tb = [], [], []

    n = 100  # boundary resolution

    # LEFT (x=0)
    ys = np.linspace(0, 1, n)
    xb.extend([0.0]*n)
    yb.extend(ys)
    Tb.extend([T_left]*n)

    # RIGHT (x=1)
    xb.extend([1.0]*n)
    yb.extend(ys)
    Tb.extend([T_right]*n)

    # BOTTOM (y=0)
    xs = np.linspace(0, 1, n)
    xb.extend(xs)
    yb.extend([0.0]*n)
    Tb.extend([T_bottom]*n)

    # TOP (y=1)
    xb.extend(xs)
    yb.extend([1.0]*n)
    Tb.extend([T_top]*n)

    x_new = np.concatenate([x, np.array(xb)])
    y_new = np.concatenate([y, np.array(yb)])
    T_new = np.concatenate([T, np.array(Tb)])

    return x_new, y_new, T_new

# ---------------- PLOTTING ----------------
for k, tval in enumerate(unique_times):

    if k % 10 != 0:
        continue

    start = k * n_nodes
    end   = (k + 1) * n_nodes

    x_t = x[start:end]
    y_t = y[start:end]
    T_t = T[start:end]

    # add artificial boundaries
    x_t, y_t, T_t = add_boundary_points(x_t, y_t, T_t)

    plt.figure(figsize=(6,5))

    triang = tri.Triangulation(x_t, y_t)

    plt.tricontourf(triang, T_t, levels=60, cmap='jet')

    plt.colorbar(label="Temperature")
    plt.title(f"Temperature at t = {tval:.5f}")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.tight_layout()
    plt.show()
