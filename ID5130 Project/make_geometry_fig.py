import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(4, 4.5))
ax.set_aspect("equal")
ax.axis("off")

# Two triangles sharing face a-b
# Face nodes
a = np.array([1.6,  0.0])   # bottom-right
b = np.array([0.0,  2.5])   # top

# Left triangle nodes  (c0 side)
p_left  = np.array([-1.5, 0.6])

# Right triangle nodes (c1 side)
p_right = np.array([ 2.2, 2.2])

# Centroids
c0 = (a + b + p_left)  / 3.0
c1 = (a + b + p_right) / 3.0

# Draw triangles
tri_left  = plt.Polygon([a, b, p_left],  fill=False, edgecolor="black", lw=1.0)
tri_right = plt.Polygon([a, b, p_right], fill=False, edgecolor="black", lw=1.0)
ax.add_patch(tri_left)
ax.add_patch(tri_right)

# Highlight shared face a-b
ax.plot([a[0], b[0]], [a[1], b[1]], "k-", lw=1.4)

# Centroids
for pt, lbl, offset in [(c0, r"$c_0$", (-0.35, -0.15)),
                         (c1, r"$c_1$",  ( 0.12, -0.15))]:
    ax.plot(*pt, "ko", ms=3.5)
    ax.text(pt[0]+offset[0], pt[1]+offset[1], lbl, fontsize=11, ha="center")

# e_xi arrow: c0 -> c1
xi_vec = c1 - c0
xi_len = np.linalg.norm(xi_vec)
xi_hat = xi_vec / xi_len
ax.annotate("", xy=c1 - 0.18*xi_hat, xytext=c0 + 0.18*xi_hat,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
mid_xi = (c0 + c1) / 2
ax.text(mid_xi[0]+0.22, mid_xi[1]-0.22, r"$\hat{e}_\xi$", fontsize=11)

# Face midpoint
f_mid = (a + b) / 2.0

# e_eta: along face from a to b
eta_vec = b - a
eta_len = np.linalg.norm(eta_vec)
eta_hat = eta_vec / eta_len
eta_start = f_mid - 0.45 * eta_hat
eta_end   = f_mid + 0.45 * eta_hat
ax.annotate("", xy=eta_end, xytext=eta_start,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
ax.text(eta_end[0]-0.05, eta_end[1]+0.12, r"$\hat{e}_\eta$", fontsize=11, ha="center")

# A_f: outward normal (perpendicular to face, pointing away from c0)
normal = np.array([eta_hat[1], -eta_hat[0]])   # rotate eta 90° CW
# ensure it points away from c0
if np.dot(normal, c1 - f_mid) < 0:
    normal = -normal
af_start = f_mid
af_end   = f_mid + 0.7 * normal
ax.annotate("", xy=af_end, xytext=af_start,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
ax.text(af_end[0]+0.12, af_end[1]+0.05, r"$\mathbf{A}_f$", fontsize=11)

# Right-angle marker between eta and A_f
s = 0.13
corner = f_mid
p1 = corner + s * eta_hat
p2 = corner + s * eta_hat + s * normal
p3 = corner + s * normal
box = plt.Polygon([corner, p1, p2, p3], fill=False, edgecolor="black", lw=0.8)
ax.add_patch(box)

# Node labels
ax.text(b[0]-0.18, b[1]+0.12, r"$b$", fontsize=12, fontweight="bold", ha="center")
ax.text(a[0]+0.18, a[1]-0.15, r"$a$", fontsize=12, fontweight="bold", ha="center")

ax.set_xlim(-2.0, 3.2)
ax.set_ylim(-0.6, 3.1)

plt.tight_layout(pad=0.2)
plt.savefig("plots/face_geometry.png", dpi=180, bbox_inches="tight",
            facecolor="white")
print("Saved plots/face_geometry.png")
