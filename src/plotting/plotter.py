import gc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from phi.jax.flow import *
from phi.vis import plot
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from tqdm.notebook import trange
from typing_utils import eval_forward_ref
import gc
import numpy as np
#%%
import pickle



def animate_speed(vx, vy, bx=None, by=None, interval=50, vmin=None, vmax=None, save_path=None
                  ,animation_title="Velocity Speed Animation"):
    """
    vx, vy: arrays (T, X, Y)
    bx/by: (min,max) bounds for x,y (optional)
    interval: ms between frames
    vmin/vmax: color scale (if None, computed from data percentiles)
    save_path: *.gif or *.mp4 to save (optional)
    """
    T, X, Y = vx.shape
    speed = np.sqrt(vx ** 2 + vy ** 2)

    if vmin is None or vmax is None:
        lo, hi = np.percentile(speed, [2, 98])  # robust scaling
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    if bx is None: bx = (0.0, float(X))
    if by is None: by = (0.0, float(Y))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(speed[0].T, origin='lower', aspect='auto',
                   extent=(bx[0], bx[1], by[0], by[1]), vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("|v|")
    ax.set_title("Speed |v|  (t=0)")
    ax.set_xlabel("X");
    ax.set_ylabel("Y")
    fig.suptitle(animation_title)

    def update(t):
        im.set_data(speed[t].T)
        ax.set_title(f"Speed |v|  (t={t})")
        return (im,)

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    if save_path:
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer=PillowWriter(fps=max(1, 1000 // interval)))
        elif save_path.lower().endswith(".mp4"):
            anim.save(save_path, writer=FFMpegWriter(fps=max(1, 1000 // interval)))
        print(f"Saved animation to {save_path}")
    return anim


# ---------- 2) Quiver (vector field) ----------
def animate_quiver(vx, vy, bx=None, by=None, step=3, interval=50, scale=None, save_path=None):
    """
    Subsampled quiver animation.
    step: take every `step` cell in x and y.
    scale: quiver scale (None lets Matplotlib choose)
    """
    T, X, Y = vx.shape
    if bx is None: bx = (0.0, float(X))
    if by is None: by = (0.0, float(Y))
    dx = (bx[1] - bx[0]) / X
    dy = (by[1] - by[0]) / Y
    xs = bx[0] + (np.arange(X) + 0.5) * dx
    ys = by[0] + (np.arange(Y) + 0.5) * dy
    Xg, Yg = np.meshgrid(xs, ys, indexing='ij')

    fig, ax = plt.subplots(figsize=(6, 4.5))
    Q = ax.quiver(Xg[::step, ::step], Yg[::step, ::step],
                  vx[0, ::step, ::step], vy[0, ::step, ::step],
                  angles='xy', scale_units='xy', scale=scale)
    ax.set_xlim(bx);
    ax.set_ylim(by)
    ax.set_xlabel("X");
    ax.set_ylabel("Y")
    ax.set_title("Velocity quiver  (t=0)")

    def update(t):
        Q.set_UVC(vx[t, ::step, ::step], vy[t, ::step, ::step])
        ax.set_title(f"Velocity quiver  (t={t})")
        return (Q,)

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    if save_path:
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer=PillowWriter(fps=max(1, 1000 // interval)))
        elif save_path.lower().endswith(".mp4"):
            anim.save(save_path, writer=FFMpegWriter(fps=max(1, 1000 // interval)))
        print(f"Saved animation to {save_path}")
    return anim

def sanity_check_vx(vx):
    """
    Plot first and last time slice of vx (velocity_x).
    vx : numpy array of shape (T, X, Y)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(vx[0].T, origin='lower', aspect='auto', cmap='coolwarm')
    axes[0].set_title("vx at t=0")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(vx[-1].T, origin='lower', aspect='auto', cmap='coolwarm')
    axes[1].set_title(f"vx at t={vx.shape[0] - 1}")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()



def plot_curl(v_trj_2d):
    # Extract vorticity at 3 frames
    curl_first = v_trj_2d.curl().values.numpy('x,y')

    fig, axes = plt.subplots(1, 1, figsize=(15, 4))
    ax = axes
    im = ax.imshow(curl_first.T, origin="lower", cmap="bwr")
    ax.set_title("Velocity Vorticity at final step")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04, label="Vorticity")

    plt.tight_layout()
    plt.show()

def plot_vorticity_snapshot(data_dict, frame, cmap="RdBu_r"):
    """
    Plot precomputed vorticity field at one time step.
    """
    X, Y, t, vort_all = data_dict["X"], data_dict["Y"], data_dict["t"], data_dict["vorticity"]
    omega = vort_all[:, :, frame]

    plt.figure(figsize=(7, 4))
    plt.contourf(X, Y, omega, levels=200, cmap=cmap)
    plt.colorbar(label="Vorticity")
    plt.title(f"Vorticity at t={t[frame]:.3f}")
    plt.xlabel("x");
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

def custom_plot(v_trj_2d,save_path=None,title="Vorticity Evolution", cmap="RdBu_r", extent=(0,200,0,100)):
    nsteps = v_trj_2d.time.size
    frames = [0, nsteps // 2, nsteps - 1]

    # Build grids
    sample = v_trj_2d.time[0].curl().values.numpy('x,y')
    nx, ny = sample.shape
    X = np.linspace(extent[0], extent[1], nx)
    Y = np.linspace(extent[2], extent[3], ny)
    X, Y = np.meshgrid(X, Y, indexing='xy')

    # Collect vorticity fields
    vort = [v_trj_2d.time[f].curl().values.numpy('x,y').T for f in frames]
    vmax = max(np.abs(v).max() for v in vort)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, v, f in zip(axes, vort, frames):
        im = ax.contourf(X, Y, v, levels=200, cmap=cmap)
        ax.set_title(f"Step {f}")
        ax.set_xlabel("x");
        ax.set_ylabel("y");
        ax.set_aspect("equal")

    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical", label="Vorticity")
    fig.suptitle(title)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)