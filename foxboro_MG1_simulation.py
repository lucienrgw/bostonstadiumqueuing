"""
MBTA Foxboro -> South Station  |  Poisson Arrivals + N(t) Adaptive Dispatch
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm

# =============================================================================
# PARAMETERS  —  edit everything here
# =============================================================================

# Service
TRAIN_CAPACITY        = 1440   # passengers per train
NUM_TRAINS            = 14     # maximum trains available
FIRST_DEPARTURE       = 30     # minutes after whistle before first train can leave

# Arrival shape  (Poisson rate lambda(t) ~ Normal)
ARRIVAL_PEAK      = 60     # minute at which fan arrival rate peaks
ARRIVAL_SIGMA     = 50     # spread (std dev) of the arrival surge in minutes
ARRIVAL_QUIET_PERIOD  = 5      # minutes of near-zero arrivals right after whistle

# N(t) dispatch policy
NT_CAPACITY_THRESHOLD = 1440   # send a train if queue >= this many passengers
NT_WAIT_THRESHOLD     = 45     # send a train if oldest passenger has waited >= this many minutes
NT_MIN_HEADWAY        = 4      # minimum minutes between consecutive dispatches

# Fixed-headway reference (original plan)
FIXED_HEADWAY         = 15     # minutes between trains in original plan

# Simulation / plot
RANDOM_SEED           = 42
PLOT_BUFFER_MIN       = 40     # extra minutes plotted beyond last departure
OUTPUT_CUMULATIVE     = "foxboro_cumulative_b.png"
OUTPUT_QUEUE_DEPTH    = "foxboro_queue_depth_b.png"

# Colors
COLOR_ARRIVAL         = "#1a6bbd"
COLOR_NT              = "#2ca44e"
COLOR_FIXED           = "#d85a30"
COLOR_TRAIN_LINE      = "#aaaaaa"

# =============================================================================
# DERIVED (do not edit)
# =============================================================================

TOTAL_PASSENGERS = NUM_TRAINS * TRAIN_CAPACITY
T_MAX            = FIRST_DEPARTURE + NUM_TRAINS * 20 + PLOT_BUFFER_MIN   # generous upper bound; actual last dispatch determined by simulation
T                = np.linspace(0, T_MAX, 4000)

# =============================================================================
# SIMULATION
# =============================================================================

def draw_arrivals():
    """Draw one Poisson arrival trace, shared by both simulations."""
    rng       = np.random.default_rng(RANDOM_SEED)
    t_minutes = np.arange(0, T_MAX + 1, dtype=float)
    raw_shape = norm.pdf(t_minutes, ARRIVAL_PEAK, ARRIVAL_SIGMA)
    lam_t     = raw_shape / raw_shape.sum() * TOTAL_PASSENGERS
    arrivals  = rng.poisson(lam_t)
    arrivals[:ARRIVAL_QUIET_PERIOD] = 0
    return arrivals

def simulate_nt(arrivals_per_min):
    """N(t) adaptive dispatch policy."""
    t_minutes = np.arange(0, T_MAX + 1, dtype=float)

    queue             = 0
    oldest_wait       = 0
    total_boarded     = 0
    trains_dispatched = 0
    last_dispatch     = -999
    dispatch_log      = []
    cum_arr_sim       = np.zeros(len(t_minutes))

    for tick in range(len(t_minutes)):
        new_arr            = arrivals_per_min[tick]
        queue             += new_arr
        cum_arr_sim[tick]  = (cum_arr_sim[tick - 1] if tick > 0 else 0) + new_arr
        oldest_wait        = oldest_wait + 1 if queue > 0 else 0

        if tick < FIRST_DEPARTURE:
            continue

        platform_ready = (tick - last_dispatch) >= NT_MIN_HEADWAY
        trigger_N      = queue >= NT_CAPACITY_THRESHOLD
        trigger_T      = oldest_wait >= NT_WAIT_THRESHOLD

        if platform_ready and (trigger_N or trigger_T) and trains_dispatched < NUM_TRAINS:
            boarded            = min(queue, TRAIN_CAPACITY)
            queue             -= boarded
            total_boarded     += boarded
            trains_dispatched += 1
            last_dispatch      = tick
            oldest_wait        = 0
            dispatch_log.append((tick, total_boarded))

    # A(t): interpolate — arrivals accumulate continuously
    cum_arr = np.interp(T, t_minutes, cum_arr_sim)

    # D(t): true step function — no interpolation
    cum_dep = np.zeros(len(T))
    for d_time, cum_b in dispatch_log:
        cum_dep[T >= d_time] = cum_b

    mean_wait = np.trapezoid(np.maximum(cum_arr - cum_dep, 0), T) / max(total_boarded, 1)

    return cum_arr, cum_dep, dispatch_log, mean_wait, trains_dispatched

def simulate_fixed(arrivals_per_min):
    """Fixed-headway reference: one train every FIXED_HEADWAY min from FIRST_DEPARTURE."""
    t_minutes     = np.arange(0, T_MAX + 1, dtype=float)
    fixed_departs = [FIRST_DEPARTURE + i * FIXED_HEADWAY for i in range(NUM_TRAINS)]
    cum_arr_sim   = np.cumsum(arrivals_per_min).astype(float)

    total_boarded = 0
    dispatch_log  = []

    for d_time in fixed_departs:
        tick    = int(d_time)
        arrived = cum_arr_sim[min(tick, len(cum_arr_sim) - 1)]
        boarded = min(arrived - total_boarded, TRAIN_CAPACITY)
        total_boarded += boarded
        dispatch_log.append((d_time, total_boarded))

    cum_dep = np.zeros(len(T))
    for d_time, cum_b in dispatch_log:
        cum_dep[T >= d_time] = cum_b

    cum_arr   = np.interp(T, t_minutes, cum_arr_sim)
    mean_wait = np.trapezoid(np.maximum(cum_arr - cum_dep, 0), T) / max(total_boarded, 1)

    return cum_dep, dispatch_log, mean_wait

# =============================================================================
# PLOTTING
# =============================================================================

def add_param_box(ax):
    lines = [
        ("Service",           ""),
        ("  Train capacity",  f"{TRAIN_CAPACITY:,} pax"),
        ("  Max trains",      f"{NUM_TRAINS}"),
        ("  First departure", f"t = {FIRST_DEPARTURE} min"),
        ("",                  ""),
        ("Arrivals",          ""),
        ("  Peak",            f"t = {ARRIVAL_PEAK} min"),
        ("  Sigma",           f"{ARRIVAL_SIGMA} min"),
        ("  Quiet period",    f"{ARRIVAL_QUIET_PERIOD} min"),
        ("",                  ""),
        ("N(t) Policy",       ""),
        ("  Queue trigger",   f">= {NT_CAPACITY_THRESHOLD:,} pax"),
        ("  Wait trigger",    f">= {NT_WAIT_THRESHOLD} min"),
        ("  Min headway",     f"{NT_MIN_HEADWAY} min"),
        ("",                  ""),
        ("Reference Plan",    ""),
        ("  Headway",         f"{FIXED_HEADWAY} min"),
    ]
    text = "\n".join(f"{label:<18}{value}" for label, value in lines)
    ax.text(
        1.02, 1.0, text,
        transform=ax.transAxes,
        fontsize=8, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7",
                  edgecolor="#cccccc", linewidth=0.8)
    )

def add_left_behind_callout(ax, cum_arr, cum_dep, color, x_frac, label):
    left_behind = int(round(cum_arr[-1] - cum_dep[-1]))
    if left_behind > 0:
        y_arr     = cum_arr[-1]
        y_dep     = cum_dep[-1]
        x_callout = T_MAX * x_frac
        ax.annotate("", xy=(x_callout, y_dep), xytext=(x_callout, y_arr),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.5))
        ax.text(x_callout - 1, (y_arr + y_dep) / 2,
                f"{left_behind:,} pax\nleft behind\n({label})",
                fontsize=8, color=color, ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, linewidth=0.8))

def plot_cumulative(cum_arr, cum_dep_nt, dispatch_log_nt, mean_wait_nt, n_trains_nt,
                    cum_dep_fixed, dispatch_log_fixed, mean_wait_fixed):
    fig, ax = plt.subplots(figsize=(15, 6), facecolor="#ffffff")

    ylim_top = TOTAL_PASSENGERS * 1.05

    # Shaded gaps
    ax.fill_between(T, cum_arr, cum_dep_nt,
                    where=(cum_arr >= cum_dep_nt),
                    color=COLOR_NT, alpha=0.06)
    ax.fill_between(T, cum_arr, cum_dep_fixed,
                    where=(cum_arr >= cum_dep_fixed),
                    color=COLOR_FIXED, alpha=0.06)

    # Curves
    ax.plot(T, cum_arr,       color=COLOR_ARRIVAL, lw=2,   label="A(t)  cumulative arrivals")
    ax.step(T, cum_dep_nt,    color=COLOR_NT,      lw=2.5, label=f"D(t)  N(t) adaptive  (mean wait {mean_wait_nt:.1f} min)",    where="pre")
    ax.step(T, cum_dep_fixed, color=COLOR_FIXED,   lw=1.8, label=f"D(t)  fixed {FIXED_HEADWAY}-min headway  (mean wait {mean_wait_fixed:.1f} min)", where="pre", ls="--")

    # Train markers — N(t)
    for i, (td, _) in enumerate(dispatch_log_nt):
        ax.axvline(td, color=COLOR_NT, lw=0.6, ls="--", alpha=0.4)
        if i % 2 == 0 or i == len(dispatch_log_nt) - 1:
            ax.text(td + 0.5, ylim_top * 0.97, f"T{i+1}",
                    fontsize=6.5, color=COLOR_NT, ha="left", va="top")

    # Left-behind callouts — slightly offset so they don't overlap
    add_left_behind_callout(ax, cum_arr, cum_dep_nt,    COLOR_NT,    0.91, "N(t)")
    add_left_behind_callout(ax, cum_arr, cum_dep_fixed, COLOR_FIXED, 0.97, "fixed")

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, ylim_top)
    ax.set_xlabel("Minutes after final whistle", fontsize=9)
    ax.set_ylabel("Cumulative passengers", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.1f}k"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title(
        f"Poisson arrivals (peak t={ARRIVAL_PEAK} min, σ={ARRIVAL_SIGMA} min)  ·  "
        f"N(t): queue≥{NT_CAPACITY_THRESHOLD} OR wait≥{NT_WAIT_THRESHOLD} min  ({n_trains_nt} trains)  ·  "
        f"Reference: fixed {FIXED_HEADWAY}-min headway ({NUM_TRAINS} trains)",
        fontsize=10, fontweight="bold"
    )

    add_param_box(ax)
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    plt.savefig(OUTPUT_CUMULATIVE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_CUMULATIVE}")


def plot_queue_depth(cum_arr, cum_dep_nt, cum_dep_fixed, mean_wait_nt, mean_wait_fixed):
    Q_nt    = np.maximum(cum_arr - cum_dep_nt,    0)
    Q_fixed = np.maximum(cum_arr - cum_dep_fixed, 0)

    fig, ax = plt.subplots(figsize=(15, 5), facecolor="#ffffff")

    ax.plot(T, Q_nt,    color=COLOR_NT,    lw=2,   label=f"N(t) adaptive  (mean wait {mean_wait_nt:.1f} min)")
    ax.plot(T, Q_fixed, color=COLOR_FIXED, lw=1.8, label=f"Fixed {FIXED_HEADWAY}-min headway  (mean wait {mean_wait_fixed:.1f} min)", ls="--")
    ax.fill_between(T, Q_nt,    alpha=0.08, color=COLOR_NT)
    ax.fill_between(T, Q_fixed, alpha=0.08, color=COLOR_FIXED)

    for Q, color, label in [(Q_nt, COLOR_NT, "N(t)"), (Q_fixed, COLOR_FIXED, "fixed")]:
        peak_idx = np.argmax(Q)
        ax.annotate(
            f"{label} peak\n{Q[peak_idx]:.0f} pax @ t={T[peak_idx]:.0f} min",
            xy=(T[peak_idx], Q[peak_idx]),
            xytext=(T[peak_idx] + 10, Q[peak_idx] - 120),
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.9)
        )

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0)
    ax.set_xlabel("Minutes after final whistle", fontsize=9)
    ax.set_ylabel("Passengers waiting on platform", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.1f}k"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title("Queue Depth  Q(t) = A(t) − D(t)  |  N(t) vs Fixed-Headway",
                 fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_QUEUE_DEPTH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_QUEUE_DEPTH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    arrivals = draw_arrivals()

    cum_arr, cum_dep_nt, log_nt, mw_nt, n_trains_nt = simulate_nt(arrivals)
    cum_dep_fixed, log_fixed, mw_fixed               = simulate_fixed(arrivals)

    plot_cumulative(cum_arr, cum_dep_nt, log_nt, mw_nt, n_trains_nt,
                    cum_dep_fixed, log_fixed, mw_fixed)
    plot_queue_depth(cum_arr, cum_dep_nt, cum_dep_fixed, mw_nt, mw_fixed)
