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
NT_CAPACITY_THRESHOLD = 1400   # send a train if queue >= this many passengers
NT_WAIT_THRESHOLD     = 90     # send a train if oldest passenger has waited >= this many minutes
NT_MIN_HEADWAY        = 8      # minimum minutes between consecutive dispatches

# Simulation / plot
RANDOM_SEED           = 42
PLOT_BUFFER_MIN       = 40     # extra minutes plotted beyond last departure
OUTPUT_CUMULATIVE     = "foxboro_cumulative.png"
OUTPUT_QUEUE_DEPTH    = "foxboro_queue_depth.png"

# Colors
COLOR_ARRIVAL         = "#1a6bbd"
COLOR_DEPARTURE       = "#2ca44e"
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

def simulate():
    rng       = np.random.default_rng(RANDOM_SEED)
    t_minutes = np.arange(0, T_MAX + 1, dtype=float)

    # Poisson rate lambda(t): normal surge shape scaled to expected total
    raw_shape = norm.pdf(t_minutes, ARRIVAL_PEAK, ARRIVAL_SIGMA)
    lam_t     = raw_shape / raw_shape.sum() * TOTAL_PASSENGERS

    arrivals_per_min = rng.poisson(lam_t)
    arrivals_per_min[:ARRIVAL_QUIET_PERIOD] = 0

    queue             = 0
    oldest_wait       = 0
    total_boarded     = 0
    trains_dispatched = 0
    last_dispatch     = -999
    dispatch_log      = []   # list of (minute, cumulative_boarded)
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

    # A(t): interpolate cumulative arrivals onto fine grid
    cum_arr = np.interp(T, t_minutes, cum_arr_sim)

    # D(t): true step function — no interpolation
    cum_dep = np.zeros(len(T))
    for d_time, cum_b in dispatch_log:
        cum_dep[T >= d_time] = cum_b

    mean_wait = np.trapezoid(np.maximum(cum_arr - cum_dep, 0), T) / max(total_boarded, 1)

    return cum_arr, cum_dep, dispatch_log, mean_wait, trains_dispatched

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
    ]
    text = "\n".join(f"{label:<18}{value}" for label, value in lines)
    ax.text(
        1.02, 1.0, text,
        transform=ax.transAxes,
        fontsize=8, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f7f7",
                  edgecolor="#cccccc", linewidth=0.8)
    )


def plot_cumulative(cum_arr, cum_dep, dispatch_log, mean_wait, n_trains):
    fig, ax = plt.subplots(figsize=(15, 6), facecolor="#ffffff")

    ax.fill_between(T, cum_arr, cum_dep,
                    where=(cum_arr >= cum_dep),
                    color=COLOR_ARRIVAL, alpha=0.08)
    ax.plot(T, cum_arr, color=COLOR_ARRIVAL,   lw=2,   label="A(t)  cumulative arrivals")
    ax.step(T, cum_dep, color=COLOR_DEPARTURE, lw=2.5, label="D(t)  cumulative departures",
            where="pre")

    ylim_top = TOTAL_PASSENGERS * 1.05
    for i, (td, _) in enumerate(dispatch_log):
        ax.axvline(td, color=COLOR_TRAIN_LINE, lw=0.8, ls="--", alpha=0.6)
        if i % 2 == 0 or i == len(dispatch_log) - 1:
            ax.text(td + 0.5, ylim_top * 0.97, f"T{i+1}",
                    fontsize=7, color="#888888", ha="left", va="top")
            
    # Callout: passengers left behind at end of service window
    left_behind = int(round(cum_arr[-1] - cum_dep[-1]))
    if left_behind > 0:
        y_arr = cum_arr[-1]
        y_dep = cum_dep[-1]
        x_callout = T_MAX * 0.97
        ax.annotate("", xy=(x_callout, y_dep), xytext=(x_callout, y_arr),
                    arrowprops=dict(arrowstyle="<->", color="#cc3333", lw=1.5))
        ax.text(x_callout - 1, (y_arr + y_dep) / 2,
                f"{left_behind:,} pax\nleft behind",
                fontsize=8, color="#cc3333", ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cc3333", linewidth=0.8))

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
        f"N(t) dispatch: queue≥{NT_CAPACITY_THRESHOLD} OR wait≥{NT_WAIT_THRESHOLD} min  ·  "
        f"{n_trains} trains used\n"
        f"mean wait ≈ {mean_wait:.1f} min (Little's Law)",
        fontsize=10, fontweight="bold"
    )

    add_param_box(ax)
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    plt.savefig(OUTPUT_CUMULATIVE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_CUMULATIVE}")


def plot_queue_depth(cum_arr, cum_dep):
    Q = np.maximum(cum_arr - cum_dep, 0)

    fig, ax = plt.subplots(figsize=(13, 5), facecolor="#ffffff")
    ax.plot(T, Q, color=COLOR_DEPARTURE, lw=2, label="Q(t)  queue depth")
    ax.fill_between(T, Q, alpha=0.10, color=COLOR_DEPARTURE)

    peak_idx = np.argmax(Q)
    ax.annotate(
        f"peak {Q[peak_idx]:.0f} pax @ t={T[peak_idx]:.0f} min",
        xy=(T[peak_idx], Q[peak_idx]),
        xytext=(T[peak_idx] + 12, Q[peak_idx] + 200),
        fontsize=8, color=COLOR_DEPARTURE,
        arrowprops=dict(arrowstyle="->", color=COLOR_DEPARTURE, lw=0.9)
    )

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0)
    ax.set_xlabel("Minutes after final whistle", fontsize=9)
    ax.set_ylabel("Passengers waiting on platform", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.1f}k"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title("Queue Depth  Q(t) = A(t) − D(t)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_QUEUE_DEPTH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_QUEUE_DEPTH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    cum_arr, cum_dep, dispatch_log, mean_wait, n_trains = simulate()
    plot_cumulative(cum_arr, cum_dep, dispatch_log, mean_wait, n_trains)
    plot_queue_depth(cum_arr, cum_dep)
