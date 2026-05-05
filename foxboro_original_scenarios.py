"""
MBTA Foxboro -> South Station  |  Original Three Scenarios
Early surge / Late surge / Uniform — fixed 15-min headway FCFS
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid

# =============================================================================
# PARAMETERS  —  edit everything here
# =============================================================================

# Service
TRAIN_CAPACITY        = 1440   # passengers per train
NUM_TRAINS            = 14     # total trains dispatched
FIRST_DEPARTURE       = 30     # minutes after whistle for first train
FIXED_HEADWAY         = 15     # minutes between trains
DWELL_TIME            = 5      # minutes (noted but not modeled in dispatch logic)

# Scenario 1 — Early surge
EARLY_SIGMA           = 22     # std dev of early-surge normal (minutes)

# Scenario 2 — Late surge
LATE_SURGE_TRAIN      = 7      # train number at which late surge peaks (1-indexed)
LATE_SIGMA            = 28     # std dev of late-surge normal (minutes)

# Scenario 3 — Uniform
UNIFORM_START         = 0      # arrivals begin at t=0

# Simulation / plot
PLOT_BUFFER_MIN       = 20     # extra minutes plotted beyond last departure
OUTPUT_PREFIX         = "foxboro_original_scenario"   # saved as foxboro_original_scenario_1.png etc.

# Colors
COLOR_ARRIVAL         = "#1a6bbd"
COLOR_DEPARTURE       = "#d85a30"
COLOR_TRAIN_LINE      = "#aaaaaa"
COLOR_LEFT_BEHIND     = "#cc3333"

# =============================================================================
# DERIVED (do not edit)
# =============================================================================

TOTAL_PASSENGERS  = NUM_TRAINS * TRAIN_CAPACITY
LAST_DEPARTURE    = FIRST_DEPARTURE + (NUM_TRAINS - 1) * FIXED_HEADWAY
T_MAX             = LAST_DEPARTURE + PLOT_BUFFER_MIN
T                 = np.linspace(0, T_MAX, 4000)
TRAIN_DEPARTS     = [FIRST_DEPARTURE + i * FIXED_HEADWAY for i in range(NUM_TRAINS)]
LATE_SURGE_PEAK   = FIRST_DEPARTURE + (LATE_SURGE_TRAIN - 1) * FIXED_HEADWAY

# =============================================================================
# ARRIVAL + DEPARTURE CURVES
# =============================================================================

def cumulative_arrivals(density, total=TOTAL_PASSENGERS):
    running = cumulative_trapezoid(density, T, initial=0)
    if running[-1] > 0:
        return running / running[-1] * total
    return running

def cumulative_departures(cum_arr):
    total_boarded = 0
    schedule      = []
    for d_time in TRAIN_DEPARTS:
        idx           = np.searchsorted(T, d_time)
        arrived       = cum_arr[min(idx, len(cum_arr) - 1)]
        boarded       = min(arrived - total_boarded, TRAIN_CAPACITY)
        total_boarded += boarded
        schedule.append((d_time, total_boarded))

    cum_dep = np.zeros(len(T))
    for d_time, cum_b in schedule:
        cum_dep[T >= d_time] = cum_b

    mean_wait = np.trapezoid(np.maximum(cum_arr - cum_dep, 0), T) / TOTAL_PASSENGERS
    return cum_dep, mean_wait

# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def add_param_box(ax):
    lines = [
        ("Service",           ""),
        ("  Train capacity",  f"{TRAIN_CAPACITY:,} pax"),
        ("  Trains",          f"{NUM_TRAINS}"),
        ("  First departure", f"t = {FIRST_DEPARTURE} min"),
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

def add_left_behind_callout(ax, cum_arr, cum_dep):
    left_behind = int(round(cum_arr[-1] - cum_dep[-1]))
    if left_behind > 0:
        y_arr     = cum_arr[-1]
        y_dep     = cum_dep[-1]
        x_callout = T_MAX * 0.97
        ax.annotate("", xy=(x_callout, y_dep), xytext=(x_callout, y_arr),
                    arrowprops=dict(arrowstyle="<->", color=COLOR_LEFT_BEHIND, lw=1.5))
        ax.text(x_callout - 1, (y_arr + y_dep) / 2,
                f"{left_behind:,} pax\nleft behind",
                fontsize=8, color=COLOR_LEFT_BEHIND, ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=COLOR_LEFT_BEHIND, linewidth=0.8))

def plot_scenario(cum_arr, cum_dep, mean_wait, title, subtitle, filename):
    fig, ax = plt.subplots(figsize=(15, 6), facecolor="#ffffff")

    ylim_top = TOTAL_PASSENGERS * 1.05

    ax.fill_between(T, cum_arr, cum_dep,
                    where=(cum_arr >= cum_dep),
                    color=COLOR_ARRIVAL, alpha=0.08)
    ax.plot(T, cum_arr, color=COLOR_ARRIVAL,   lw=2,   label="A(t)  cumulative arrivals")
    ax.step(T, cum_dep, color=COLOR_DEPARTURE, lw=2.5, label=f"D(t)  cumulative departures  (mean wait {mean_wait:.1f} min)",
            where="pre")

    for i, td in enumerate(TRAIN_DEPARTS):
        ax.axvline(td, color=COLOR_TRAIN_LINE, lw=0.8, ls="--", alpha=0.6)
        if i % 2 == 0 or i == NUM_TRAINS - 1:
            ax.text(td + 0.5, ylim_top * 0.97, f"T{i+1}",
                    fontsize=7, color="#888888", ha="left", va="top")

    add_left_behind_callout(ax, cum_arr, cum_dep)

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, ylim_top)
    ax.set_xlabel("Minutes after final whistle", fontsize=9)
    ax.set_ylabel("Cumulative passengers", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.1f}k"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(0.99, 0.04, subtitle, transform=ax.transAxes,
            fontsize=8, color="#666666", ha="right", va="bottom", style="italic")

    add_param_box(ax)
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # Scenario 1 — Early surge: Normal peak at first departure
    d1      = norm.pdf(T, loc=FIRST_DEPARTURE, scale=EARLY_SIGMA)
    ca1     = cumulative_arrivals(d1)
    cd1, mw1 = cumulative_departures(ca1)
    plot_scenario(
        ca1, cd1, mw1,
        title   = f"Scenario 1 — Early surge: Normal peak at t = {FIRST_DEPARTURE} min  (σ = {EARLY_SIGMA} min)",
        subtitle= "Most fans rush to the platform immediately; queue clears quickly",
        filename= f"{OUTPUT_PREFIX}_1.png"
    )

    # Scenario 2 — Late surge: Normal peak at train 7 departure
    d2      = norm.pdf(T, loc=LATE_SURGE_PEAK, scale=LATE_SIGMA)
    ca2     = cumulative_arrivals(d2)
    cd2, mw2 = cumulative_departures(ca2)
    plot_scenario(
        ca2, cd2, mw2,
        title   = f"Scenario 2 — Late surge: Normal peak at t = {LATE_SURGE_PEAK} min  (train {LATE_SURGE_TRAIN}, σ = {LATE_SIGMA} min)",
        subtitle= "Fans linger in stadium; large backlog builds then clears after midpoint",
        filename= f"{OUTPUT_PREFIX}_2.png"
    )

    # Scenario 3 — Uniform arrivals across full service window
    inside  = (T >= UNIFORM_START) & (T <= LAST_DEPARTURE)
    d3      = np.where(inside, 1.0 / (LAST_DEPARTURE - UNIFORM_START), 0.0)
    ca3     = cumulative_arrivals(d3)
    cd3, mw3 = cumulative_departures(ca3)
    plot_scenario(
        ca3, cd3, mw3,
        title   = f"Scenario 3 — Uniform arrivals from t = {UNIFORM_START} to t = {LAST_DEPARTURE} min",
        subtitle= "Steady trickle throughout; departure curve tracks A(t) closely with constant lag",
        filename= f"{OUTPUT_PREFIX}_3.png"
    )
