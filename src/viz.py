"""Fan chart for a SimulationResult. See docs/ for rationale and the
short-vs-long-horizon trust styling."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def fan_chart(
    result,
    title: str = "",
    outpath: str | None = None,
    show_horizon_caveat: bool = True,
):
    """Draw a percentile fan from a SimulationResult and save a PNG.

    Bands: 5-95 (outer) and 25-75 (inner), with the median line and the start
    price reference. One generator per chart by design.
    """
    pp = result.percentile_paths((5, 25, 50, 75, 95))
    t = np.arange(result.price_paths.shape[1])
    years = result.horizon / 252.0

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.fill_between(t, pp[5], pp[95], color="#3b82f6", alpha=0.15,
                    label="5–95% range")
    ax.fill_between(t, pp[25], pp[75], color="#3b82f6", alpha=0.30,
                    label="25–75% range")
    ax.plot(t, pp[50], color="#1d4ed8", lw=2, label="median")
    ax.axhline(result.start_price, color="#64748b", ls="--", lw=1,
               label=f"start ${result.start_price:,.0f}")

    ax.set_xlabel("trading days ahead")
    ax.set_ylabel("price ($)")
    ax.set_title(title or "Simulated price distribution")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.margins(x=0)
    ax.grid(True, alpha=0.2)

    # Honest framing: long horizons are scenario, not forecast.
    if show_horizon_caveat:
        note = ("scenario, not forecast — drift-dominated, untestable"
                if years > 3 else
                "wide dispersion is honest; the band is the message")
        ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8, style="italic", color="#94a3b8")

    fig.tight_layout()
    if outpath is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        outpath = str(RESULTS_DIR / "fan_chart.png")
    fig.savefig(outpath, facecolor="white")
    plt.close(fig)
    return outpath
