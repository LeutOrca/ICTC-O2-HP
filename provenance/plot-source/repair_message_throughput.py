"""UVU/RadialUVU edge throughput versus angular order, separate parity cohorts."""
from pathlib import Path
import importlib
import repair_plot_helpers as plot_helpers
OUT = Path(__file__).resolve().parent


def main():
    helpers = importlib.reload(plot_helpers)
    setup, data, message_figure = helpers.setup, helpers.data, helpers.message_figure
    setup()
    rows=data()
    canonical=message_figure(rows,"canonical")
    canonical.savefig(OUT / "O2_UVU_Radial_canonical_repair20261122.pdf")
    canonical.savefig(OUT / "O2_UVU_Radial_canonical_repair20261122.png")
    full=message_figure(rows,"full")
    full.savefig(OUT / "O2_UVU_Radial_full_repair20261122.pdf")
    full.savefig(OUT / "O2_UVU_Radial_full_repair20261122.png")


if __name__ == "__main__":
    main()
