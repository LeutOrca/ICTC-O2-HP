"""UVU/RadialUVU edge throughput versus angular order, separate parity cohorts."""
from pathlib import Path
import importlib
import plot_helpers
OUT = Path(__file__).resolve().parent


def main():
    helpers = importlib.reload(plot_helpers)
    setup, data, message_figure = helpers.setup, helpers.data, helpers.message_figure
    setup()
    rows=data()
    canonical=message_figure(rows,"canonical")
    canonical.savefig(OUT / "O2_UVU_Radial_canonical.pdf")
    canonical.savefig(OUT / "O2_UVU_Radial_canonical.png")
    full=message_figure(rows,"full")
    full.savefig(OUT / "O2_UVU_Radial_full.pdf")
    full.savefig(OUT / "O2_UVU_Radial_full.png")


if __name__ == "__main__":
    main()
