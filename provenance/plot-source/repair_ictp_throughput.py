"""Native ICTP O3 UVU throughput; never an O2 speedup denominator."""
from pathlib import Path
import importlib
import repair_plot_helpers as plot_helpers

OUT = Path(__file__).resolve().parent


def main():
    helpers = importlib.reload(plot_helpers)
    setup, data, ictp_figure = helpers.setup, helpers.data, helpers.ictp_figure
    setup()
    figure = ictp_figure(data())
    figure.savefig(OUT / "ICTP_atoms_edges_repair20261122.pdf")
    figure.savefig(OUT / "ICTP_atoms_edges_repair20261122.png")


if __name__ == "__main__":
    main()
