"""SO2Linear C64, E16384, Lmax=Mmax=1..8, immutable matched weights."""
from pathlib import Path
import importlib
import repair_plot_helpers as plot_helpers
OUT = Path(__file__).resolve().parent


def main():
    helpers = importlib.reload(plot_helpers)
    setup, data, so2_figure = helpers.setup, helpers.data, helpers.so2_figure
    setup()
    figure=so2_figure(data())
    figure.savefig(OUT / "SO2Linear_edges_repair20261122.pdf")
    figure.savefig(OUT / "SO2Linear_edges_repair20261122.png")


if __name__ == "__main__":
    main()
