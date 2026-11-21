"""SO2Linear C64, E16384, Lmax=Mmax=1..8, immutable matched weights."""
from pathlib import Path
from plot_helpers import setup, data, so2_figure
OUT = Path(__file__).resolve().parent


def main():
    setup()
    figure=so2_figure(data())
    figure.savefig(OUT / "SO2Linear_edges.pdf")
    figure.savefig(OUT / "SO2Linear_edges.png")


if __name__ == "__main__":
    main()
