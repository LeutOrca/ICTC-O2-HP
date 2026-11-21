"""Native ICTP O3 UVU throughput; never an O2 speedup denominator."""
from pathlib import Path
from plot_helpers import setup, data, ictp_figure

OUT = Path(__file__).resolve().parent


def main():
    setup()
    figure = ictp_figure(data())
    figure.savefig(OUT / "ICTP_atoms_edges.pdf")
    figure.savefig(OUT / "ICTP_atoms_edges.png")


if __name__ == "__main__":
    main()
