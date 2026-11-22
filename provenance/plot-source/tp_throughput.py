"""Fixed N/E, fixed C, angular1..8. ICTP is a different-equation cost reference."""
from pathlib import Path
import importlib
import plot_helpers
OUT = Path(__file__).resolve().parent


def main():
    helpers = importlib.reload(plot_helpers)
    setup, data, tp_figure = helpers.setup, helpers.data, helpers.tp_figure
    setup()
    rows=data()
    atoms=tp_figure(rows,4096)
    atoms.savefig(OUT / "O2_TP_atoms.pdf")
    atoms.savefig(OUT / "O2_TP_atoms.png")
    edges=tp_figure(rows,16384)
    edges.savefig(OUT / "O2_TP_edges.pdf")
    edges.savefig(OUT / "O2_TP_edges.png")


if __name__ == "__main__":
    main()
