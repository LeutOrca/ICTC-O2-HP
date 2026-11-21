"""Direct local frozen-weight HPX / GitHub public O2-HP paired speedups."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from plot_helpers import setup, finish
from speedups import curve, style

OUT = Path(__file__).resolve().parent


def main():
    setup()
    with (OUT/"data/speedups.csv").open() as stream:
        data = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2,2,figsize=(150/25.4,112.5/25.4),sharey="row")
    for col, count in enumerate((4096,16384)):
        for row, metric in enumerate(("forward","forward_vjp")):
            values = [r for r in data if r["family"]=="tp" and int(r["rows"])==count and r["metric"]==metric]
            for parity, color, line in (("canonical","#CC79A7","-"),("full","#0072B2","--")):
                selected = [r for r in values if r["parity"]==parity]
                curve(axes[row,col],selected,"local-hpx","public",f"HPX / GitHub O2-HP, {parity}",color,line)
            style(axes[row,col],("N=4096" if col==0 else "E=16384")+" / "+("forward" if row==0 else "+ input VJP"))
    finish(fig,axes)
    fig.savefig(OUT/"HPX_vs_GitHub_O2HP.pdf")
    fig.savefig(OUT/"HPX_vs_GitHub_O2HP.png")


if __name__ == "__main__":
    main()
