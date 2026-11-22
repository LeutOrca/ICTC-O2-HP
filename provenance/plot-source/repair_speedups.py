"""Only same-equation ratios; no ICTP/O2 or SO2/TP acceleration ratios."""
import csv
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import repair_plot_helpers as plot_helpers

OUT = Path(__file__).resolve().parent


def curve(ax, rows, candidate, baseline, label, color, style="-"):
    points = {int(r["order"]):r for r in rows
              if r["candidate"]==candidate and r["baseline"]==baseline}
    if not points:
        return
    values = [float(points[n]["speedup"]) if n in points else np.nan for n in range(1,9)]
    ax.plot(range(1,9), values, marker="o", markersize=3.5, color=color,
            linestyle=style, label=label)


def style(ax, title):
    ax.set_title(title)
    ax.set_xticks(range(1,9))
    ax.set_xlim(.8,8.2)
    ax.set_xlabel("Maximum angular order")
    ax.set_ylabel("Same-equation speedup")
    ax.tick_params(which="both",top=True,right=True)


def main():
    helpers = importlib.reload(plot_helpers)
    setup, finish = helpers.setup, helpers.finish
    setup()
    with (OUT / "data/repair_speedups.csv").open() as stream:
        data=list(csv.DictReader(stream))
    fig,axes=plt.subplots(2,2,figsize=(150/25.4,112.5/25.4))
    for col,count in enumerate((4096,16384)):
        for row,metric in enumerate(("forward","forward_vjp")):
            values=[r for r in data if r["family"]=="tp" and int(r["rows"])==count and r["metric"]==metric]
            for parity,linestyle in (("canonical","-"),("full","--")):
                selected=[r for r in values if r["parity"]==parity]
                curve(axes[row,col],selected,"public","eqx-o2",f"O2-HP / Eqx-O2, {parity}","#009E73",linestyle)
                curve(axes[row,col],selected,"public","cue-o2",f"O2-HP / cuEq, {parity}","#0072B2",linestyle)
            style(axes[row,col],("N=4096" if count==4096 else "E=16384")+" / "+("forward" if row==0 else "+ input VJP"))
    finish(fig,axes)
    fig.savefig(OUT / "O2_TP_same_equation_speedup_repair20261122.pdf")
    fig.savefig(OUT / "O2_TP_same_equation_speedup_repair20261122.png")
    own,axes=plt.subplots(2,2,figsize=(150/25.4,150/25.4))
    for row,metric in enumerate(("forward","forward_vjp")):
        values=[r for r in data if r["metric"]==metric]
        so2=[r for r in values if r["family"]=="so2"]
        curve(axes[row,0],so2,"so2-cuda2","so2-native","Own CUDA / native SO2Linear","#D55E00")
        curve(axes[row,0],so2,"so2-triton","so2-native","Own Triton / native SO2Linear","#0072B2","--")
        for parity,linestyle in (("canonical","-"),("full","--")):
            radial=[r for r in values if r["family"]=="radial" and r["parity"]==parity]
            curve(axes[row,1],radial,"public","public-unfused",f"Fused / unfused radial, {parity}","#009E73",linestyle)
        radial=[r for r in values if r["family"]=="radial" and r["parity"]=="full"]
        curve(axes[row,1],radial,"public-repair","public-unfused-repair","Streamed / unfused repair, full","#882255",":")
        style(axes[row,0],"SO2Linear / "+("forward" if row==0 else "+ input VJP"))
        style(axes[row,1],"Radial UVU / "+("forward" if row==0 else "+ input VJP"))
    finish(own,axes)
    own.subplots_adjust(top=.82, bottom=.13, hspace=.65)
    own.savefig(OUT / "SO2_Radial_same_equation_speedup_repair20261122.pdf")
    own.savefig(OUT / "SO2_Radial_same_equation_speedup_repair20261122.png")


if __name__ == "__main__":
    main()
