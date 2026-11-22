"""Shared plotting only; all scientific aggregation is a separate Slurm job."""
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path(__file__).resolve().parent
ROLES = {
    "local-hpx": ("HPX 5f278ef (fresh output)", "#CC79A7", "-", "P"),
    "public": ("O2-HP official (= fork)", "#D55E00", "-", "o"),
    "eqx-o2": ("Eqx-O2 CUDA", "#009E73", "-", "s"),
    "cue-o2": ("cuEq 0.13, exact O2", "#0072B2", "-", "^"),
    "public-unfused": ("O2-HP unfused radial", "#56B4E9", "--", "v"),
    "public-repair": ("O2-HP bounded/streamed repair", "#882255", "--", "P"),
    "public-unfused-repair": ("O2-HP bounded unfused repair", "#332288", "-.", "^"),
    "eqx-o2-repair": ("Eqx-O2 + own affine input VJP", "#009E73", ":", "D"),
    "ictp-native": ("ICTP O3 UVU (cost ref.)", "#777777", ":", "D"),
    "so2-native": ("Native SO2Linear", "#222222", "-", "s"),
    "so2-cuda2": ("Generic CUDA (c591b16)", "#D55E00", "-", "o"),
    "so2-triton": ("Generic Triton (c591b16)", "#0072B2", "--", "^"),
}


def setup():
    font = OUT / "Times.TTF"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    mpl.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
        "axes.labelsize": 9, "axes.labelweight": "bold", "axes.titlesize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "axes.linewidth": .75, "lines.linewidth": 1., "xtick.direction": "in",
        "ytick.direction": "in", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman",
        "mathtext.bf": "Times New Roman", "mathtext.cal": "Times New Roman"})


def data():
    with (OUT / "data/summary.csv").open() as stream:
        return list(csv.DictReader(stream))


def select(rows, family, count, parity, metric):
    return [r for r in rows if r["family"] == family and int(r["rows"]) == count
            and r["parity"] == parity and r["metric"] == metric]


def line(ax, rows, role, value="throughput_rows_s", label=None):
    points = {int(r["order"]): r for r in rows if r["role"] == role}
    if not points:
        return
    name, color, style, marker = ROLES[role]
    values = [float(points[n][value]) if n in points else np.nan for n in range(1,9)]
    curve, = ax.plot(range(1,9), values, label=label or name, color=color,
                     linestyle=style, marker=marker, markersize=3.5)
    curve.set_gid(f"{role}-{value}")


def axes_style(ax, unit, title=None):
    if title:
        ax.set_title(title)
    ax.set_xticks(range(1,9))
    ax.set_xlim(.8,8.2)
    ax.set_xlabel("Maximum angular order")
    ax.set_ylabel(f"Throughput ({unit}/s)")
    ax.set_yscale("log")
    ax.tick_params(top=True, right=True, which="major")
    ax.minorticks_off()


def finish(fig, axes):
    items = {}
    for ax in np.asarray(axes).flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            items.setdefault(label, handle)
    fig.legend(list(items.values()), list(items), loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(.5,1.), columnspacing=1.3,
               handlelength=2.3)
    fig.subplots_adjust(left=.13, right=.98, bottom=.16, top=.82,
                        hspace=.60, wspace=.42)


def tp_figure(rows, count):
    fig, axes = plt.subplots(2,2,figsize=(150/25.4,112.5/25.4))
    unit = "atoms" if count == 4096 else "edges"
    for col, parity in enumerate(("canonical", "full")):
        for row, metric in enumerate(("forward", "forward_vjp")):
            values = select(rows,"tp",count,parity,metric)
            for role in ("local-hpx","public","eqx-o2","cue-o2"):
                line(axes[row,col],values,role)
            reference = select(rows,"ictp",count,"natural",metric)
            line(axes[row,col], reference, "ictp-native")
            title = ("Canonical O(2)" if parity == "canonical" else "Full-parity O(2)")
            title += " / " + ("forward" if row == 0 else "forward + input VJP")
            axes_style(axes[row,col],unit,title)
    finish(fig,axes)
    fig.text(.5,.018,f"H100, FP32 | C=64, {'N' if count==4096 else 'E'}={count} | CUDA Graph off",
             ha="center",fontsize=8)
    return fig


def message_figure(rows, parity):
    height = 150 if parity == "full" else 112.5
    fig, axes = plt.subplots(2,2,figsize=(150/25.4,height/25.4))
    for col, family in enumerate(("uvu","radial")):
        for row, metric in enumerate(("forward","forward_vjp")):
            values=select(rows,family,16384,parity,metric)
            for role in ("public","eqx-o2","public-unfused","public-repair","public-unfused-repair","eqx-o2-repair"):
                line(axes[row,col],values,role)
            title=("UVU" if family=="uvu" else "Radial UVU")
            title+=" / "+("forward" if row==0 else "forward + input VJP")
            axes_style(axes[row,col],"edges",title)
    finish(fig,axes)
    if parity == "full":
        fig.subplots_adjust(top=.82, bottom=.13, hspace=.65)
    fig.text(.5,.018,"H100, FP32 | E=16384, C=64, V=1 | radial H=64 | Graph off",
             ha="center",fontsize=8)
    return fig


def so2_figure(rows):
    fig, axes=plt.subplots(1,2,figsize=(150/25.4,84.375/25.4))
    for col, metric in enumerate(("forward","forward_vjp")):
        values=select(rows,"so2",16384,"natural",metric)
        for role in ("so2-native","so2-cuda2","so2-triton"):
            line(axes[col],values,role)
        axes_style(axes[col],"edges","Forward" if col==0 else "Forward + input VJP")
    finish(fig,axes)
    fig.subplots_adjust(top=.76,bottom=.22,wspace=.32)
    fig.text(.5,.018,"H100, FP32 | E=16384, C=64 | Lmax=Mmax | Graph off",
             ha="center",fontsize=8)
    return fig


def ictp_figure(rows):
    fig, axes = plt.subplots(2,2,figsize=(150/25.4,112.5/25.4))
    for col, count in enumerate((4096,16384)):
        unit = "atoms" if count == 4096 else "edges"
        for row, metric in enumerate(("forward","forward_vjp")):
            values = select(rows,"ictp",count,"natural",metric)
            line(axes[row,col],values,"ictp-native",label="Native ICTP, natural parity")
            title = ("N=4096" if col==0 else "E=16384") + " / " + ("forward" if row==0 else "+ input VJP")
            axes_style(axes[row,col],unit,title)
            axes[row,col].axvspan(3.5,8.2,color="#eeeeee",zorder=0)
            axes[row,col].text(.69,.80,"Orders 4–8\nunsupported",ha="center",va="center",
                              fontsize=8,transform=axes[row,col].transAxes,color="#555555")
    finish(fig,axes)
    fig.text(.5,.018,"H100, FP32 | C=64, natural parity | CUDA Graph off",ha="center",fontsize=8)
    return fig
