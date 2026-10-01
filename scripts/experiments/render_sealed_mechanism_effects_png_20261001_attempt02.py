#!/usr/bin/env python3
"""Render PNG previews from the exact values frozen in figure attempt03."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SOURCE_MANIFEST_SHA="cb362e2b789867c78120f8c4c506620b2437b3e8dd6eb6466c877b982a4b1272"
SOURCE_MANIFEST=Path("runs/recovered_20260930/independent_validation/mechanism_figures_20261001_attempt03/figure_manifest.json")
OUT_DEFAULT=Path("runs/recovered_20260930/independent_validation/mechanism_figures_20261001_attempt04")
SUPPORT_COLORS={"full":"#49566b","valid":"#8b6faf","guard15":"#198c8c","guard20":"#d17a22"}
SUPPORT_LABELS={"full":"full","valid":"valid","guard15":"guard 15","guard20":"guard 20"}
CONTRASTS=("interior_q10_minus_q00","boundary_q01_minus_q00","joint_q11_minus_q00")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def ci(point:dict[str,Any])->tuple[float,float]:return float(point["ci99"][0]),float(point["ci99"][1])
def x_limits(points:list[dict[str,Any]])->tuple[float,float]:
    low=min(0.0,min(ci(p)[0] for p in points)); high=max(0.0,max(ci(p)[1] for p in points)); span=max(high-low,1e-9)
    return low-0.08*span,high+0.08*span

def plot_rms(ax,points:list[dict[str,Any]],geo:str,stat:str)->None:
    subset=[p for p in points if p["geometry"]==geo and p["statistic"]==stat]
    ordered=[(cohort,metric) for cohort in ("Male","Female") for metric in ("C","C_anchor")]
    subset.sort(key=lambda p:ordered.index((p["cohort"],p["metric"])))
    colors={"Male":"#2463a6","Female":"#bb4b3f"}; markers={"C":"o","C_anchor":"D"}
    ys=list(range(len(subset)))
    for y,p in zip(ys,subset):
        estimate=float(p["estimate"]); low,high=ci(p)
        ax.errorbar(estimate,y,xerr=[[estimate-low],[high-estimate]],fmt=markers[p["metric"]],
                    color=colors[p["cohort"]],ecolor=colors[p["cohort"]],elinewidth=1.7,capsize=3.5,markersize=6,zorder=3)
    ax.set_yticks(ys,[f'{p["cohort"]} ({p["speakers"]}) · {"C" if p["metric"]=="C" else "C anchor"}' for p in subset],fontsize=10.5)
    ax.invert_yaxis(); ax.axvline(0,color="#202733",linewidth=1.2,linestyle="--",zorder=1)
    ax.set_xlim(*x_limits(subset)); ax.grid(axis="x",color="#dfe3e9",linewidth=.8); ax.set_axisbelow(True)
    stat_title="observed G contrast" if stat=="observed" else "observed minus mean time permutation"
    ax.set_title(f'{geo.capitalize()} geometry · {stat_title}',loc="left",fontsize=13,fontweight="bold",pad=10)
    ax.set_xlabel("SyncNet C contrast (distance units)",fontsize=10)
    for spine in ("top","right"):ax.spines[spine].set_visible(False)

def plot_mel(ax,points:list[dict[str,Any]],geo:str,stat:str)->None:
    subset=[p for p in points if p["geometry"]==geo and p["statistic"]==stat]
    contrast_order={name:i for i,name in enumerate(CONTRASTS)}
    support_order={"full":0,"valid":1,"guard15":2,"guard20":3}
    subset.sort(key=lambda p:(contrast_order[p["contrast"]],support_order[p["support"]]))
    ys=list(range(len(subset)))
    for y,p in zip(ys,subset):
        estimate=float(p["estimate"]); low,high=ci(p); color=SUPPORT_COLORS[p["support"]]
        ax.errorbar(estimate,y,xerr=[[estimate-low],[high-estimate]],fmt="o",color=color,ecolor=color,
                    elinewidth=1.5,capsize=2.7,markersize=5.3,zorder=3)
    ax.set_yticks(ys,[f'{p["contrast_label"]} · {SUPPORT_LABELS[p["support"]]}' for p in subset],fontsize=9.1)
    ax.invert_yaxis(); ax.axvline(0,color="#202733",linewidth=1.2,linestyle="--",zorder=1)
    ax.set_xlim(*x_limits(subset)); ax.grid(axis="x",color="#dfe3e9",linewidth=.8); ax.set_axisbelow(True)
    stat_title="observed contrast" if stat=="observed" else "observed minus mean time permutation"
    ax.set_title(f'{geo.capitalize()} geometry · {stat_title}',loc="left",fontsize=13,fontweight="bold",pad=10)
    ax.set_xlabel("SyncNet contrast (distance units)",fontsize=10)
    for spine in ("top","right"):ax.spines[spine].set_visible(False)

def main()->None:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--manifest",type=Path,default=SOURCE_MANIFEST);ap.add_argument("--output-dir",type=Path,default=OUT_DEFAULT);args=ap.parse_args()
    manifest=args.manifest.resolve(strict=True)
    if sha(manifest)!=SOURCE_MANIFEST_SHA:raise ValueError("figure attempt03 manifest SHA mismatch")
    data=json.loads(manifest.read_text(encoding="utf-8"));values=data["plotted_values"]
    rms=values["rms"]; mel_c=values["mel_C"];mel_anchor=values["mel_C_anchor"]
    if len(rms)!=16 or len(mel_c)!=48 or len(mel_anchor)!=48:raise ValueError("frozen plot point counts differ")
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    figures={}
    fig,axes=plt.subplots(2,2,figsize=(18,9),constrained_layout=False)
    fig.subplots_adjust(left=.14,right=.99,bottom=.14,top=.86,wspace=.40,hspace=.48)
    for i,(stat,geo) in enumerate((("observed","raw"),("observed","unit"),("observed_minus_null","raw"),("observed_minus_null","unit"))):plot_rms(axes.flat[i],rms,geo,stat)
    fig.suptitle("RMS generator contrast and time-pairing excess",fontsize=18,fontweight="bold")
    cohort_handles=[Patch(facecolor="#2463a6",label="Male · 40 speakers"),Patch(facecolor="#bb4b3f",label="Female · 20 speakers"),
                    Line2D([0],[0],marker="o",color="#333",linestyle="none",label="C"),Line2D([0],[0],marker="D",color="#333",linestyle="none",label="C anchor")]
    fig.legend(handles=cohort_handles,loc="upper center",bbox_to_anchor=(.5,.925),ncol=4,frameon=False,fontsize=10)
    fig.text(.01,.018,"Full-A cutoff, guard20. Time-pairing excess = observed contrast − mean of 100 cached full-A index permutations. 99% paired speaker-bootstrap intervals.",fontsize=9,color="#505a68")
    name="rms_generator_G_observed_and_excess.png";path=out/name
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=100,bbox_inches="tight",facecolor="white");plt.close(fig);figures[name]={"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size,"dimensions_px":[1800,None]}
    for metric,points,suffix in (("C",mel_c,"C"),("C anchor",mel_anchor,"C_anchor")):
        fig,axes=plt.subplots(2,2,figsize=(18,11),constrained_layout=False)
        fig.subplots_adjust(left=.18,right=.99,bottom=.14,top=.90,wspace=.37,hspace=.42)
        for i,(stat,geo) in enumerate((("observed","raw"),("observed","unit"),("observed_minus_null","raw"),("observed_minus_null","unit"))):plot_mel(axes.flat[i],points,geo,stat)
        fig.suptitle(f"MEL partition contrasts · {metric}",fontsize=18,fontweight="bold")
        fig.legend(handles=[Patch(facecolor=SUPPORT_COLORS[k],label=SUPPORT_LABELS[k]) for k in ("full","valid","guard15","guard20")],
                   loc="upper center",bbox_to_anchor=(.5,.945),ncol=4,frameon=False,fontsize=10)
        fig.text(.01,.018,"Labels preserve the frozen interior-only, boundary-only, and joint contrasts; no phonetic class or mediation percentage is inferred. 99% paired speaker-bootstrap intervals.",fontsize=9,color="#505a68")
        name=f"mel_partition_{suffix}_observed_and_excess.png";path=out/name
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=100,bbox_inches="tight",facecolor="white");plt.close(fig);figures[name]={"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size,"dimensions_px":[1800,None]}
    report={"schema_version":1,"status":"complete","renderer":"Matplotlib Agg; values copied verbatim from frozen figure attempt03 JSON","source_manifest":{"path":str(manifest),"sha256":sha(manifest)},"images":figures}
    rp=out/"png_manifest.json"
    if rp.exists():raise FileExistsError(rp)
    rp.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"complete","images":figures,"manifest":str(rp),"manifest_sha256":sha(rp)},ensure_ascii=False,separators=(",",":")))
if __name__=="__main__":main()