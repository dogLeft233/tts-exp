#!/usr/bin/env python3
"""Render sealed mechanism summaries as vector forest plots; no recomputation of effects."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

MALE_SHA = "7299436cf1605c920605f42b2bd4e1feb5e6cba82052f5f1f3bab3a29d0eece6"
FEMALE_SHA = "1bfa675eafce3595ac07b8a8d88d6d7f4efe75901ad092a3b8b447ddaf3da20a"
MEL_SHA = "112ff916b3c6b1790f3cf3db1b75721f08f84486aca0e16d828ec432bf4bc6fb"
MALE_DEFAULT = Path("runs/recovered_20260930/independent_validation/native_cached_null_attempt05/root_read_level_GE_I.json")
FEMALE_DEFAULT = Path("runs/recovered_20260930/independent_validation/female_cached_null_addendum_attempt02/independent_female_cached_null.json")
MEL_DEFAULT = Path("runs/recovered_20260930/independent_validation/mel_partition_consumer_20261001_attempt07/independent_mel_report.json")
OUT_DEFAULT = Path("runs/recovered_20260930/independent_validation/mechanism_figures_20261001")
SUPPORTS = ("full", "valid", "guard15", "guard20")
SUPPORT_LABEL = {"full": "full", "valid": "valid", "guard15": "guard 15", "guard20": "guard 20"}
SUPPORT_COLOR = {"full": "#49566b", "valid": "#8b6faf", "guard15": "#198c8c", "guard20": "#d17a22"}
CONTRASTS = (
    ("interior_q10_minus_q00", "interior only · q10−q00"),
    ("boundary_q01_minus_q00", "boundary only · q01−q00"),
    ("joint_q11_minus_q00", "joint · q11−q00"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path, expected: str, label: str) -> Any:
    path = path.resolve(strict=True)
    actual = _sha(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ci(row: dict[str, Any], field: str) -> tuple[float, float]:
    ci = row[field]
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError(f"bad CI in {field}")
    return float(ci[0]), float(ci[1])


def collect_rms(male: list[dict[str, Any]], female: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [
        row for row in male
        if row.get("domain") == "supplement_full_audio_cutoff"
        and row.get("face") == "primary"
        and row.get("geo") in ("raw", "unit")
        and row.get("support") == "guard20"
        and row.get("source") == "natural"
        and row.get("contrast") == "G_visual_fixed_minus_raw_audio_raw"
        and row.get("metric") in ("C", "C_anchor")
    ]
    if len(selected) != 4:
        raise ValueError(f"expected 4 male RMS rows, got {len(selected)}")
    points: list[dict[str, Any]] = []
    for geo in ("raw", "unit"):
        for metric in ("C", "C_anchor"):
            matches = [r for r in selected if r["geo"] == geo and r["metric"] == metric]
            if len(matches) != 1:
                raise ValueError(f"duplicate/missing male row {geo}/{metric}")
            r = matches[0]
            for stat, value_key, ci_key in (("observed", "obs", "obs_ci"), ("observed_minus_null", "excess", "excess_ci")):
                lo, hi = _ci(r, ci_key)
                points.append({"cohort": "Male", "speakers": int(r["speakers"]), "geometry": geo, "metric": metric,
                               "statistic": stat, "estimate": float(r[value_key]), "ci99": [lo, hi],
                               "source": "sealed main retry3 native RMS generator G, full-A cutoff, guard20"})

    results = female["results"]["generator_G"]
    for geo in ("raw", "unit"):
        for metric in ("C", "C_anchor"):
            item = results["full_A_cutoff"][geo]["guard20"][metric]["natural"]["q10_minus_q00"]
            for stat, container in (("observed", item["observed"]), ("observed_minus_null", item["observed_minus_mean_null"])):
                lo, hi = _ci(container, "speaker_bootstrap_99_ci")
                points.append({"cohort": "Female", "speakers": int(container["intended_speaker_count"]), "geometry": geo, "metric": metric,
                               "statistic": stat, "estimate": float(container["complete_pair_equal_speaker_mean"]), "ci99": [lo, hi],
                               "source": "sealed female cached RAW-N RMS generator G, full-A cutoff, guard20"})
    if len(points) != 16:
        raise ValueError(f"expected 16 RMS plot points, got {len(points)}")
    return points


def collect_mel(mel: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    stats = mel["aggregate_statistics"]
    if stats["clip_count"] != 80 or stats["speaker_count"] != 40:
        raise ValueError("MEL source summary is not the frozen 80-clip/40-speaker cohort")
    points = []
    for contrast_key, contrast_label in CONTRASTS:
        for support in SUPPORTS:
            for geometry in ("raw", "unit"):
                observed = stats["contrasts"][contrast_key][geometry][support][metric]
                lo, hi = _ci(observed["bootstrap"], "ci99")
                points.append({"contrast": contrast_key, "contrast_label": contrast_label, "support": support,
                               "geometry": geometry, "statistic": "observed", "metric": metric,
                               "estimate": float(observed["equal_speaker_mean"]), "ci99": [lo, hi],
                               "speakers": int(observed["speaker_count"])})
                excess = stats["temporal_null_contrasts"][contrast_key][geometry][support][metric]["observed_minus_mean_null"]
                lo, hi = _ci(excess["bootstrap"], "ci99")
                points.append({"contrast": contrast_key, "contrast_label": contrast_label, "support": support,
                               "geometry": geometry, "statistic": "observed_minus_null", "metric": metric,
                               "estimate": float(excess["equal_speaker_mean"]), "ci99": [lo, hi],
                               "speakers": int(excess["speaker_count"])})
    expected = len(CONTRASTS) * len(SUPPORTS) * 2 * 2
    if len(points) != expected:
        raise ValueError(f"MEL point count mismatch: {len(points)} != {expected}")
    return points


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _panel_svg(title: str, subtitle: str, points: list[dict[str, Any]], x: int, y: int,
               width: int, height: int, *, label_field: str, color_field: str,
               colors: dict[str, str], marker_field: str | None = None,
               marker_map: dict[str, str] | None = None) -> str:
    left, right, top, bottom = 190, 28, 65, 64
    px0, px1 = x + left, x + width - right
    py0, py1 = y + top, y + height - bottom
    labels = [str(p[label_field]) for p in points]
    low = min(float(p["ci99"][0]) for p in points)
    high = max(float(p["ci99"][1]) for p in points)
    # Keep the null reference in view even when all intervals are positive.
    low = min(low, 0.0)
    high = max(high, 0.0)
    span = max(high - low, 1e-6)
    low -= span * 0.08
    high += span * 0.08
    def sx(v: float) -> float: return px0 + (v - low) / (high - low) * (px1 - px0)
    n = max(len(points), 1)
    step = (py1 - py0) / n
    pieces = [f'<text x="{x}" y="{y+22}" class="ptitle">{esc(title)}</text>',
              f'<text x="{x}" y="{y+43}" class="psub">{esc(subtitle)}</text>']
    # vertical grid and ticks
    for i in range(5):
        value = low + (high - low) * i / 4
        xx = sx(value)
        dash = ' stroke-dasharray="4 4"' if abs(value) < (high-low)/100 else ""
        color = "#7b7b7b" if abs(value) < (high-low)/100 else "#d9dee7"
        pieces.append(f'<line x1="{xx:.2f}" y1="{py0-8}" x2="{xx:.2f}" y2="{py1+2}" stroke="{color}" stroke-width="1"{dash}/>')
        pieces.append(f'<text x="{xx:.2f}" y="{py1+22}" text-anchor="middle" class="tick">{value:.3f}</text>')
    pieces.append(f'<line x1="{px0}" y1="{py1+2}" x2="{px1}" y2="{py1+2}" stroke="#404854" stroke-width="1.2"/>')
    for idx, point in enumerate(points):
        yy = py0 + (idx + 0.5) * step
        label = str(point[label_field])
        color_key = str(point[color_field])
        color = colors[color_key]
        estimate = float(point["estimate"])
        lo, hi = map(float, point["ci99"])
        xlo, xhi, xp = sx(lo), sx(hi), sx(estimate)
        pieces.append(f'<text x="{px0-9}" y="{yy+4:.2f}" text-anchor="end" class="rowlabel">{esc(label)}</text>')
        pieces.append(f'<line x1="{xlo:.2f}" y1="{yy:.2f}" x2="{xhi:.2f}" y2="{yy:.2f}" stroke="{color}" stroke-width="2.4"/>')
        pieces.append(f'<line x1="{xlo:.2f}" y1="{yy-4:.2f}" x2="{xlo:.2f}" y2="{yy+4:.2f}" stroke="{color}" stroke-width="1.5"/>')
        pieces.append(f'<line x1="{xhi:.2f}" y1="{yy-4:.2f}" x2="{xhi:.2f}" y2="{yy+4:.2f}" stroke="{color}" stroke-width="1.5"/>')
        marker = marker_map.get(str(point[marker_field]), "circle") if marker_field and marker_map else "circle"
        if marker == "diamond":
            pieces.append(f'<path d="M {xp:.2f} {yy-5:.2f} L {xp+5:.2f} {yy:.2f} L {xp:.2f} {yy+5:.2f} L {xp-5:.2f} {yy:.2f} Z" fill="{color}" stroke="white" stroke-width="0.7"><title>{esc(label)}: {estimate:.5f}, 99% CI [{lo:.5f}, {hi:.5f}]</title></path>')
        else:
            pieces.append(f'<circle cx="{xp:.2f}" cy="{yy:.2f}" r="4.3" fill="{color}" stroke="white" stroke-width="0.7"><title>{esc(label)}: {estimate:.5f}, 99% CI [{lo:.5f}, {hi:.5f}]</title></circle>')
    pieces.append(f'<text x="{(px0+px1)/2:.2f}" y="{y+height-10}" text-anchor="middle" class="xlabel">SyncNet C contrast (distance units)</text>')
    return "\n".join(pieces)


def svg_document(title: str, subtitle: str, width: int, height: int, panels: list[str], footer: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">
<title>{esc(title)}</title><desc>{esc(subtitle)}</desc>
<style>
text{{font-family:Arial,Helvetica,sans-serif;fill:#202733}} .title{{font-size:24px;font-weight:700}} .subtitle{{font-size:13px;fill:#495566}} .ptitle{{font-size:16px;font-weight:700}} .psub{{font-size:11px;fill:#5b6574}} .rowlabel{{font-size:11px}} .tick{{font-size:10px;fill:#626d7b}} .xlabel{{font-size:11px;fill:#394454}} .foot{{font-size:11px;fill:#505a68}}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="34" y="34" class="title">{esc(title)}</text>
<text x="34" y="55" class="subtitle">{esc(subtitle)}</text>
{''.join(panels)}
<text x="34" y="{height-16}" class="foot">{esc(footer)}</text>
</svg>'''


def render_rms(points: list[dict[str, Any]]) -> str:
    w, h, gap = 1450, 790, 24
    panel_w, panel_h = (w - 2*34 - gap)//2, 340
    panels=[]
    for row, stat in enumerate(("observed", "observed_minus_null")):
        for col, geo in enumerate(("raw", "unit")):
            cohort_order=("Male","Female")
            metric_order=("C","C_anchor")
            selected=[p for cohort in cohort_order for metric in metric_order for p in points
                      if p["cohort"]==cohort and p["geometry"]==geo and p["statistic"]==stat and p["metric"]==metric]
            label_map={"C":"C","C_anchor":"C anchor"}
            for p in selected: p["plot_label"]=f'{p["cohort"]} ({p["speakers"]}) · {label_map[p["metric"]]}'
            title=f'{geo.capitalize()} geometry · ' + ("observed G contrast" if stat=="observed" else "observed − mean time permutation")
            subtitle="Fixed-RMS natural generator contrast; full-A lag domain, guard20 support"
            px=34+col*(panel_w+gap); py=82+row*(panel_h+gap)
            panels.append(_panel_svg(title,subtitle,selected,px,py,panel_w,panel_h,label_field="plot_label",color_field="cohort",
                         colors={"Male":"#2463a6","Female":"#bb4b3f"},marker_field="metric",marker_map={"C":"circle","C_anchor":"diamond"}))
    return svg_document("RMS generator contrast and time-pairing excess",
        "Male (40 speakers) and female (20 speakers); point estimates and paired speaker-bootstrap 99% CIs.",w,h,panels,
        "Null-adjusted value = observed contrast minus the mean over 100 cached full-A index permutations; C and fixed-column C anchor are shown separately.")


def render_mel(points: list[dict[str, Any]], metric: str) -> str:
    w, h, gap = 1540, 1090, 24
    panel_w, panel_h=(w-2*34-gap)//2, 470
    panels=[]
    contrast_order=[x[0] for x in CONTRASTS]
    contrast_labels={x[0]:x[1] for x in CONTRASTS}
    for row, stat in enumerate(("observed","observed_minus_null")):
        for col, geo in enumerate(("raw","unit")):
            selected=[]
            for contrast in contrast_order:
                for support in SUPPORTS:
                    matches=[p for p in points if p["contrast"]==contrast and p["support"]==support and p["geometry"]==geo and p["statistic"]==stat]
                    if len(matches)!=1: raise ValueError(f"MEL plot row cardinality differs: {contrast}/{support}/{geo}/{stat}")
                    p=matches[0]
                    p["plot_label"]=f'{contrast_labels[contrast]} · {SUPPORT_LABEL[support]}'
                    selected.append(p)
            title=f'{geo.capitalize()} geometry · ' + ("observed" if stat=="observed" else "observed − mean time permutation")
            subtitle=f'MEL level partition · {metric}; 80 clips / 40 speakers; all four frozen supports'
            px=34+col*(panel_w+gap); py=82+row*(panel_h+gap)
            panels.append(_panel_svg(title,subtitle,selected,px,py,panel_w,panel_h,label_field="plot_label",color_field="support",colors=SUPPORT_COLOR))
    footer="Labels name the frozen interior-only, boundary-only, and joint contrasts; no phonetic class or causal percentage is inferred. 99% paired speaker-bootstrap CIs."
    return svg_document(f"MEL partition contrasts · {metric}",
        "Observed C and cached-time-permutation excess are separated; natural RAW audio embeddings are fixed across cells.",w,h,panels,footer)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--male",type=Path,default=MALE_DEFAULT)
    parser.add_argument("--female",type=Path,default=FEMALE_DEFAULT)
    parser.add_argument("--mel",type=Path,default=MEL_DEFAULT)
    parser.add_argument("--output-dir",type=Path,default=OUT_DEFAULT)
    args=parser.parse_args()
    male=_read(args.male,MALE_SHA,"male native cached report")
    female=_read(args.female,FEMALE_SHA,"female cached report")
    mel=_read(args.mel,MEL_SHA,"MEL independent report")
    if not isinstance(male,list): raise ValueError("male report should be a list of contrast rows")
    if female.get("status")!="COMPLETE" or mel.get("status")!="PASS": raise ValueError("an input report is not complete")
    rms=collect_rms(male,female)
    mel_c=collect_mel(mel,"C")
    mel_anchor=collect_mel(mel,"C_anchor")
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    outputs={
      "rms_generator_G_observed_and_excess.svg":render_rms(rms),
      "mel_partition_C_observed_and_excess.svg":render_mel(mel_c,"C"),
      "mel_partition_C_anchor_observed_and_excess.svg":render_mel(mel_anchor,"C anchor"),
    }
    manifest={"schema_version":1,"status":"complete","figure_source_sha256":_sha(Path(__file__).resolve()),
      "inputs":{"male_native_full_A_contrasts":{"path":str(args.male.resolve()),"sha256":_sha(args.male)},
                 "female_cached_null":{"path":str(args.female.resolve()),"sha256":_sha(args.female)},
                 "mel_partition_independent_report":{"path":str(args.mel.resolve()),"sha256":_sha(args.mel)}},
      "ci":"paired speaker bootstrap, percentile 99% intervals as sealed in reports","statistic":"observed and observed-minus-mean 100-permutation null are separate panels; no inferential recomputation",
      "figures":{},"plotted_values":{"rms":rms,"mel_C":mel_c,"mel_C_anchor":mel_anchor}}
    for name,svg in outputs.items():
        path=out/name
        if path.exists(): raise FileExistsError(f"refusing to overwrite figure: {path}")
        path.write_text(svg,encoding="utf-8",newline="\n")
        manifest["figures"][name]={"path":str(path),"sha256":_sha(path),"bytes":path.stat().st_size,"format":"SVG"}
    manifest_path=out/"figure_manifest.json"
    if manifest_path.exists(): raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"status":"complete","outputs":manifest["figures"],"manifest":str(manifest_path),"manifest_sha256":_sha(manifest_path)},ensure_ascii=False,separators=(",",":")))

if __name__=="__main__": main()
