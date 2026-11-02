"""Export descriptive paired pilot plots from completed CPU analysis only."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    result = json.loads((args.run / 'analysis.json').read_text())
    names = result['primary_contrasts_C']
    fig, axes = plt.subplots(1, len(names), figsize=(11.8, 4.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        summary = result['metrics']['C']['contrasts'][name]
        x = np.asarray(summary['values'])
        ax.scatter(np.arange(1, len(x) + 1), x, color='#315b89', s=28, label='Source-speaker mean')
        ax.axhline(0, color='#888888', linestyle='--', linewidth=1)
        ax.axhline(summary['mean'], color='#be5d27', label='Mean paired difference')
        lo, hi = summary['ci97_5']
        ax.axhspan(lo, hi, color='#be5d27', alpha=.12, label='97.5% bootstrap CI of mean')
        ax.set_title(f'{name}: {summary["mean"]:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        ax.set_xlabel('Fixed source-speaker order')
        ax.set_ylabel('Paired Sync-C difference')
        ax.set_xticks(np.arange(1, len(x) + 1))
        ax.grid(axis='y', alpha=.15)
        ax.legend(fontsize=7.5, loc='best')
    fig.suptitle(f'Official SyncNet pilot: {result["coverage"]["complete_speakers"]} source speakers; all planned draws retained')
    fig.tight_layout()
    dest = args.run / 'paired_sync_c.png'
    if dest.exists():
        raise FileExistsError(dest)
    fig.savefig(dest, dpi=180)
    fig.savefig(args.run / 'paired_sync_c.pdf')
    plt.close(fig)
    print(str(dest))


if __name__ == '__main__':
    main()
