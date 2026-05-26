"""
plot_eval.py
============
Generate ROUGE-L evaluation chart comparing base model vs fine-tuned model
per question type — equivalent to the GitHub joupark/facultyHandbook
Step 10 evaluation visualization.

Usage:
    python plot_eval.py
    python plot_eval.py --out eval/rouge_chart.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

VALID_PATH   = 'data/valid.jsonl'
RESULTS_PATH = 'eval/results.jsonl'

TYPE_LABELS = {
    'factual':                 'Factual',
    'procedural':              'Procedural',
    'eligibility':             'Eligibility',
    'scenario':                'Scenario',
    'rights_responsibilities': 'Rights &\nResponsibilities',
    'comparative':             'Comparative',
    'timeline_deadline':       'Timeline /\nDeadline',
    'committee_role':          'Committee\nRole',
}

TYPE_ORDER = list(TYPE_LABELS.keys())


def load_data(valid_path: str, results_path: str) -> list[dict]:
    valid   = [json.loads(l) for l in open(valid_path,   encoding='utf-8')]
    results = [json.loads(l) for l in open(results_path, encoding='utf-8')]
    assert len(valid) == len(results), (
        f"Mismatch: valid.jsonl has {len(valid)} rows, results.jsonl has {len(results)}"
    )
    merged = []
    for v, r in zip(valid, results):
        merged.append({
            'question_type': v.get('question_type', 'unknown'),
            'base_rougeL':   r['base_rougeL'],
            'ft_rougeL':     r['ft_rougeL'],
        })
    return merged


def aggregate_by_type(data: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list] = defaultdict(lambda: {'base': [], 'ft': []})
    for row in data:
        qt = row['question_type']
        buckets[qt]['base'].append(row['base_rougeL'])
        buckets[qt]['ft'].append(row['ft_rougeL'])

    agg = {}
    for qt, vals in buckets.items():
        agg[qt] = {
            'base_mean': np.mean(vals['base']),
            'ft_mean':   np.mean(vals['ft']),
            'n':         len(vals['base']),
        }
    return agg


def plot(agg: dict, out_path: str, n_total: int) -> None:
    # Order by TYPE_ORDER, append any unexpected types at the end
    ordered = [qt for qt in TYPE_ORDER if qt in agg]
    ordered += [qt for qt in agg if qt not in ordered]

    labels     = [TYPE_LABELS.get(qt, qt) for qt in ordered]
    base_vals  = [agg[qt]['base_mean'] for qt in ordered]
    ft_vals    = [agg[qt]['ft_mean']   for qt in ordered]
    n_vals     = [agg[qt]['n']         for qt in ordered]

    x     = np.arange(len(ordered))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    bars_base = ax.bar(x - width/2, base_vals, width,
                       label='Base Model',
                       color='#4a90d9', alpha=0.85, zorder=3)
    bars_ft   = ax.bar(x + width/2, ft_vals,   width,
                       label='Fine-tuned (CBU)',
                       color='#e05c5c', alpha=0.85, zorder=3)

    # Value labels on bars
    for bar in bars_base:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.3f}', ha='center', va='bottom',
                fontsize=8, color='#aaaaaa')
    for bar in bars_ft:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.3f}', ha='center', va='bottom',
                fontsize=8, color='white', fontweight='bold')

    # Sample size annotation below each group
    for i, n in enumerate(n_vals):
        ax.text(x[i], -0.025, f'n={n}', ha='center', va='top',
                fontsize=7.5, color='#888888')

    ax.set_xlabel('Question Type', color='#cccccc', fontsize=11, labelpad=24)
    ax.set_ylabel('ROUGE-L Score', color='#cccccc', fontsize=11)
    ax.set_title(
        'ROUGE-L: Base Model vs Fine-tuned (CBU Faculty Manual)\n'
        f'{n_total} validation examples  ·  Gemma 4 E4B  ·  300 QA pairs  ·  1500 iters',
        color='white', fontsize=12, fontweight='bold', pad=14,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='#cccccc', fontsize=9)
    ax.set_ylim(0, max(ft_vals + base_vals) * 1.25)
    ax.tick_params(colors='#888888', which='both')
    ax.yaxis.label.set_color('#cccccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.grid(axis='y', color='#333333', linestyle='--', alpha=0.6, zorder=0)

    # Overall averages in legend
    overall_base = np.mean(base_vals)
    overall_ft   = np.mean(ft_vals)
    pct = (overall_ft - overall_base) / overall_base * 100

    patch_base = mpatches.Patch(color='#4a90d9', alpha=0.85,
                                label=f'Base Model  (avg {overall_base:.4f})')
    patch_ft   = mpatches.Patch(color='#e05c5c', alpha=0.85,
                                label=f'Fine-tuned  (avg {overall_ft:.4f}  ·  {pct:+.1f}%)')
    ax.legend(handles=[patch_base, patch_ft],
              facecolor='#1a1d27', edgecolor='#444444',
              labelcolor='#dddddd', fontsize=10, loc='upper right')

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'[plot_eval] Chart saved → {out_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid',   default=VALID_PATH)
    parser.add_argument('--results', default=RESULTS_PATH)
    parser.add_argument('--out',     default='eval/rouge_chart.png')
    args = parser.parse_args()

    data = load_data(args.valid, args.results)
    agg  = aggregate_by_type(data)

    print(f'[plot_eval] {len(data)} examples across {len(agg)} question types\n')
    print(f'  {"Type":<28} {"n":>3}  {"Base":>7}  {"FT":>7}  {"Delta":>8}')
    print(f'  {"-"*28} {"---":>3}  {"-------":>7}  {"-------":>7}  {"--------":>8}')
    for qt in TYPE_ORDER:
        if qt not in agg:
            continue
        v = agg[qt]
        d = v["ft_mean"] - v["base_mean"]
        sign = '+' if d >= 0 else ''
        print(f'  {TYPE_LABELS[qt]:<28} {v["n"]:>3}  {v["base_mean"]:>7.4f}  '
              f'{v["ft_mean"]:>7.4f}  {sign+f"{d:.4f}":>8}')

    plot(agg, args.out, len(data))


if __name__ == '__main__':
    main()
