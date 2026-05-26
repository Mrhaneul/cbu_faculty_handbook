"""
formatter.py
============
Stage 4a: Format seeds.jsonl into Gemma 4 chat-template training examples.

Produces train.jsonl and valid.jsonl for MLX training.
File names match MLX convention (valid.jsonl, NOT val.jsonl).

Gemma 4 chat format with thinking trace:
    <bos><|system|>
    {system_prompt}
    <|end|>
    <|user|>
    {question}
    <|end|>
    <|assistant|>
    <|channel>thought
    {thinking_trace}
    <channel|>
    {answer}
    <|end|>

Input  : data/seeds.jsonl
Output : data/train.jsonl, data/valid.jsonl

Usage:
    python formatter.py
    python formatter.py --seeds data/seeds.jsonl --train data/train.jsonl --val data/valid.jsonl
    python formatter.py --val-ratio 0.10
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

VAL_RATIO   = 0.10
RANDOM_SEED = 42

CATEGORY_LABELS: dict = {
    'foundation':          'CBU Mission and Values',
    'academic_freedom':    'Academic Freedom and Intellectual Property',
    'employment':          'Faculty Employment and Appointments',
    'conduct':             'Faculty Conduct Policies',
    'tenure_promotion':    'Tenure and Promotion',
    'leave_benefits':      'Leave Policies and Benefits',
    'compensation':        'Salary and Compensation',
    'academic_procedures': 'Academic Procedures',
    'grievances':          'Grievances and Dispute Resolution',
    'governance':          'Faculty Governance and Committees',
    'appendix':            'Glossary and Appendices',
    'general':             'General Faculty Policy',
    'unknown':             'Faculty Policy',
}

SYSTEM_TEMPLATE = (
    "You are a knowledgeable CBU Faculty Affairs assistant with expertise in the "
    "California Baptist University Employee Manual — Faculty Section (Updated 01/2026). "
    "You help faculty understand university policies, procedures, rights, and responsibilities. "
    "Topic area: {category_label}. Audience: {audience}. "
    "Place all policy citations at the END of your answer in "
    "[Reference: CBU Faculty Manual, Policy X.XXX] format — NEVER at the beginning."
)


def build_system_prompt(meta: dict) -> str:
    cat   = meta.get('category', 'general')
    label = CATEGORY_LABELS.get(cat, 'Faculty Policy')
    aud   = meta.get('audience', 'all faculty').replace('_', ' ')
    return SYSTEM_TEMPLATE.format(category_label=label, audience=aud)


def format_example(seed: dict) -> Optional[dict]:
    q    = seed.get('question',      '').strip()
    tt   = seed.get('thinking_trace','').strip()
    ans  = seed.get('answer',        '').strip()
    meta = seed.get('metadata',      {})

    if not q or not ans:
        return None

    system = build_system_prompt(meta)

    if tt:
        assistant_block = f'<|channel>thought\n{tt}\n<channel|>\n{ans}'
    else:
        assistant_block = ans

    text = (
        f'<bos><|system|>\n{system}\n<|end|>\n'
        f'<|user|>\n{q}\n<|end|>\n'
        f'<|assistant|>\n{assistant_block}\n<|end|>'
    )

    return {
        'text':          text,
        'qa_id':         seed.get('qa_id'),
        'question_type': seed.get('question_type'),
        'policy_num':    seed.get('source_policy'),
        'category':      meta.get('category'),
        'audience':      meta.get('audience'),
    }


def format_and_split(
    seeds_path: str,
    train_path: str,
    val_path:   str,
    val_ratio:  float = VAL_RATIO,
) -> tuple[int, int]:
    seeds = []
    with open(seeds_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seeds.append(json.loads(line))
            except Exception:
                pass
    print(f'[formatter] Loaded {len(seeds)} seeds from {seeds_path}')

    examples = []
    skipped  = 0
    for seed in seeds:
        ex = format_example(seed)
        if ex:
            examples.append(ex)
        else:
            skipped += 1
    if skipped:
        print(f'[formatter] Skipped {skipped} malformed seeds')

    random.seed(RANDOM_SEED)
    random.shuffle(examples)

    n_val   = max(1, int(len(examples) * val_ratio))
    n_train = len(examples) - n_val
    train   = examples[:n_train]
    val     = examples[n_train:]

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, 'w', encoding='utf-8') as f:
        for ex in train:
            f.write(json.dumps(ex) + '\n')

    with open(val_path, 'w', encoding='utf-8') as f:
        for ex in val:
            f.write(json.dumps(ex) + '\n')

    qt_dist  = Counter(ex.get('question_type', 'unknown') for ex in train)
    cat_dist = Counter(ex.get('category', 'unknown')      for ex in train)

    print(f'\n[formatter] Split: {n_train} train | {n_val} valid')
    print('[formatter] Question type distribution (train):')
    for qt, cnt in sorted(qt_dist.items(), key=lambda x: -x[1]):
        print(f'  {qt:<28}: {cnt}')
    print('[formatter] Category distribution (train):')
    for cat, cnt in sorted(cat_dist.items(), key=lambda x: -x[1]):
        print(f'  {cat:<28}: {cnt}')
    print(f'\n[formatter] Train → {train_path}')
    print(f'[formatter] Valid → {val_path}')

    return n_train, n_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 4a: Format seeds into Gemma 4 chat-template training data'
    )
    parser.add_argument('--seeds',     default='data/seeds.jsonl')
    parser.add_argument('--train',     default='data/train.jsonl')
    parser.add_argument('--val',       default='data/valid.jsonl',
                        help='Validation output (MLX expects valid.jsonl, default)')
    parser.add_argument('--val-ratio', type=float, default=VAL_RATIO)
    args = parser.parse_args()
    format_and_split(args.seeds, args.train, args.val, args.val_ratio)
