"""
trainer.py
==========
Stage 4b: MLX LoRA fine-tuning of Gemma 4 E4B on CBU Faculty Manual data.

Wraps mlx_lm.lora for Apple Silicon training.
Requires the caimll_finetuning conda environment.

Training data must be in data/train.jsonl and data/valid.jsonl
(produced by formatter.py — note: valid.jsonl NOT val.jsonl).

Default model: mlx-community/gemma-4-E4B-it-4bit

Usage:
    python trainer.py
    python trainer.py --iters 1500 --out outputs/cbu-gemma4-e4b-v1
    python trainer.py --dry-run    # print command without running
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────

BASE_MODEL      = 'mlx-community/gemma-4-E4B-it-4bit'
OUTPUT_DIR      = 'outputs/cbu-gemma4-e4b-mlx-v1'
DATA_DIR        = 'data'
ITERS           = 1500      # ~5 epochs over ~300 examples
LEARNING_RATE   = 1e-5
BATCH_SIZE      = 1
GRAD_ACCUM      = 4         # effective batch = 4
LORA_RANK       = 8
LORA_SCALE      = 20.0
LORA_LAYERS     = 16        # number of layers to apply LoRA
MAX_SEQ_LENGTH  = 1024      # longer than ATP (1024 > 512) to fit policy reasoning
SAVE_EVERY      = 100
STEPS_PER_EVAL  = 200
STEPS_PER_REPORT = 10
SEED            = 0


def parse_args():
    p = argparse.ArgumentParser(
        description='Stage 4b: MLX LoRA SFT of Gemma 4 E4B on CBU Faculty Manual'
    )
    p.add_argument('--model',          default=BASE_MODEL)
    p.add_argument('--data',           default=DATA_DIR)
    p.add_argument('--out',            default=OUTPUT_DIR,
                   help='Adapter output directory')
    p.add_argument('--iters',          type=int,   default=ITERS)
    p.add_argument('--lr',             type=float, default=LEARNING_RATE)
    p.add_argument('--batch',          type=int,   default=BATCH_SIZE)
    p.add_argument('--grad-accum',     type=int,   default=GRAD_ACCUM)
    p.add_argument('--lora-rank',      type=int,   default=LORA_RANK)
    p.add_argument('--lora-layers',    type=int,   default=LORA_LAYERS)
    p.add_argument('--max-seq-length', type=int,   default=MAX_SEQ_LENGTH)
    p.add_argument('--resume',         default=None,
                   help='Path to adapter checkpoint to resume from')
    p.add_argument('--dry-run',        action='store_true',
                   help='Print the training command without running it')
    return p.parse_args()


def check_data(data_dir: str) -> None:
    train = Path(data_dir) / 'train.jsonl'
    valid = Path(data_dir) / 'valid.jsonl'
    if not train.exists():
        print(f'[trainer] ERROR: {train} not found. Run formatter.py first.')
        sys.exit(1)
    if not valid.exists():
        print(f'[trainer] ERROR: {valid} not found. Run formatter.py first.')
        sys.exit(1)
    n_train = sum(1 for _ in open(train, encoding='utf-8'))
    n_valid = sum(1 for _ in open(valid, encoding='utf-8'))
    print(f'[trainer] Train: {n_train} examples | Valid: {n_valid} examples')

    # Sanity-check first example
    with open(train, encoding='utf-8') as f:
        sample = json.loads(f.readline())['text']
    has_channel = '<|channel>thought' in sample
    has_ref     = '[Reference: CBU Faculty Manual' in sample
    has_end     = '<|end|>' in sample
    print(f'[trainer] Thinking channel : {"OK" if has_channel else "MISSING"}')
    print(f'[trainer] Reference citation: {"OK" if has_ref     else "MISSING"}')
    print(f'[trainer] End token        : {"OK" if has_end     else "MISSING"}')


def build_command(args) -> list[str]:
    cmd = [
        sys.executable, '-m', 'mlx_lm', 'lora',
        '--train',
        '--model',                  args.model,
        '--data',                   args.data,
        '--adapter-path',           args.out,
        '--iters',                  str(args.iters),
        '--learning-rate',          str(args.lr),
        '--batch-size',             str(args.batch),
        '--grad-accumulation-steps',str(args.grad_accum),
        '--num-layers',             str(args.lora_layers),
        '--max-seq-length',         str(args.max_seq_length),
        '--save-every',             str(SAVE_EVERY),
        '--steps-per-eval',         str(STEPS_PER_EVAL),
        '--steps-per-report',       str(STEPS_PER_REPORT),
        '--seed',                   str(SEED),
        '--grad-checkpoint',
    ]
    if args.resume:
        cmd += ['--resume-adapter-file', args.resume]
    return cmd


def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 56)
    print('  CBU Faculty Manual — Gemma 4 E4B MLX Training')
    print('=' * 56)
    print(f'  Model          : {args.model}')
    print(f'  Data dir       : {args.data}')
    print(f'  Output dir     : {args.out}')
    print(f'  Iterations     : {args.iters}')
    print(f'  Learning rate  : {args.lr}')
    print(f'  Batch size     : {args.batch} x {args.grad_accum} = {args.batch * args.grad_accum} effective')
    print(f'  LoRA layers    : {args.lora_layers}')
    print(f'  Max seq length : {args.max_seq_length}')
    if args.resume:
        print(f'  Resuming from  : {args.resume}')
    print('=' * 56 + '\n')

    check_data(args.data)

    cmd = build_command(args)
    print('[trainer] Command:')
    print('  ' + ' \\\n    '.join(cmd))

    if args.dry_run:
        print('\n[trainer] DRY RUN — not executing.')
        return

    print('\n[trainer] Starting training...\n')
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f'\n[trainer] Training exited with code {result.returncode}')
        sys.exit(result.returncode)

    print(f'\n[trainer] Training complete. Adapter → {args.out}')
    print('\nNext step — fuse and export to GGUF:')
    print(f'  python burn_gguf.py --adapter {args.out}')


if __name__ == '__main__':
    main()
