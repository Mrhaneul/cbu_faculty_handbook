"""
run_pipeline.py
===============
Orchestrator for the CBU Faculty Manual fine-tuning pipeline.

Stages:
  1  chunker.py   — PDF → data/chunks.jsonl
  2  enricher.py  — chunks → data/enriched.jsonl
  3  generator.py — enriched → data/seeds.jsonl
  4  formatter.py — seeds → data/train.jsonl + data/valid.jsonl
  5  trainer.py   — MLX LoRA training → outputs/cbu-gemma4-e4b-mlx-v1/

Usage:
    python run_pipeline.py               # run all stages
    python run_pipeline.py --stages 1,2  # only chunk + enrich
    python run_pipeline.py --stages 3    # only generate (resume-safe)
    python run_pipeline.py --stages 4,5  # only format + train
    python run_pipeline.py --resume      # skip stages whose output already exists
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

STAGES = {
    1: ('chunker.py',   'data/chunks.jsonl'),
    2: ('enricher.py',  'data/enriched.jsonl'),
    3: ('generator.py', 'data/seeds.jsonl'),
    4: ('formatter.py', 'data/train.jsonl'),
    5: ('trainer.py',   'outputs/cbu-gemma4-e4b-mlx-v1'),
    6: ('evaluator.py', 'eval/results.jsonl'),
}


def run_stage(stage_num: int, extra_args: list[str], resume: bool) -> None:
    script, output = STAGES[stage_num]
    output_path = Path(output)

    if resume and output_path.exists():
        size = output_path.stat().st_size if output_path.is_file() else -1
        if size != 0:
            print(f'[pipeline] Stage {stage_num} ({script}) — output exists, skipping.')
            return

    print(f'\n[pipeline] ── Stage {stage_num}: {script} {"─"*40}')
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script)] + extra_args,
        check=False,
    )
    if result.returncode != 0:
        print(f'[pipeline] Stage {stage_num} failed (exit {result.returncode}). Stopping.')
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description='CBU Faculty Manual fine-tuning pipeline orchestrator'
    )
    parser.add_argument('--stages', default='1,2,3,4,5,6',
                        help='Comma-separated stage numbers to run (default: all)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip stages whose output file already exists')
    parser.add_argument('--target', type=int, default=600,
                        help='QA generation target (stage 3 only, default: 600)')
    parser.add_argument('--model', default='gemma4:latest',
                        help='Ollama model for generation/enrichment')
    parser.add_argument('--iters', type=int, default=1500,
                        help='MLX training iterations (stage 5 only, default: 1500)')
    args = parser.parse_args()

    stages = [int(s.strip()) for s in args.stages.split(',')]

    for s in stages:
        extra: list[str] = []
        if s == 2:
            extra = ['--model', args.model]
        elif s == 3:
            extra = ['--model', args.model, '--target', str(args.target)]
        elif s == 5:
            extra = ['--iters', str(args.iters)]
        elif s == 6:
            extra = ['--out', 'eval/results.jsonl']
        run_stage(s, extra, args.resume)

    print('\n[pipeline] All requested stages complete.')


if __name__ == '__main__':
    main()
