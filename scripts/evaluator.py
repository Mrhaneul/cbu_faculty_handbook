"""
evaluator.py
============
Stage 6: Evaluate fine-tuned adapter against base model using ROUGE-L.

Mirrors the GitHub joupark/facultyHandbook Step 10 evaluation approach,
adapted for MLX (Apple Silicon) instead of transformers/PEFT.

Steps:
  10A — Load base model (no adapter), generate answers on valid set
  10B — Load fine-tuned model (with adapter), generate answers on same set
  10C — Compute ROUGE-L for both, print comparison

Input : data/valid.jsonl  (produced by formatter.py)
Output: printed report + optional eval/results.jsonl

Usage:
    python evaluator.py
    python evaluator.py --adapter outputs/cbu-gemma4-e4b-mlx-v1
    python evaluator.py --adapter outputs/cbu-gemma4-e4b-mlx-v1 --out eval/results.jsonl
    python evaluator.py --max-tokens 512 --max-examples 20
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# ── Defaults ──────────────────────────────────────────────────
BASE_MODEL   = 'mlx-community/gemma-4-E4B-it-4bit'
ADAPTER_DIR  = 'outputs/cbu-gemma4-e4b-mlx-v1'
VALID_PATH   = 'data/valid.jsonl'
MAX_TOKENS   = 512
MAX_EXAMPLES = None   # None = use all

SYSTEM_PROMPT = (
    "You are a knowledgeable CBU Faculty Affairs assistant with expertise in the "
    "California Baptist University Employee Manual — Faculty Section (Updated 01/2026). "
    "You help faculty understand university policies, procedures, rights, and responsibilities. "
    "Place all policy citations at the END of your answer in "
    "[Reference: CBU Faculty Manual, Policy X.XXX] format — NEVER at the beginning."
)


# ── Data loading ───────────────────────────────────────────────

def load_valid(path: str, max_n: Optional[int] = None) -> list[dict]:
    examples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except Exception:
                pass
    if max_n:
        examples = examples[:max_n]
    return examples


def extract_question_and_answer(example: dict) -> tuple[str, str]:
    """Pull question and reference answer out of a formatted training example."""
    text = example.get('text', '')

    # Extract question from between <|user|> and <|end|>
    q_start = text.find('<|user|>')
    q_end   = text.find('<|end|>', q_start)
    question = ''
    if q_start != -1 and q_end != -1:
        question = text[q_start + len('<|user|>'):q_end].strip()

    # Extract reference answer — after <channel|> (skip thinking trace) or after <|assistant|>
    ref_answer = ''
    chan_end = text.find('<channel|>')
    if chan_end != -1:
        # Everything after <channel|> up to the final <|end|>
        after_chan = text[chan_end + len('<channel|>'):].strip()
        end_tok = after_chan.rfind('<|end|>')
        ref_answer = after_chan[:end_tok].strip() if end_tok != -1 else after_chan
    else:
        # No thinking trace — take everything after <|assistant|>
        a_start = text.find('<|assistant|>')
        a_end   = text.rfind('<|end|>')
        if a_start != -1 and a_end > a_start:
            ref_answer = text[a_start + len('<|assistant|>'):a_end].strip()

    return question, ref_answer


# ── Inference ──────────────────────────────────────────────────

def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {'role': 'system',  'content': SYSTEM_PROMPT},
        {'role': 'user',    'content': question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: manual Gemma 4 format
        return (
            f'<bos><|system|>\n{SYSTEM_PROMPT}\n<|end|>\n'
            f'<|user|>\n{question}\n<|end|>\n'
            f'<|assistant|>\n'
        )


def run_inference(model, tokenizer, questions: list[str],
                  max_tokens: int, label: str) -> list[str]:
    from mlx_lm import generate as mlx_generate
    responses = []
    for i, q in enumerate(questions, 1):
        prompt = build_prompt(tokenizer, q)
        print(f'  [{label}] {i}/{len(questions)} generating...', end='\r', flush=True)
        resp = mlx_generate(model, tokenizer, prompt=prompt,
                            max_tokens=max_tokens, verbose=False)
        # Strip any prompt echo if present
        if resp.startswith(prompt):
            resp = resp[len(prompt):]
        responses.append(resp.strip())
    print(f'  [{label}] Done.{" " * 30}')
    return responses


# ── ROUGE-L scoring ────────────────────────────────────────────

def rouge_l_scores(references: list[str], predictions: list[str]) -> list[float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return [
        scorer.score(ref, pred)['rougeL'].fmeasure
        for ref, pred in zip(references, predictions)
    ]


# ── Report ─────────────────────────────────────────────────────

def print_report(questions: list[str], references: list[str],
                 base_preds: list[str], ft_preds: list[str],
                 base_scores: list[float], ft_scores: list[float]) -> None:
    n = len(questions)
    avg_base = sum(base_scores) / n if n else 0.0
    avg_ft   = sum(ft_scores)   / n if n else 0.0
    delta    = avg_ft - avg_base

    print('\n' + '=' * 60)
    print('  CBU Faculty Manual — Evaluation Results (ROUGE-L)')
    print('=' * 60)
    print(f'  Examples evaluated : {n}')
    print(f'  Base model ROUGE-L : {avg_base:.4f}')
    print(f'  Fine-tuned ROUGE-L : {avg_ft:.4f}')
    sign = '+' if delta >= 0 else ''
    print(f'  Improvement        : {sign}{delta:.4f}  ({sign}{delta/max(avg_base,1e-9)*100:.1f}%)')
    print('=' * 60)

    print('\n  Per-example breakdown:')
    print(f'  {"#":<4} {"Base":>7} {"FT":>7} {"Delta":>7}  Question (truncated)')
    print(f'  {"-"*4} {"-"*7} {"-"*7} {"-"*7}  {"-"*40}')
    for i, (q, bs, fs) in enumerate(zip(questions, base_scores, ft_scores), 1):
        d = fs - bs
        sign = '+' if d >= 0 else ''
        print(f'  {i:<4} {bs:>7.4f} {fs:>7.4f} {sign+f"{d:.4f}":>7}  {q[:60]}')

    print('\n  Sample comparison (first example):')
    if questions:
        print(f'\n  Q: {questions[0]}')
        print(f'\n  Reference:\n    {references[0][:300]}...' if len(references[0]) > 300 else f'\n  Reference:\n    {references[0]}')
        print(f'\n  Base model:\n    {base_preds[0][:300]}...' if len(base_preds[0]) > 300 else f'\n  Base model:\n    {base_preds[0]}')
        print(f'\n  Fine-tuned:\n    {ft_preds[0][:300]}...' if len(ft_preds[0]) > 300 else f'\n  Fine-tuned:\n    {ft_preds[0]}')
    print()


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Stage 6: ROUGE-L evaluation of base vs fine-tuned CBU model'
    )
    parser.add_argument('--model',        default=BASE_MODEL)
    parser.add_argument('--adapter',      default=ADAPTER_DIR,
                        help='Adapter directory (outputs/cbu-gemma4-e4b-mlx-v1)')
    parser.add_argument('--valid',        default=VALID_PATH)
    parser.add_argument('--out',          default=None,
                        help='Optional JSONL path to save per-example results')
    parser.add_argument('--max-tokens',   type=int, default=MAX_TOKENS)
    parser.add_argument('--max-examples', type=int, default=MAX_EXAMPLES)
    args = parser.parse_args()

    # ── 10A: Validate inputs ──────────────────────────────────
    if not Path(args.valid).exists():
        print(f'[evaluator] ERROR: {args.valid} not found. Run formatter.py first.')
        sys.exit(1)
    if not Path(args.adapter).exists():
        print(f'[evaluator] ERROR: adapter {args.adapter} not found. Run trainer.py first.')
        sys.exit(1)

    examples = load_valid(args.valid, args.max_examples)
    if not examples:
        print(f'[evaluator] ERROR: no examples found in {args.valid}')
        sys.exit(1)
    print(f'[evaluator] Loaded {len(examples)} validation examples from {args.valid}')

    questions, references = [], []
    for ex in examples:
        q, ref = extract_question_and_answer(ex)
        if q and ref:
            questions.append(q)
            references.append(ref)
    print(f'[evaluator] Parsed {len(questions)} question/answer pairs\n')

    from mlx_lm import load as mlx_load

    # ── 10B: Base model inference ─────────────────────────────
    print('[evaluator] ── Step 10B: Loading base model (no adapter)...')
    base_model, base_tok = mlx_load(args.model)
    base_preds = run_inference(base_model, base_tok, questions, args.max_tokens, 'base')
    del base_model  # free memory before loading fine-tuned

    # ── 10B: Fine-tuned model inference ──────────────────────
    print('[evaluator] ── Step 10B: Loading fine-tuned model (with adapter)...')
    ft_model, ft_tok = mlx_load(args.model, adapter_path=args.adapter)
    ft_preds = run_inference(ft_model, ft_tok, questions, args.max_tokens, 'fine-tuned')
    del ft_model

    # ── 10C: ROUGE-L scoring ──────────────────────────────────
    print('[evaluator] ── Step 10C: Computing ROUGE-L scores...')
    base_scores = rouge_l_scores(references, base_preds)
    ft_scores   = rouge_l_scores(references, ft_preds)

    print_report(questions, references, base_preds, ft_preds, base_scores, ft_scores)

    # ── Optional: save results ────────────────────────────────
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for i, (q, ref, bp, fp, bs, fs) in enumerate(
                zip(questions, references, base_preds, ft_preds, base_scores, ft_scores)
            ):
                f.write(json.dumps({
                    'example':       i + 1,
                    'question':      q,
                    'reference':     ref,
                    'base_pred':     bp,
                    'ft_pred':       fp,
                    'base_rougeL':   round(bs, 6),
                    'ft_rougeL':     round(fs, 6),
                    'delta':         round(fs - bs, 6),
                }) + '\n')
        print(f'[evaluator] Results saved → {args.out}')


if __name__ == '__main__':
    main()
