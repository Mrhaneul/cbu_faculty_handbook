"""
generator.py
============
Stage 3: Synthetic QA generation with thinking traces for CBU Faculty Manual.

Calls gemma4:latest (or :31b) via Ollama at temperature=0.7.
Seeds file is written in append mode — safe to interrupt and resume.

Input  : data/enriched.jsonl
Output : data/seeds.jsonl  (or --out for per-machine shards)

Output schema per line:
{
    qa_id, source_policy, question_type,
    question, thinking_trace, answer,
    metadata: {category, section, content_type, audience,
               has_deadlines, difficulty, citation_policy}
}

CRITICAL RULES:
  1. Temperature MUST be 0.7 for generation.
  2. NEVER start answer with "According to the CBU" — citation goes at END.
  3. NEVER use 'w' mode for seeds file — always append ('a').
  4. Thinking trace must cite the policy number (e.g., "Policy 3.101").
  5. Answer MUST end with [Reference: CBU Faculty Manual, Policy X.XXX].
  6. Question must be ≥ 15 words.

Usage:
    python generator.py
    python generator.py --enriched data/enriched.jsonl --out data/seeds.jsonl --target 600
    python generator.py --model gemma4:31b --target 800
"""

import re
import json
import random
import argparse
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

# ── Config ────────────────────────────────────────────────────

OLLAMA_MODEL = 'gemma4:latest'
OLLAMA_URL   = 'http://localhost:11434/api/generate'
GEN_TEMP     = 0.7
RANDOM_SEED  = 42
TARGET_PAIRS = 600
MACHINE_ID   = 'm0'

MAX_THINKING_WORDS = 300
MIN_QUESTION_WORDS = 15
MIN_ANSWER_WORDS   = 40

# Section weights from POLICY_MAP (higher = more QA pairs generated)
SECTION_WEIGHTS: dict[str, float] = {
    'foundation':          2.0,
    'academic_freedom':    2.5,
    'employment':          2.5,
    'conduct':             2.0,
    'tenure_promotion':    3.0,
    'leave_benefits':      2.5,
    'compensation':        2.5,
    'academic_procedures': 2.0,
    'grievances':          2.5,
    'governance':          1.5,
    'appendix':            1.0,
    'unknown':             1.0,
}

# 8 question types — weights reflect how useful each type is for a faculty assistant
QUESTION_TYPES: dict[str, float] = {
    'factual':                 1.5,
    'procedural':              2.0,
    'eligibility':             2.0,
    'scenario':                2.0,
    'rights_responsibilities': 1.5,
    'comparative':             1.0,
    'timeline_deadline':       1.5,
    'committee_role':          1.0,
}

QT_NAMES = list(QUESTION_TYPES.keys())
QT_PROBS = [QUESTION_TYPES[t] for t in QT_NAMES]

CITATION_RE  = re.compile(r'\[Reference:\s*CBU Faculty Manual', re.IGNORECASE)
POLICY_REF_RE = re.compile(r'\b(?:Policy\s+)?(?:3|4)\.\d+', re.IGNORECASE)
BAD_OPENING  = re.compile(r'^According to (?:the )?CBU', re.IGNORECASE)


# ── Ollama helpers ─────────────────────────────────────────────

def coerce_str(val) -> str:
    if isinstance(val, str):  return val
    if isinstance(val, dict): return json.dumps(val)
    if isinstance(val, list): return ' '.join(str(v) for v in val)
    return str(val)


def generate(prompt: str, system: str = '', temperature: float = GEN_TEMP) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                'model':  OLLAMA_MODEL,
                'prompt': prompt,
                'system': system,
                'stream': False,
                'options': {'temperature': temperature, 'top_k': 64,
                            'top_p': 0.95, 'num_predict': 2048},
            },
            timeout=300,
        )
        return coerce_str(resp.json().get('response', ''))
    except Exception as e:
        print(f'  [generator] Ollama error: {e}')
        return ''


def strip_thinking_tags(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'Thinking\.\.\..*?done thinking\.', '', text, flags=re.DOTALL)
    return text.strip()


def extract_json(text: str) -> Optional[dict]:
    text = strip_thinking_tags(text)
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL)
    for blk in sorted(matches, key=len, reverse=True):
        try: return json.loads(blk)
        except Exception: pass
    return None


# ── Question-type instructions ─────────────────────────────────

TYPE_INSTRUCTIONS: dict[str, str] = {
    'factual': (
        'Generate a FACTUAL question testing recall of a specific policy rule, '
        'requirement, or definition stated verbatim in the source text.'
    ),
    'procedural': (
        'Generate a PROCEDURAL question asking WHAT STEPS a faculty member must follow '
        'to complete the specific process described in the source text.'
    ),
    'eligibility': (
        'Generate an ELIGIBILITY question asking who QUALIFIES for a benefit, leave, '
        'or status described in the source text, and under what conditions.'
    ),
    'scenario': (
        'Generate a SCENARIO-BASED question that presents a realistic situation a CBU '
        'faculty member might face and asks what the policy says they should do. '
        'Include a concrete scenario in the question.'
    ),
    'rights_responsibilities': (
        'Generate a RIGHTS AND RESPONSIBILITIES question asking what a faculty member '
        'is explicitly entitled to or required to do under this policy.'
    ),
    'comparative': (
        'Generate a COMPARATIVE question asking how this policy applies differently '
        'to two distinct faculty types, statuses, or circumstances described in the text.'
    ),
    'timeline_deadline': (
        'Generate a TIMELINE or DEADLINE question asking when something must be done, '
        'how long a process takes, or what consequences follow from missing a deadline '
        '— based only on timing information explicitly stated in the source text.'
    ),
    'committee_role': (
        'Generate a COMMITTEE ROLE question asking about the purpose, composition, '
        'authority, or responsibilities of the committee described in the source text.'
    ),
}

GENERATION_SYSTEM = """\
You are an expert California Baptist University Faculty Affairs specialist generating \
high-quality training data for a CBU Faculty Manual AI assistant.

ABSOLUTE RULES:
1. Ground every answer ONLY in the provided source text. No outside knowledge.
2. The answer MUST end with [Reference: CBU Faculty Manual, Policy X.XXX] — NEVER at the start.
3. NEVER begin the answer with "According to the CBU Faculty Manual" — citations go at the END.
4. The thinking_trace MUST reference the policy number (e.g., "Policy 3.101" or "section 3.101").
5. thinking_trace must be ≤ 300 words. Be concise and practical.
6. The question must be ≥ 15 words and specific enough to require knowing the policy.
7. Output valid JSON ONLY — no prose before or after.
"""

GENERATION_TEMPLATE = """\
SOURCE TEXT (Policy {policy_num}: {policy_title}, Section: {section}, Audience: {audience}):
\"\"\"
{text}
\"\"\"

TASK: {instruction}

Respond with this exact JSON structure:
{{
  "question": "<your question (≥15 words)>",
  "thinking_trace": "<reasoning: cite policy number, ≤300 words>",
  "answer": "<substantive answer (≥40 words), ending with [Reference: CBU Faculty Manual, Policy {policy_num}]>",
  "difficulty": "<basic|intermediate|advanced>",
  "citation_policy": "{policy_num}"
}}
"""

COMPARATIVE_TEMPLATE = """\
SOURCE A (Policy {policy_num_a}: {policy_title_a}):
\"\"\"
{text_a}
\"\"\"

SOURCE B (Policy {policy_num_b}: {policy_title_b}):
\"\"\"
{text_b}
\"\"\"

TASK: {instruction}

Respond with this exact JSON structure:
{{
  "question": "<comparative question using both sources (≥15 words)>",
  "thinking_trace": "<reasoning citing both policy numbers, ≤300 words>",
  "answer": "<answer comparing both policies (≥40 words), ending with [Reference: CBU Faculty Manual, Policy {policy_num_a} and {policy_num_b}]>",
  "difficulty": "<basic|intermediate|advanced>",
  "citation_policy": "{policy_num_a}"
}}
"""


# ── Task scheduling ────────────────────────────────────────────

def build_task_list(chunks: list[dict], target: int,
                    seed: int = RANDOM_SEED) -> list[tuple]:
    rng = random.Random(seed)

    by_section: dict[str, list] = {}
    for c in chunks:
        by_section.setdefault(c.get('category', 'general'), []).append(c)

    # Weight pool by section_weight
    pool: list[dict] = []
    for c in chunks:
        w    = SECTION_WEIGHTS.get(c.get('category', 'general'), 1.5)
        reps = max(1, round(w))
        pool.extend([c] * reps)
    rng.shuffle(pool)

    tasks = []
    for chunk in itertools.cycle(pool):
        if len(tasks) >= target:
            break

        qt = rng.choices(QT_NAMES, weights=QT_PROBS, k=1)[0]

        chunk_b = None
        if qt == 'comparative':
            same_sec = by_section.get(chunk.get('category', 'general'), [])
            candidates = [
                c for c in same_sec
                if c.get('chunk_id') != chunk.get('chunk_id')
                and c.get('policy_num') != chunk.get('policy_num')
            ]
            if candidates:
                chunk_b = rng.choice(candidates)
            else:
                qt = 'factual'

        tasks.append((chunk, qt, chunk_b))
    return tasks


# ── QA validation ──────────────────────────────────────────────

def validate_qa(pair: dict) -> tuple[bool, str]:
    q   = pair.get('question',      '').strip()
    tt  = pair.get('thinking_trace','').strip()
    ans = pair.get('answer',        '').strip()

    if not q:                             return False, 'empty question'
    if not ans:                           return False, 'empty answer'
    if len(q.split()) < MIN_QUESTION_WORDS:
        return False, f'question too short ({len(q.split())} words)'
    if len(ans.split()) < MIN_ANSWER_WORDS:
        return False, f'answer too short ({len(ans.split())} words)'
    if len(tt.split()) > MAX_THINKING_WORDS:
        return False, f'thinking trace too long ({len(tt.split())} words)'
    if BAD_OPENING.match(ans):
        return False, "answer starts with forbidden opener"
    if not CITATION_RE.search(ans):
        return False, 'missing [Reference: CBU Faculty Manual, ...] in answer'
    if not POLICY_REF_RE.search(tt):
        return False, 'thinking trace does not cite a policy number'
    if 'SOURCE TEXT' in q.upper() or 'SOURCE A' in q.upper():
        return False, 'prompt leaked into question'
    return True, 'ok'


# ── Main generation loop ───────────────────────────────────────

def _build_prompt(chunk_a: dict, qt: str,
                  chunk_b: Optional[dict]) -> tuple[str, str]:
    """Returns (prompt_text, primary_policy_num)."""
    pn_a  = chunk_a.get('policy_num', '')
    meta_a = chunk_a.get('metadata', {})

    if qt == 'comparative' and chunk_b:
        pn_b = chunk_b.get('policy_num', '')
        prompt = COMPARATIVE_TEMPLATE.format(
            policy_num_a   = pn_a,
            policy_title_a = chunk_a.get('policy_title', ''),
            text_a         = chunk_a.get('text', '')[:600],
            policy_num_b   = pn_b,
            policy_title_b = chunk_b.get('policy_title', ''),
            text_b         = chunk_b.get('text', '')[:600],
            instruction    = TYPE_INSTRUCTIONS[qt],
        )
    else:
        prompt = GENERATION_TEMPLATE.format(
            policy_num   = pn_a,
            policy_title = chunk_a.get('policy_title', ''),
            section      = chunk_a.get('section', ''),
            audience     = meta_a.get('audience', 'all_faculty'),
            text         = chunk_a.get('text', '')[:900],
            instruction  = TYPE_INSTRUCTIONS[qt],
        )
    return prompt, pn_a


def _process_task(task_args: tuple) -> Optional[tuple]:
    i, total, chunk_a, qt, chunk_b, resume_key, _ = task_args
    label = f'[{i+1}/{total}] {chunk_a.get("policy_num")} {qt}'
    prompt, pn = _build_prompt(chunk_a, qt, chunk_b)
    raw  = generate(prompt, GENERATION_SYSTEM, GEN_TEMP)
    pair = extract_json(raw) if raw else None

    if pair is None:
        print(f'  {label} ... FAILED (no JSON)', flush=True)
        return None

    passes, reason = validate_qa(pair)
    if not passes:
        print(f'  {label} ... FILTERED ({reason})', flush=True)
        return None

    meta_a = chunk_a.get('metadata', {})
    record = {
        'qa_id':          None,
        'source_policy':  pn,
        'question_type':  qt,
        'question':       pair['question'].strip(),
        'thinking_trace': pair.get('thinking_trace', '').strip(),
        'answer':         pair['answer'].strip(),
        'metadata': {
            'category':        meta_a.get('category', chunk_a.get('category', 'general')),
            'section':         meta_a.get('section',  chunk_a.get('section',  'Unknown')),
            'content_type':    meta_a.get('content_type', 'general'),
            'audience':        meta_a.get('audience', 'all_faculty'),
            'has_deadlines':   meta_a.get('has_deadlines', False),
            'difficulty':      pair.get('difficulty', 'intermediate'),
            'citation_policy': pair.get('citation_policy', pn),
        },
    }
    return resume_key, record, label


def run(enriched_path: str, seeds_path: str, target: int = TARGET_PAIRS,
        seed: int = RANDOM_SEED, workers: int = 1) -> int:
    Path(seeds_path).parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    with open(enriched_path, encoding='utf-8') as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except Exception:
                pass
    print(f'[generator] Loaded {len(chunks)} enriched chunks')
    print(f'[generator] Model: {OLLAMA_MODEL}  Machine: {MACHINE_ID}  '
          f'Seed: {seed}  Workers: {workers}')

    done_keys: set = set()
    if Path(seeds_path).exists():
        with open(seeds_path, encoding='utf-8') as f:
            for line in f:
                try:
                    s = json.loads(line)
                    done_keys.add(s.get('source_policy', '') + s.get('question_type', ''))
                except Exception:
                    pass
        print(f'[generator] Resuming — {len(done_keys)} already saved')

    task_budget = round(target * 1.35)
    tasks = build_task_list(chunks, task_budget, seed=seed)
    print(f'[generator] Scheduled {len(tasks)} tasks → target {target} pairs '
          f'(+35% filter buffer)\n')

    pending = []
    for i, (chunk_a, qt, chunk_b) in enumerate(tasks):
        rk = chunk_a.get('policy_num', '') + qt
        if rk not in done_keys:
            pending.append((i, len(tasks), chunk_a, qt, chunk_b, rk, len(done_keys)))

    skipped = len(tasks) - len(pending)
    print(f'[generator] {skipped} already done, {len(pending)} pending\n')

    lock      = threading.Lock()
    generated = 0
    filtered  = 0

    with open(seeds_path, 'a', encoding='utf-8') as out_f:

        def _write_if_needed(result: Optional[tuple]) -> bool:
            nonlocal generated, filtered
            if result is None:
                with lock:
                    filtered += 1
                return False
            resume_key, record, label = result
            with lock:
                if generated >= target:
                    return False
                record['qa_id'] = f'{MACHINE_ID}-{generated + len(done_keys) + 1:04d}'
                out_f.write(json.dumps(record) + '\n')
                out_f.flush()
                done_keys.add(resume_key)
                generated += 1
                print(f'  {label} ... OK  [{generated}/{target}]', flush=True)
            return True

        if workers == 1:
            for task_args in pending:
                if generated >= target:
                    break
                _write_if_needed(_process_task(task_args))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {}
                it = iter(pending)
                for task_args in it:
                    if generated >= target:
                        break
                    futures[pool.submit(_process_task, task_args)] = True
                    if len(futures) >= workers * 4:
                        break
                while futures:
                    done_f = next(__import__('concurrent.futures', fromlist=['as_completed'])
                                  .as_completed(futures))
                    del futures[done_f]
                    _write_if_needed(done_f.result())
                    if generated < target:
                        for task_args in it:
                            futures[pool.submit(_process_task, task_args)] = True
                            if len(futures) >= workers * 4:
                                break

    print(f'\n[generator] Done — {generated} saved | '
          f'{skipped} already done | {filtered} filtered')
    print(f'[generator] Seeds → {seeds_path}')
    return generated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 3: Generate thinking-trace QA pairs from CBU Faculty Manual chunks'
    )
    parser.add_argument('--enriched', default='data/enriched.jsonl')
    parser.add_argument('--out',      default='data/seeds.jsonl')
    parser.add_argument('--model',    default=OLLAMA_MODEL)
    parser.add_argument('--target',   type=int, default=TARGET_PAIRS)
    parser.add_argument('--seed',     type=int, default=RANDOM_SEED)
    parser.add_argument('--machine-id', default=MACHINE_ID)
    parser.add_argument('--workers',  type=int, default=1,
                        help='Concurrent Ollama requests (default 1)')
    args = parser.parse_args()
    OLLAMA_MODEL = args.model
    MACHINE_ID   = args.machine_id
    run(args.enriched, args.out, args.target, seed=args.seed, workers=args.workers)
