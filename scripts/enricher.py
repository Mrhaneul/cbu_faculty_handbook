"""
enricher.py
===========
Stage 2: Classify each CBU Faculty Manual chunk with grounded metadata.

Calls Ollama at temperature=0.1 (classification task). Category, section, and
section_weight are authoritative from chunker.py's POLICY_MAP — never overridden
by the LLM. The LLM classifies the content_type, audience, and whether the
chunk has actionable deadlines or explicit rights language.

Input  : data/chunks.jsonl
Output : data/enriched.jsonl

Usage:
    python enricher.py
    python enricher.py --chunks data/chunks.jsonl --out data/enriched.jsonl
    python enricher.py --max 20   # quick test on first 20 chunks
"""

import re
import json
import argparse
import requests
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────

OLLAMA_MODEL  = 'gemma4:latest'
OLLAMA_URL    = 'http://localhost:11434/api/generate'
CLASSIFY_TEMP = 0.1   # MUST stay at 0.1 — deterministic classification

VALID_CONTENT_TYPES = {
    'policy',           # statement of a rule or standard
    'procedure',        # step-by-step process
    'eligibility',      # who qualifies and under what conditions
    'rights',           # entitlements of faculty
    'definition',       # explanation of a term or concept
    'committee_info',   # committee purpose, composition, or responsibilities
    'general',          # introductory or miscellaneous
}

VALID_AUDIENCES = {
    'all_faculty',
    'full_time',
    'part_time',
    'adjunct',
    'tenured',
    'untenured',
    'department_chair',
    'administration',
    'general',
}


# ── Ollama helpers ─────────────────────────────────────────────

def coerce_str(val) -> str:
    if isinstance(val, str):  return val
    if isinstance(val, dict): return json.dumps(val)
    if isinstance(val, list): return ' '.join(str(v) for v in val)
    return str(val)


def generate(prompt: str, system: str = '', temperature: float = CLASSIFY_TEMP) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                'model':  OLLAMA_MODEL,
                'prompt': prompt,
                'system': system,
                'stream': False,
                'options': {'temperature': temperature, 'top_k': 40,
                            'top_p': 0.95, 'num_predict': 256},
            },
            timeout=120,
        )
        return coerce_str(resp.json().get('response', ''))
    except Exception as e:
        print(f'  [enricher] Ollama error: {e}')
        return ''


def extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None


# ── Classification prompt ──────────────────────────────────────

CLASSIFY_SYSTEM = (
    'You are an HR/academic affairs analyst classifying excerpts from the '
    'California Baptist University Faculty Manual. Classify ONLY based on '
    'what the source text explicitly states. Respond with valid JSON only, no prose.'
)

CLASSIFY_TEMPLATE = '''\
Classify this excerpt from the CBU Faculty Manual. Choose ONLY from the allowed values.

POLICY: {policy_num} — {policy_title}
SECTION: {section}

TEXT:
"""
{text}
"""

Respond with JSON:
{{
  "content_type" : "<policy|procedure|eligibility|rights|definition|committee_info|general>",
  "audience"     : "<all_faculty|full_time|part_time|adjunct|tenured|untenured|department_chair|administration|general>",
  "has_deadlines": <true|false>,
  "has_rights"   : <true|false>
}}

Rules:
- "audience" is the primary audience explicitly addressed by this text.
- "has_deadlines" is true only if specific time limits or dates are stated.
- "has_rights" is true if the text grants explicit entitlements to faculty.
- Do NOT infer information not present in the source text.
'''


# ── Validation and fallback ────────────────────────────────────

def coerce_field(value, valid_set: set, default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in valid_set:
        return value.strip().lower()
    return default


def validate_and_fix(meta: dict, chunk: dict) -> dict:
    return {
        # category/section/weight come from POLICY_MAP — never from LLM
        'category':       chunk.get('category', 'general'),
        'section':        chunk.get('section', 'Unknown'),
        'section_weight': chunk.get('section_weight', 1.5),
        # LLM-classified fields
        'content_type':  coerce_field(meta.get('content_type', ''),  VALID_CONTENT_TYPES, 'general'),
        'audience':      coerce_field(meta.get('audience', ''),       VALID_AUDIENCES,     'general'),
        'has_deadlines': bool(meta.get('has_deadlines', False)),
        'has_rights':    bool(meta.get('has_rights', False)),
    }


def heuristic_metadata(chunk: dict) -> dict:
    """Keyword-based fallback when Ollama fails."""
    text = chunk.get('text', '').lower()

    content_type = 'general'
    if any(w in text for w in ['step 1', 'step 2', 'must submit', 'shall submit',
                                'the following steps', 'procedure:', 'process:']):
        content_type = 'procedure'
    elif any(w in text for w in ['eligible', 'eligibility', 'qualifies', 'qualify',
                                  'to be eligible', 'must have']):
        content_type = 'eligibility'
    elif any(w in text for w in ['entitled to', 'has the right', 'right to', 'rights of']):
        content_type = 'rights'
    elif any(w in text for w in ['committee', 'council', 'board', 'membership',
                                  'purpose:', 'composition:']):
        content_type = 'committee_info'
    elif any(w in text for w in ['means ', 'defined as', 'definition:', 'is defined',
                                  'refers to', 'glossary']):
        content_type = 'definition'
    elif any(w in text for w in ['policy:', 'policy states', 'the university shall',
                                  'shall be', 'will be', 'are required']):
        content_type = 'policy'

    audience = 'all_faculty'
    if any(w in text for w in ['adjunct', 'part-time']):
        audience = 'adjunct'
    elif any(w in text for w in ['full-time faculty', 'contracted faculty']):
        audience = 'full_time'
    elif any(w in text for w in ['tenured faculty', 'tenured member']):
        audience = 'tenured'
    elif any(w in text for w in ['department chair', 'dept. chair', 'dean']):
        audience = 'department_chair'

    deadline_words = ['within', 'days', 'by the end of', 'deadline', 'no later than',
                       'must be submitted', 'weeks', 'annual', 'semester', 'year']
    rights_words   = ['entitled to', 'has the right', 'right to appeal', 'may request',
                       'academic freedom', 'protection']

    return {
        'category':       chunk.get('category', 'general'),
        'section':        chunk.get('section', 'Unknown'),
        'section_weight': chunk.get('section_weight', 1.5),
        'content_type':   content_type,
        'audience':       audience,
        'has_deadlines':  any(w in text for w in deadline_words),
        'has_rights':     any(w in text for w in rights_words),
    }


# ── Main loop ──────────────────────────────────────────────────

def run(chunks_path: str, out_path: str, max_chunks: Optional[int] = None) -> int:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    with open(chunks_path, encoding='utf-8') as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except Exception:
                pass
    print(f'[enricher] Loaded {len(chunks)} chunks')

    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f'[enricher] Capped at {max_chunks} (--max)')

    done_ids: set = set()
    if Path(out_path).exists():
        with open(out_path, encoding='utf-8') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line).get('chunk_id'))
                except Exception:
                    pass
        print(f'[enricher] Resuming — {len(done_ids)} already enriched')

    llm_ok = 0
    heuristic = 0

    with open(out_path, 'a', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            cid = chunk.get('chunk_id')
            if cid in done_ids:
                continue

            label = f'[{i+1}/{len(chunks)}] {chunk["policy_num"]} chunk_{chunk["chunk_index"]}'
            print(f'  {label}', end=' ... ', flush=True)

            prompt = CLASSIFY_TEMPLATE.format(
                policy_num   = chunk['policy_num'],
                policy_title = chunk['policy_title'],
                section      = chunk['section'],
                text         = chunk['text'][:900],
            )
            raw    = generate(prompt, CLASSIFY_SYSTEM)
            parsed = extract_json(raw) if raw else None

            if parsed:
                meta = validate_and_fix(parsed, chunk)
                llm_ok += 1
                print('OK')
            else:
                meta = heuristic_metadata(chunk)
                heuristic += 1
                print('HEURISTIC')

            enriched = {**chunk, 'metadata': meta}
            f.write(json.dumps(enriched) + '\n')
            f.flush()
            done_ids.add(cid)

    total = llm_ok + heuristic
    print(f'\n[enricher] Done — {total} enriched ({llm_ok} LLM, {heuristic} heuristic)')
    print(f'[enricher] Output → {out_path}')
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 2: Classify CBU Faculty Manual chunks with metadata'
    )
    parser.add_argument('--chunks', default='data/chunks.jsonl')
    parser.add_argument('--out',    default='data/enriched.jsonl')
    parser.add_argument('--max',    type=int, default=None,
                        help='Cap number of chunks (for testing)')
    parser.add_argument('--model',  default=OLLAMA_MODEL)
    args = parser.parse_args()
    OLLAMA_MODEL = args.model
    run(args.chunks, args.out, args.max)
