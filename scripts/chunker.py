"""
chunker.py
==========
Stage 1: Parse CBU Faculty Manual PDF into policy-level chunks.

Uses pdftotext (no -layout flag) for clean text extraction.
Policy IDs follow the handbook's numbering scheme: 3.000, 3.101, etc.
Each distinct policy (or sub-section thereof) becomes one chunk.

Long policies (> MAX_CHUNK_WORDS) are split at internal section headings
(POLICY:, PROCEDURE:, PURPOSE:, CONDITIONS:, etc.).

Output schema per line of chunks.jsonl:
{
    chunk_id, policy_num, chunk_index, text,
    policy_title, section, category,
    page_range, word_count, effective_date,
    has_numbered_list, cross_references
}

Usage:
    python chunker.py
    python chunker.py --pdf cbu_faculty.pdf --out data/chunks.jsonl
"""

import re
import json
import argparse
import subprocess
from pathlib import Path
from collections import Counter
from typing import Optional

# ── Policy map — title, section group, and category ──────────

POLICY_MAP: dict[str, dict] = {
    "3.000": {"title": "Introduction and Statement of Anti-Discrimination",      "section": "Foundation",            "category": "foundation"},
    "3.001": {"title": "Values of California Baptist University",                "section": "Foundation",            "category": "foundation"},
    "3.002": {"title": "Mission and Philosophy",                                 "section": "Foundation",            "category": "foundation"},
    "3.003": {"title": "Statement of Faith",                                     "section": "Foundation",            "category": "foundation"},
    "3.004": {"title": "Academic Freedom and Responsibility",                    "section": "Foundation",            "category": "academic_freedom"},
    "3.005": {"title": "Intellectual Property Rights",                           "section": "Foundation",            "category": "academic_freedom"},
    "3.100": {"title": "Employment Overview",                                    "section": "Employment",            "category": "employment"},
    "3.101": {"title": "Recruitment",                                            "section": "Employment",            "category": "employment"},
    "3.102": {"title": "Faculty Appointments",                                   "section": "Employment",            "category": "employment"},
    "3.103": {"title": "Faculty Responsibilities",                               "section": "Employment",            "category": "employment"},
    "3.104": {"title": "Faculty Roles",                                          "section": "Employment",            "category": "employment"},
    "3.105": {"title": "Faculty Awards",                                         "section": "Employment",            "category": "employment"},
    "3.106": {"title": "Load",                                                   "section": "Employment",            "category": "employment"},
    "3.107": {"title": "Termination",                                            "section": "Employment",            "category": "employment"},
    "3.108": {"title": "Emeritus Status",                                        "section": "Employment",            "category": "employment"},
    "3.109": {"title": "Adjunct Faculty: Definitions, Roles and Duties",         "section": "Employment",            "category": "employment"},
    "3.110": {"title": "Student Honor Code Violations",                          "section": "Employment",            "category": "conduct"},
    "3.200": {"title": "Promotion and Tenure Overview",                          "section": "Tenure and Promotion",  "category": "tenure_promotion"},
    "3.201": {"title": "Promotion",                                              "section": "Tenure and Promotion",  "category": "tenure_promotion"},
    "3.202": {"title": "Authority to Grant/Terminate Tenure and Purpose of Tenure", "section": "Tenure and Promotion", "category": "tenure_promotion"},
    "3.203": {"title": "Eligibility for Earning Tenure",                         "section": "Tenure and Promotion",  "category": "tenure_promotion"},
    "3.204": {"title": "Qualifications for Earning Promotion and Tenure",        "section": "Tenure and Promotion",  "category": "tenure_promotion"},
    "3.205": {"title": "Sequence of Procedures of Tenure",                       "section": "Tenure and Promotion",  "category": "tenure_promotion"},
    "3.206": {"title": "Post Tenure Review",                                     "section": "Tenure and Promotion",  "category": "tenure_promotion"},
    "3.207": {"title": "Annual Evaluation of Faculty and Merit Pay",             "section": "Tenure and Promotion",  "category": "tenure_promotion"},
    "3.300": {"title": "Reduced Load",                                           "section": "Leave and Benefits",    "category": "leave_benefits"},
    "3.301": {"title": "Sabbatical",                                             "section": "Leave and Benefits",    "category": "leave_benefits"},
    "3.302": {"title": "Leaves of Absence",                                      "section": "Leave and Benefits",    "category": "leave_benefits"},
    "3.303": {"title": "Transitional Retirement",                                "section": "Leave and Benefits",    "category": "leave_benefits"},
    "3.304": {"title": "Educational Assistance (Advance Study Tuition Reimbursement)", "section": "Leave and Benefits", "category": "leave_benefits"},
    "3.305": {"title": "Sick Leave",                                             "section": "Leave and Benefits",    "category": "leave_benefits"},
    "3.400": {"title": "Salary and Compensation",                                "section": "Compensation",          "category": "compensation"},
    "3.500": {"title": "Academic Procedures Overview",                           "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.501": {"title": "Absences",                                               "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.502": {"title": "Student Records",                                        "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.503": {"title": "Class Hour/Location Changes",                            "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.504": {"title": "Records and Reports",                                    "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.505": {"title": "Field Trips",                                            "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.506": {"title": "Library Services",                                       "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.507": {"title": "Meetings",                                               "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.508": {"title": "Office Hours",                                           "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.509": {"title": "Problem Resolution",                                     "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.511": {"title": "Student Workers",                                        "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.512": {"title": "Textbook Orders",                                        "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.513": {"title": "Jury Duty",                                              "section": "Academic Procedures",   "category": "academic_procedures"},
    "3.600": {"title": "Resolution of Grievances and Disputes",                  "section": "Grievances",            "category": "grievances"},
    "3.700": {"title": "Faculty Organization and Governance",                    "section": "Governance",            "category": "governance"},
    "3.701": {"title": "Admissions and Re-Entry Committee",                      "section": "Governance",            "category": "governance"},
    "3.702": {"title": "Assessment Committee",                                   "section": "Governance",            "category": "governance"},
    "3.703": {"title": "Athletic Committee",                                     "section": "Governance",            "category": "governance"},
    "3.704": {"title": "Chapel Committee",                                       "section": "Governance",            "category": "governance"},
    "3.706": {"title": "Faculty Appeals Committee",                              "section": "Governance",            "category": "governance"},
    "3.707": {"title": "Faculty Development Committee",                          "section": "Governance",            "category": "governance"},
    "3.709": {"title": "Graduate Curriculum Committee",                          "section": "Governance",            "category": "governance"},
    "3.710": {"title": "Library Committee",                                      "section": "Governance",            "category": "governance"},
    "3.711": {"title": "President's Advisory Committee",                         "section": "Governance",            "category": "governance"},
    "3.712": {"title": "Promotion and Tenure Committee",                         "section": "Governance",            "category": "governance"},
    "3.713": {"title": "Provost's Council",                                      "section": "Governance",            "category": "governance"},
    "3.714": {"title": "Institutional Review Board",                             "section": "Governance",            "category": "governance"},
    "3.715": {"title": "Staff Advisory Council",                                 "section": "Governance",            "category": "governance"},
    "3.716": {"title": "Student Appeals Committee",                              "section": "Governance",            "category": "governance"},
    "3.717": {"title": "Student Services Committee",                             "section": "Governance",            "category": "governance"},
    "3.718": {"title": "Education Committee",                                    "section": "Governance",            "category": "governance"},
    "3.719": {"title": "Undergraduate Curriculum Committee",                     "section": "Governance",            "category": "governance"},
    "4.0":   {"title": "Glossary",                                               "section": "Appendix",              "category": "appendix"},
}

SECTION_WEIGHTS: dict[str, float] = {
    "foundation":          2.0,
    "academic_freedom":    2.5,
    "employment":          2.5,
    "conduct":             2.0,
    "tenure_promotion":    3.0,
    "leave_benefits":      2.5,
    "compensation":        2.5,
    "academic_procedures": 2.0,
    "grievances":          2.5,
    "governance":          1.5,
    "appendix":            1.0,
}

# ── Extraction constants ───────────────────────────────────────

MAX_CHUNK_WORDS = 500
MIN_CHUNK_WORDS = 30

# Regex to find standalone policy number line (e.g. "3.101" or "4.0")
_STANDALONE_POLICY_RE = re.compile(r'^((?:3|4)\.\d{1,3})\s*(?:\d+\s*)?$')

# Regex to find effective date
EFFECTIVE_DATE_RE = re.compile(
    r'(?:Effective Date|Updated)[:\s]+([0-9/\-]+)',
    re.IGNORECASE,
)

# Internal section headers that act as sub-chunk split points
SECTION_HEADER_RE = re.compile(
    r'^(?:POLICY|PROCEDURE|PURPOSE|CONDITIONS|DEFINITIONS?|TYPES?|ELIGIBILITY|'
    r'COMPENSATION|BENEFITS?|REQUIREMENTS?|RESPONSIBILITIES|AUTHORITY|OVERVIEW|'
    r'SCOPE|BACKGROUND|RATIONALE|GUIDELINES?|LIMITATIONS?|RIGHTS|SANCTIONS?):\s*$',
    re.IGNORECASE | re.MULTILINE,
)

# Noise patterns to strip from page content
NOISE_PATTERNS = [
    re.compile(r'Effective Date[:\s]+.*?\n', re.IGNORECASE),
    re.compile(r'Updated[:\s]+.*?\n', re.IGNORECASE),
    re.compile(r'^Section:\s*\n?\s*\d+\s*$', re.MULTILINE),
    re.compile(r'^Policy\s*\n?\s*Number[:\s#]*\n?\s*(?:3|4)\.\d+.*?\n', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^Page[:\s]+[\d\s]+$', re.MULTILINE),
    re.compile(r'Subject:\s*.*?\n', re.IGNORECASE),
    re.compile(r'Responsible Department:\s*.*?\n', re.IGNORECASE),
    re.compile(r'Academic Affairs\s*\n'),
    re.compile(r'Human Resources\s*\n'),
    re.compile(r'\n{3,}', ),
]

CROSS_REF_RE = re.compile(r'\bPolicy\s+(3\.\d+|4\.\d+)', re.IGNORECASE)
NUMBERED_LIST_RE = re.compile(r'^\s*\d+\.\s+\S', re.MULTILINE)


# ── Text extraction ────────────────────────────────────────────

def _detect_policy_num(page_text: str) -> Optional[str]:
    """
    Scan a page line-by-line for a standalone policy number.
    Only runs on pages that contain 'Effective Date:' or 'Updated:',
    which are the CBU policy header pages.
    The PDF uses 5+ header formats where 'Policy', 'Number', and the
    number itself may appear on separate lines with 'Page:' interspersed.
    In all formats the bare number (e.g. '3.101') appears on its own line.
    """
    if 'Effective Date:' not in page_text and 'Updated:' not in page_text:
        return None
    for line in page_text.split('\n'):
        m = _STANDALONE_POLICY_RE.match(line.strip())
        if m:
            return m.group(1)
    return None


def extract_text(pdf_path: str) -> str:
    result = subprocess.run(
        ['pdftotext', pdf_path, '-'],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f'pdftotext failed: {result.stderr}')
    words = len(result.stdout.split())
    pages = result.stdout.count('\x0c') + 1
    print(f'[chunker] Extracted {words:,} words across {pages} pages')
    return result.stdout


def strip_page_header(page_text: str) -> str:
    """Remove the boilerplate header table from a page, returning only content."""
    # Find 'Responsible Department:' — content starts after the dept name that follows it
    idx = page_text.find('Responsible Department:')
    if idx == -1:
        return page_text.strip()
    after = page_text[idx + len('Responsible Department:'):]
    # Skip 1-2 lines (dept name, maybe a blank) then take the rest
    lines = after.split('\n')
    content_lines = []
    skipped = 0
    for line in lines:
        stripped = line.strip()
        if skipped < 2:
            if stripped.lower() in {'academic affairs', 'human resources',
                                    'provost', 'president', 'student affairs',
                                    'finance', 'it services', 'athletics'}:
                skipped += 1
                continue
            if not stripped:
                skipped += 1
                continue
            # If non-empty and not a dept name, content has started
            skipped = 2
        content_lines.append(line)
    return '\n'.join(content_lines).strip()


# ── Chunking ───────────────────────────────────────────────────

def split_into_sub_chunks(text: str, max_words: int) -> list[str]:
    """
    If text exceeds max_words, split at SECTION_HEADER_RE boundaries.
    Always returns at least one element.
    """
    if len(text.split()) <= max_words:
        return [text]

    # Find all section header positions
    boundaries = [m.start() for m in SECTION_HEADER_RE.finditer(text)]
    if not boundaries:
        # No internal headers; hard-split at paragraph boundaries
        paras = re.split(r'\n{2,}', text)
        chunks, cur = [], []
        for p in paras:
            if len(' '.join(cur).split()) + len(p.split()) > max_words and cur:
                chunks.append('\n\n'.join(cur).strip())
                cur = [p]
            else:
                cur.append(p)
        if cur:
            chunks.append('\n\n'.join(cur).strip())
        return [c for c in chunks if c.strip()]

    parts = []
    prev = 0
    for b in boundaries:
        if b > prev:
            parts.append(text[prev:b])
        prev = b
    parts.append(text[prev:])

    # Merge very short parts with next
    result, buf = [], ''
    for p in parts:
        if buf:
            candidate = buf + '\n\n' + p
        else:
            candidate = p
        if len(candidate.split()) > max_words and buf:
            result.append(buf.strip())
            buf = p
        else:
            buf = candidate
    if buf.strip():
        result.append(buf.strip())
    return [r for r in result if r.strip()] or [text]


def chunk_handbook(raw_text: str) -> list[dict]:
    """
    Parse raw pdftotext output into policy-level chunks.
    Groups pages by policy number, strips headers, merges content,
    then splits oversized blocks at internal section headers.
    """
    pages = raw_text.split('\x0c')

    # ── Pass 1: assign each page to a policy number ───────────
    page_records: list[dict] = []
    cur_policy = None
    cur_date   = None

    for pg_idx, page in enumerate(pages):
        detected = _detect_policy_num(page)
        if detected:
            cur_policy = detected
            d = EFFECTIVE_DATE_RE.search(page)
            cur_date = d.group(1).strip() if d else None

        if cur_policy is None:
            continue  # skip cover page / TOC before first policy

        content = strip_page_header(page)
        if not content.strip():
            continue

        page_records.append({
            'policy_num':    cur_policy,
            'effective_date':cur_date,
            'page_num':      pg_idx + 1,
            'content':       content,
        })

    # ── Pass 2: merge pages with same policy number ────────────
    merged: dict[str, dict] = {}   # policy_num → {content, pages, date}
    for rec in page_records:
        pn = rec['policy_num']
        if pn not in merged:
            merged[pn] = {
                'policy_num':     pn,
                'effective_date': rec['effective_date'],
                'pages':          [rec['page_num']],
                'content':        rec['content'],
            }
        else:
            merged[pn]['pages'].append(rec['page_num'])
            merged[pn]['content'] += '\n\n' + rec['content']

    # ── Pass 3: split oversized policies, emit chunks ─────────
    chunks: list[dict] = []
    chunk_id = 0

    for pn in sorted(merged.keys(), key=lambda x: [int(p) if p.isdigit() else p
                                                    for p in re.split(r'[.]', x)]):
        rec = merged[pn]
        info = POLICY_MAP.get(pn, {
            'title':    f'Policy {pn}',
            'section':  'Unknown',
            'category': 'unknown',
        })
        sub_texts = split_into_sub_chunks(rec['content'], MAX_CHUNK_WORDS)

        for sub_idx, text in enumerate(sub_texts):
            wc = len(text.split())
            if wc < MIN_CHUNK_WORDS:
                continue

            chunks.append({
                'chunk_id':       chunk_id,
                'policy_num':     pn,
                'chunk_index':    sub_idx,
                'text':           text,
                'policy_title':   info['title'],
                'section':        info['section'],
                'category':       info['category'],
                'section_weight': SECTION_WEIGHTS.get(info['category'], 1.5),
                'page_range':     rec['pages'],
                'word_count':     wc,
                'effective_date': rec['effective_date'],
                'has_numbered_list': bool(NUMBERED_LIST_RE.search(text)),
                'cross_references': list(set(CROSS_REF_RE.findall(text))),
            })
            chunk_id += 1

    return chunks


# ── Reporting ──────────────────────────────────────────────────

def print_report(chunks: list[dict]) -> None:
    cat_counts = Counter(c['category'] for c in chunks)
    total_words = sum(c['word_count'] for c in chunks)
    print(f'\n[chunker] ── Chunk Report {"─"*40}')
    print(f'  Total chunks : {len(chunks)}')
    print(f'  Total words  : {total_words:,}')
    print(f'  Avg words    : {total_words // max(len(chunks), 1)}')
    print(f'\n  By category:')
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f'    {cat:<25}: {cnt}')
    print()


# ── Entry point ───────────────────────────────────────────────

def run(pdf_path: str, out_path: str) -> list[dict]:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    raw    = extract_text(pdf_path)
    chunks = chunk_handbook(raw)
    print_report(chunks)

    with open(out_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    print(f'[chunker] Saved {len(chunks)} chunks → {out_path}')
    return chunks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 1: Parse CBU Faculty Manual PDF into policy chunks'
    )
    parser.add_argument('--pdf', default='cbu_faculty.pdf',
                        help='Path to CBU Faculty Manual PDF')
    parser.add_argument('--out', default='data/chunks.jsonl',
                        help='Output JSONL path (default: data/chunks.jsonl)')
    args = parser.parse_args()
    run(args.pdf, args.out)
