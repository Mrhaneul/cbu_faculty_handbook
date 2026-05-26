# CBU Faculty Manual — Fine-Tuning Runbook

End-to-end guide for running the CBU Faculty Manual fine-tuning pipeline.
Adapts the ATP 2-01.3 pipeline architecture for a university HR/policy document.

---

## Environment

Use the existing `caimll_finetuning` conda environment — it has everything needed.

```bash
# Always use the explicit Python path — conda run picks up homebrew Python instead
PYTHON=/Users/csdsstudent/miniconda3/envs/caimll_finetuning/bin/python
```

Or activate and use normally:
```bash
source /Users/csdsstudent/miniconda3/etc/profile.d/conda.sh
conda activate caimll_finetuning
```

---

## Pipeline Overview

| Stage | Script | Input | Output |
|---|---|---|---|
| 1 | `chunker.py` | `cbu_faculty.pdf` | `data/chunks.jsonl` |
| 2 | `enricher.py` | `data/chunks.jsonl` | `data/enriched.jsonl` |
| 3 | `generator.py` | `data/enriched.jsonl` | `data/seeds.jsonl` |
| 4 | `formatter.py` | `data/seeds.jsonl` | `data/train.jsonl` + `data/valid.jsonl` |
| 5 | `trainer.py` | `data/` | `outputs/cbu-gemma4-e4b-mlx-v1/` |
| 6 | `evaluator.py` | `data/valid.jsonl` + adapter | `eval/results.jsonl` |

---

## Stage 1 — Chunk the PDF

```bash
$PYTHON chunker.py --pdf cbu_faculty.pdf --out data/chunks.jsonl
```

Expected output: **~80–100 chunks** from 105 pages, grouped by policy number.
Policies > 500 words are split at internal section headers (POLICY:, PROCEDURE:, etc.).

---

## Stage 2 — Enrich Chunks

```bash
$PYTHON enricher.py --chunks data/chunks.jsonl --out data/enriched.jsonl
```

Calls Ollama `gemma4:latest` at **temperature=0.1** to classify:
- `content_type`: policy | procedure | eligibility | rights | definition | committee_info
- `audience`: all_faculty | full_time | part_time | adjunct | tenured | department_chair
- `has_deadlines`: whether the chunk states specific time limits
- `has_rights`: whether the chunk grants explicit faculty entitlements

> Safe to interrupt and resume — uses append mode.

To use gemma4:31b for better classification quality:
```bash
$PYTHON enricher.py --model gemma4:31b
```

---

## Stage 3 — Generate QA Seeds

```bash
$PYTHON generator.py --target 600 --out data/seeds.jsonl
```

Generates 8 question types per chunk:
- `factual`, `procedural`, `eligibility`, `scenario`
- `rights_responsibilities`, `comparative`, `timeline_deadline`, `committee_role`

Target: **600 QA pairs** (adjustable via `--target`).
Higher-stakes sections (tenure_promotion, grievances, leave) are oversampled.

> Safe to interrupt and resume — uses append mode.

For faster generation with a second model (if available):
```bash
$PYTHON generator.py --model gemma4:31b --target 600
```

---

## Stage 4 — Format for Training

```bash
$PYTHON formatter.py --seeds data/seeds.jsonl --train data/train.jsonl --val data/valid.jsonl
```

Outputs **Gemma 4 chat template** with thinking traces:
```
<bos><|system|>
You are a knowledgeable CBU Faculty Affairs assistant...
<|end|>
<|user|>
{question}
<|end|>
<|assistant|>
<|channel>thought
{thinking_trace}
<channel|>
{answer} [Reference: CBU Faculty Manual, Policy X.XXX]
<|end|>
```

Expected split: **~540 train / ~60 valid** (from 600 seeds, 10% validation).

> MLX requires `valid.jsonl` (not `val.jsonl`) — formatter.py outputs the correct name.

---

## Stage 5 — Train

```bash
$PYTHON trainer.py --iters 1500 --out outputs/cbu-gemma4-e4b-mlx-v1
```

Training config:
| Parameter | Value |
|---|---|
| Base model | `mlx-community/gemma-4-E4B-it-4bit` |
| Framework | MLX LoRA |
| Iterations | 1500 |
| LoRA rank | 8 |
| LoRA scale | 20.0 |
| LoRA layers | 16 |
| Learning rate | 1e-5 |
| Batch size | 1 (grad_accum=4, effective=4) |
| Max seq length | 1024 |
| Save every | 100 steps |

### Resume training

```bash
$PYTHON trainer.py \
  --iters 2000 \
  --resume outputs/cbu-gemma4-e4b-mlx-v1/0001500_adapters.safetensors
```

### Dry run (check command without running)

```bash
$PYTHON trainer.py --dry-run
```

---

## Run Everything at Once

```bash
$PYTHON run_pipeline.py --target 600 --iters 1500
```

Skip stages with existing output (resume mode):
```bash
$PYTHON run_pipeline.py --resume
```

Run specific stages only:
```bash
$PYTHON run_pipeline.py --stages 3,4,5   # re-generate, re-format, re-train
```

---

## Stage 6 — Evaluate (ROUGE-L)

Mirrors Step 10 from joupark/facultyHandbook — compares base model vs fine-tuned adapter on the held-out validation set.

```bash
$PYTHON evaluator.py --adapter outputs/cbu-gemma4-e4b-mlx-v1
```

With saved results:
```bash
$PYTHON evaluator.py \
  --adapter outputs/cbu-gemma4-e4b-mlx-v1 \
  --out eval/results.jsonl
```

Outputs:
- Base model ROUGE-L average
- Fine-tuned ROUGE-L average
- Per-example delta table
- Sample answer comparison (first example)

> With only 8 QA pairs, valid.jsonl has 1 example — enough to verify the pipeline works but not statistically meaningful. Re-run with `--target 600` for a proper evaluation.

---

## Quick Test Run (8 Questions)

End-to-end pipeline test with minimal data — validates every stage before committing to a full run.

```bash
# Stages 1-2 only need to run once (output already exists after first run)
$PYTHON run_pipeline.py --stages 3,4,5,6 \
  --target 8 \
  --iters 100
```

Or step by step:
```bash
$PYTHON generator.py --target 8
$PYTHON formatter.py
$PYTHON trainer.py --iters 100
$PYTHON evaluator.py --adapter outputs/cbu-gemma4-e4b-mlx-v1 --out eval/results.jsonl
```

> **Why `--iters 100` for 8 examples?** With 7 training examples and batch_size=1, 1500 iterations = ~214 passes over the same data (massive overfit). Use 100 iterations for the test run; scale to 1500 for the full 600-pair run.

---

## Export to GGUF (for Ollama)

After training, run from the ATP pipeline's burn_gguf.py adapted for CBU:

```bash
cd /Users/csdsstudent/atp_finetuning
$PYTHON burn_gguf.py \
  --adapter /Users/csdsstudent/cbu_faculty_handbook/outputs/cbu-gemma4-e4b-mlx-v1 \
  --out /Users/csdsstudent/cbu_faculty_handbook/burns/cbu-gemma4-e4b-mlx-v1 \
  --system-prompt "You are a knowledgeable CBU Faculty Affairs assistant..."
```

Or copy burn_gguf.py to this folder:
```bash
cp /Users/csdsstudent/atp_finetuning/burn_gguf.py /Users/csdsstudent/cbu_faculty_handbook/
$PYTHON burn_gguf.py --adapter outputs/cbu-gemma4-e4b-mlx-v1
```

Then register with Ollama:
```bash
cd burns/cbu-gemma4-e4b-mlx-v1
ollama create cbu-faculty-gemma4 -f Modelfile
ollama run cbu-faculty-gemma4
```

---

## Key Rules (do not change without reason)

1. **Never fp16 with Gemma 4** — MLX uses bf16 automatically on Apple Silicon
2. **Never overwrite seeds.jsonl** — always append mode (generator.py enforces this)
3. **Citation format** — `[Reference: CBU Faculty Manual, Policy X.XXX]` at END of answer
4. **Never start answers with "According to the CBU"** — citations go at the END
5. **valid.jsonl not val.jsonl** — MLX convention; formatter.py enforces this
6. **Stop Ollama-heavy models before training** if low on unified memory

---

## Differences from ATP 2-01.3 Pipeline

| Aspect | ATP 2-01.3 | CBU Faculty Manual |
|---|---|---|
| Chunk ID | Paragraph number (3-24) | Policy number (3.101) |
| Document size | ~300+ pages | 105 pages |
| Training target | ~4,000 QA pairs | ~600 QA pairs |
| Generator model | gemma4:31b (vLLM on Spark) | gemma4:latest (Ollama on Mac) |
| Distribution | 5 machines | Single machine |
| Question types | IPB-specific (9 types) | Policy-specific (8 types) |
| Citation format | [Reference: ATP 2-01.3, para X-Y] | [Reference: CBU Faculty Manual, Policy X.XXX] |
| Training iterations | 2500 | 1500 |
| Max seq length | 512 | 1024 |
