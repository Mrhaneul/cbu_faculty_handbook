# CBU Faculty Manual — Fine-Tuned LLM: Project Summary Report

**Date:** May 21, 2026  
**Project:** CAIMLL — CBU Faculty Manual QA Fine-Tuning  
**Model:** Gemma 4 E4B (4-bit, Apple Silicon / MLX)

---

## Overview

We fine-tuned a local language model on the California Baptist University Employee Manual (Faculty Section, Updated 01/2026) to serve as a policy-aware assistant for faculty affairs questions. The model runs entirely on-device — no cloud API required.

---

## Pipeline Summary

| Stage | Description | Output |
|---|---|---|
| 1 — Chunk | Split PDF into policy-level sections | 90 chunks, 11 categories |
| 2 — Enrich | Tag each chunk with metadata | 90 enriched chunks |
| 3 — Generate | Produce QA pairs via Ollama (Gemma 4) | 300 QA pairs |
| 4 — Format | Split into train / validation sets | 276 train, 30 valid |
| 5 — Train | LoRA fine-tuning via MLX (1,500 iterations) | Adapter weights |
| 6 — Evaluate | ROUGE-L scoring: base model vs fine-tuned | +77.4% improvement |
| 7 — Export | Merge adapter → GGUF for Ollama deployment | 5.0 GB Q4\_K\_M model |

---

## Dataset

- **Source document:** CBU Faculty Manual (Faculty Section, 01/2026)
- **Chunks extracted:** 90 policy sections across 11 categories
- **QA pairs generated:** 300 (target matched GitHub reference pipeline)
- **Training set:** 276 examples (92%)
- **Validation set:** 30 examples (8%)
- **Question types covered:** 8 (factual, procedural, eligibility, scenario, rights & responsibilities, comparative, timeline/deadline, committee role)

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `mlx-community/gemma-4-E4B-it-4bit` |
| Method | LoRA (rank 8, α 16) |
| Iterations | 1,500 |
| Learning rate | 1e-4 |
| Batch size | 4 |
| Best checkpoint | Iter 800 (val loss 0.954) |
| Final checkpoint | Iter 1,500 (val loss 1.512) |
| Hardware | Apple Silicon (MLX) |

> **Note:** Val loss was lowest at iteration 800. The final checkpoint shows signs of overfitting. Production deployment should consider using the iter 800 adapter (`0000800_adapters.safetensors`).

---

## Evaluation Results (ROUGE-L)

**Overall — 30 validation examples**

| Metric | Score |
|---|---|
| Base model average | 0.1153 |
| Fine-tuned average | 0.2044 |
| Improvement | **+0.0891 (+77.4%)** |
| Examples improved | **28 / 30 (93%)** |

**By question type**

| Question Type | n | Base | Fine-tuned | Delta |
|---|---|---|---|---|
| Timeline / Deadline | 5 | 0.1225 | 0.2603 | **+0.1378** |
| Committee Role | 2 | 0.1067 | 0.2204 | **+0.1137** |
| Rights & Responsibilities | 5 | 0.1216 | 0.2303 | **+0.1087** |
| Procedural | 5 | 0.1098 | 0.2022 | **+0.0924** |
| Eligibility | 3 | 0.1116 | 0.1962 | **+0.0846** |
| Scenario | 7 | 0.1183 | 0.1739 | **+0.0556** |
| Factual | 2 | 0.0777 | 0.1172 | **+0.0396** |
| Comparative | 1 | 0.1570 | 0.1882 | **+0.0312** |

Full chart: `eval/rouge_chart.png`

---

## Deployment

The model is exported and ready for local Ollama deployment:

```bash
cd burns/cbu-gemma4-e4b-mlx-v1
ollama create cbu-faculty-gemma4 -f Modelfile
ollama run cbu-faculty-gemma4
```

| Artifact | Details |
|---|---|
| GGUF file | `burns/cbu-gemma4-e4b-mlx-v1/model.Q4_K_M.gguf` |
| File size | 5.0 GB |
| Quantization | Q4\_K\_M |
| Ollama model name | `cbu-faculty-gemma4` |
| Inference temp | 0.1 (deterministic) |
| Context window | 4,096 tokens |

**System prompt behavior:** The model is instructed to place all policy citations at the end of answers in `[Reference: CBU Faculty Manual, Policy X.XXX]` format and to clearly state when a question falls outside the manual's scope.

---

## Known Limitations

- **Hallucination risk:** ROUGE-L improvement does not eliminate hallucination. One example (Faculty Appeals Committee question) produced an incorrect "Board of Trustees" reference. Additional evaluation with human review is recommended before faculty-facing deployment.
- **Overfitting signal:** Best val loss was at iter 800, not iter 1,500. Using the iter-800 checkpoint may yield better out-of-distribution generalization.
- **ROUGE-L scope:** ROUGE-L measures n-gram overlap with reference answers; it does not measure factual accuracy or citation correctness directly.
- **Dataset size:** 300 QA pairs covers the manual broadly but not exhaustively. Edge-case policies may not be well-represented.

---

## Next Steps (Optional)

1. **Use best checkpoint** — swap `adapters.safetensors` for `0000800_adapters.safetensors` and re-burn GGUF
2. **Human evaluation** — spot-check model answers against the manual for a sample of 20–30 questions
3. **Expand dataset** — generate additional QA pairs targeting under-represented categories (Comparative: n=1)
4. **Deploy & test** — run `ollama create` and pilot with a small group of faculty users
