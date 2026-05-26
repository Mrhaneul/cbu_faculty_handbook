"""
burn_gguf.py
============
Merge the CBU Faculty Manual LoRA adapter into the base model and
export GGUF for Ollama deployment.

Mac/Apple Silicon version — uses mlx_lm fuse instead of Unsloth.

CRITICAL — Gemma 4 specific:
  - NEVER fp16 — Gemma 4 activations overflow fp16. Always bf16 or f32.
  - NEVER hardcode base model path — read from adapter_config.json.

Flow:
  1. mlx_lm fuse     — dequantize + merge adapter → HF model weights
  2. convert_hf_to_gguf.py  — HF weights → BF16 GGUF
  3. llama-quantize  — BF16 GGUF → Q4_K_M GGUF (or other quant)
  4. Write Modelfile for Ollama

Note: mlx_lm fuse --export-gguf does not support Gemma 4 — must go through
HF format first, then convert_hf_to_gguf.py (which does support Gemma4).

Quant options:
    q4_k_m  — default, best size/quality (~3–5 GB)
    q8_0    — higher quality, larger
    q5_k_m  — middle ground
    f16     — keep BF16 GGUF without quantization

Usage:
    python burn_gguf.py
    python burn_gguf.py --adapter outputs/cbu-gemma4-e4b-mlx-v1
    python burn_gguf.py --adapter outputs/cbu-gemma4-e4b-mlx-v1 --quant q8_0
    python burn_gguf.py --adapter outputs/cbu-gemma4-e4b-mlx-v1 --out burns/cbu-v1

After the burn:
    cd burns/cbu-gemma4-e4b-mlx-v1
    ollama create cbu-faculty-gemma4 -f Modelfile
    ollama run cbu-faculty-gemma4
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────

DEFAULT_ADAPTER = Path("outputs") / "cbu-gemma4-e4b-mlx-v1"

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable CBU Faculty Affairs assistant with expertise in the "
    "California Baptist University Employee Manual — Faculty Section (Updated 01/2026). "
    "You help faculty understand university policies, procedures, rights, and responsibilities. "
    "Place all policy citations at the END of your answer in "
    "[Reference: CBU Faculty Manual, Policy X.XXX] format — NEVER at the beginning. "
    "If a question cannot be answered from the CBU Faculty Manual, say so clearly."
)

QUANT_MAP = {
    "f16":    None,
    "q8_0":   "Q8_0",
    "q4_k_m": "Q4_K_M",
    "q5_k_m": "Q5_K_M",
}

# Homebrew llama.cpp tools (installed via `brew install llama.cpp`)
LLAMA_QUANTIZE        = Path("/opt/homebrew/bin/llama-quantize")
CONVERT_HF_TO_GGUF    = Path("/opt/homebrew/bin/convert_hf_to_gguf.py")


# ── Helpers ───────────────────────────────────────────────────

def read_adapter_config(adapter_path: Path) -> dict:
    cfg = adapter_path / "adapter_config.json"
    if not cfg.exists():
        print(f"ERROR: {cfg} not found. Is this a valid adapter directory?")
        sys.exit(1)
    with open(cfg, encoding="utf-8") as f:
        return json.load(f)


# ── Step 1a: fuse adapter → merged HF weights via mlx_lm ─────

def fuse_to_hf(
    adapter_path: Path,
    base_model:   str,
    merged_dir:   Path,
) -> Path:
    print(f"\n[1/3] Fusing adapter into base model (dequantize → merged HF weights)...")
    print(f"  Base model  : {base_model}")
    print(f"  Adapter     : {adapter_path}")
    print(f"  Merge output: {merged_dir}")

    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model",        base_model,
        "--adapter-path", str(adapter_path),
        "--save-path",    str(merged_dir),
        "--dequantize",   # dequantize 4-bit base → bf16 before saving
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: mlx_lm fuse failed (exit {result.returncode})")
        sys.exit(result.returncode)

    if not (merged_dir / "config.json").exists():
        print(f"ERROR: merged HF model not found at {merged_dir}")
        sys.exit(1)

    print(f"  Merged HF model saved → {merged_dir}")
    return merged_dir


# ── Step 1b: convert merged HF weights → BF16 GGUF ───────────

def convert_to_bf16_gguf(
    merged_dir: Path,
    gguf_path:  Path,
) -> Path:
    print(f"\n[2/3] Converting merged HF weights → BF16 GGUF...")
    print(f"  Input : {merged_dir}")
    print(f"  Output: {gguf_path}")

    if not CONVERT_HF_TO_GGUF.exists():
        print(f"ERROR: {CONVERT_HF_TO_GGUF} not found. Install with: brew install llama.cpp")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(CONVERT_HF_TO_GGUF),
         str(merged_dir),
         "--outfile", str(gguf_path),
         "--outtype", "bf16"],
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: convert_hf_to_gguf.py failed (exit {result.returncode})")
        sys.exit(result.returncode)

    # Clean up merged HF dir — no longer needed
    shutil.rmtree(merged_dir, ignore_errors=True)

    size_gb = gguf_path.stat().st_size / 1e9
    print(f"  BF16 GGUF saved ({size_gb:.2f} GB)")
    return gguf_path


# ── Step 3: quantize GGUF ─────────────────────────────────────

def quantize_gguf(bf16_gguf: Path, output_dir: Path, quant: str) -> Path:
    quant_type = QUANT_MAP[quant]

    if quant_type is None:
        print(f"\n[3/3] Quant=f16 — keeping BF16 GGUF as final output.")
        return bf16_gguf

    if not LLAMA_QUANTIZE.exists():
        print(f"ERROR: llama-quantize not found at {LLAMA_QUANTIZE}")
        print("Install with: brew install llama.cpp")
        sys.exit(1)

    final_gguf = output_dir / f"model.{quant.upper()}.gguf"
    print(f"\n[3/3] Quantizing BF16 GGUF → {quant_type}...")
    result = subprocess.run(
        [str(LLAMA_QUANTIZE), str(bf16_gguf), str(final_gguf), quant_type],
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: llama-quantize failed (exit {result.returncode})")
        sys.exit(result.returncode)

    # Remove intermediate BF16 GGUF to save disk space
    bf16_gguf.unlink()

    size_gb = final_gguf.stat().st_size / 1e9
    print(f"  Quantized GGUF saved: {final_gguf} ({size_gb:.2f} GB)")
    return final_gguf


# ── Step 3: Modelfile ─────────────────────────────────────────

def write_modelfile(
    output_dir:    Path,
    gguf_file:     Path,
    model_name:    str,
    system_prompt: str,
) -> None:
    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(
        f'FROM ./{gguf_file.name}\n\n'
        f'SYSTEM """{system_prompt}"""\n\n'
        f"PARAMETER temperature 0.1\n"
        f"PARAMETER top_p 0.9\n"
        f"PARAMETER top_k 64\n"
        f"PARAMETER num_ctx 4096\n"
        f"PARAMETER num_predict 2048\n"
        f"PARAMETER repeat_penalty 1.1\n",
        encoding="utf-8",
    )
    print(f"\n  Modelfile written → {modelfile_path}")
    print("\n" + "=" * 56)
    print("  To deploy in Ollama:")
    print(f"    cd {output_dir}")
    print(f"    ollama create {model_name} -f Modelfile")
    print(f"    ollama run {model_name}")
    print("=" * 56)


# ── Main ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge CBU Gemma 4 E4B LoRA adapter → GGUF for Ollama (Mac/MLX)"
    )
    parser.add_argument(
        "--adapter", default=str(DEFAULT_ADAPTER),
        help=f"LoRA adapter directory (default: {DEFAULT_ADAPTER})",
    )
    parser.add_argument(
        "--quant", default="q4_k_m", choices=sorted(QUANT_MAP),
        help="Quantization format (default: q4_k_m)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: burns/<adapter_name>)",
    )
    parser.add_argument(
        "--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
    )
    parser.add_argument(
        "--model-name", default="cbu-faculty-gemma4",
        help="Ollama model name (default: cbu-faculty-gemma4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    adapter_path = Path(args.adapter).expanduser().resolve()
    if not adapter_path.exists():
        print(f"ERROR: {adapter_path} not found")
        sys.exit(1)

    output_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else Path.cwd() / "burns" / adapter_path.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config     = read_adapter_config(adapter_path)
    base_model = config.get("base_model_name_or_path", "mlx-community/gemma-4-E4B-it-4bit")

    print("=" * 56)
    print("  CBU Faculty Manual — Gemma 4 E4B GGUF Burn")
    print("=" * 56)
    print(f"  Adapter    : {adapter_path}")
    print(f"  Base model : {base_model}")
    print(f"  Quant      : {args.quant}")
    print(f"  Output     : {output_dir}")
    print("=" * 56)

    # Step 1 — fuse adapter → merged HF weights
    merged_dir = output_dir / "merged_hf"
    fuse_to_hf(adapter_path, base_model, merged_dir)

    # Step 2 — convert HF weights → BF16 GGUF
    bf16_gguf = output_dir / "model.BF16.gguf"
    convert_to_bf16_gguf(merged_dir, bf16_gguf)

    # Step 3 — quantize
    final_gguf = quantize_gguf(bf16_gguf, output_dir, args.quant)

    # Step 3 — Modelfile
    write_modelfile(output_dir, final_gguf, args.model_name, args.system_prompt)


if __name__ == "__main__":
    main()
