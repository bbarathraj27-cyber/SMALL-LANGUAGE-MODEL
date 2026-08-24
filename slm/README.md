# my-slm

A from-scratch Small Language Model (SLM): a ~100M-parameter decoder-only
Transformer trained end-to-end — data pipeline, pretraining, SFT,
preference alignment, safety training, and quantized inference —
sized to run on a single consumer GPU.

## Architecture (first version)

| | |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | ~100M |
| Context length | 2048 |
| Vocabulary | 32K (BPE) |
| Precision | BF16 / FP16 |
| Normalization | RMSNorm |
| Position encoding | RoPE |
| FFN | SwiGLU |
| Optimizer | AdamW |
| Training objective | Causal LM |
| Post-training | SFT → Preference optimization |
| Deployment | Quantized inference |

## Workflow

The project follows 20 phases, grouped into three stages:

**Data pipeline** — collect → clean → filter → deduplicate → tokenize → shard
**Model development** — architecture → implementation → tiny-model test → pretraining → base-model eval
**Post-training & deployment** — instruction data → SFT → preference/alignment → safety → red-teaming → eval → quantize/tools/RAG → inference optimization → deployment → continuous eval

See the full phase list and diagrams in the project planning notes.

## Project structure

```
my-slm/
├── configs/            # model.yaml, data.yaml, training.yaml
├── data/                # raw -> cleaned -> deduplicated -> train/validation/test
├── tokenizer/         # BPE tokenizer training + trained artifacts
├── model/              # Transformer building blocks (embeddings, RoPE, RMSNorm,
│                       #   attention, SwiGLU, transformer block, LM head)
├── dataset/            # PyTorch Dataset/DataLoader for pretraining + instruction data
├── preprocessing/  # Data pipeline: collect, clean, filter, deduplicate,
│                       #   split, tokenize, shard  (PHASE 2-6)
├── training/           # Pretraining loop, loss, optimizer, scheduler, checkpointing
├── sft/                    # Supervised fine-tuning data prep, training, eval
├── evaluation/       # Perplexity, benchmarks, general eval harness
├── optimization/   # Quantization, export, inference benchmarking
├── inference/         # Generation, sampling, KV cache, chat interface
├── checkpoints/     # pretraining / sft / final model weights (gitignored)
├── logs/                  # training / evaluation logs (gitignored)
├── tests/                 # unit tests for every module above
└── scripts/             # thin CLI entry points wrapping the modules above
```

## Data pipeline (`preprocessing/`)

Each stage is both an importable module and a standalone CLI, and reads
shared defaults from `configs/data.yaml`:

```bash
# PHASE 2: collect
python -m preprocessing.collect --input-dir ./sources --output data/raw/corpus.jsonl

# PHASE 3: clean
python -m preprocessing.clean --input data/raw/corpus.jsonl --output data/cleaned/corpus.jsonl

# quality filter
python -m preprocessing.filter --input data/cleaned/corpus.jsonl --output data/cleaned/corpus.filtered.jsonl

# PHASE 4: deduplicate (exact + MinHash near-dedup)
python -m preprocessing.deduplicate --input data/cleaned/corpus.filtered.jsonl --output data/deduplicated/corpus.jsonl

# split into train/validation/test
python -m preprocessing.split --input data/deduplicated/corpus.jsonl \
    --train-out data/train/train.jsonl --val-out data/validation/val.jsonl --test-out data/test/test.jsonl

# PHASE 6: tokenize + shard
python -m preprocessing.tokenize --input data/train/train.jsonl --output data/train/train.tok.jsonl
python -m preprocessing.shard --input data/train/train.tok.jsonl --output-dir data/train/shards
```

Or drive the whole thing through the wrapper scripts in `scripts/` (each
supports `--dry-run` to sanity-check resolved args before touching data):

```bash
python scripts/prepare_data.py --config configs/data.yaml
python scripts/train_tokenizer.py --config configs/data.yaml
python scripts/tokenize_data.py --config configs/data.yaml
python scripts/train.py --config configs/training.yaml
python scripts/train_sft.py --config configs/training.yaml
python scripts/evaluate.py --checkpoint checkpoints/final
python scripts/run_inference.py --checkpoint checkpoints/final --prompt "Hello"
```

## System requirements

| Component | Minimum (dev) | Recommended (50-100M SLM) | Comfortable |
|---|---|---|---|
| CPU | 6 cores | 8-16 cores | 16+ cores |
| RAM | 16 GB | 32 GB | 64 GB+ |
| GPU | 8 GB VRAM | 16-24 GB VRAM | 24-48 GB VRAM |
| Storage | 100 GB free | 500 GB NVMe SSD | 1 TB+ NVMe |
| Python | 3.10-3.12 | 3.11/3.12 | Same |
| OS | Linux/WSL2 | Ubuntu/WSL2 | Linux |

## Status

| Module | Status |
|---|---|
| configs, model, preprocessing, dataset, tokenizer, training, sft, evaluation, inference, optimization, scripts | ✅ done |
| README.md, .gitignore | ✅ done |

## Tests

```bash
pytest tests/ -v
```

## License

TBD.
