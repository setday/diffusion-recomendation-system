# RecSys — Diffusion Recommendation System

A diffusion-based recommendation system and training pipelines for research and experiments.

Features
- Diffusion model for recommendation ranking and generation
- Tokenizer training and quantization utilities
- Config-driven training pipelines and dataloaders

Quickstart

Prerequisites

- Python 3.8+ (recommended 3.10+)
- CUDA/cuDNN for GPU training (optional)
- Install Python dependencies:

```bash
pip install -r requirements.txt
```

Prepare data

Configure datasets and dataloaders in the `config/dataloaders` and `config/datasets` YAML files.

Train tokenizer (optional)

```bash
python train_tokenizer.py --config config/tokenizer/opq.yaml
```

Train diffusion model

```bash
python train_diffusion.py --config config/train.yaml
```

Repository structure (important files)

- `train_diffusion.py` — main training entry for diffusion models
- `train_tokenizer.py` — tokenizer training/quantization script
- `config/` — all YAML config presets for datasets, models, and pipelines
- `src/models/diffgrm.py` — diffusion model implementation
- `src/tokenizers/` — tokenizer implementations
- `src/pipelines/` — training & inference pipelines

Configuration

All runtime settings live under the `config/` folder. Copy and adapt the YAML presets for your experiments.

Usage notes

- Inspect and edit the YAML files to set dataset paths, hyperparameters, and logging.
- Logs and checkpoints are created by the training pipelines according to config settings.

Contributing

PRs, issues and experiments are welcome. For substantial changes open an issue first to discuss design.

License

See the `LICENSE` file at the project root.
