# SAQR AI

SAQR is an open-source Egyptian language-model project built from scratch.

Roadmap: SAQR-100M -> 500M -> 1B -> 3B -> 7B -> 30B.

## Current stage

SAQR-100M foundation. A clean PyTorch decoder-only Transformer training stack with randomly initialized weights.

## Google Colab

    git clone https://github.com/majsjsj/saqr-ai.git
    cd saqr-ai
    pip install -r requirements.txt

Prepare data/train.txt, then:

    python tokenizer.py --input data/train.txt --output tokenizer
    python train.py --config configs/saqr_100m.json --data data/train.txt

Generate:

    python generate.py --checkpoint checkpoints/last.pt --prompt "مصر"

## Principles

- Train from scratch.
- Reproduce experiments.
- Evaluate every generation before scaling.
- Keep the architecture scalable.
- Never equate parameter count with intelligence.

Experimental research project. Apache-2.0.
