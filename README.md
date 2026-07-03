# LAYER

LAYER is a focused InstructPix2Pix identity-layer discovery repo.

Current scope: **Milestone 1 — Identity Layer Scan only**.

This milestone searches for frozen InstructPix2Pix internal layers whose pooled activations separate same-identity image pairs from different-identity image pairs. It does not implement or run a geometry attack yet.

## What is included

- Layer inventory for `timbrooks/instruct-pix2pix`
- Forward-hook activation shape capture
- Pooled activation extraction
- Same-identity / different-identity pair scoring
- Layer/timestep ranking
- HTML and Markdown scan reports

## What is excluded

- No pixel noise
- No adversarial patches
- No finetuning
- No LoRA
- No SPSA/CEM/black-box attack
- No landmarks
- No face alignment or face detection
- No geometry optimization in Milestone 1

All model weights are frozen. The scan only reads internal activations.

## Default data source

The default scripts use MAT data:

```text
/home/interns/Desktop/mat/data/face_002/
/home/interns/Desktop/mat/data/face_005/
```

The default auto-manifest uses available image variants such as `instruct_512.png`, `master_1024.png`, and `flux_768.png` to create development same-identity pairs. This is useful for validating the scan pipeline, but it is not a final identity benchmark. A richer identity dataset can be added later.

## A6000 command sequence

```bash
cd /home/interns/Desktop/layer
git pull origin main

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.inventory_instruct_layers \
  --root /home/interns/Desktop/layer \
  --mat-root /home/interns/Desktop/mat

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.extract_identity_activations \
  --root /home/interns/Desktop/layer \
  --mat-root /home/interns/Desktop/mat

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.score_identity_layers \
  --root /home/interns/Desktop/layer

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.summarize_identity_layer_scan \
  --root /home/interns/Desktop/layer
```

## Outputs

```text
identity_layers/outputs/layer_inventory/
  layer_inventory.json
  layer_inventory.csv
  activation_shapes.csv
  model_tree.txt
  recommended_initial_layers.json

identity_layers/outputs/activation_scan/
  embeddings/
  embeddings_index.csv
  activation_statistics.csv
  extraction_manifest.json
  failures.jsonl

identity_layers/outputs/identity_scores/
  pair_scores.csv
  layer_scores.csv
  top_identity_layers.csv
  identity_score_summary.json

identity_layers/outputs/reports/
  identity_layer_scan_report.html
  identity_layer_scan_report.md
  graphs/
```

## Notes

Milestone 1 is meant to answer: which InstructPix2Pix internal activations look identity-separable enough to be useful targets for a later gradient/geometry phase?

The later attack phase should only be built after these scan outputs are inspected.
