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

The default scripts now use the committed Wikimedia identity-probe dataset:

```text
identity_layers/datasets/wikimedia_identity_probe/
```

Current dataset size:

- 15 identities
- 3 images per identity
- 45 total images
- 45 same-identity pairs
- 945 different-identity pairs

The images were selected from Wikimedia Commons and resized to 512×512 using generic letterbox padding. No face detection, face alignment, or landmarks are used. Per-file license and source metadata is stored in:

```text
identity_layers/datasets/wikimedia_identity_probe/source_attribution.csv
```

The old MAT auto-manifest fallback still exists for development, but the default configs point to the Wikimedia dataset.

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
  pooled_embeddings/
  pooled_embeddings_index.csv
  activation_statistics.csv
  resolved_timesteps.json
  fixed_noise_metadata.json
  prompt_embeddings_metadata.json
  scan_conditions.csv
  extraction_manifest.json
  failures.jsonl

identity_layers/outputs/identity_scores/
  pair_scores.csv
  verification_metrics.csv
  layer_identity_scores.csv
  layer_timestep_scores.csv
  layer_prompt_scores.csv
  ranked_layers.json
  identity_score_summary.json
  same_vs_different_distributions.png
  identity_separation_by_layer.png
  identity_auc_by_layer.png
  layer_timestep_heatmap.png

identity_layers/outputs/reports/
  identity_layer_scan_report.html
  identity_layer_scan_report.md
  graphs/

identity_layers/outputs/baseline_comparison/
  baseline_comparison.csv
  baseline_comparison_summary.json
  baseline_comparison.png
  instruct_vs_arcface_correlation.csv

identity_layers/outputs/gradient_scan/
  gradient_sanity.csv
  layer_gradient_rankings.csv
  selected_layers.json
  rejected_layers.json
  gradient_norm_by_layer.png
  short_step_Z_curves.png
  gradient_scan_summary.json
```

## Notes

Milestone 1 is meant to answer: which InstructPix2Pix internal activations look identity-separable enough to be useful targets for a later gradient/geometry phase?

The later attack phase should only be built after these scan outputs are inspected.

## Stage-A continuation

After Milestone 1, run Phase A6/A7 with:

```text
instructions/a6000_phase_a6_a7_commands.md
```

A6 compares the ranked Instruct layers against simple baselines. A7 performs only a short gradient sanity scan to check whether identity-sensitive layers produce finite gradients to geometry parameters. It is not a full attack.
