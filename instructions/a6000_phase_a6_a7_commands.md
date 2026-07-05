# LAYER Phase A6/A7 commands

Run these only after Milestone 1 is complete:

```text
identity_layers/outputs/layer_inventory/
identity_layers/outputs/activation_scan/
identity_layers/outputs/identity_scores/
identity_layers/outputs/reports/
```

## A6 — baseline comparison

```bash
cd /home/interns/Desktop/layer
git pull origin main

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.run_baseline_comparison \
  --root /home/interns/Desktop/layer
```

Outputs:

```text
identity_layers/outputs/baseline_comparison/baseline_comparison.csv
identity_layers/outputs/baseline_comparison/baseline_comparison_summary.json
identity_layers/outputs/baseline_comparison/baseline_comparison.png
identity_layers/outputs/baseline_comparison/instruct_vs_arcface_correlation.csv
```

Note: ArcFace is not required for this run. If ArcFace embeddings are unavailable, the correlation CSV records that explicitly.

## A7 — short gradient sanity scan

```bash
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.test_layer_gradients \
  --root /home/interns/Desktop/layer
```

Default A7 settings:

```text
max_layers = 5
max_cases = 3
prompt = add black sunglasses
timestep_index = 6
steps = 5
learning_rate = 0.03
```

Outputs:

```text
identity_layers/outputs/gradient_scan/gradient_sanity.csv
identity_layers/outputs/gradient_scan/layer_gradient_rankings.csv
identity_layers/outputs/gradient_scan/selected_layers.json
identity_layers/outputs/gradient_scan/rejected_layers.json
identity_layers/outputs/gradient_scan/gradient_norm_by_layer.png
identity_layers/outputs/gradient_scan/short_step_Z_curves.png
identity_layers/outputs/gradient_scan/gradient_scan_summary.json
```

Push after running:

```bash
git status -sb
git add identity_layers/outputs/baseline_comparison identity_layers/outputs/gradient_scan
git commit -m "Add Stage A baseline and gradient sanity outputs"
git push origin main
```

Do not run a full geometry attack yet. A7 is only a short sanity gate.
