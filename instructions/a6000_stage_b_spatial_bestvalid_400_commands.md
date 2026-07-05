# LAYER Stage B spatial best-valid 400 commands

This is the next run after the preservation-checkpoint run failed.

## Why this run exists

The preservation-aware run:

```text
loss = -Z + 1.0 * MSE(perturbed, original)
```

was too conservative. It preserved the input almost perfectly, but every run had negative `Z_increase`.

This run removes the preservation counter-loss and keeps the part that we still want:

```text
loss = -Z
```

but final artifacts are selected from the best valid checkpoint, not blindly from the last iteration.

## Config

```text
identity_layers/configs/stage_b_spatial_bestvalid_400.json
```

Matrix:

```text
layer:
  unet.conv_in

prompts:
  add headphones
  add black sunglasses

images:
  barack_obama_01
  joe_biden_01
  michelle_obama_01
  kamala_harris_01

iterations:
  400
```

Geometry:

```text
spatial only
FFT phase disabled
max combined displacement <= 5 px
```

Checkpoint selection:

```text
enabled
use best valid checkpoint for final images
iteration >= 1
input SSIM >= 0.97
input PSNR >= 26.0
combined max displacement <= 5.0 px
score = Z increase from initial
```

## Run on A6000

```bash
cd /home/interns/Desktop/layer
git pull origin main

mkdir -p logs

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.run_stage_b_smoke \
  --root /home/interns/Desktop/layer \
  --config /home/interns/Desktop/layer/identity_layers/configs/stage_b_spatial_bestvalid_400.json \
  --output-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_spatial_bestvalid_400 \
  2>&1 | tee logs/layer_stage_b_spatial_bestvalid_400.log

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.summarize_stage_b \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_spatial_bestvalid_400 \
  2>&1 | tee -a logs/layer_stage_b_spatial_bestvalid_400.log

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.build_report \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_spatial_bestvalid_400 \
  --output-root /home/interns/Desktop/layer/identity_layers/outputs/reports/stage_b_spatial_bestvalid_400 \
  2>&1 | tee -a logs/layer_stage_b_spatial_bestvalid_400.log
```

## Expected ETA

Expected runtime:

```text
8 runs x 400 iterations
roughly 10-20 minutes on the A6000
```

## Push after running

```bash
git status -sb
git add \
  identity_layers/outputs/stage_b_spatial_bestvalid_400 \
  identity_layers/outputs/reports/stage_b_spatial_bestvalid_400 \
  logs/layer_stage_b_spatial_bestvalid_400.log \
  logs.txt
git commit -m "Add Stage B spatial best-valid outputs" || true
git push origin main
```

If `logs.txt` is missing or unchanged, remove it from `git add`.

## Files to inspect first

```text
identity_layers/outputs/stage_b_spatial_bestvalid_400/stage_b_top_sheet.jpg
identity_layers/outputs/stage_b_spatial_bestvalid_400/stage_b_decision_report.md
identity_layers/outputs/stage_b_spatial_bestvalid_400/stage_b_all_runs.csv
identity_layers/outputs/reports/stage_b_spatial_bestvalid_400/layer_stage_b_report.pdf
identity_layers/outputs/reports/stage_b_spatial_bestvalid_400/layer_stage_b_report.html
```

Inside each run folder:

```text
summary.json
best_metadata.json
comparison_sheet.png
history.csv
```

Important fields:

```text
final_source
selected_checkpoint_iter
selected_checkpoint_score
selected_checkpoint_input_ssim
selected_checkpoint_max_disp_px
Z_increase
output_ssim
output_l2
```

## Decision rule

Interesting only if:

```text
final_source = best_valid_checkpoint
Z_increase > 0
input SSIM >= 0.97
perturbed input remains visually acceptable
clean edit succeeds
perturbed edit visibly weakens/fails
```

If the edit still clearly succeeds, call it weak or metric-only.

If this run is still weak, do not do another broad Stage B sweep. The better next step would be a tiny refinement around the best case or a return to ArcFace-correlated deeper-layer gradient probing.
