# LAYER Stage B preservation-aware checkpoint run

This is the next run after:

- `stage_b_broad_400`
- `stage_b_constrained_spatial_200`
- `stage_b_constrained_low_fft_200`

The broad run proved that unconstrained `unet.conv_in` can move the editor, but mostly by destroying the input. The constrained runs showed that spatial-only geometry is stable but weak, and low FFT is essentially inactive.

This run keeps the useful part:

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
```

But it changes the optimizer behavior:

```text
Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))
loss = -Z + 1.0 * MSE(perturbed, original)
```

Final artifacts are generated from the best valid checkpoint, not blindly from the final iteration.

Best valid checkpoint rule:

```text
iteration >= 10
input SSIM >= 0.97
input PSNR >= 26.0
combined max displacement <= 5.0 px
score = Z increase from initial
```

FFT phase is disabled. This is intentional. The previous low-FFT run was too inactive, and the broad FFT run caused invalid green/phase-corrupted inputs.

## Run on A6000

```bash
cd /home/interns/Desktop/layer
git pull origin main

mkdir -p logs

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.run_stage_b_smoke \
  --root /home/interns/Desktop/layer \
  --config /home/interns/Desktop/layer/identity_layers/configs/stage_b_preservation_checkpoint_250.json \
  --output-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_preservation_checkpoint_250 \
  2>&1 | tee logs/layer_stage_b_preservation_checkpoint_250.log

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.summarize_stage_b \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_preservation_checkpoint_250 \
  2>&1 | tee -a logs/layer_stage_b_preservation_checkpoint_250.log

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.build_report \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_preservation_checkpoint_250 \
  --output-root /home/interns/Desktop/layer/identity_layers/outputs/reports/stage_b_preservation_checkpoint_250 \
  2>&1 | tee -a logs/layer_stage_b_preservation_checkpoint_250.log
```

## Expected ETA

Expected runtime:

```text
8 runs x 250 iterations
roughly 6-12 minutes on the A6000
```

If the machine is busy or Hugging Face cache reloads are slow, give it a little more.

## Push after running

```bash
git status -sb
git add \
  identity_layers/outputs/stage_b_preservation_checkpoint_250 \
  identity_layers/outputs/reports/stage_b_preservation_checkpoint_250 \
  logs/layer_stage_b_preservation_checkpoint_250.log \
  logs.txt
git commit -m "Add Stage B preservation checkpoint outputs" || true
git push origin main
```

If `logs.txt` is missing or unchanged, remove it from `git add`.

## Files to inspect first

```text
identity_layers/outputs/stage_b_preservation_checkpoint_250/stage_b_top_sheet.jpg
identity_layers/outputs/stage_b_preservation_checkpoint_250/stage_b_decision_report.md
identity_layers/outputs/stage_b_preservation_checkpoint_250/stage_b_all_runs.csv
identity_layers/outputs/reports/stage_b_preservation_checkpoint_250/layer_stage_b_report.pdf
identity_layers/outputs/reports/stage_b_preservation_checkpoint_250/layer_stage_b_report.html
```

Inside each run folder, inspect:

```text
summary.json
best_metadata.json
comparison_sheet.png
history.csv
```

Useful fields:

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

Call a result interesting only if:

```text
final_source = best_valid_checkpoint
Z_increase > 0
input SSIM >= 0.97
perturbed input is visually acceptable
clean edit succeeds
perturbed edit visibly weakens/fails
```

If the perturbed edit still clearly performs the requested edit, call it weak or metric-only.

If this run only gives weak candidates, do not do another broad sweep immediately. The next step would be either:

```text
1. make a tiny Kamala-only refinement around the two cases that previously moved, or
2. revisit deeper ArcFace-correlated layers with a modified gradient probe.
```
