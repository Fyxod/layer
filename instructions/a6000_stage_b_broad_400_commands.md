# LAYER Stage B broad 400-iteration commands

Use this for the controlled broader pass after the Stage-B smoke.

## Run matrix

```text
layers:
  unet.conv_in
  vae_image_latent
  unet.mid_block
  unet.up_blocks.0

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

Total:

```text
4 layers × 2 prompts × 4 images = 32 runs
```

Expected ETA from the previous 50-iteration smoke:

```text
approximately 25–40 minutes
```

This estimate includes final clean/perturbed InstructPix2Pix edit generation.

## Why these layers

Primary Stage-B targets:

```text
unet.conv_in
vae_image_latent
```

Diagnostic additions:

```text
unet.mid_block
unet.up_blocks.0
```

The diagnostic additions had better ArcFace pairwise-distance correlation, but weak A7 gradient movement. This broad run is a cheap controlled check, not a claim that those layers are good attack targets.

## Run

```bash
cd /home/interns/Desktop/layer
git pull origin main

mkdir -p logs

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.run_stage_b_smoke \
  --root /home/interns/Desktop/layer \
  --config /home/interns/Desktop/layer/identity_layers/configs/stage_b_broad_400.json \
  --output-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_broad_400 \
  2>&1 | tee logs/layer_stage_b_broad_400.log

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.summarize_stage_b \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_broad_400 \
  2>&1 | tee -a logs/layer_stage_b_broad_400.log
```

## Push after running

```bash
git status -sb
git add \
  identity_layers/outputs/stage_b_broad_400 \
  logs/layer_stage_b_broad_400.log \
  logs.txt
git commit -m "Add Stage B broad 400-iteration outputs"
git push origin main
```

If `logs.txt` is unchanged or missing, remove it from `git add`.

## Files to inspect first

```text
identity_layers/outputs/stage_b_broad_400/stage_b_summary.json
identity_layers/outputs/stage_b_broad_400/stage_b_all_runs.csv
identity_layers/outputs/stage_b_broad_400/stage_b_top_runs.csv
identity_layers/outputs/stage_b_broad_400/stage_b_decision_report.md
identity_layers/outputs/stage_b_broad_400/stage_b_top_sheet.jpg
```

Then inspect the top individual sheets under:

```text
identity_layers/outputs/stage_b_broad_400/runs/
```

## Decision rule

Look for cases where:

```text
clean edit succeeds
perturbed input remains recognizable enough
perturbed edit visibly weakens/fails
Z increases without severe foldover/clamp saturation
```

If the top runs only increase Z by making the input ugly or globally color-shifted, do not call them successes.
