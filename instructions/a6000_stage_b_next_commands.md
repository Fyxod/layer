# LAYER Stage B next commands and implementation target

Current status:

- Stage A Milestone 1/A6/A7 is complete.
- ArcFace baseline comparison has been run.
- Do not rerun the broad activation scan unless the dataset/model setup changes.
- Do not start a large attack sweep yet.

## Stage A conclusion to carry forward

Recommended first Stage-B targets:

```text
vae_image_latent
unet.conv_in
```

Reason:

- These two layers had the strongest usable combination of:
  - identity separation in the Stage-A scan
  - finite nonzero geometry gradients in A7
  - short-step `Z` increase under the gradient sanity probe

Do not target the whole ranked layer list in the first Stage-B run.

## Important note for later

ArcFace correlation showed that some deeper layers had stronger pairwise-distance correlation with ArcFace than `vae_image_latent` / `unet.conv_in`, especially:

```text
unet.mid_block
unet.up_blocks.0
```

But these layers failed the current A7 practical gradient gate: finite rows, but zero useful short-probe gradient movement.

Remember this rule:

```text
Start Stage B with vae_image_latent and unet.conv_in only.
If those fail, revisit the ArcFace-correlated deeper layers with a modified gradient probe.
```

## Implemented Stage-B smoke files

```text
layer/identity_layers/stage_b_attack.py
layer/scripts/run_stage_b_smoke.py
layer/scripts/summarize_stage_b.py
identity_layers/configs/stage_b_smoke.json
```

Suggested outputs:

```text
identity_layers/outputs/stage_b_smoke/
  stage_b_all_runs.csv
  stage_b_summary.json
  stage_b_decision_report.md
  stage_b_top_sheet.jpg
```

## Stage-B smoke design

Use only:

```text
layers:
  - vae_image_latent
  - unet.conv_in

prompts:
  - add black sunglasses
  - add headphones

cases:
  use a small subset from the existing identity probe dataset
```

Keep it short:

```text
iterations: 25 to 50
starts: 1
execution: sequential
```

Use the same geometry-family style as A7/WOOD:

```text
TPS
Delaunay / piecewise
rolling shutter
DCT low-frequency warp
FFT phase
```

One backward pass should update all active geometry parameter blocks.

## Objective naming

Use the scalar name:

```text
Z
loss = -Z
```

For Stage B, `Z` should measure targeted mismatch at the selected frozen InstructPix2Pix layer.

The first smoke should compare:

```text
selected_layer(original_image)
vs
selected_layer(perturbed_image)
```

Do not include a visual counter-loss in the first smoke unless explicitly requested later.

## Hard exclusions

Still forbidden:

```text
pixel noise
adversarial patches
finetuning
LoRA
model-weight training
SPSA/CEM/black-box search
landmarks
face alignment / face detection
```

Only optimize differentiable geometry parameters.

## A6000 Stage-B smoke commands

Run this after pulling the implementation.

```bash
cd /home/interns/Desktop/layer
git pull origin main

mkdir -p logs

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.run_stage_b_smoke \
  --root /home/interns/Desktop/layer \
  --config /home/interns/Desktop/layer/identity_layers/configs/stage_b_smoke.json \
  --layers vae_image_latent unet.conv_in \
  --prompts "add black sunglasses" "add headphones" \
  --iters 50 \
  --max-cases 3 \
  --output-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_smoke \
  2>&1 | tee logs/layer_stage_b_smoke.log

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.summarize_stage_b \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_smoke \
  2>&1 | tee -a logs/layer_stage_b_smoke.log
```

Push after running:

```bash
git status -sb
git add identity_layers/outputs/stage_b_smoke logs/layer_stage_b_smoke.log logs.txt
git commit -m "Add Stage B targeted layer smoke outputs"
git push origin main
```

## What to inspect after pushing

```text
identity_layers/outputs/stage_b_smoke/stage_b_summary.json
identity_layers/outputs/stage_b_smoke/stage_b_all_runs.csv
identity_layers/outputs/stage_b_smoke/stage_b_top_runs.csv
identity_layers/outputs/stage_b_smoke/stage_b_decision_report.md
identity_layers/outputs/stage_b_smoke/stage_b_top_sheet.jpg
logs/layer_stage_b_smoke.log
```

## Decision gate

Only proceed to a 150-iteration main test if the smoke shows:

```text
Z increases
finite gradients
reasonable PSNR/SSIM
no severe foldover or clamp saturation
```

If both `vae_image_latent` and `unet.conv_in` fail this smoke, then revisit the ArcFace-correlated deeper layers with a modified gradient probe instead of forcing this objective family.
