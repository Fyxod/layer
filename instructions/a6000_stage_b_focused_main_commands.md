# LAYER Stage B focused main commands

Use this after inspecting the Stage-B smoke outputs.

## Smoke interpretation

The Stage-B smoke found one notable visual candidate:

```text
layer: unet.conv_in
prompt: add headphones
image_id: barack_obama_01
```

The clean edit clearly adds headphones. The perturbed edit is much weaker and does not clearly preserve the same headphone edit.

This is not yet a final success:

- the perturbed input is visibly color/geometry shifted
- input SSIM is only around `0.932`
- the run was only 50 iterations
- identity metrics have not yet been evaluated on the edited outputs

But it is enough to justify one focused 150-iteration main run.

## Do not run a broad sweep

Do not expand to all layers.

Do not target attention layers yet.

Do not run multi-layer or multi-timestep objectives yet.

First run the focused `unet.conv_in / add headphones` test.

## Focused main run

```bash
cd /home/interns/Desktop/layer
git pull origin main

mkdir -p logs

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.run_stage_b_smoke \
  --root /home/interns/Desktop/layer \
  --config /home/interns/Desktop/layer/identity_layers/configs/stage_b_smoke.json \
  --layers unet.conv_in \
  --prompts "add headphones" \
  --image-ids barack_obama_01 joe_biden_01 \
  --iters 150 \
  --output-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_main_unet_conv_in_headphones \
  2>&1 | tee logs/layer_stage_b_main_unet_conv_in_headphones.log

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.summarize_stage_b \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_main_unet_conv_in_headphones \
  2>&1 | tee -a logs/layer_stage_b_main_unet_conv_in_headphones.log
```

## Optional control

Only run this if the focused headphones run finishes cleanly and you have spare time.

```bash
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.run_stage_b_smoke \
  --root /home/interns/Desktop/layer \
  --config /home/interns/Desktop/layer/identity_layers/configs/stage_b_smoke.json \
  --layers unet.conv_in \
  --prompts "add black sunglasses" \
  --image-ids barack_obama_01 joe_biden_01 \
  --iters 150 \
  --output-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_control_unet_conv_in_sunglasses \
  2>&1 | tee logs/layer_stage_b_control_unet_conv_in_sunglasses.log

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.summarize_stage_b \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_control_unet_conv_in_sunglasses \
  2>&1 | tee -a logs/layer_stage_b_control_unet_conv_in_sunglasses.log
```

## Push after running

```bash
git status -sb
git add \
  identity_layers/outputs/stage_b_main_unet_conv_in_headphones \
  identity_layers/outputs/stage_b_control_unet_conv_in_sunglasses \
  logs/layer_stage_b_main_unet_conv_in_headphones.log \
  logs/layer_stage_b_control_unet_conv_in_sunglasses.log \
  logs.txt
git commit -m "Add Stage B focused unet conv-in main outputs"
git push origin main
```

If the optional control folder/log does not exist, remove those two paths from `git add`.

## Files to inspect first

```text
identity_layers/outputs/stage_b_main_unet_conv_in_headphones/stage_b_top_sheet.jpg
identity_layers/outputs/stage_b_main_unet_conv_in_headphones/stage_b_decision_report.md
identity_layers/outputs/stage_b_main_unet_conv_in_headphones/stage_b_all_runs.csv
identity_layers/outputs/stage_b_main_unet_conv_in_headphones/runs/unet_conv_in/add_headphones/barack_obama_01/comparison_sheet.png
```

## Decision rule

If the 150-iteration run strengthens the headphone failure while keeping the perturbed input acceptable, the next step is Stage-C evaluation:

```text
ArcFace / DeepFace identity metrics on:
  original vs perturbed
  clean edited vs perturbed edited

final edit metrics:
  clean edit vs perturbed edit
  input preservation
  geometry diagnostics
```

If it only increases distortion without strengthening the visible edit failure, do not keep deepening this exact objective blindly.
