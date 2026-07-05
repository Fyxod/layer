# LAYER Stage B constrained follow-up commands

Use this after the broad 400-iteration run.

## Why this exists

The broad 400-iteration run showed that unconstrained `unet.conv_in` can maximize `Z`, but it does so by destroying the input:

```text
input SSIM collapsed as low as ~0.01
PSNR fell to ~8–11 dB
max displacement hit the 8 px hard cap
FFT phase saturated at ±pi
perturbed images became green/phase-corrupted
```

Those runs are not valid final attacks. They are diagnostic evidence that the current `loss = -Z` objective has a destructive loophole.

The next check is whether the effect survives under much tighter hard constraints.

## Main constrained run: spatial-only

This disables FFT phase entirely and tightens spatial ranges.

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
  200

constraints:
  max_combined_disp_px = 4.0
  fft_phase = disabled
  lower DCT/TPS/Delaunay/Rolling limits
```

Run:

```bash
cd /home/interns/Desktop/layer
git pull origin main

mkdir -p logs

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.run_stage_b_smoke \
  --root /home/interns/Desktop/layer \
  --config /home/interns/Desktop/layer/identity_layers/configs/stage_b_constrained_spatial_200.json \
  --output-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_constrained_spatial_200 \
  2>&1 | tee logs/layer_stage_b_constrained_spatial_200.log

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.summarize_stage_b \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_constrained_spatial_200 \
  2>&1 | tee -a logs/layer_stage_b_constrained_spatial_200.log
```

## Optional constrained run: tiny FFT

Only run this if the spatial-only run is too weak or if you want to test whether a very small FFT phase helps without causing green corruption.

This keeps FFT enabled but limits it to:

```text
fft_phase_limit_rad = 0.35
```

Run:

```bash
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.run_stage_b_smoke \
  --root /home/interns/Desktop/layer \
  --config /home/interns/Desktop/layer/identity_layers/configs/stage_b_constrained_low_fft_200.json \
  --output-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_constrained_low_fft_200 \
  2>&1 | tee logs/layer_stage_b_constrained_low_fft_200.log

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.summarize_stage_b \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_constrained_low_fft_200 \
  2>&1 | tee -a logs/layer_stage_b_constrained_low_fft_200.log
```

## Expected ETA

Spatial-only:

```text
8 runs × 200 iterations
approximately 5–10 minutes
```

Tiny-FFT optional:

```text
8 runs × 200 iterations
approximately 5–10 minutes
```

## Push after running

```bash
git status -sb
git add \
  identity_layers/outputs/stage_b_constrained_spatial_200 \
  identity_layers/outputs/stage_b_constrained_low_fft_200 \
  logs/layer_stage_b_constrained_spatial_200.log \
  logs/layer_stage_b_constrained_low_fft_200.log \
  logs.txt
git commit -m "Add Stage B constrained follow-up outputs"
git push origin main
```

If the optional low-FFT run was not run, remove those paths from `git add`.

If `logs.txt` is unchanged or missing, remove it from `git add`.

## Files to inspect first

```text
identity_layers/outputs/stage_b_constrained_spatial_200/stage_b_top_sheet.jpg
identity_layers/outputs/stage_b_constrained_spatial_200/stage_b_decision_report.md
identity_layers/outputs/stage_b_constrained_spatial_200/stage_b_all_runs.csv
```

Optional low-FFT:

```text
identity_layers/outputs/stage_b_constrained_low_fft_200/stage_b_top_sheet.jpg
identity_layers/outputs/stage_b_constrained_low_fft_200/stage_b_decision_report.md
identity_layers/outputs/stage_b_constrained_low_fft_200/stage_b_all_runs.csv
```

## Decision rule

Treat a candidate as interesting only if:

```text
input SSIM remains roughly >= 0.90
perturbed input remains visually recognizable
clean edit succeeds
perturbed edit visibly weakens/fails
the effect is not only a green/color-corruption artifact
```

If constrained runs do not preserve the effect, the current `unet.conv_in` layer objective is probably not enough by itself.

Then the next step should be either:

```text
1. add best-valid checkpoint selection / early stopping by input SSIM
2. add a tiny visual/identity-preservation constraint
3. revisit ArcFace-correlated deeper layers with a modified gradient probe
```
