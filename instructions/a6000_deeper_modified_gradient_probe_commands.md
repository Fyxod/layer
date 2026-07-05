# LAYER deeper modified gradient probe commands

This is the next diagnostic after the Stage B `unet.conv_in` runs.

## Why this exists

The Stage B runs show:

```text
unet.conv_in has usable gradient signal
but the final edit effect is weak / metric-level
```

Earlier baseline comparison also showed that some deeper layers had stronger ArcFace correlation:

```text
unet.mid_block
unet.up_blocks.0
```

However, the original A7 gradient sanity test used only:

```text
pool activation -> L2 normalize -> cosine distance
```

That can erase spatial/scale information in deeper UNet activations. This probe reruns A7 with modified objective variants before deciding whether deeper layers are genuinely unusable.

## Config

```text
identity_layers/configs/gradient_scan_deeper_modified.json
```

Layers:

```text
vae_image_latent
unet.conv_in
unet.mid_block
unet.up_blocks.0
```

Objective variants:

```text
pooled_cosine
mean_std_cosine
normalized_activation_mse
```

Prompts:

```text
add black sunglasses
add headphones
```

Images:

```text
barack_obama_01
joe_biden_01
michelle_obama_01
kamala_harris_01
```

This is still a short gradient probe:

```text
steps = 8
FFT disabled
spatial geometry only
```

## Run on A6000

```bash
cd /home/interns/Desktop/layer
git pull origin main

mkdir -p logs

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.test_layer_gradients \
  --root /home/interns/Desktop/layer \
  --config /home/interns/Desktop/layer/identity_layers/configs/gradient_scan_deeper_modified.json \
  --output-dir /home/interns/Desktop/layer/identity_layers/outputs/gradient_scan_deeper_modified \
  2>&1 | tee logs/layer_gradient_scan_deeper_modified.log
```

## Expected ETA

This is:

```text
4 layers x 3 objective variants x 2 prompts x 4 images x 9 forward/backward steps
```

Expected runtime:

```text
roughly 5-20 minutes on the A6000
```

If `normalized_activation_mse` is slower for deeper layers, let it finish unless it OOMs.

## Push after running

```bash
git status -sb
git add \
  identity_layers/outputs/gradient_scan_deeper_modified \
  logs/layer_gradient_scan_deeper_modified.log \
  logs.txt
git commit -m "Add deeper modified gradient probe outputs" || true
git push origin main
```

If `logs.txt` is missing or unchanged, remove it from `git add`.

## Files to inspect first

```text
identity_layers/outputs/gradient_scan_deeper_modified/layer_gradient_rankings.csv
identity_layers/outputs/gradient_scan_deeper_modified/gradient_sanity.csv
identity_layers/outputs/gradient_scan_deeper_modified/gradient_failures.csv
identity_layers/outputs/gradient_scan_deeper_modified/selected_layers.json
identity_layers/outputs/gradient_scan_deeper_modified/rejected_layers.json
identity_layers/outputs/gradient_scan_deeper_modified/short_step_Z_curves.png
identity_layers/outputs/gradient_scan_deeper_modified/gradient_norm_by_layer.png
```

## Decision rule

A deeper-layer objective is worth a small Stage B run only if it has:

```text
finite gradients
positive mean Z increase
nonzero total gradient norm
no obvious numerical failure
```

If `unet.mid_block` or `unet.up_blocks.0` still do not increase Z under any objective variant, then the deeper ArcFace-correlated layers are not useful with the current differentiable geometry family.

If one deeper layer/objective variant passes, run a tiny Stage B attack only for that specific pair. Do not broaden immediately.
