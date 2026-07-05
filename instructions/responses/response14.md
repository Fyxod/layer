# Response 14 - Stage B preservation checkpoint output review

Checked latest push:

```text
48842ed Add Stage B preservation checkpoint outputs
```

Output folder:

```text
identity_layers/outputs/stage_b_preservation_checkpoint_250/
```

Report folder:

```text
identity_layers/outputs/reports/stage_b_preservation_checkpoint_250/
```

## Completion state

The run completed:

```text
8 / 8 runs done
```

All final artifacts were generated from:

```text
final_source = best_valid_checkpoint
```

So the best-valid checkpoint mechanism worked mechanically.

## Main result

This run is negative.

All 8 runs had negative `Z_increase`.

Mean values:

```text
mean Z_increase:           about -0.000424
mean final input SSIM:     about 0.99987
mean final input PSNR:     about 49.29 dB
mean output SSIM:          about 0.99777
mean max displacement:     about 0.39 px
```

The input was preserved extremely well, but the optimizer did not increase the target layer distance.

## Why this happened

For every run, the highest Z was at iteration 0.

Example pattern:

```text
iteration 0:
  Z is highest
  input SSIM already high
  displacement about 0.58 px

later iterations:
  MSE-to-original term pulls perturbation back toward identity
  displacement shrinks
  Z collapses toward zero
```

The preservation loss was too strong:

```text
loss = -Z + 1.0 * MSE(perturbed, original)
```

At this scale, the input-preservation gradient dominates the weak `unet.conv_in` Z gradient. The optimizer mostly learns to undo the initialization instead of increasing Z.

The checkpoint selector did what it was told:

```text
iteration >= 10
input SSIM >= 0.97
input PSNR >= 26
max displacement <= 5 px
pick highest Z increase
```

But because all valid checkpoints after iteration 10 had negative Z increase, it selected the least-negative checkpoint.

## Visual read

The image strips are clean but not useful as attacks.

The largest output-difference case was:

```text
unet.conv_in / add headphones / barack_obama_01
```

But visually:

```text
clean edit:      headphones visible
perturbed edit:  headphones still visible
```

So this is metric-only / weak, not a real visible edit failure.

## Logs note

`logs.txt` in the repo appears to contain older Stage A setup logs, not the latest Stage B preservation run log.

The pushed artifacts are enough to analyze the run, but the fresh `logs/layer_stage_b_preservation_checkpoint_250.log` was not visible in the tracked files, likely because `*.log` is ignored.

## Conclusion

Do not rerun this same preservation config.

The current result tells us:

```text
input_preservation_weight = 1.0 is too strong
```

It overcorrects and kills the only available Stage B signal.

## Recommended next step

Use checkpoint selection, but remove or greatly reduce the preservation counter-loss.

Best next config should be:

```text
layer:
  unet.conv_in

loss:
  loss = -Z

checkpoint selection:
  enabled
  input SSIM >= 0.97
  max displacement <= 5 px
  select highest Z increase

FFT:
  disabled

iterations:
  300 or 400
```

This is basically:

```text
constrained_spatial
+ best-valid checkpoint selection
+ longer run
+ no preservation MSE term
```

Reason:

The previous spatial-only constrained run actually increased Z for the Kamala cases while preserving the input:

```text
kamala_harris_01 / add black sunglasses:
  Z increase about +0.00223
  input SSIM about 0.990

kamala_harris_01 / add headphones:
  Z increase about +0.00205
  input SSIM about 0.994
```

The preservation run failed because it removed that movement.

If we want one more compact run, make it a no-preservation best-valid-checkpoint run. If that still only gives weak metric changes, then Stage B `unet.conv_in` is probably exhausted.
