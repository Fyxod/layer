# Response 16 - Stage B spatial best-valid 400 output review

Checked latest push:

```text
e3ee00f Add Stage B spatial best-valid outputs
```

Output folder:

```text
identity_layers/outputs/stage_b_spatial_bestvalid_400/
```

Report folder:

```text
identity_layers/outputs/reports/stage_b_spatial_bestvalid_400/
```

## Completion state

The run completed:

```text
8 / 8 runs done
```

Report artifacts exist:

```text
identity_layers/outputs/reports/stage_b_spatial_bestvalid_400/layer_stage_b_report.pdf
identity_layers/outputs/reports/stage_b_spatial_bestvalid_400/layer_stage_b_report.html
identity_layers/outputs/reports/stage_b_spatial_bestvalid_400/layer_stage_b_report.md
```

No missing artifacts were reported.

## Main result

This run is better than the preservation run and confirms the earlier diagnosis:

```text
loss = -Z
FFT disabled
best-valid checkpoint selection enabled
```

works better than:

```text
loss = -Z + 1.0 * MSE(perturbed, original)
```

However, it still does not produce a convincing visible edit failure.

## Numeric summary

Across all 8 runs:

```text
mean Z_increase:              about +0.000527
max Z_increase:               about +0.002133
mean final input SSIM:        about 0.9960
minimum final input SSIM:     about 0.9831
mean output SSIM:             about 0.9920
minimum output SSIM:          about 0.9775
mean max displacement:        about 1.30 px
maximum displacement:         about 4.49 px
```

All final artifacts used:

```text
final_source = best_valid_checkpoint
```

So checkpoint selection worked.

## The only positive-Z cases

Only the Kamala cases moved:

### `kamala_harris_01 / add black sunglasses`

```text
Z increase:        about +0.002133
selected iter:     76
input SSIM:        about 0.9924
input PSNR:        about 29.69 dB
max displacement:  about 2.43 px
output SSIM:       about 0.9916
```

Visual read:

```text
clean edit:      sunglasses clearly added
perturbed edit:  sunglasses still clearly added
```

Conclusion:

```text
metric-only / weak, not a visible edit failure
```

### `kamala_harris_01 / add headphones`

```text
Z increase:        about +0.002084
selected iter:     223
input SSIM:        about 0.9831
input PSNR:        about 26.23 dB
max displacement:  about 4.49 px
output SSIM:       about 0.9775
```

History note:

The raw highest-Z point was around iteration 206:

```text
Z increase:        about +0.00215
input PSNR:        about 25.76 dB
valid checkpoint:  false
```

The selector correctly chose a slightly later/lower-Z checkpoint that passed the PSNR threshold.

Visual read:

```text
clean edit:      headphones visible but not extremely strong
perturbed edit:  headphones still visible / shifted / changed
```

Conclusion:

```text
weak candidate at best, not a clean visible failure
```

## Other six cases

The other six cases had:

```text
Z increase = 0
```

They mostly stayed at initialization-level perturbation:

```text
max displacement about 0.58 px
input SSIM about 0.9987-0.9990
```

These are not useful attack candidates.

## Conclusion

The best-valid checkpoint strategy is now working and should be kept.

The useful Stage B signal is real but narrow:

```text
unet.conv_in
Kamala image
especially headphones / sunglasses
```

But the current geometry/layer objective still produces only metric-level or weak visual changes. It does not yet produce a convincing edit failure.

## Recommendation

Do not run another broad 8-case Stage B sweep with the same setup.

If we want one final Stage B check, make it tiny and focused:

```text
image:
  kamala_harris_01

prompts:
  add headphones
  add black sunglasses

layer:
  unet.conv_in

loss:
  loss = -Z

FFT:
  disabled

checkpoint selection:
  enabled

iterations:
  600
```

Loosen only the existence threshold slightly for the headphone case:

```text
input SSIM >= 0.98
input PSNR >= 25.5
max displacement <= 5.0 px
```

But the more honest decision is:

```text
Stage B found a weak, image-specific signal but not a strong final-edit failure.
```

If the project needs the next materially different move, go back to the note from earlier:

```text
Stage B should start with vae_image_latent and unet.conv_in only. If those fail, then revisit ArcFace-correlated deeper layers with a modified gradient probe.
```

At this point, `unet.conv_in` has been tested enough to say it is not giving a clean visible failure under the current geometry family.
