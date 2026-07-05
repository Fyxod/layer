# Response 12 - constrained Stage B reports and interpretation

## What was checked

Pulled latest `main` after commit `b409dd7 Add Stage B constrained follow-up outputs`.

Two new constrained Stage B output folders were present:

- `identity_layers/outputs/stage_b_constrained_spatial_200`
- `identity_layers/outputs/stage_b_constrained_low_fft_200`

Both completed 8/8 runs.

## Reports created

Generated full HTML, Markdown, and PDF reports for both constrained runs:

- `identity_layers/outputs/reports/stage_b_constrained_spatial_200/layer_stage_b_report.html`
- `identity_layers/outputs/reports/stage_b_constrained_spatial_200/layer_stage_b_report.md`
- `identity_layers/outputs/reports/stage_b_constrained_spatial_200/layer_stage_b_report.pdf`
- `identity_layers/outputs/reports/stage_b_constrained_low_fft_200/layer_stage_b_report.html`
- `identity_layers/outputs/reports/stage_b_constrained_low_fft_200/layer_stage_b_report.md`
- `identity_layers/outputs/reports/stage_b_constrained_low_fft_200/layer_stage_b_report.pdf`

The PDFs were rendered with Poppler for a quick visual sanity check. Each report is 7 pages and about 1.2 MB.

## extra_ignore

Added `extra_ignore/` to `.gitignore`.

Local ignored copies were placed in:

- `extra_ignore/2/stage_b_constrained_spatial_200_report`
- `extra_ignore/3/stage_b_constrained_low_fft_200_report`

These local folders are intentionally not tracked by Git.

## Main observations

### `stage_b_constrained_spatial_200`

This run avoids the previous green/destructive failure mode.

Summary:

- runs completed: 8/8
- invalid/destroyed inputs: 0
- mean input SSIM: about 0.9975
- worst input SSIM: about 0.9903
- max displacement: up to about 3.58 px
- mean output SSIM: about 0.9948
- lowest output SSIM: about 0.9898

Only two runs show meaningful Z increase:

1. `unet.conv_in / add black sunglasses / kamala_harris_01`
   - final Z about 0.00274
   - Z increase about 0.00223
   - input SSIM about 0.9903
   - output SSIM about 0.9898
   - max displacement about 3.58 px

2. `unet.conv_in / add headphones / kamala_harris_01`
   - final Z about 0.00256
   - Z increase about 0.00205
   - input SSIM about 0.9940
   - output SSIM about 0.9918
   - max displacement about 3.21 px

The other six runs are essentially neutral: final Z equals initial Z and max displacement stays near the tiny initialization level.

Interpretation: spatial-only constraints made the run stable, but the actual final-edit disruption is weak.

### `stage_b_constrained_low_fft_200`

This run is stable but basically inactive.

Summary:

- runs completed: 8/8
- invalid/destroyed inputs: 0
- Z increase: 0 for all runs
- mean input SSIM: about 0.99945
- max displacement: about 0.395 px for all runs
- mean output SSIM: about 0.9952

The lower FFT cap prevented the earlier green/destructive phase collapse, but it also removed almost all attack movement.

Interpretation: low-FFT is too constrained in this setup. It is safe but not useful as currently configured.

## Current conclusion

The broad 400-run result showed that unconstrained `unet.conv_in` can strongly affect the edit pipeline, but mostly by destroying the input.

The constrained follow-ups show:

- spatial-only constraints preserve the input but produce only weak edit disruption;
- low-FFT constraints preserve the input but produce no useful optimization movement.

So the useful signal is real but narrow: `unet.conv_in` remains the only layer with practical gradient signal, but the current objective needs a better way to allow meaningful edit disruption without collapsing into invalid image artifacts.

## Recommended next step

Do not spend more time on the current low-FFT config.

The next useful run should be a small preservation-aware `unet.conv_in` run:

- keep `unet.conv_in` only;
- keep the same 4 identities and 2 prompts;
- disable full FFT-phase saturation;
- use spatial components only or tiny FFT;
- add a small visual/input-preservation counter-loss or early stopping by input SSIM;
- select the best checkpoint by a combined rule, not simply final iteration:
  - input SSIM >= 0.97 or 0.98,
  - output SSIM as low as possible,
  - Z increased,
  - no obvious color collapse.

The most promising case to watch remains:

- `kamala_harris_01 / add headphones`
- `kamala_harris_01 / add black sunglasses`

These are the only constrained runs where Z actually increased while the input remained visually usable.
