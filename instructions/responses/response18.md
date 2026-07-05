# Response 18: deeper modified gradient probe review

Checked the latest pushed outputs in:

`identity_layers/outputs/gradient_scan_deeper_modified/`

## What was run

The deeper modified gradient probe tested:

- layers:
  - `vae_image_latent`
  - `unet.conv_in`
  - `unet.mid_block`
  - `unet.up_blocks.0`
- objective variants:
  - `pooled_cosine`
  - `mean_std_cosine`
  - `normalized_activation_mse`
- prompts:
  - `add black sunglasses`
  - `add headphones`
- images:
  - `barack_obama_01`
  - `joe_biden_01`
  - `michelle_obama_01`
  - `kamala_harris_01`

There were no recorded failures.

## Main result

The deeper ArcFace-correlated layers did not become useful after the modified objectives.

`unet.mid_block` and `unet.up_blocks.0` stayed flat:

- `mean_Z_increase = 0`
- final gradient norm effectively `0`
- no useful short-step movement under `pooled_cosine`, `mean_std_cosine`, or `normalized_activation_mse`

So the earlier suspicion was tested: these deeper layers correlate with ArcFace in the passive layer scan, but they are not practically reachable by the current differentiable geometry objective.

## Selected rows

The scan selected only:

| layer | objective variant | prompt | mean Z increase | mean final Z | mean final grad norm |
|---|---|---:|---:|---:|---:|
| `unet.conv_in` | `pooled_cosine` | `add headphones` | `0.000223` | `0.000752` | `0.000164` |
| `unet.conv_in` | `pooled_cosine` | `add black sunglasses` | `0.000204` | `0.000733` | `0.000049` |

`vae_image_latent / pooled_cosine` also moved, but was weaker:

| layer | objective variant | prompt | mean Z increase |
|---|---|---:|---:|
| `vae_image_latent` | `pooled_cosine` | `add headphones` | `0.000182` |
| `vae_image_latent` | `pooled_cosine` | `add black sunglasses` | `0.000123` |

All `mean_std_cosine` and `normalized_activation_mse` variants were effectively dead for this probe.

## Important per-case detail

The positive signal is still narrow and mostly Kamala-specific.

Best individual short-step cases:

| layer | prompt | image | Z increase | final SSIM | final PSNR | max disp px |
|---|---|---|---:|---:|---:|---:|
| `unet.conv_in` | `add headphones` | `kamala_harris_01` | `0.000893` | `0.997856` | `35.21` | `1.08` |
| `unet.conv_in` | `add black sunglasses` | `kamala_harris_01` | `0.000817` | `0.997484` | `34.51` | `1.13` |
| `vae_image_latent` | `add headphones` | `kamala_harris_01` | `0.000730` | `0.998912` | `38.15` | `0.96` |
| `vae_image_latent` | `add black sunglasses` | `kamala_harris_01` | `0.000493` | `0.998326` | `36.28` | `1.28` |

The other identities were basically flat in the short probe.

## Interpretation

This confirms the current picture:

1. `unet.conv_in / pooled_cosine` is the only consistently usable gradient target.
2. `vae_image_latent / pooled_cosine` has a smaller signal, but previous longer runs did not turn it into a convincing final-edit failure.
3. The deeper layers (`unet.mid_block`, `unet.up_blocks.0`) should not be used for another expensive Stage B attack in the current form.
4. The modified objectives did not unlock a new route.

## Recommendation

Do not launch a larger run with `unet.mid_block` or `unet.up_blocks.0`.

The next useful step is not another broad optimization run. The evidence now says the current InstructPix2Pix geometry-only/layer-targeted route produces measurable internal movement, but not a convincing visible edit failure.

If we continue, it should be either:

1. a final consolidated report over Stage A + Stage B, or
2. a deliberately different objective design, not more iterations/layers of the same family.

