# Stage B targeted layer smoke report

This is a short layer-targeted geometry smoke, not a full attack sweep.

## Layers tested

- `unet.conv_in`
- `unet.mid_block`
- `unet.up_blocks.0`
- `vae_image_latent`

## Summary

- Runs completed: 32
- Objective: `Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))`
- Loss: `loss = -Z`
- Trainable values: geometry parameters only

## Top runs by Z increase

| layer | image | prompt | initial Z | final Z | Z increase | input SSIM | max disp px | output SSIM |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| unet.conv_in | barack_obama_01 | add black sunglasses | 0.00166106 | 0.330863 | 0.329202 | 0.04973 | 8 | -0.004205 |
| unet.conv_in | barack_obama_01 | add headphones | 0.00166106 | 0.33081 | 0.329149 | 0.0161 | 8 | -0.05994 |
| unet.conv_in | kamala_harris_01 | add headphones | 0.00208163 | 0.304021 | 0.30194 | 0.01206 | 8 | -0.02564 |
| unet.conv_in | kamala_harris_01 | add black sunglasses | 0.00208163 | 0.302753 | 0.300671 | 0.08558 | 8 | -0.0164 |
| unet.conv_in | joe_biden_01 | add headphones | 0.00106478 | 0.220617 | 0.219552 | 0.09114 | 8 | 0.1068 |
| unet.conv_in | joe_biden_01 | add black sunglasses | 0.00106478 | 0.217374 | 0.216309 | 0.09515 | 8 | -0.3214 |
| vae_image_latent | joe_biden_01 | add headphones | 0.000706255 | 0.0119851 | 0.0112789 | 0.8164 | 7.957 | 0.8005 |
| vae_image_latent | joe_biden_01 | add black sunglasses | 0.000706255 | 0.00105798 | 0.000351727 | 0.9674 | 3.964 | 0.9733 |
| vae_image_latent | barack_obama_01 | add black sunglasses | 0.000736952 | 0.000794351 | 5.73993e-05 | 0.9547 | 2.308 | 0.9485 |
| vae_image_latent | barack_obama_01 | add headphones | 0.000736952 | 0.000794053 | 5.71012e-05 | 0.9547 | 2.308 | 0.9066 |
| unet.conv_in | michelle_obama_01 | add black sunglasses | 0.000913262 | 0.000913262 | 0 | 0.9776 | 1.921 | 0.9809 |
| unet.conv_in | michelle_obama_01 | add headphones | 0.000913262 | 0.000913262 | 0 | 0.9776 | 1.921 | 0.9681 |
| unet.mid_block | barack_obama_01 | add black sunglasses | 1.47223e-05 | 1.47223e-05 | 0 | 0.9768 | 1.921 | 0.9749 |
| unet.mid_block | joe_biden_01 | add black sunglasses | 1.52588e-05 | 1.52588e-05 | 0 | 0.9819 | 1.921 | 0.9781 |
| unet.mid_block | kamala_harris_01 | add black sunglasses | 1.96099e-05 | 1.96099e-05 | 0 | 0.9844 | 1.921 | 0.9859 |
| unet.mid_block | michelle_obama_01 | add black sunglasses | 1.15633e-05 | 1.15633e-05 | 0 | 0.9776 | 1.921 | 0.9809 |
| unet.mid_block | barack_obama_01 | add headphones | 1.4782e-05 | 1.4782e-05 | 0 | 0.9768 | 1.921 | 0.9311 |
| unet.mid_block | joe_biden_01 | add headphones | 1.53184e-05 | 1.53184e-05 | 0 | 0.9819 | 1.921 | 0.9622 |
| unet.mid_block | kamala_harris_01 | add headphones | 2.03252e-05 | 2.03252e-05 | 0 | 0.9844 | 1.921 | 0.9813 |
| unet.mid_block | michelle_obama_01 | add headphones | 1.15633e-05 | 1.15633e-05 | 0 | 0.9776 | 1.921 | 0.9681 |

## Decision note

Use this smoke only as Gate 4 evidence. Proceed to a 150-iteration run only if Z increases without numerical failure and the geometry diagnostics remain acceptable.
