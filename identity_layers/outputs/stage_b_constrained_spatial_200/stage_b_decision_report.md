# Stage B targeted layer smoke report

This is a short layer-targeted geometry smoke, not a full attack sweep.

## Layers tested

- `unet.conv_in`

## Summary

- Runs completed: 8
- Objective: `Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))`
- Loss: `loss = -Z`
- Trainable values: geometry parameters only

## Top runs by Z increase

| layer | image | prompt | initial Z | final Z | Z increase | input SSIM | max disp px | output SSIM |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| unet.conv_in | kamala_harris_01 | add black sunglasses | 0.000512123 | 0.0027433 | 0.00223118 | 0.9903 | 3.581 | 0.9898 |
| unet.conv_in | kamala_harris_01 | add headphones | 0.000512123 | 0.00255907 | 0.00204694 | 0.994 | 3.209 | 0.9918 |
| unet.conv_in | barack_obama_01 | add black sunglasses | 0.000404596 | 0.000404596 | 0 | 0.9992 | 0.4317 | 0.9983 |
| unet.conv_in | joe_biden_01 | add black sunglasses | 0.000263572 | 0.000263572 | 0 | 0.9994 | 0.4317 | 0.9985 |
| unet.conv_in | michelle_obama_01 | add black sunglasses | 0.000190496 | 0.000190496 | 0 | 0.9992 | 0.4317 | 0.9982 |
| unet.conv_in | barack_obama_01 | add headphones | 0.000404596 | 0.000404596 | 0 | 0.9992 | 0.4317 | 0.9942 |
| unet.conv_in | joe_biden_01 | add headphones | 0.000263572 | 0.000263572 | 0 | 0.9994 | 0.4317 | 0.9909 |
| unet.conv_in | michelle_obama_01 | add headphones | 0.000190496 | 0.000190496 | 0 | 0.9992 | 0.4317 | 0.9964 |

## Decision note

Use this smoke only as Gate 4 evidence. Proceed to a 150-iteration run only if Z increases without numerical failure and the geometry diagnostics remain acceptable.
