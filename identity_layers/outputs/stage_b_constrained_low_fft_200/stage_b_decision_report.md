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
| unet.conv_in | barack_obama_01 | add black sunglasses | 0.000336707 | 0.000336707 | 0 | 0.9993 | 0.3955 | 0.9985 |
| unet.conv_in | joe_biden_01 | add black sunglasses | 0.000218511 | 0.000218511 | 0 | 0.9995 | 0.3955 | 0.9988 |
| unet.conv_in | kamala_harris_01 | add black sunglasses | 0.000403106 | 0.000403106 | 0 | 0.9997 | 0.3955 | 0.9992 |
| unet.conv_in | michelle_obama_01 | add black sunglasses | 0.000154138 | 0.000154138 | 0 | 0.9993 | 0.3955 | 0.9986 |
| unet.conv_in | barack_obama_01 | add headphones | 0.000336707 | 0.000336707 | 0 | 0.9993 | 0.3955 | 0.9755 |
| unet.conv_in | joe_biden_01 | add headphones | 0.000218511 | 0.000218511 | 0 | 0.9995 | 0.3955 | 0.9975 |
| unet.conv_in | kamala_harris_01 | add headphones | 0.000403106 | 0.000403106 | 0 | 0.9997 | 0.3955 | 0.9988 |
| unet.conv_in | michelle_obama_01 | add headphones | 0.000154138 | 0.000154138 | 0 | 0.9993 | 0.3955 | 0.9946 |

## Decision note

Use this smoke only as Gate 4 evidence. Proceed to a 150-iteration run only if Z increases without numerical failure and the geometry diagnostics remain acceptable.
