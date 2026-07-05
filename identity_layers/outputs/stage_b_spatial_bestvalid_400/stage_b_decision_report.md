# Stage B targeted layer smoke report

This is a short layer-targeted geometry smoke, not a full attack sweep.

## Layers tested

- `unet.conv_in`

## Summary

- Runs completed: 8
- Objective: `Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))`
- Loss: `loss = -Z`
- Input preservation weights observed: `0`
- Best-valid checkpoint selection: `enabled`
- Trainable values: geometry parameters only

## Top runs by Z increase

| layer | image | prompt | source | initial Z | final Z | Z increase | input SSIM | max disp px | output SSIM |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| unet.conv_in | kamala_harris_01 | add black sunglasses | best_valid_checkpoint | 0.000786662 | 0.00291955 | 0.00213289 | 0.9924 | 2.427 | 0.9916 |
| unet.conv_in | kamala_harris_01 | add headphones | best_valid_checkpoint | 0.000786662 | 0.00287032 | 0.00208366 | 0.9831 | 4.488 | 0.9775 |
| unet.conv_in | barack_obama_01 | add black sunglasses | best_valid_checkpoint | 0.000620842 | 0.000620842 | 0 | 0.9987 | 0.5777 | 0.9974 |
| unet.conv_in | joe_biden_01 | add black sunglasses | best_valid_checkpoint | 0.000403523 | 0.000403523 | 0 | 0.999 | 0.5777 | 0.9939 |
| unet.conv_in | michelle_obama_01 | add black sunglasses | best_valid_checkpoint | 0.000303149 | 0.000303149 | 0 | 0.9987 | 0.5777 | 0.9976 |
| unet.conv_in | barack_obama_01 | add headphones | best_valid_checkpoint | 0.000620842 | 0.000620842 | 0 | 0.9987 | 0.5777 | 0.9957 |
| unet.conv_in | joe_biden_01 | add headphones | best_valid_checkpoint | 0.000403523 | 0.000403523 | 0 | 0.999 | 0.5777 | 0.9884 |
| unet.conv_in | michelle_obama_01 | add headphones | best_valid_checkpoint | 0.000303149 | 0.000303149 | 0 | 0.9987 | 0.5777 | 0.9943 |

## Decision note

Use this run as targeted Stage-B evidence. Treat final images as candidates only if Z increases, input preservation remains visually acceptable, and final edited outputs show a real visible edit change rather than just metric movement.
