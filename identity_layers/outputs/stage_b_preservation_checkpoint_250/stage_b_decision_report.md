# Stage B targeted layer smoke report

This is a short layer-targeted geometry smoke, not a full attack sweep.

## Layers tested

- `unet.conv_in`

## Summary

- Runs completed: 8
- Objective: `Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))`
- Loss: `loss = -Z + input_preservation_weight * MSE(perturbed, original)`
- Input preservation weights observed: `1`
- Best-valid checkpoint selection: `enabled`
- Trainable values: geometry parameters only

## Top runs by Z increase

| layer | image | prompt | source | initial Z | final Z | Z increase | input SSIM | max disp px | output SSIM |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| unet.conv_in | michelle_obama_01 | add black sunglasses | best_valid_checkpoint | 0.000303149 | 5.43594e-05 | -0.00024879 | 0.9998 | 0.3995 | 0.9995 |
| unet.conv_in | michelle_obama_01 | add headphones | best_valid_checkpoint | 0.000303149 | 5.42402e-05 | -0.000248909 | 0.9998 | 0.3995 | 0.9954 |
| unet.conv_in | joe_biden_01 | add black sunglasses | best_valid_checkpoint | 0.000403523 | 0.000107169 | -0.000296354 | 0.9999 | 0.3346 | 0.9996 |
| unet.conv_in | joe_biden_01 | add headphones | best_valid_checkpoint | 0.000403523 | 0.000107169 | -0.000296354 | 0.9999 | 0.3346 | 0.9983 |
| unet.conv_in | barack_obama_01 | add headphones | best_valid_checkpoint | 0.000620842 | 0.000117421 | -0.000503421 | 0.9998 | 0.3696 | 0.9907 |
| unet.conv_in | barack_obama_01 | add black sunglasses | best_valid_checkpoint | 0.000620842 | 0.000117242 | -0.0005036 | 0.9998 | 0.3696 | 0.9994 |
| unet.conv_in | kamala_harris_01 | add headphones | best_valid_checkpoint | 0.000786662 | 0.000141621 | -0.000645041 | 0.9999 | 0.4585 | 0.9995 |
| unet.conv_in | kamala_harris_01 | add black sunglasses | best_valid_checkpoint | 0.000786662 | 0.000140309 | -0.000646353 | 0.9999 | 0.4585 | 0.9997 |

## Decision note

Use this run as targeted Stage-B evidence. Treat final images as candidates only if Z increases, input preservation remains visually acceptable, and final edited outputs show a real visible edit change rather than just metric movement.
