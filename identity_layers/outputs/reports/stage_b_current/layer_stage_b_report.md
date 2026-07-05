# LAYER: InstructPix2Pix Identity-Layer Geometry Results

Stage-B targeted layer optimization with Stage-A identity-layer context

Author: Parth Katiyar

## Method

`Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))`

`loss = -Z`

No visual counter-loss was used.

## Run matrix

| layers | prompts | images | runs | iterations | layer names |
| --- | --- | --- | --- | --- | --- |
| 4.0000 | 2.0000 | 4.0000 | 32.000 | 400 | unet.conv_in, unet.mid_block, unet.up_blocks.0, vae_image_latent |

## Aggregate results

| layer | prompt | runs | mean dZ | max dZ | mean input SSIM | invalid |
| --- | --- | --- | --- | --- | --- | --- |
| unet.conv_in | add black sunglasses | 4.0000 | 0.2115 | 0.3292 | 0.302 | 3.0000 |
| unet.conv_in | add headphones | 4.0000 | 0.2127 | 0.3291 | 0.2742 | 3.0000 |
| unet.mid_block | add black sunglasses | 4.0000 | 0 | 0 | 0.9802 | 0 |
| unet.mid_block | add headphones | 4.0000 | 0 | 0 | 0.9802 | 0 |
| unet.up_blocks.0 | add black sunglasses | 4.0000 | 0 | 0 | 0.9802 | 0 |
| unet.up_blocks.0 | add headphones | 4.0000 | 0 | 0 | 0.9802 | 0 |
| vae_image_latent | add black sunglasses | 4.0000 | 0.0001023 | 0.0003517 | 0.971 | 0 |
| vae_image_latent | add headphones | 4.0000 | 0.002834 | 0.01128 | 0.9333 | 1.0000 |

## Per-run final values

| validity | layer | prompt | image | dZ | input SSIM | output SSIM | max disp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| invalid_input_destroyed | unet.conv_in | add black sunglasses | barack_obama_01 | 0.3292 | 0.04973 | -0.004205 | 8.0000 |
| invalid_input_destroyed | unet.conv_in | add headphones | barack_obama_01 | 0.3291 | 0.0161 | -0.05994 | 8.0000 |
| invalid_input_destroyed | unet.conv_in | add headphones | kamala_harris_01 | 0.3019 | 0.01206 | -0.02564 | 8.0000 |
| invalid_input_destroyed | unet.conv_in | add black sunglasses | kamala_harris_01 | 0.3007 | 0.08558 | -0.0164 | 8.0000 |
| invalid_input_destroyed | unet.conv_in | add headphones | joe_biden_01 | 0.2196 | 0.09114 | 0.1068 | 8.0000 |
| invalid_input_destroyed | unet.conv_in | add black sunglasses | joe_biden_01 | 0.2163 | 0.09515 | -0.3214 | 8.0000 |
| invalid_fft_saturated | vae_image_latent | add headphones | joe_biden_01 | 0.01128 | 0.8164 | 0.8005 | 7.9569 |
| valid_or_visually_check | vae_image_latent | add black sunglasses | joe_biden_01 | 0.0003517 | 0.9674 | 0.9733 | 3.9636 |
| valid_or_visually_check | vae_image_latent | add black sunglasses | barack_obama_01 | 5.74e-05 | 0.9547 | 0.9485 | 2.3085 |
| valid_or_visually_check | vae_image_latent | add headphones | barack_obama_01 | 5.71e-05 | 0.9547 | 0.9066 | 2.3085 |
| valid_or_visually_check | unet.conv_in | add black sunglasses | michelle_obama_01 | 0 | 0.9776 | 0.9809 | 1.9212 |
| valid_or_visually_check | unet.conv_in | add headphones | michelle_obama_01 | 0 | 0.9776 | 0.9681 | 1.9212 |
| valid_or_visually_check | unet.mid_block | add black sunglasses | barack_obama_01 | 0 | 0.9768 | 0.9749 | 1.9212 |
| valid_or_visually_check | unet.mid_block | add black sunglasses | joe_biden_01 | 0 | 0.9819 | 0.9781 | 1.9212 |
| valid_or_visually_check | unet.mid_block | add black sunglasses | kamala_harris_01 | 0 | 0.9844 | 0.9859 | 1.9212 |
| valid_or_visually_check | unet.mid_block | add black sunglasses | michelle_obama_01 | 0 | 0.9776 | 0.9809 | 1.9212 |
| valid_or_visually_check | unet.mid_block | add headphones | barack_obama_01 | 0 | 0.9768 | 0.9311 | 1.9212 |
| valid_or_visually_check | unet.mid_block | add headphones | joe_biden_01 | 0 | 0.9819 | 0.9622 | 1.9212 |
| valid_or_visually_check | unet.mid_block | add headphones | kamala_harris_01 | 0 | 0.9844 | 0.9813 | 1.9212 |
| valid_or_visually_check | unet.mid_block | add headphones | michelle_obama_01 | 0 | 0.9776 | 0.9681 | 1.9212 |
| valid_or_visually_check | unet.up_blocks.0 | add black sunglasses | barack_obama_01 | 0 | 0.9768 | 0.9749 | 1.9212 |
| valid_or_visually_check | unet.up_blocks.0 | add black sunglasses | joe_biden_01 | 0 | 0.9819 | 0.9781 | 1.9212 |
| valid_or_visually_check | unet.up_blocks.0 | add black sunglasses | kamala_harris_01 | 0 | 0.9844 | 0.9859 | 1.9212 |
| valid_or_visually_check | unet.up_blocks.0 | add black sunglasses | michelle_obama_01 | 0 | 0.9776 | 0.9809 | 1.9212 |
| valid_or_visually_check | unet.up_blocks.0 | add headphones | barack_obama_01 | 0 | 0.9768 | 0.9311 | 1.9212 |
| valid_or_visually_check | unet.up_blocks.0 | add headphones | joe_biden_01 | 0 | 0.9819 | 0.9622 | 1.9212 |
| valid_or_visually_check | unet.up_blocks.0 | add headphones | kamala_harris_01 | 0 | 0.9844 | 0.9813 | 1.9212 |
| valid_or_visually_check | unet.up_blocks.0 | add headphones | michelle_obama_01 | 0 | 0.9776 | 0.9681 | 1.9212 |
| valid_or_visually_check | vae_image_latent | add black sunglasses | kamala_harris_01 | 0 | 0.9844 | 0.9859 | 1.9212 |
| valid_or_visually_check | vae_image_latent | add black sunglasses | michelle_obama_01 | 0 | 0.9776 | 0.9809 | 1.9212 |
| valid_or_visually_check | vae_image_latent | add headphones | kamala_harris_01 | 0 | 0.9844 | 0.9813 | 1.9212 |
| valid_or_visually_check | vae_image_latent | add headphones | michelle_obama_01 | 0 | 0.9776 | 0.9681 | 1.9212 |

## Image strips

### unet.conv_in / add black sunglasses / barack_obama_01

![strip](assets/strips/unet_conv_in_add_black_sunglasses_barack_obama_01.png)

### unet.conv_in / add headphones / barack_obama_01

![strip](assets/strips/unet_conv_in_add_headphones_barack_obama_01.png)

### unet.conv_in / add headphones / kamala_harris_01

![strip](assets/strips/unet_conv_in_add_headphones_kamala_harris_01.png)

### unet.conv_in / add black sunglasses / kamala_harris_01

![strip](assets/strips/unet_conv_in_add_black_sunglasses_kamala_harris_01.png)

### unet.conv_in / add headphones / joe_biden_01

![strip](assets/strips/unet_conv_in_add_headphones_joe_biden_01.png)

### unet.conv_in / add black sunglasses / joe_biden_01

![strip](assets/strips/unet_conv_in_add_black_sunglasses_joe_biden_01.png)

### vae_image_latent / add headphones / joe_biden_01

![strip](assets/strips/vae_image_latent_add_headphones_joe_biden_01.png)

### vae_image_latent / add black sunglasses / joe_biden_01

![strip](assets/strips/vae_image_latent_add_black_sunglasses_joe_biden_01.png)

### vae_image_latent / add black sunglasses / barack_obama_01

![strip](assets/strips/vae_image_latent_add_black_sunglasses_barack_obama_01.png)

### vae_image_latent / add headphones / barack_obama_01

![strip](assets/strips/vae_image_latent_add_headphones_barack_obama_01.png)

### unet.conv_in / add black sunglasses / michelle_obama_01

![strip](assets/strips/unet_conv_in_add_black_sunglasses_michelle_obama_01.png)

### unet.conv_in / add headphones / michelle_obama_01

![strip](assets/strips/unet_conv_in_add_headphones_michelle_obama_01.png)

### unet.mid_block / add black sunglasses / barack_obama_01

![strip](assets/strips/unet_mid_block_add_black_sunglasses_barack_obama_01.png)

### unet.mid_block / add black sunglasses / joe_biden_01

![strip](assets/strips/unet_mid_block_add_black_sunglasses_joe_biden_01.png)

### unet.mid_block / add black sunglasses / kamala_harris_01

![strip](assets/strips/unet_mid_block_add_black_sunglasses_kamala_harris_01.png)

### unet.mid_block / add black sunglasses / michelle_obama_01

![strip](assets/strips/unet_mid_block_add_black_sunglasses_michelle_obama_01.png)

### unet.mid_block / add headphones / barack_obama_01

![strip](assets/strips/unet_mid_block_add_headphones_barack_obama_01.png)

### unet.mid_block / add headphones / joe_biden_01

![strip](assets/strips/unet_mid_block_add_headphones_joe_biden_01.png)

### unet.mid_block / add headphones / kamala_harris_01

![strip](assets/strips/unet_mid_block_add_headphones_kamala_harris_01.png)

### unet.mid_block / add headphones / michelle_obama_01

![strip](assets/strips/unet_mid_block_add_headphones_michelle_obama_01.png)

### unet.up_blocks.0 / add black sunglasses / barack_obama_01

![strip](assets/strips/unet_up_blocks_0_add_black_sunglasses_barack_obama_01.png)

### unet.up_blocks.0 / add black sunglasses / joe_biden_01

![strip](assets/strips/unet_up_blocks_0_add_black_sunglasses_joe_biden_01.png)

### unet.up_blocks.0 / add black sunglasses / kamala_harris_01

![strip](assets/strips/unet_up_blocks_0_add_black_sunglasses_kamala_harris_01.png)

### unet.up_blocks.0 / add black sunglasses / michelle_obama_01

![strip](assets/strips/unet_up_blocks_0_add_black_sunglasses_michelle_obama_01.png)

### unet.up_blocks.0 / add headphones / barack_obama_01

![strip](assets/strips/unet_up_blocks_0_add_headphones_barack_obama_01.png)

### unet.up_blocks.0 / add headphones / joe_biden_01

![strip](assets/strips/unet_up_blocks_0_add_headphones_joe_biden_01.png)

### unet.up_blocks.0 / add headphones / kamala_harris_01

![strip](assets/strips/unet_up_blocks_0_add_headphones_kamala_harris_01.png)

### unet.up_blocks.0 / add headphones / michelle_obama_01

![strip](assets/strips/unet_up_blocks_0_add_headphones_michelle_obama_01.png)

### vae_image_latent / add black sunglasses / kamala_harris_01

![strip](assets/strips/vae_image_latent_add_black_sunglasses_kamala_harris_01.png)

### vae_image_latent / add black sunglasses / michelle_obama_01

![strip](assets/strips/vae_image_latent_add_black_sunglasses_michelle_obama_01.png)

### vae_image_latent / add headphones / kamala_harris_01

![strip](assets/strips/vae_image_latent_add_headphones_kamala_harris_01.png)

### vae_image_latent / add headphones / michelle_obama_01

![strip](assets/strips/vae_image_latent_add_headphones_michelle_obama_01.png)

## Graphs

### Cross-run graphs

#### Z vs iteration

![Z vs iteration](assets/graphs/z_vs_iteration.png)

#### Loss vs iteration

![Loss vs iteration](assets/graphs/loss_vs_iteration.png)

#### Input SSIM vs iteration

![Input SSIM vs iteration](assets/graphs/input_ssim_vs_iteration.png)

#### Input PSNR vs iteration

![Input PSNR vs iteration](assets/graphs/input_psnr_vs_iteration.png)

#### Combined max displacement vs iteration

![Combined max displacement vs iteration](assets/graphs/combined_max_disp_vs_iteration.png)

#### FFT spatial delta MSE vs iteration

![FFT spatial delta MSE vs iteration](assets/graphs/fft_delta_mse_vs_iteration.png)

#### Component magnitude vs iteration

![Component magnitude vs iteration](assets/graphs/component_magnitude_vs_iteration.png)

#### Z increase vs input SSIM

![Z increase vs input SSIM](assets/graphs/z_increase_vs_input_ssim.png)
